"""FSDP 기반 학습 루프.

- FSDP full sharding + bf16 mixed precision
- TransformerBlock 단위 auto-wrap
- (선택) activation checkpointing
- gradient accumulation, grad clipping, warmup+cosine LR
- 분산 체크포인트 저장/재개 (sharded state dict)
"""
from __future__ import annotations

import functools
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,
)

# 분산 체크포인트(DCP) — 버전에 따라 없을 수 있어 가드
try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict, set_state_dict,
    )
    _DCP_OK = True
except Exception:  # noqa
    _DCP_OK = False

from .config import ModelConfig, TrainConfig
from .model import Transformer, TransformerBlock
from . import utils
from .utils import rank0_print


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 train_loader, train_sampler=None, val_loader=None):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.val_loader = val_loader

        self.rank = utils.get_rank()
        self.world_size = utils.get_world_size()
        self.local_rank = utils.get_local_rank()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.dtype = utils.resolve_dtype(train_cfg.dtype)

        torch.manual_seed(train_cfg.seed + self.rank)

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.step = 0

        # 파라미터 수 (MFU 추정에 사용)
        base = self.model.module if hasattr(self.model, "module") else self.model
        self._n_params = base.num_params() if isinstance(base, Transformer) \
            else model_cfg.estimate_params()

        self._init_wandb()

        if train_cfg.resume:
            self._load_checkpoint(train_cfg.resume)

    # ------------------------------------------------------------------
    def _init_wandb(self):
        """rank0에서만 wandb 초기화. 미설치/미설정 시 조용히 건너뜀."""
        self._wandb = None
        if not self.tcfg.wandb_project or not utils.is_master():
            return
        try:
            import wandb
        except ImportError:
            rank0_print("[wandb] 미설치 — 로깅 비활성화 (pip install wandb)")
            return
        cfg = {**self.mcfg.__dict__, **self.tcfg.__dict__,
               "world_size": self.world_size, "n_params": self._n_params}
        wandb.init(project=self.tcfg.wandb_project, name=self.tcfg.wandb_run, config=cfg)
        self._wandb = wandb
        rank0_print(f"[wandb] 로깅 활성화: project={self.tcfg.wandb_project}")

    def _wandb_log(self, metrics: dict, step: int):
        """wandb + 로컬 metrics.jsonl 동시 기록 (rank0)."""
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if utils.is_master():
            import json
            os.makedirs(self.tcfg.out_dir, exist_ok=True)
            rec = {"step": step, **metrics}
            with open(os.path.join(self.tcfg.out_dir, "metrics.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------
    def _build_model(self):
        # meta device에 모델을 만들면 대형 모델 초기화 시 CPU RAM 절약 가능하나,
        # 여기서는 명확성을 위해 일반 초기화 후 FSDP로 샤딩합니다.
        model = Transformer(self.mcfg)
        rank0_print(f"모델 파라미터: {model.num_params()/1e9:.3f}B "
                    f"(non-emb {model.num_params(True)/1e9:.3f}B)")

        # SFT 등: FSDP 래핑 전에 사전학습 가중치만 로드
        if self.tcfg.init_from:
            if os.path.exists(self.tcfg.init_from):
                ckpt = torch.load(self.tcfg.init_from, map_location="cpu")
                sd = ckpt.get("model", ckpt)
                sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                      for k, v in sd.items()}
                missing, unexpected = model.load_state_dict(sd, strict=False)
                rank0_print(f"init_from 로드: {self.tcfg.init_from} "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                rank0_print(f"[경고] init_from 경로 없음: {self.tcfg.init_from}")

        if not torch.cuda.is_available() or self.world_size == 1 and "RANK" not in os.environ:
            # 단일 프로세스(디버깅): FSDP 없이 그대로 사용
            model = model.to(self.device)
            if self.tcfg.activation_checkpointing:
                self._apply_act_ckpt(model)
            self._maybe_compile(model)
            self._fsdp = False
            return model

        # ---- FSDP 래핑 ----
        mp = MixedPrecision(
            param_dtype=self.dtype,
            reduce_dtype=self.dtype,
            buffer_dtype=self.dtype,
        )
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        strategy = {
            "full": ShardingStrategy.FULL_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
            "none": ShardingStrategy.NO_SHARD,
        }[self.tcfg.fsdp_sharding]

        cpu_offload = None
        if self.tcfg.cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp,
            sharding_strategy=strategy,
            device_id=self.local_rank,
            cpu_offload=cpu_offload,
            use_orig_params=True,      # torch.compile 및 optimizer 그룹 호환
            limit_all_gathers=True,
        )
        if self.tcfg.activation_checkpointing:
            self._apply_act_ckpt(model)
        self._maybe_compile(model)
        self._fsdp = True
        return model

    def _apply_act_ckpt(self, model):
        wrapper = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=lambda m: isinstance(m, TransformerBlock),
        )

    def _maybe_compile(self, model):
        if self.tcfg.compile and hasattr(torch, "compile"):
            try:
                # in-place로 forward를 감쌈
                model.forward = torch.compile(model.forward)  # type: ignore
                rank0_print("torch.compile 적용됨")
            except Exception as e:  # noqa
                rank0_print(f"torch.compile 실패, 무시: {e}")

    def _build_optimizer(self):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # FSDP(use_orig_params=True)에서도 model.parameters()로 그룹 구성 가능
        base = self.model.module if hasattr(self.model, "module") else self.model
        if isinstance(base, Transformer):
            return base.configure_optimizer(
                self.tcfg.lr, self.tcfg.weight_decay,
                (self.tcfg.beta1, self.tcfg.beta2), device_type)
        # FSDP로 감싼 경우: 파라미터 그룹을 직접 구성
        decay, no_decay = [], []
        for _, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        try:
            return torch.optim.AdamW(
                groups, lr=self.tcfg.lr,
                betas=(self.tcfg.beta1, self.tcfg.beta2),
                fused=(device_type == "cuda"))
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(
                groups, lr=self.tcfg.lr, betas=(self.tcfg.beta1, self.tcfg.beta2))

    # ------------------------------------------------------------------
    def _set_lr(self, step: int) -> float:
        lr = utils.get_lr(step, self.tcfg.warmup_steps, self.tcfg.lr_decay_steps,
                          self.tcfg.lr, self.tcfg.min_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def _autocast(self):
        if self.dtype == torch.float32 or not torch.cuda.is_available():
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    # ------------------------------------------------------------------
    def train(self):
        self.model.train()
        loader_iter = iter(self.train_loader)
        t0 = time.time()
        tokens_per_step = (self.tcfg.micro_batch_size * self.tcfg.grad_accum_steps
                           * self.world_size * self.tcfg.seq_len)

        while self.step < self.tcfg.max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.step)

            lr = self._set_lr(self.step)
            self.optimizer.zero_grad(set_to_none=True)

            loss_accum = 0.0
            for micro in range(self.tcfg.grad_accum_steps):
                try:
                    x, y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    x, y = next(loader_iter)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with self._autocast():
                    _, loss = self.model(x, y)
                    loss = loss / self.tcfg.grad_accum_steps
                loss.backward()
                loss_accum += loss.item()

            # grad clipping
            if self._fsdp:
                grad_norm = self.model.clip_grad_norm_(self.tcfg.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip)

            self.optimizer.step()
            self.step += 1

            # 로깅
            if self.step % self.tcfg.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.time() - t0
                t0 = time.time()
                tok_per_sec = tokens_per_step * self.tcfg.log_every / dt
                mfu = self._mfu(tokens_per_step, dt / self.tcfg.log_every)
                rank0_print(
                    f"step {self.step:>6} | loss {loss_accum:.4f} | "
                    f"lr {lr:.2e} | gnorm {float(grad_norm):.2f} | "
                    f"{tok_per_sec/1e3:.1f}k tok/s | mfu {mfu*100:.1f}%")
                self._wandb_log({
                    "train/loss": loss_accum,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "perf/tokens_per_sec": tok_per_sec,
                    "perf/mfu": mfu,
                    "train/tokens": self.step * tokens_per_step,
                }, self.step)

            # 평가
            if self.val_loader and self.step % self.tcfg.eval_every == 0:
                self.evaluate()
                self.model.train()

            # 저장
            if self.step % self.tcfg.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint(final=True)
        if self._wandb is not None:
            self._wandb.finish()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = []
        it = iter(self.val_loader)
        for _ in range(self.tcfg.eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast():
                _, loss = self.model(x, y)
            losses.append(loss.item())
        mean = sum(losses) / max(1, len(losses))
        if utils.is_dist():
            t = torch.tensor(mean, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            mean = t.item()
        ppl = torch.exp(torch.tensor(mean)).item()
        rank0_print(f"  [eval] step {self.step} | val_loss {mean:.4f} | ppl {ppl:.2f}")
        self._wandb_log({"val/loss": mean, "val/ppl": ppl}, self.step)
        return mean

    # ------------------------------------------------------------------
    def _mfu(self, tokens_per_step: int, dt_per_step: float) -> float:
        """Model FLOPs Utilization 추정. GPU peak FLOPS는 감지 실패 시 0 반환."""
        peak = self._peak_flops()
        if peak <= 0 or dt_per_step <= 0:
            return 0.0
        return utils.estimate_mfu(
            self._n_params, tokens_per_step, dt_per_step,
            self.mcfg.n_layers, self.mcfg.n_heads, self.mcfg.head_dim,
            self.tcfg.seq_len, peak * self.world_size)

    def _peak_flops(self) -> float:
        """GPU별 대략적 bf16 peak FLOPS (알려진 모델만; 그 외 0)."""
        if not torch.cuda.is_available():
            return 0.0
        name = torch.cuda.get_device_name(self.local_rank).upper()
        # bf16/TF32 텐서코어 대략치 (제조사 스펙, sparsity 제외)
        table = {"H100": 989e12, "A100": 312e12, "A800": 312e12,
                 "H800": 989e12, "V100": 125e12, "L40": 181e12, "4090": 165e12}
        for key, val in table.items():
            if key in name:
                return val
        return 0.0

    # ------------------------------------------------------------------
    # 체크포인트
    #
    # FSDP: torch.distributed.checkpoint(DCP)로 sharded 저장 → 각 rank가 자기
    #       샤드를 병렬로 씀. 모델+옵티마이저+step을 모두 담아 학습 재개 가능.
    #       저장 단위는 디렉터리(out/ckpt_{tag}/). meta.json에 step/model_cfg.
    #       final 시엔 추론/배포용 consolidated full state dict(.pt)도 추가 생성.
    # 비FSDP: 단일 .pt 파일에 model+optimizer+step.
    # ------------------------------------------------------------------
    def _meta_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, "meta.json")

    def save_checkpoint(self, final: bool = False):
        import json
        out = self.tcfg.out_dir
        os.makedirs(out, exist_ok=True)
        tag = "final" if final else f"step{self.step}"

        if self._fsdp and _DCP_OK:
            ckpt_dir = os.path.join(out, f"ckpt_{tag}")
            if utils.is_master():
                os.makedirs(ckpt_dir, exist_ok=True)
            if utils.is_dist():
                dist.barrier()
            # sharded state dict (distributed-aware)
            model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=ckpt_dir)
            if utils.is_master():
                meta = {"step": self.step, "model_cfg": self.mcfg.__dict__,
                        "data_state": self._loader_state()}
                with open(self._meta_path(ckpt_dir), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            rank0_print(f"sharded 체크포인트 저장: {ckpt_dir}")

            if final:
                self._save_consolidated(os.path.join(out, "ckpt_final_full.pt"))
        else:
            path = os.path.join(out, f"ckpt_{tag}.pt")
            if utils.is_master():
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": self.step,
                    "model_cfg": self.mcfg.__dict__,
                    "data_state": self._loader_state(),
                }, path)
            rank0_print(f"체크포인트 저장: {path}")

    def _loader_state(self):
        """ResumableLoader면 위치 상태를 반환, 아니면 None."""
        if hasattr(self.train_loader, "state_dict"):
            try:
                return self.train_loader.state_dict()
            except Exception:  # noqa
                return None
        return None

    def _restore_loader(self, data_state):
        if data_state and hasattr(self.train_loader, "load_state_dict"):
            try:
                self.train_loader.load_state_dict(data_state)
                rank0_print(f"데이터로더 위치 복원: pos={data_state.get('pos')}")
            except Exception as e:  # noqa
                rank0_print(f"[경고] 데이터로더 위치 복원 실패: {e}")

    def _save_consolidated(self, path: str):
        """rank0에 full state dict를 모아 단일 .pt로 저장 (추론/배포용)."""
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            model_sd = self.model.state_dict()
        if utils.is_master():
            torch.save({"model": model_sd, "step": self.step,
                        "model_cfg": self.mcfg.__dict__}, path)
            rank0_print(f"consolidated 모델 저장: {path}")

    def _load_checkpoint(self, path: str):
        import json
        if not os.path.exists(path):
            rank0_print(f"[경고] resume 경로 없음: {path}")
            return

        if os.path.isdir(path) and self._fsdp and _DCP_OK:
            # sharded 재개: 현재 구조로 템플릿을 만든 뒤 in-place 로드
            model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
            state = {"model": model_sd, "optim": optim_sd}
            dcp.load(state, checkpoint_id=path)
            set_state_dict(self.model, self.optimizer,
                           model_state_dict=state["model"],
                           optim_state_dict=state["optim"])
            meta_p = self._meta_path(path)
            if os.path.exists(meta_p):
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.step = meta.get("step", 0)
                self._restore_loader(meta.get("data_state"))
            rank0_print(f"sharded 재개: {path} (step {self.step})")
        else:
            # 단일 .pt 재개 (비FSDP 또는 consolidated)
            ckpt = torch.load(path, map_location="cpu")
            base = self.model
            base.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt and not self._fsdp:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except Exception as e:  # noqa
                    rank0_print(f"[경고] 옵티마이저 상태 복원 실패: {e}")
            self.step = ckpt.get("step", 0)
            self._restore_loader(ckpt.get("data_state"))
            rank0_print(f"재개: {path} (step {self.step})")
