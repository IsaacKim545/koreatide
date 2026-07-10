"""모델 및 학습 설정 데이터클래스.

JSON 파일과 상호 변환 가능하며, CLI에서 일부 필드를 override 할 수 있습니다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Optional


@dataclass
class ModelConfig:
    # 어휘/시퀀스
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # 트랜스포머 크기
    dim: int = 4096            # hidden size
    n_layers: int = 48
    n_heads: int = 32          # query heads
    n_kv_heads: int = 8        # GQA: key/value heads (n_heads의 약수)
    intermediate_size: int = 14336  # SwiGLU FFN 중간 차원

    # 정규화 / 위치인코딩
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    # dropout (사전학습에서는 보통 0)
    dropout: float = 0.0

    # 임베딩/헤드
    tie_embeddings: bool = False   # lm_head를 임베딩과 공유할지

    # 초기화
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        assert self.dim % self.n_heads == 0, "dim은 n_heads로 나누어져야 합니다."
        assert self.n_heads % self.n_kv_heads == 0, "n_heads는 n_kv_heads의 배수여야 합니다."

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def estimate_params(self) -> int:
        """대략적인 파라미터 수 추정."""
        v, d, L = self.vocab_size, self.dim, self.n_layers
        h, kv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        inter = self.intermediate_size
        emb = v * d
        # attention: q(d*d) + k(d*kv*hd) + v(d*kv*hd) + o(d*d)
        attn = d * d + d * (kv * hd) + d * (kv * hd) + d * d
        mlp = 3 * d * inter          # gate, up, down
        norms = 2 * d                 # 두 개의 RMSNorm (weight만)
        per_layer = attn + mlp + norms
        head = 0 if self.tie_embeddings else v * d
        final_norm = d
        return emb + L * per_layer + final_norm + head


@dataclass
class TrainConfig:
    # 데이터
    data_path: str = "data/train.bin"
    val_path: Optional[str] = None
    seq_len: int = 4096

    # 배치
    micro_batch_size: int = 1          # GPU당 마이크로배치
    grad_accum_steps: int = 8          # 유효 배치 = micro * accum * world_size
    max_steps: int = 100_000

    # 옵티마이저 (AdamW)
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # 스케줄 (warmup + cosine decay)
    warmup_steps: int = 2000
    lr_decay_steps: int = 100_000      # 보통 max_steps와 동일

    # 정밀도 / 메모리
    dtype: str = "bfloat16"            # bfloat16 | float16 | float32
    activation_checkpointing: bool = True
    compile: bool = True               # torch.compile

    # FSDP
    fsdp_sharding: str = "full"        # full | hybrid | none
    cpu_offload: bool = False

    # 체크포인트 / 로깅
    out_dir: str = "runs/default"
    save_every: int = 2000
    eval_every: int = 2000
    eval_iters: int = 100
    log_every: int = 10
    seed: int = 1337

    # 재개 / 초기화
    resume: Optional[str] = None       # 체크포인트 경로 (모델+옵티마이저+step 복원)
    init_from: Optional[str] = None    # 사전학습 가중치만 로드 (SFT 시작점, 옵티마이저 초기화)

    # wandb (선택)
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None


def _filter_kwargs(cls, d: dict) -> dict:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


def load_configs(path: str) -> tuple[ModelConfig, TrainConfig]:
    """JSON 파일에서 {"model": {...}, "train": {...}} 를 읽어 두 config를 반환."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    model = ModelConfig(**_filter_kwargs(ModelConfig, raw.get("model", {})))
    train = TrainConfig(**_filter_kwargs(TrainConfig, raw.get("train", {})))
    return model, train


def dump_configs(model: ModelConfig, train: TrainConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": asdict(model), "train": asdict(train)}, f, indent=2, ensure_ascii=False)


def apply_overrides(cfg: Any, overrides: dict) -> None:
    """CLI override(dict)를 config에 in-place 적용. None 값은 무시."""
    valid = {f.name for f in fields(cfg)}
    for k, v in overrides.items():
        if v is None:
            continue
        if k in valid:
            setattr(cfg, k, v)
