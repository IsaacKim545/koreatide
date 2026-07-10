"""배치 추론 엔진 + 동적 배처.

- InferenceEngine.generate_batch: 서로 다른 길이의 프롬프트를 left-padding으로
  묶어 KV 캐시 + attention mask로 배치 생성. 시퀀스별 EOS 조기 종료.
- DynamicBatcher: 동시 요청을 짧은 시간창 동안 모아 한 번에 처리(throughput↑).

HTTP 서버는 scripts/serve.py 참고.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import List, Optional

import torch

from .config import ModelConfig
from .model import Transformer, KVCache


class InferenceEngine:
    def __init__(self, model: Transformer, tokenizer, device=None,
                 dtype: torch.dtype = torch.bfloat16):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.pad_id = getattr(tokenizer, "pad_id", 0) or 0
        self.eos_id = getattr(tokenizer, "eos_id", None)
        self.max_seq_len = model.cfg.max_seq_len

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128,
                       temperature: float = 0.8, top_k: Optional[int] = 50,
                       add_bos: bool = True) -> List[str]:
        """프롬프트 문자열 리스트 → 생성 문자열 리스트."""
        if not prompts:
            return []
        enc = [self.tok.encode(p, bos=add_bos, eos=False) for p in prompts]
        out_ids = self._generate_ids(enc, max_new_tokens, temperature, top_k)
        return [self.tok.decode(ids) for ids in out_ids]

    @torch.no_grad()
    def _generate_ids(self, enc: List[List[int]], max_new_tokens: int,
                      temperature: float, top_k: Optional[int]) -> List[List[int]]:
        B = len(enc)
        lengths = [len(e) for e in enc]
        L = max(lengths)
        L = min(L, self.max_seq_len - 1)  # 최소 1토큰 생성 여지

        # left-padding: 실제 토큰을 오른쪽으로 정렬
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long, device=self.device)
        pad_mask = torch.zeros((B, L), dtype=torch.bool, device=self.device)  # True=real
        for i, e in enumerate(enc):
            e = e[-L:]
            input_ids[i, L - len(e):] = torch.tensor(e, dtype=torch.long, device=self.device)
            pad_mask[i, L - len(e):] = True

        cache = KVCache(self.model.cfg.n_layers)

        # ---- prefill ----
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=self.device))
        key_real = pad_mask[:, None, None, :]                 # (B,1,1,L)
        mask = causal[None, None] & key_real                  # (B,1,L,L)
        # 모든 query 행이 최소 자기 자신은 보도록 (pad query NaN 방지; 출력은 버려짐)
        mask = mask | torch.eye(L, dtype=torch.bool, device=self.device)[None, None]

        with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                            enabled=self.device.type == "cuda"):
            logits, _ = self.model(input_ids, cache=cache, start_pos=0, attn_mask=mask)
        next_logits = logits[:, -1, :]                        # 마지막 위치=항상 real
        next_id = self.model._sample(next_logits, temperature, top_k)  # (B,1)

        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, 1, dtype=torch.bool, device=self.device)
        # 현재까지의 key 실존 마스크 (B, K). prefill 이후 K=L.
        key_real_flat = pad_mask.clone()

        for step in range(max_new_tokens):
            # 이번에 뽑은 토큰 기록 (완료된 시퀀스는 무시)
            for i in range(B):
                if not bool(finished[i]):
                    generated[i].append(int(next_id[i, 0]))
            if self.eos_id is not None:
                finished = finished | (next_id == self.eos_id)
            if bool(finished.all()):
                break
            pos = cache.seq_len
            if pos >= self.max_seq_len:
                break

            # decode: 새 토큰 1개. 새 key는 항상 real → mask에 True 추가.
            key_real_flat = torch.cat(
                [key_real_flat, torch.ones(B, 1, dtype=torch.bool, device=self.device)], dim=1)
            dec_mask = key_real_flat[:, None, None, :]        # (B,1,1,K+1)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype,
                                enabled=self.device.type == "cuda"):
                logits, _ = self.model(next_id, cache=cache, start_pos=pos, attn_mask=dec_mask)
            next_id = self.model._sample(logits[:, -1, :], temperature, top_k)

        # EOS에서 잘라내기
        results = []
        for i in range(B):
            ids = generated[i]
            if self.eos_id is not None and self.eos_id in ids:
                ids = ids[:ids.index(self.eos_id)]
            results.append(ids)
        return results


# ---------------------------------------------------------------------------
# 동적 배처: 동시 요청을 모아 한 번에 처리
# ---------------------------------------------------------------------------
@dataclass
class _Req:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: Optional[int]
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None


class DynamicBatcher:
    """백그라운드 워커가 큐에서 요청을 모아 engine.generate_batch로 처리.

    같은 배치로 묶으려면 생성 파라미터가 같아야 하므로, 파라미터별로 그룹화합니다.
    """

    def __init__(self, engine: InferenceEngine, max_batch: int = 16,
                 max_wait_ms: int = 20):
        self.engine = engine
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.q: "Queue[_Req]" = Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, prompt: str, max_new_tokens: int = 128,
               temperature: float = 0.8, top_k: Optional[int] = 50) -> str:
        req = _Req(prompt, max_new_tokens, temperature, top_k)
        self.q.put(req)
        req.event.wait()
        return req.result

    def _collect(self) -> List[_Req]:
        batch: List[_Req] = []
        try:
            batch.append(self.q.get(timeout=0.5))
        except Empty:
            return batch
        deadline = time.time() + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(self.q.get(timeout=remaining))
            except Empty:
                break
        return batch

    def _loop(self):
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            # 동일 파라미터끼리 그룹화
            groups: dict = {}
            for r in batch:
                key = (r.max_new_tokens, r.temperature, r.top_k)
                groups.setdefault(key, []).append(r)
            for (mnt, temp, tk), reqs in groups.items():
                try:
                    outs = self.engine.generate_batch(
                        [r.prompt for r in reqs], mnt, temp, tk)
                except Exception as e:  # noqa
                    outs = [f"[error] {e}"] * len(reqs)
                for r, o in zip(reqs, outs):
                    r.result = o
                    r.event.set()

    def stop(self):
        self._stop.set()


def load_engine(ckpt_path: str, tokenizer_path: str, device=None) -> InferenceEngine:
    """.pt 체크포인트(consolidated) + 토크나이저 → InferenceEngine."""
    from .tokenizer import Tokenizer
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(mcfg)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    tok = Tokenizer(tokenizer_path)
    return InferenceEngine(model, tok, device=device)
