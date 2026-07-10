"""분산 학습/체크포인트/로깅 유틸리티."""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 분산 초기화
# ---------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_master() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    """torchrun 환경변수로 프로세스 그룹 초기화. (rank, world_size, local_rank) 반환.

    분산 환경이 아니면 (0,1,0) 반환하고 아무것도 초기화하지 않습니다.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def rank0_print(*args, **kwargs) -> None:
    if is_master():
        print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# LR 스케줄: linear warmup + cosine decay to min_lr
# ---------------------------------------------------------------------------
def get_lr(step: int, warmup_steps: int, decay_steps: int,
           lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= decay_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# dtype 헬퍼
# ---------------------------------------------------------------------------
def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


# ---------------------------------------------------------------------------
# 처리량 계산 (MFU 추정용)
# ---------------------------------------------------------------------------
def estimate_mfu(model_params: int, tokens_per_step: int, dt: float,
                 n_layers: int, n_heads: int, head_dim: int, seq_len: int,
                 peak_flops: float) -> float:
    """대략적인 Model FLOPs Utilization 추정 (PaLM 방식 근사)."""
    # 6*N (fwd+bwd) per token + 어텐션 항
    flops_per_token = 6 * model_params + 12 * n_layers * n_heads * head_dim * seq_len
    flops_per_step = flops_per_token * tokens_per_step
    achieved = flops_per_step / dt
    return achieved / peak_flops
