#!/usr/bin/env python
"""학습 진입점.

단일 GPU/CPU (디버깅):
    python scripts/train.py --config configs/small.json --data data/train.bin --max-steps 20

멀티 GPU (단일 노드):
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/10b.json --data data/train.bin --out runs/10b
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.config import load_configs, apply_overrides, dump_configs  # noqa: E402
from src.data import make_dataloader, ResumableLoader  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src import utils  # noqa: E402
from src.utils import rank0_print  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help="train .bin (config override)")
    ap.add_argument("--val", default=None, help="validation .bin")
    ap.add_argument("--out", default=None, help="출력 디렉터리 (config override)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--wandb-project", default=None, help="설정 시 W&B 로깅 활성화")
    ap.add_argument("--wandb-run", default=None, help="W&B run 이름")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader 워커 수 (Windows/spawn에서는 0 권장)")
    ap.add_argument("--data-mode", default="chunked", choices=["chunked", "random"],
                    help="chunked=비겹침 청크+셔플(권장), random=무작위 윈도우")
    ap.add_argument("--loader", default="resumable", choices=["resumable", "dataloader"],
                    help="resumable=정확한 위치 복원(권장), dataloader=torch DataLoader")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    rank, world_size, local_rank = utils.setup_distributed()

    model_cfg, train_cfg = load_configs(args.config)
    apply_overrides(train_cfg, {
        "data_path": args.data,
        "val_path": args.val,
        "out_dir": args.out,
        "max_steps": args.max_steps,
        "resume": args.resume,
        "wandb_project": args.wandb_project,
        "wandb_run": args.wandb_run,
    })
    if args.no_compile:
        train_cfg.compile = False

    # TF32 허용 (Ampere+): matmul 가속
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rank0_print("=" * 60)
    rank0_print(f"world_size={world_size} | device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    rank0_print(f"추정 파라미터: {model_cfg.estimate_params()/1e9:.3f}B")
    rank0_print(f"seq_len={train_cfg.seq_len} micro_bs={train_cfg.micro_batch_size} "
                f"accum={train_cfg.grad_accum_steps}")
    eff = (train_cfg.micro_batch_size * train_cfg.grad_accum_steps
           * world_size * train_cfg.seq_len)
    rank0_print(f"유효 배치: {eff:,} tokens/step")
    rank0_print("=" * 60)

    distributed = world_size > 1
    if args.loader == "resumable" and args.data_mode == "chunked":
        train_loader = ResumableLoader(
            train_cfg.data_path, train_cfg.seq_len, train_cfg.micro_batch_size,
            world_size=world_size, rank=rank, seed=train_cfg.seed)
        train_sampler = None
        rank0_print("데이터로더: ResumableLoader (정확한 위치 복원)")
    else:
        train_loader, train_sampler = make_dataloader(
            train_cfg.data_path, train_cfg.seq_len, train_cfg.micro_batch_size,
            distributed=distributed, seed=train_cfg.seed, num_workers=args.num_workers,
            mode=args.data_mode)

    val_loader = None
    if train_cfg.val_path:
        val_loader, _ = make_dataloader(
            train_cfg.val_path, train_cfg.seq_len, train_cfg.micro_batch_size,
            distributed=distributed, seed=train_cfg.seed, num_workers=1,
            mode="chunked")

    if utils.is_master():
        os.makedirs(train_cfg.out_dir, exist_ok=True)
        dump_configs(model_cfg, train_cfg, os.path.join(train_cfg.out_dir, "config.json"))

    trainer = Trainer(model_cfg, train_cfg, train_loader, train_sampler, val_loader)
    try:
        trainer.train()
    finally:
        if hasattr(train_loader, "stop"):
            train_loader.stop()
        utils.cleanup_distributed()


if __name__ == "__main__":
    main()
