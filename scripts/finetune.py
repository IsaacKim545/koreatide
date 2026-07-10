#!/usr/bin/env python
"""대화형 AI 만들기 — SFT(지도 미세조정) 진입점.

사전학습된 베이스 모델을 대화 데이터로 미세조정합니다. assistant 응답 토큰에만
loss가 걸리도록 마스킹하며(chat 템플릿), 나머지는 사전학습 트레이너를 그대로 재사용합니다.

데이터 형식 (jsonl, 각 줄):
    {"messages": [{"role":"system","content":"..."},
                  {"role":"user","content":"..."},
                  {"role":"assistant","content":"..."}, ...]}

사용 예 (단일 GPU 디버깅):
    python scripts/finetune.py --config configs/small.json \
        --sft-data data/chat.jsonl --tokenizer tokenizer/ \
        --init-from runs/small/ckpt_final_full.pt --out runs/small-sft --max-steps 50

멀티 GPU:
    torchrun --nproc_per_node=8 scripts/finetune.py --config configs/10b.json \
        --sft-data data/chat.jsonl --tokenizer tokenizer/ \
        --init-from runs/10b/ckpt_final_full.pt --out runs/10b-sft
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.config import load_configs, apply_overrides, dump_configs  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402
from src.chat import ChatTemplate, SFTDataset, make_sft_collate  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src import utils  # noqa: E402
from src.utils import rank0_print  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sft-data", required=True, help="대화 jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--init-from", default=None, help="사전학습 베이스 .pt (consolidated)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--system", default="You are a helpful assistant.",
                    help="기본 시스템 프롬프트")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Windows/spawn 환경에서는 0 권장")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    rank, world_size, local_rank = utils.setup_distributed()

    model_cfg, train_cfg = load_configs(args.config)
    # SFT 기본값: 더 낮은 lr, 짧은 warmup (설정 파일 값 위에 덮어씀)
    apply_overrides(train_cfg, {
        "out_dir": args.out,
        "max_steps": args.max_steps,
        "resume": args.resume,
        "init_from": args.init_from,
        "wandb_project": args.wandb_project,
    })
    if args.no_compile:
        train_cfg.compile = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = Tokenizer(args.tokenizer)
    if not tok.has_chat_tokens:
        rank0_print("[오류] 토크나이저에 대화 특수토큰(<|user|> 등)이 없습니다.\n"
                    "       CHAT_TOKENS 포함해 토크나이저를 재학습하세요 "
                    "(train_tokenizer.py는 자동 포함).")
        utils.cleanup_distributed()
        sys.exit(1)

    template = ChatTemplate(tok, default_system=args.system)
    ds = SFTDataset(args.sft_data, template, max_len=train_cfg.seq_len)
    rank0_print(f"SFT 예시 수: {len(ds)}")

    collate = make_sft_collate(pad_id=tok.pad_id)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=True, seed=train_cfg.seed, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=train_cfg.micro_batch_size, sampler=sampler,
        shuffle=(sampler is None), collate_fn=collate,
        num_workers=args.num_workers, drop_last=True,
        persistent_workers=args.num_workers > 0)

    if utils.is_master():
        os.makedirs(train_cfg.out_dir, exist_ok=True)
        dump_configs(model_cfg, train_cfg, os.path.join(train_cfg.out_dir, "config.json"))

    trainer = Trainer(model_cfg, train_cfg, train_loader, sampler, None)
    try:
        trainer.train()
    finally:
        utils.cleanup_distributed()


if __name__ == "__main__":
    main()
