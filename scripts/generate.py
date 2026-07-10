#!/usr/bin/env python
"""학습된 체크포인트로 텍스트 생성.

사용 예:
    python scripts/generate.py --ckpt runs/10b/ckpt_final.pt \
        --tokenizer tokenizer/ --prompt "옛날 옛적에" --max-new-tokens 200
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.config import ModelConfig  # noqa: E402
from src.model import Transformer  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(mcfg)
    # torch.compile / FSDP 접두사 제거
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), mcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true", help="KV 캐시 비활성화")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer(args.tokenizer)
    model, mcfg = load_model(args.ckpt, device)

    ids = tok.encode(args.prompt, bos=True, eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k,
                             use_cache=not args.no_cache, eos_id=tok.eos_id)

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
