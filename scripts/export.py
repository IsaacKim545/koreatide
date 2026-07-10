#!/usr/bin/env python
"""체크포인트 후처리: torch.compile/FSDP 접두사 정리 후 단일 state_dict 저장.

사용 예:
    python scripts/export.py --ckpt runs/10b/ckpt_final.pt --out runs/10b/model.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def clean_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cleaned = clean_state_dict(ckpt["model"])
    payload = {"model": cleaned, "model_cfg": ckpt.get("model_cfg", {}),
               "step": ckpt.get("step", 0)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(payload, args.out)
    n = sum(v.numel() for v in cleaned.values())
    print(f"저장 완료: {args.out} ({n/1e9:.3f}B params)")


if __name__ == "__main__":
    main()
