#!/usr/bin/env python
"""BPE 토크나이저 학습.

사용 예:
    python scripts/train_tokenizer.py --input data/corpus.jsonl \
        --vocab-size 32000 --out tokenizer/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import train_bpe_from_iterator  # noqa: E402
from src.data import iter_texts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl 또는 txt 코퍼스")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out", default="tokenizer/")
    args = ap.parse_args()

    print(f"토크나이저 학습 시작: {args.input} (vocab={args.vocab_size})")
    path = train_bpe_from_iterator(
        iter_texts(args.input, args.text_key),
        vocab_size=args.vocab_size,
        out_dir=args.out,
        min_frequency=args.min_frequency,
    )
    print(f"저장 완료: {path}")


if __name__ == "__main__":
    main()
