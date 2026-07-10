#!/usr/bin/env python
"""코퍼스를 토큰화하여 .bin memmap으로 직렬화.

사용 예:
    python scripts/prepare_data.py --input data/corpus.jsonl \
        --tokenizer tokenizer/ --out data/train.bin
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import Tokenizer  # noqa: E402
from src.data import tokenize_to_bin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer 디렉터리 또는 tokenizer.json")
    ap.add_argument("--out", required=True, help="출력 .bin 경로")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--no-eos", action="store_true", help="문서 끝 EOS를 넣지 않음")
    args = ap.parse_args()

    tok = Tokenizer(args.tokenizer)
    print(f"토크나이저 로드 완료 (vocab={tok.vocab_size})")
    tokenize_to_bin(
        args.input, tok, args.out,
        text_key=args.text_key,
        append_eos=not args.no_eos,
    )


if __name__ == "__main__":
    main()
