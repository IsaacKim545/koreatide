#!/usr/bin/env python
"""대화형 AI — 인터랙티브 채팅 REPL.

SFT로 미세조정한 모델과 대화합니다. 대화 이력을 유지하며 chat 템플릿으로
매 턴 입력을 구성하고, KV 캐시로 응답을 생성합니다(<|eot|>에서 정지).

사용 예:
    python scripts/chat.py --ckpt runs/small-sft/ckpt_final_full.pt --tokenizer tokenizer/

명령: /reset 대화 초기화 · /system <텍스트> 시스템 프롬프트 변경 · /exit 종료
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.config import ModelConfig  # noqa: E402
from src.model import Transformer  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402
from src.chat import ChatTemplate  # noqa: E402


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(mcfg)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), mcfg


def trim_history(template, messages, max_input_len):
    """입력 토큰이 너무 길면 오래된 user/assistant 쌍부터 제거 (system은 유지)."""
    while True:
        ids = template.build_inference_ids(messages)
        if len(ids) <= max_input_len:
            return ids
        # system 다음의 가장 오래된 non-system 메시지 제거
        drop = next((i for i, m in enumerate(messages) if m["role"] != "system"), None)
        if drop is None or len(messages) <= 1:
            return ids[-max_input_len:]
        messages.pop(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer(args.tokenizer)
    if not tok.has_chat_tokens:
        print("[오류] 토크나이저에 대화 특수토큰이 없습니다. SFT용 토크나이저인지 확인하세요.")
        sys.exit(1)
    model, mcfg = load_model(args.ckpt, device)
    template = ChatTemplate(tok, default_system=args.system)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    messages = []
    print("대화를 시작하세요. (/reset, /system <텍스트>, /exit)\n")

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = []
            print("[대화 초기화됨]\n")
            continue
        if user.startswith("/system "):
            template.default_system = user[len("/system "):].strip()
            messages = []
            print("[시스템 프롬프트 변경 · 대화 초기화]\n")
            continue

        messages.append({"role": "user", "content": user})
        max_input = mcfg.max_seq_len - args.max_new_tokens - 4
        ids = trim_history(template, messages, max_input)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=device.type == "cuda"):
            out = model.generate(x, args.max_new_tokens, args.temperature, args.top_k,
                                 use_cache=True, eos_id=tok.eot_id)
        gen = out[0, len(ids):].tolist()
        # 정지 토큰에서 자르기
        for stop in (tok.eot_id, tok.eos_id):
            if stop is not None and stop in gen:
                gen = gen[:gen.index(stop)]
        reply = tok.decode(gen).strip()
        print(f"AI > {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
