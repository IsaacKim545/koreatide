#!/usr/bin/env python
"""엔드투엔드 스모크 테스트 (torch 필요, GPU 불필요).

소형 config로 모델을 만들고 forward/backward가 도는지, loss가 감소하는지,
generate가 동작하는지 확인합니다.

    python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from src.config import ModelConfig  # noqa: E402
from src.model import Transformer  # noqa: E402


def main():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, max_seq_len=64, dim=128, n_layers=2,
                      n_heads=4, n_kv_heads=2, intermediate_size=256,
                      tie_embeddings=True)
    model = Transformer(cfg)
    print(f"파라미터: {model.num_params()/1e6:.2f}M / 추정 {cfg.estimate_params()/1e6:.2f}M")

    B, T = 2, 32
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))

    # forward
    logits, loss = model(x, y)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape
    print(f"forward OK: logits {tuple(logits.shape)}, loss {loss.item():.4f}")

    # 몇 스텝 학습 → loss 감소 확인 (단일 배치 과적합)
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "cpu")
    first = loss.item()
    for _ in range(30):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    last = loss.item()
    print(f"overfit: {first:.4f} → {last:.4f}")
    assert last < first, "loss가 감소하지 않음"

    # generate
    out = model.generate(x[:, :4], max_new_tokens=8, temperature=0.8, top_k=10)
    assert out.shape == (B, 12), out.shape
    print(f"generate OK: {tuple(out.shape)}")

    # KV 캐시 동등성: greedy(temp=0)에서 캐시 유/무 결과가 같아야 함
    torch.manual_seed(0)
    a = model.generate(x[:, :4], max_new_tokens=10, temperature=0.0, use_cache=True)
    torch.manual_seed(0)
    b = model.generate(x[:, :4], max_new_tokens=10, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "KV 캐시 결과가 비캐시와 불일치"
    print(f"KV 캐시 동등성 OK: {tuple(a.shape)}")

    # 배치 추론(left-pad + mask) 동등성:
    # 서로 다른 길이 프롬프트를 배치로 greedy 생성한 결과가
    # 각각 단독 greedy 생성과 같아야 함 (패딩/마스크 정확성).
    from src.serve import InferenceEngine

    class _Tok:  # 최소 토크나이저 스텁 (id 리스트를 그대로 사용)
        vocab_size = cfg.vocab_size
        pad_id, eos_id, bos_id = 0, None, 1
        def encode(self, t, bos=False, eos=False): return t
        def decode(self, ids): return ids

    eng = InferenceEngine(model, _Tok(), device=torch.device("cpu"), dtype=torch.float32)
    p_long = list(range(5, 15))     # 길이 10
    p_short = list(range(20, 24))   # 길이 4
    batched = eng._generate_ids([p_long, p_short], max_new_tokens=6,
                                temperature=0.0, top_k=None)
    single_long = eng._generate_ids([p_long], 6, 0.0, None)[0]
    single_short = eng._generate_ids([p_short], 6, 0.0, None)[0]
    assert batched[0] == single_long, (batched[0], single_long)
    assert batched[1] == single_short, (batched[1], single_short)
    print(f"배치 추론 동등성 OK: long={batched[0][:3]}... short={batched[1][:3]}...")

    print("\n스모크 테스트 통과 ✅")


if __name__ == "__main__":
    main()
