"""Llama 계열 decoder-only 트랜스포머.

구성: RMSNorm (pre-norm), RoPE, Grouped-Query Attention, SwiGLU FFN.
어텐션은 torch.nn.functional.scaled_dot_product_attention 을 사용하여
가능한 경우 FlashAttention 커널로 실행됩니다. 추론 시 KV 캐시를 지원합니다.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 정규화는 fp32로 계산 후 원래 dtype으로 복귀 (수치 안정성)
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float,
                          device=None, dtype=torch.float32):
    """RoPE용 cos/sin 캐시를 미리 계산. shape: (max_seq_len, head_dim)."""
    assert head_dim % 2 == 0, "head_dim은 짝수여야 합니다."
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                  # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """q, k: (B, n_heads, T, head_dim). cos/sin: (T, head_dim) — 현재 위치 슬라이스."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1,1,T,hd)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out.type_as(q), k_out.type_as(k)


class KVCache:
    """레이어별 K/V 텐서를 보관하는 단순 캐시 (증분 디코딩용)."""

    def __init__(self, n_layers: int):
        self.k: list[Optional[torch.Tensor]] = [None] * n_layers
        self.v: list[Optional[torch.Tensor]] = [None] * n_layers

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        # k, v: (B, n_kv_heads, T, hd). 시간축(2)으로 이어붙임.
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None, layer_idx: int = 0,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """cos/sin은 현재 토큰들의 절대 위치에 해당하는 슬라이스여야 합니다.

        cache가 주어지면 새 k/v를 캐시에 이어붙이고 전체 과거 k/v로 어텐션합니다.
        attn_mask(bool, True=참여)가 주어지면 causal 대신 이 마스크를 사용합니다
        (패딩 배치 추론용). shape는 (B,1,T_q,T_k)로 브로드캐스트 가능해야 합니다.
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 새 토큰에만 RoPE 적용 (과거 k는 캐시에 이미 적용된 채 저장됨)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        # GQA: kv 헤드를 query 헤드 수만큼 반복
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        k_len = k.shape[2]
        if attn_mask is not None:
            # 명시적 마스크 사용 (패딩/커스텀). is_causal과 병용 불가.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        elif T == k_len:
            # 학습 또는 prefill: 표준 causal
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            # 증분 디코딩(T < k_len): 새 쿼리는 모든 과거 키를 볼 수 있음 → 마스크 불필요
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w_down = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None, layer_idx: int = 0,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer_idx, attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE 캐시 (버퍼로 저장, state_dict에는 포함하지 않음)
        cos, sin = precompute_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 잔차 투영에 대한 스케일 초기화 (GPT-2 방식)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None, start_pos: int = 0,
                attn_mask: Optional[torch.Tensor] = None):
        """idx: (B, T) long. targets: (B, T) long 또는 None.

        cache/start_pos는 증분 디코딩용. start_pos는 idx의 첫 토큰의 절대 위치.
        attn_mask는 패딩 배치 추론용(bool, True=참여, (B,1,T_q,T_k)).
        반환: targets가 주어지면 (logits, loss), 아니면 (logits, None).
        """
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, \
            f"위치 {start_pos + T} > max_seq_len {self.cfg.max_seq_len}"
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        x = self.drop(self.tok_emb(idx))
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i, attn_mask)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        else:
            logits = self.lm_head(x)
            return logits, None

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def configure_optimizer(self, lr, weight_decay, betas, device_type="cuda"):
        """decay/no-decay 파라미터 그룹을 나누어 AdamW 생성."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:      # 행렬(가중치)만 weight decay
                decay.append(p)
            else:                 # bias, RMSNorm weight 등
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 use_cache: bool = True, eos_id: Optional[int] = None) -> torch.Tensor:
        """KV 캐시 기반 자기회귀 생성.

        use_cache=False로 두면 캐시 없이(매 스텝 전체 재계산) 동작하며,
        greedy에서는 캐시 버전과 동일한 결과를 내야 합니다(동등성 검증용).
        """
        self.eval()
        B = idx.shape[0]

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.max_seq_len:]
                logits, _ = self(idx_cond)
                next_id = self._sample(logits[:, -1, :], temperature, top_k)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

        cache = KVCache(self.cfg.n_layers)
        # prefill: 프롬프트 전체를 한 번에 처리
        logits, _ = self(idx, cache=cache, start_pos=0)
        next_id = self._sample(logits[:, -1, :], temperature, top_k)
        idx = torch.cat((idx, next_id), dim=1)

        finished = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens - 1):
            pos = cache.seq_len              # 다음 토큰의 절대 위치
            if pos >= self.cfg.max_seq_len:
                break
            logits, _ = self(next_id, cache=cache, start_pos=pos)
            next_id = self._sample(logits[:, -1, :], temperature, top_k)
            if eos_id is not None:
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
                finished = finished | (next_id == eos_id)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return idx
