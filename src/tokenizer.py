"""BPE 토크나이저 학습/로드 래퍼 (HuggingFace `tokenizers` 기반).

특수 토큰: <unk>, <s> (BOS), </s> (EOS), <pad>.
학습된 토크나이저는 디렉터리에 tokenizer.json 으로 저장됩니다.
"""
from __future__ import annotations

import os
from typing import Iterable, Iterator, List, Optional

# 기본 특수토큰 + 대화(chat) 역할 토큰.
# 역할 토큰을 special_tokens로 학습해두면 BPE가 이를 단일 토큰으로 취급합니다.
CHAT_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|eot|>"]
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"] + CHAT_TOKENS


def train_bpe(files: List[str], vocab_size: int, out_dir: str,
              min_frequency: int = 2) -> str:
    """텍스트 파일 목록으로 BPE 토크나이저를 학습하고 저장. 저장 경로 반환."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files, trainer)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(path)
    return path


def train_bpe_from_iterator(iterator: Iterable[str], vocab_size: int, out_dir: str,
                            min_frequency: int = 2) -> str:
    """문자열 이터레이터로부터 학습 (대용량 jsonl 스트리밍용)."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(path)
    return path


class Tokenizer:
    """학습된 tokenizer.json 을 감싸는 얇은 래퍼."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer as HFTokenizer
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        self._tok = HFTokenizer.from_file(path)
        self.bos_id = self._tok.token_to_id("<s>")
        self.eos_id = self._tok.token_to_id("</s>")
        self.pad_id = self._tok.token_to_id("<pad>")
        self.unk_id = self._tok.token_to_id("<unk>")
        # 대화 역할 토큰 id (미학습 토크나이저면 None)
        self.system_id = self._tok.token_to_id("<|system|>")
        self.user_id = self._tok.token_to_id("<|user|>")
        self.assistant_id = self._tok.token_to_id("<|assistant|>")
        self.eot_id = self._tok.token_to_id("<|eot|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str):
        return self._tok.token_to_id(token)

    @property
    def has_chat_tokens(self) -> bool:
        return None not in (self.system_id, self.user_id, self.assistant_id, self.eot_id)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = self._tok.encode(text).ids
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)
