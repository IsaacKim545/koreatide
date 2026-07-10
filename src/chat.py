"""대화(chat) 템플릿 + SFT 데이터셋.

대화 형식 (토큰):
    <s>
    <|system|> {system}   <|eot|>
    <|user|> {user}       <|eot|>
    <|assistant|> {reply} <|eot|>
    ...

- 추론: 마지막에 <|assistant|>까지 붙여(build_inference_ids) 모델이 응답을 생성.
- 학습(SFT): assistant 응답(+종료 <|eot|>) 토큰만 loss에 반영, 나머지는 -100 마스킹.

역할 토큰은 tokenizer의 특수토큰(<|system|> 등)으로, 각각 단일 id입니다.
"""
from __future__ import annotations

import json
from typing import List, Dict, Optional, Tuple

IGNORE_INDEX = -100


class ChatTemplate:
    def __init__(self, tokenizer, default_system: Optional[str] = None):
        assert tokenizer.has_chat_tokens, \
            "토크나이저에 대화 특수토큰이 없습니다. CHAT_TOKENS로 재학습이 필요합니다."
        self.tok = tokenizer
        self.default_system = default_system
        self.bos = tokenizer.bos_id
        self.eot = tokenizer.eot_id
        self.role_id = {
            "system": tokenizer.system_id,
            "user": tokenizer.user_id,
            "assistant": tokenizer.assistant_id,
        }

    # ------------------------------------------------------------------
    def _encode_text(self, text: str) -> List[int]:
        return self.tok.encode(text, bos=False, eos=False)

    def build_inference_ids(self, messages: List[Dict[str, str]]) -> List[int]:
        """messages=[{"role","content"}...] → 생성용 입력 ids (끝에 <|assistant|>)."""
        msgs = self._with_system(messages)
        ids: List[int] = [self.bos]
        for m in msgs:
            ids.append(self.role_id[m["role"]])
            ids.extend(self._encode_text(" " + m["content"].strip() + " "))
            ids.append(self.eot)
        ids.append(self.role_id["assistant"])   # 생성 프롬프트
        return ids

    def build_training_example(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """대화 → (token_ids, labels). assistant 응답+eot만 label, 나머지는 -100.

        labels[i]는 token_ids[i] 위치가 "예측 대상"인지로 채워지며(같은 값),
        데이터셋 collate에서 한 칸 shift하여 (input, target)으로 만듭니다.
        """
        msgs = self._with_system(messages)
        ids: List[int] = [self.bos]
        labels: List[int] = [IGNORE_INDEX]

        def add(tok_ids: List[int], target: bool):
            ids.extend(tok_ids)
            labels.extend(tok_ids if target else [IGNORE_INDEX] * len(tok_ids))

        for m in msgs:
            add([self.role_id[m["role"]]], target=False)     # 역할 마커는 마스킹
            content = self._encode_text(" " + m["content"].strip() + " ")
            is_asst = m["role"] == "assistant"
            add(content, target=is_asst)                     # assistant만 학습
            add([self.eot], target=is_asst)                  # 종료 토큰도 학습(멈춤 학습)
        return ids, labels

    def _with_system(self, messages):
        if self.default_system and (not messages or messages[0]["role"] != "system"):
            return [{"role": "system", "content": self.default_system}] + list(messages)
        return list(messages)


# ---------------------------------------------------------------------------
# SFT 데이터셋 (대화 jsonl → (input, label))
# ---------------------------------------------------------------------------
def read_conversations(path: str) -> List[List[Dict[str, str]]]:
    """jsonl 로드. 각 줄은 {"messages":[{"role","content"},...]} 또는 메시지 리스트."""
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj["messages"] if isinstance(obj, dict) and "messages" in obj else obj
            convs.append(msgs)
    return convs


class SFTDataset:
    """대화 예시를 (input_ids, labels)로 토큰화하는 map-style 데이터셋.

    torch가 필요하므로 지연 import. collate_fn과 함께 사용.
    """

    def __init__(self, path: str, template: ChatTemplate, max_len: int = 4096):
        self.template = template
        self.max_len = max_len
        self.convs = read_conversations(path)

    def __len__(self):
        return len(self.convs)

    def __getitem__(self, idx: int):
        import torch
        ids, labels = self.template.build_training_example(self.convs[idx])
        ids = ids[: self.max_len]
        labels = labels[: self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SFTCollator:
    """가변 길이 예시를 배치 max로 right-padding. input/target을 한 칸 shift.

    모듈 최상위 클래스이므로 pickle 가능 → Windows(spawn) DataLoader 워커와 호환.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        # 각 예시: (ids[L], labels[L]) → x=ids[:-1], y=labels[1:]
        xs, ys = [], []
        for ids, labels in batch:
            xs.append(ids[:-1])
            ys.append(labels[1:])
        maxlen = max(x.size(0) for x in xs)
        B = len(xs)
        X = torch.full((B, maxlen), self.pad_id, dtype=torch.long)
        Y = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            X[i, : x.size(0)] = x
            Y[i, : y.size(0)] = y
        return X, Y


def make_sft_collate(pad_id: int):
    """하위 호환: pickle 가능한 SFTCollator 인스턴스를 반환."""
    return SFTCollator(pad_id)
