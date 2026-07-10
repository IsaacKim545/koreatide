"""데이터 파이프라인.

1) 코퍼스(jsonl/txt)를 토큰화하여 단일 uint 배열(.bin memmap)로 직렬화.
2) 학습 시 memmap을 seq_len 청크로 샘플링하는 dataset.

- .bin 은 문서 사이에 EOS를 넣어 이어붙인 토큰 스트림입니다.
- 토큰 dtype은 vocab_size에 따라 uint16(<65536) 또는 uint32 자동 선택.

데이터셋 종류:
- ChunkedDataset  : 비겹침 청크 + DistributedSampler 셔플/샤딩 (권장, 기본)
- RandomWindowDataset : 무작위 오프셋 윈도우 (옵션, 옛 방식)
"""
from __future__ import annotations

import json
import os
import threading
from queue import Queue
from typing import Iterator, Optional

import numpy as np
import torch


def _dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size < 2 ** 16 else np.uint32


def iter_texts(path: str, text_key: str = "text") -> Iterator[str]:
    """jsonl(각 줄 {"text": ...}) 또는 .txt(줄 단위) 스트리밍."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj[text_key] if isinstance(obj, dict) else str(obj)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.rstrip("\n")


def tokenize_to_bin(input_path: str, tokenizer, out_path: str,
                    text_key: str = "text", append_eos: bool = True,
                    report_every: int = 10000) -> int:
    """코퍼스를 토큰화해 out_path(.bin)로 직렬화. 총 토큰 수 반환.

    메모리에 다 올리지 않도록 청크 단위로 파일에 append 합니다.
    """
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total = 0
    n_docs = 0
    buf = []
    BUF_LIMIT = 1_000_000  # 토큰 버퍼 크기

    with open(out_path, "wb") as fout:
        for text in iter_texts(input_path, text_key):
            ids = tokenizer.encode(text, bos=False, eos=append_eos)
            buf.extend(ids)
            n_docs += 1
            if len(buf) >= BUF_LIMIT:
                np.asarray(buf, dtype=dtype).tofile(fout)
                total += len(buf)
                buf = []
            if n_docs % report_every == 0:
                print(f"  {n_docs} docs, {total + len(buf)} tokens...", flush=True)
        if buf:
            np.asarray(buf, dtype=dtype).tofile(fout)
            total += len(buf)

    meta = {"total_tokens": total, "dtype": np.dtype(dtype).name,
            "vocab_size": tokenizer.vocab_size}
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"완료: {n_docs} docs, {total} tokens → {out_path}")
    return total


def _load_meta(bin_path: str) -> dict:
    meta_path = bin_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"dtype": "uint16"}


class ChunkedDataset(torch.utils.data.Dataset):
    """비겹침 (seq_len+1) 청크를 결정적으로 인덱싱하는 map-style 데이터셋.

    청크 i는 토큰 [i*seq_len : i*seq_len + seq_len + 1] 를 덮으며,
    마지막 토큰은 다음 위치 예측용 라벨로만 사용됩니다(모델 입력엔 겹치지 않음).

    셔플과 rank 샤딩은 DataLoader의 DistributedSampler가 담당하므로,
    __getitem__은 순수하게 결정적입니다 → 에폭별 재현성 + 전 코퍼스 균등 커버리지.
    """

    def __init__(self, bin_path: str, seq_len: int):
        meta = _load_meta(bin_path)
        self.dtype = np.dtype(meta["dtype"])
        self.data = np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        self.n_chunks = (self.n_tokens - 1) // seq_len  # +1은 shifted target용
        assert self.n_chunks > 0, "코퍼스가 seq_len보다 짧습니다."

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start: start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class RandomWindowDataset(torch.utils.data.Dataset):
    """무작위 오프셋으로 (seq_len+1) 윈도우를 뽑는 데이터셋(옛 방식, 옵션).

    커버리지 보장은 없지만 매우 큰 코퍼스에서 간단히 쓰기 좋습니다.
    idx별로 결정적인 rng를 써서 재현 가능합니다.
    """

    def __init__(self, bin_path: str, seq_len: int, length: Optional[int] = None,
                 seed: int = 1337):
        meta = _load_meta(bin_path)
        self.dtype = np.dtype(meta["dtype"])
        self.data = np.memmap(bin_path, dtype=self.dtype, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        assert self.n_tokens > seq_len + 1, "코퍼스가 seq_len보다 짧습니다."
        self.length = length if length is not None else self.n_tokens // seq_len
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        max_start = self.n_tokens - (self.seq_len + 1)
        start = int(rng.integers(0, max_start))
        chunk = self.data[start: start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


# 하위 호환 별칭
PackedDataset = RandomWindowDataset


def make_dataloader(bin_path: str, seq_len: int, micro_batch_size: int,
                    distributed: bool = False, seed: int = 1337,
                    num_workers: int = 2, length: Optional[int] = None,
                    mode: str = "chunked"):
    """DataLoader 생성.

    mode="chunked": 비겹침 청크 + 셔플/샤딩 (권장, 기본).
    mode="random":  무작위 윈도우 샘플링.

    반환: (loader, sampler). sampler는 분산일 때만 non-None (set_epoch 호출용).
    """
    if mode == "chunked":
        ds = ChunkedDataset(bin_path, seq_len)
    elif mode == "random":
        ds = RandomWindowDataset(bin_path, seq_len, length=length, seed=seed)
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=True, seed=seed, drop_last=True)
    else:
        # 단일 프로세스에서도 셔플 (chunked는 순서가 결정적이므로 셔플 필요)
        sampler = torch.utils.data.RandomSampler(
            ds, generator=torch.Generator().manual_seed(seed))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader, (sampler if distributed else None)


class ResumableLoader:
    """결정적·샤딩된 무한 배치 로더 (정확한 위치 복원 지원).

    - 전역 청크 인덱스를 에폭별 시드 셔플로 순회하며, rank마다 disjoint 슬라이스를
      가져갑니다(데이터 병렬). 진행 위치(소비한 전역 샘플 수)를 저장/복원하면
      학습 재개 시 데이터로더가 정확히 이어서 공급합니다.
    - 백그라운드 프리페치 스레드로 I/O를 학습과 겹칩니다.

    사용:
        loader = ResumableLoader(bin_path, seq_len, mb, world_size, rank, seed)
        loader.load_state_dict({"pos": saved_pos})   # (선택) 재개
        for x, y in loader:  # 무한 반복
            ...
        sd = loader.state_dict()  # 체크포인트에 저장
    """

    def __init__(self, bin_path: str, seq_len: int, micro_batch_size: int,
                 world_size: int = 1, rank: int = 0, seed: int = 1337,
                 prefetch: int = 4):
        self.ds = ChunkedDataset(bin_path, seq_len)
        self.n = self.ds.n_chunks
        self.mb = micro_batch_size
        self.ws = max(1, world_size)
        self.rank = rank
        self.seed = seed
        self._perm_epoch = -1
        self._perm = None
        self._produce = 0            # 워커가 생성한 전역 위치
        self._consumed = 0           # 메인이 소비한 전역 위치 (저장 대상)
        self._q: "Queue" = Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._started = False
        self._thread: Optional[threading.Thread] = None

    # -- 결정적 인덱싱 --
    def _perm_for(self, epoch: int):
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n)
            self._perm_epoch = epoch
        return self._perm

    def _chunk_index(self, global_i: int) -> int:
        epoch = global_i // self.n
        within = global_i % self.n
        return int(self._perm_for(epoch)[within])

    def _plan(self):
        base = self._produce + self.rank * self.mb
        idxs = [self._chunk_index(base + j) for j in range(self.mb)]
        self._produce += self.ws * self.mb
        return idxs, self._produce

    def _worker(self):
        while not self._stop.is_set():
            idxs, after = self._plan()
            xs, ys = [], []
            for i in idxs:
                x, y = self.ds[i]
                xs.append(x)
                ys.append(y)
            batch = (torch.stack(xs), torch.stack(ys), after)
            # stop 이벤트 중에도 블로킹 방지
            while not self._stop.is_set():
                try:
                    self._q.put(batch, timeout=0.5)
                    break
                except Exception:
                    continue

    def _ensure_started(self):
        if not self._started:
            self._produce = self._consumed
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            self._started = True

    def __iter__(self):
        self._ensure_started()
        return self

    def __next__(self):
        self._ensure_started()
        x, y, after = self._q.get()
        self._consumed = after
        return x, y

    # -- 체크포인트 --
    def state_dict(self) -> dict:
        return {"pos": self._consumed, "seed": self.seed}

    def load_state_dict(self, sd: dict):
        assert not self._started, "반복 시작 전에 load_state_dict를 호출해야 합니다."
        self._consumed = int(sd.get("pos", 0))
        self._produce = self._consumed

    def stop(self):
        self._stop.set()
