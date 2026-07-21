"""관측소별 조차 기준선 — 상대 세기 판정.

조차의 절대값은 관측소마다 스케일이 다릅니다(인천 대사리 900cm+, 묵호 30cm 남짓).
전국 고정 기준을 쓰면 서해는 늘 '큰 조차', 동해는 영원히 '작은 조차'가 됩니다.
여기서는 관측소별 소조차~대조차 구간 안에서 오늘이 어디쯤인지를 상대 위치로 봅니다.

기준선 데이터: data/station_ranges.json (scripts/build_ranges.py 로 생성)
    {"DT_0001": {"name":"인천","neap":305,"spring":872,...}, ...}
파일이 없거나 해당 관측소가 없으면 기존 고정 기준으로 자동 대체합니다.
"""
from __future__ import annotations

import json
import os
from typing import Optional

# 상대 위치 경계. 0=소조차, 1=대조차
BIG_AT = 0.66
SMALL_AT = 0.33

# 기준선이 없을 때 쓰는 전국 고정 기준(cm) — 서해 중심의 대략값
FALLBACK_BIG = 600
FALLBACK_SMALL = 200

_cache: Optional[dict] = None


def _path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "data", "station_ranges.json")


def load_baselines(force: bool = False) -> dict:
    """station_ranges.json 을 읽어 캐시. 없으면 빈 dict."""
    global _cache
    if _cache is not None and not force:
        return _cache
    path = _path()
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:  # noqa
            data = {}
    _cache = data
    return data


def get_baseline(code: Optional[str]) -> Optional[dict]:
    """관측소 기준선 dict 또는 None."""
    if not code:
        return None
    b = load_baselines().get(code)
    if not b:
        return None
    neap, spring = b.get("neap"), b.get("spring")
    # 구간이 뒤집혔거나 너무 좁으면 상대 판정이 무의미
    if neap is None or spring is None or spring - neap < 5:
        return None
    return b


def relative_position(range_cm: float, code: Optional[str]) -> Optional[float]:
    """조차의 상대 위치(0=소조차, 1=대조차). 기준선이 없으면 None.

    구간 밖의 값은 0~1로 자르지 않고 그대로 둡니다(예: 관측 이래 최대 조차면 1.1).
    호출부에서 필요하면 clamp 하세요.
    """
    b = get_baseline(code)
    if not b:
        return None
    neap, spring = b["neap"], b["spring"]
    return (range_cm - neap) / (spring - neap)


def classify(range_cm: float, code: Optional[str] = None) -> str:
    """조수 세기 3단계: "big" | "mid" | "small".

    관측소 기준선이 있으면 상대 위치로, 없으면 전국 고정 기준으로 판정합니다.
    """
    pos = relative_position(range_cm, code)
    if pos is None:
        if range_cm >= FALLBACK_BIG:
            return "big"
        if range_cm < FALLBACK_SMALL:
            return "small"
        return "mid"
    if pos >= BIG_AT:
        return "big"
    if pos <= SMALL_AT:
        return "small"
    return "mid"


def range_label(range_cm: float, code: Optional[str] = None) -> str:
    """조차를 사람이 읽는 라벨로. 관측소 기준선이 있으면 그 관측소 기준."""
    pos = relative_position(range_cm, code)
    if pos is None:
        if range_cm >= 700:
            return "매우 큰 조차(사리급)"
        if range_cm >= 400:
            return "큰 조차"
        if range_cm >= 200:
            return "보통 조차"
        return "작은 조차(조금급)"
    if pos >= 0.9:
        return "이 지점 기준 매우 큰 조차"
    if pos >= BIG_AT:
        return "이 지점 기준 큰 조차"
    if pos > SMALL_AT:
        return "이 지점 기준 보통 조차"
    if pos > 0.1:
        return "이 지점 기준 작은 조차"
    return "이 지점 기준 매우 작은 조차"
