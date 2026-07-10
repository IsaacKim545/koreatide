"""물때 질의 자연어 파서 (규칙 기반).

한국어 질문에서 관측소(지점)와 날짜를 추출합니다. LLM이 아니라 결정론적 규칙이라
정확하고 예측 가능합니다.

예:
    "내일 인천 물때 알려줘"        → (DT_0001, 내일 날짜)
    "부산 7월 15일 만조 시간"      → (DT_0005, 2026-07-15)
    "모레 목포"                    → (DT_0007, 모레)
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

from .stations import load_stations

_WEEKDAYS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}


# 관측소 이름 접미사(약칭 매칭용): '거제도'→'거제', '동해항'→'동해', '영종대교'→'영종'
_SUFFIXES = ("신항", "대교", "항", "도")


def parse_station(text: str) -> Optional[str]:
    """텍스트에서 관측소 코드를 추출.

    1) 코드 직접(DT_0001)  2) 정식 이름 포함(긴 이름 우선)
    3) 접미사(도/항/대교/신항)를 뗀 핵심어 매칭 — '거제'로 '거제도'를 찾음(핵심어 2자↑).
    """
    st = load_stations()
    m = re.search(r"\b([A-Z]{2}_\d{4})\b", text)
    if m and m.group(1) in st:
        return m.group(1)

    # 1차: 정식 이름이 텍스트에 포함 (가장 긴 이름 우선)
    best, best_len = None, 0
    for code, name in st.items():
        if name in text and len(name) > best_len:
            best, best_len = code, len(name)
    if best:
        return best

    # 2차: 접미사 제거 핵심어 매칭 (핵심어 2자 이상, 가장 긴 핵심어 우선)
    for code, name in st.items():
        core = name
        for suf in _SUFFIXES:
            if core.endswith(suf) and len(core) - len(suf) >= 2:
                core = core[:-len(suf)]
                break
        if core != name and len(core) >= 2 and core in text and len(core) > best_len:
            best, best_len = code, len(core)
    return best


def parse_date_opt(text: str, today: Optional[date] = None) -> Optional[date]:
    """텍스트에서 날짜를 추출. 날짜 표현이 없으면 None(오늘로 대체하지 않음).

    대화에서 이전 턴의 날짜를 이어받기 위해 '명시 여부'를 구분합니다.
    """
    if today is None:
        today = datetime.now().date()

    # 상대 표현
    rel = {"그저께": -2, "어제": -1, "오늘": 0, "금일": 0, "내일": 1,
           "명일": 1, "모레": 2, "글피": 3}
    for k, off in rel.items():
        if k in text:
            return today + timedelta(days=off)

    # "N일 뒤/후/전"
    m = re.search(r"(\d+)\s*일\s*(뒤|후|전)", text)
    if m:
        n = int(m.group(1))
        return today + timedelta(days=(n if m.group(2) != "전" else -n))

    # ISO / 구분자 YYYY-MM-DD, YYYY.MM.DD, YYYY/MM/DD
    m = re.search(r"(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # "M월 D일" (연도 없으면 올해; 이미 지난 날짜면 내년)
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        y = today.year
        cand = _safe_date(y, mo, d)
        if cand and cand < today - timedelta(days=180):
            cand = _safe_date(y + 1, mo, d)
        if cand:
            return cand

    # "D일" 단독 (이번 달; 지났으면 다음 달)
    m = re.search(r"(?<!\d)(\d{1,2})\s*일(?!\s*(뒤|후|전))", text)
    if m:
        d = int(m.group(1))
        cand = _safe_date(today.year, today.month, d)
        if cand and cand < today:
            # 다음 달
            ny, nm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
            cand = _safe_date(ny, nm, d) or cand
        if cand:
            return cand

    # 요일 (이번/다음 해당 요일; 오늘 포함 이후 가장 가까운 날)
    for wd, idx in _WEEKDAYS.items():
        if re.search(wd + r"요일", text):
            delta = (idx - today.weekday()) % 7
            return today + timedelta(days=delta)

    return None


def parse_date(text: str, today: Optional[date] = None) -> date:
    """텍스트에서 날짜를 추출. 없으면 오늘."""
    if today is None:
        today = datetime.now().date()
    return parse_date_opt(text, today) or today


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_query(text: str, today: Optional[date] = None) -> Tuple[Optional[str], date]:
    """질의에서 (관측소코드, 날짜)를 반환. 관측소를 못 찾으면 (None, 날짜)."""
    return parse_station(text), parse_date(text, today)
