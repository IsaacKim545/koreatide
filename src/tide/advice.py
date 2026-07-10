"""낚시·갯벌 적기 안내 생성.

물때(사리/조금)·조차·만조/간조 시각으로부터 실용적인 코멘트를 만듭니다.
값은 KHOA 공식 예측 기반이지만, 조언 문구는 일반적 경향이므로 참고용입니다.
안전 안내(갯벌/갯골)를 항상 덧붙입니다.
"""
from __future__ import annotations

from typing import List, Optional


def _fmt(t) -> str:
    return t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)


def make_advice(mul: Optional[dict], extrema: List[dict], range_cm: float) -> dict:
    """조언 dict 반환.

    mul: {"name","phase",...} 또는 None
    extrema: [{"type":"고조"/"저조","time":datetime,"level":cm}, ...]
    range_cm: 그날 최대 조차
    """
    highs = [e for e in extrema if e["type"] == "고조"]
    lows = [e for e in extrema if e["type"] == "저조"]
    low_times = [_fmt(e["time"]) for e in lows]
    high_times = [_fmt(e["time"]) for e in highs]

    phase = mul["phase"] if mul else None
    is_spring = (phase == "사리(대조)") or range_cm >= 600
    is_neap = (phase == "조금(소조)") or range_cm < 200

    tags = []
    lines = []

    if mul:
        lines.append(f"오늘은 '{mul['name']}'({mul['phase']}) 물때입니다.")

    # 조수 세기
    if is_spring:
        tags.append("사리(큰물)")
        lines.append(f"조차가 {range_cm:.0f}cm로 커서 물이 많이 들고 많이 빠집니다.")
    elif is_neap:
        tags.append("조금(작은물)")
        lines.append(f"조차가 {range_cm:.0f}cm로 작아 물 움직임이 잔잔합니다.")
    else:
        tags.append("중간물")
        lines.append(f"조차는 {range_cm:.0f}cm로 보통 수준입니다.")

    # 갯벌
    if low_times:
        wide = "넓게 " if is_spring else ""
        lines.append(f"갯벌 체험: 간조 {', '.join(low_times)} 무렵 갯벌이 {wide}드러납니다.")
        tags.append("갯벌")

    # 낚시 물돌이 (만조·간조 전후 1~2시간 입질 활발)
    windows = []
    for e in sorted(extrema, key=lambda x: x["time"]):
        t = e["time"]
        if hasattr(t, "strftime"):
            lo = (t.replace(second=0, microsecond=0))
            windows.append(f"{_fmt(t)} 앞뒤 1~2시간")
    if windows:
        lines.append("낚시: 물때가 바뀌는 " + ", ".join(windows[:4]) +
                     " 무렵 입질이 활발한 편입니다.")
        tags.append("낚시")

    # 안전
    if is_spring:
        lines.append("⚠ 사리 무렵은 갯골 물살이 빨라, 갯벌 진입 시 만조 시각을 꼭 확인하세요.")

    return {
        "summary": " ".join(lines),
        "lines": lines,
        "tags": tags,
        "mudflat_low_times": low_times,
        "high_times": high_times,
        "is_spring": is_spring,
        "is_neap": is_neap,
    }
