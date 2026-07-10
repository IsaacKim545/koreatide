"""물때(음력 기반 조석 위상) 계산.

서해안 '7물때식'(15일 주기)을 기본으로 합니다. 음력일로 '몇물/조금/무시'를 정하고,
사리(대조)·조금(소조) 위상을 표시합니다. 지역별로 하루 차이가 나므로(예: 남해·동해
8물때식) offset 파라미터로 보정할 수 있습니다.

주의: '몇물'은 관례적 명칭이라 지역차가 있습니다. 실제 조수 세기의 객관적 지표는
조차(만조höhe−간조높이)이며, KHOA 예측조위로 정확히 계산됩니다(khoa.py).

음력 변환에는 korean_lunar_calendar 패키지가 필요합니다:
    pip install korean_lunar_calendar
없으면 물때 이름은 생략되고 만조/간조 정보만 제공됩니다.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

# 15일 주기 물때 이름 (서해 7물때식). p = ((음력일-1) % 15) + 1
_MUL_NAMES = {
    1: "7물", 2: "8물", 3: "9물", 4: "10물", 5: "11물", 6: "12물", 7: "13물",
    8: "조금", 9: "무시", 10: "1물", 11: "2물", 12: "3물", 13: "4물",
    14: "5물", 15: "6물",
}
# 위상: 사리(대조) / 조금(소조) / 중간
_PHASE = {1: "사리(대조)", 2: "사리(대조)", 15: "사리(대조)", 14: "사리(대조)",
          7: "조금(소조)", 8: "조금(소조)", 9: "조금(소조)"}


def _lunar_day(d: date):
    """양력 date → (음력 연,월,일). korean_lunar_calendar 필요. 실패 시 None."""
    try:
        from korean_lunar_calendar import KoreanLunarCalendar
    except ImportError:
        return None
    cal = KoreanLunarCalendar()
    cal.setSolarDate(d.year, d.month, d.day)
    return (cal.lunarYear, cal.lunarMonth, cal.lunarDay)


def get_mulddae(d, offset: int = 0) -> Optional[dict]:
    """물때 정보를 반환. korean_lunar_calendar 미설치 시 None.

    반환: {"lunar":(y,m,d), "lunar_day":int, "name":str, "phase":str}
    offset: 지역 보정(일 단위). 서해=0, 지역에 따라 ±1.
    """
    if isinstance(d, datetime):
        d = d.date()
    lunar = _lunar_day(d)
    if lunar is None:
        return None
    _, _, lday = lunar
    p = ((lday - 1 + offset) % 15) + 1
    return {
        "lunar": lunar,
        "lunar_day": lday,
        "name": _MUL_NAMES[p],
        "phase": _PHASE.get(p, "중간"),
    }


def range_phase(tide_range_cm: float) -> str:
    """조차(cm)로 대조/소조 대략 판정 (객관적 지표, 참고용)."""
    if tide_range_cm >= 700:
        return "매우 큰 조차(사리급)"
    if tide_range_cm >= 400:
        return "큰 조차"
    if tide_range_cm >= 200:
        return "보통 조차"
    return "작은 조차(조금급)"
