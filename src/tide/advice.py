"""낚시·갯벌 적기 안내 생성.

물때(사리/조금)·조차·만조/간조 시각으로부터 실용적인 코멘트를 만듭니다.
값은 KHOA 공식 예측 기반이지만, 조언 문구는 일반적 경향이므로 참고용입니다.
안전 안내(갯벌/갯골)를 항상 덧붙입니다.
"""
from __future__ import annotations

from typing import List, Optional

from .baseline import classify, relative_position


def _fmt(t) -> str:
    return t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)


def make_advice(mul: Optional[dict], extrema: List[dict], range_cm: float,
                code: Optional[str] = None) -> dict:
    """조언 dict 반환.

    mul: {"name","phase",...} 또는 None
    extrema: [{"type":"고조"/"저조","time":datetime,"level":cm}, ...]
    range_cm: 그날 최대 조차
    code: 관측소 코드. 주면 그 관측소의 조차 기준선으로 세기를 판정합니다.
          (없으면 전국 고정 기준으로 대체 — baseline.py 참고)
    """
    highs = [e for e in extrema if e["type"] == "고조"]
    lows = [e for e in extrema if e["type"] == "저조"]
    low_times = [_fmt(e["time"]) for e in lows]
    high_times = [_fmt(e["time"]) for e in highs]

    # 물때 이름(음력 위상)과 조수 세기(조차)는 서로 다른 것을 재는 값이다.
    #   - 위상: 달의 주기에서 온 관례적 명칭. 지역차가 있음(mulddae.py 주석 참고).
    #   - 조차: KHOA 예측조위로 계산한 객관적 세기.
    # 둘은 대개 일치하지만 어긋나는 날이 있으므로(예: 위상은 조금인데 조차 616cm)
    # 각각 따로 판정하고, 문구에서 모순 없이 연결한다.
    phase = mul["phase"] if mul else None
    phase_spring = (phase == "사리(대조)")
    phase_neap = (phase == "조금(소조)")

    # 조차 세기 — 서로 배타적인 3단계.
    # 관측소별 기준선(소조차~대조차)이 있으면 그 안에서의 상대 위치로 판정한다.
    # 조차 절대값은 지역차가 극심해서(인천 900cm+ vs 묵호 30cm) 전국 고정 기준을
    # 쓰면 서해는 늘 '큼', 동해는 늘 '작음'으로 굳는다.
    strength = classify(range_cm, code)
    rel_pos = relative_position(range_cm, code)
    has_baseline = rel_pos is not None

    # 위상과 조차가 어긋나는가
    mismatch = (phase_spring and strength == "small") or (phase_neap and strength == "big")

    tags = []
    lines = []

    if mul:
        tags.append("사리(큰물)" if phase_spring
                    else "조금(작은물)" if phase_neap else "중간물")
        lines.append(f"오늘은 '{mul['name']}'({mul['phase']}) 물때입니다.")

    # 조수 세기 — 위상과 어긋나면 '다만'으로 이어 붙여 모순을 없앤다.
    # 기준선이 있으면 '이 지점 기준'을 밝힌다. 같은 300cm라도 인천에선 작은 조차,
    # 동해안에선 있을 수 없는 큰 조차이므로 비교 대상을 명시해야 오해가 없다.
    head = "다만 " if mismatch else ""
    scope = "이 지점 기준으로 " if has_baseline else ""
    if strength == "big":
        tags.append("큰 조차")
        lines.append(f"{head}조차는 {range_cm:.0f}cm로 {scope}큰 편이라 "
                     "물이 많이 들고 많이 빠집니다.")
    elif strength == "small":
        tags.append("작은 조차")
        lines.append(f"{head}조차는 {range_cm:.0f}cm로 {scope}작아 "
                     "물 움직임이 잔잔합니다.")
    else:
        tags.append("보통 조차")
        lines.append(f"{head}조차는 {range_cm:.0f}cm로 {scope}보통 수준입니다.")

    # 이후 판단은 실제 세기(조차) 기준. 안전·갯벌 안내는 객관적 지표를 따른다.
    is_spring = strength == "big"
    is_neap = strength == "small"

    # 갯벌
    if low_times:
        wide = "넓게 " if is_spring else ""
        lines.append(f"갯벌 체험: 간조 {', '.join(low_times)} 무렵 갯벌이 {wide}드러납니다.")

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

    # 안전 — 위상 이름이 아니라 실제 조차를 근거로 경고한다.
    # (위상이 '조금'이어도 조차가 크면 갯골 물살은 실제로 빠르다)
    if is_spring:
        lines.append("⚠ 조차가 큰 날은 갯골 물살이 빨라, 갯벌 진입 시 만조 시각을 꼭 확인하세요.")

    return {
        "summary": " ".join(lines),
        "lines": lines,
        "tags": tags,
        "mudflat_low_times": low_times,
        "high_times": high_times,
        # 실제 조수 세기(조차 기준) — 안전·갯벌 안내의 근거
        "is_spring": is_spring,
        "is_neap": is_neap,
        "strength": strength,          # "big" | "mid" | "small"
        "range_cm": round(range_cm),
        # 관측소 기준선 대비 상대 위치 (0=소조차, 1=대조차). 기준선 없으면 None
        "rel_pos": round(rel_pos, 3) if has_baseline else None,
        "has_baseline": has_baseline,
        # 음력 위상(물때 이름) 기준 — 배지 표시용
        "phase_spring": phase_spring,
        "phase_neap": phase_neap,
        # 위상과 조차가 어긋난 날인지
        "mismatch": mismatch,
    }
