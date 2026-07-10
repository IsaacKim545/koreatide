"""구조화된 물때 리포트 (CLI·웹 공용).

day_report / week_report 는 KHOA 조회 + 물때 + 조언을 합쳐 JSON 직렬화 가능한
dict를 만듭니다. 웹 API와 CLI가 같은 데이터를 씁니다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from .khoa import KhoaTideClient
from .mulddae import get_mulddae, range_phase
from .advice import make_advice
from .stations import station_name

_WD = ["월", "화", "수", "목", "금", "토", "일"]
_HL_KO = {"고조": "만조", "저조": "간조"}


def _downsample(series, max_points: int = 48):
    """예측조위 시계열을 그래프용으로 다운샘플. [{'time','level','min'}] 반환.

    min = 자정(00:00)부터의 분 (그래프 x축 위치용).
    """
    n = len(series)
    pts = series if n <= max_points else series[:: max(1, n // max_points)]
    if n and pts[-1] is not series[-1]:
        pts = list(pts) + [series[-1]]
    out = []
    for t, p, _o in pts:
        out.append({"time": t.strftime("%H:%M"),
                    "min": t.hour * 60 + t.minute,
                    "level": round(p)})
    return out


def day_report(client: KhoaTideClient, code: str, d: date,
               offset: int = 0, sample: bool = False,
               include_series: bool = True) -> dict:
    """하루치 구조화 리포트."""
    day = client.get_day(code, d.strftime("%Y%m%d"), sample=sample)
    meta = day["meta"]
    ex = day["extrema"]
    highs = [e for e in ex if e["type"] == "고조"]
    lows = [e for e in ex if e["type"] == "저조"]
    range_cm = (max(h["level"] for h in highs) - min(l["level"] for l in lows)) \
        if highs and lows else 0.0

    mul = get_mulddae(datetime(d.year, d.month, d.day), offset=offset)
    advice = make_advice(mul, ex, range_cm)

    mul_out = None
    if mul:
        mul_out = {"name": mul["name"], "phase": mul["phase"],
                   "lunar": f"{mul['lunar'][1]}.{mul['lunar'][2]}"}

    out = {
        "date": d.isoformat(),
        "weekday": _WD[d.weekday()],
        "mulddae": mul_out,
        "extrema": [{"type": _HL_KO.get(e["type"], e["type"]),
                     "time": e["time"].strftime("%H:%M"),
                     "level": round(e["level"])} for e in ex],
        "range_cm": round(range_cm),
        "range_label": range_phase(range_cm),
        "advice": advice,
    }
    if include_series:
        out["series"] = _downsample(day.get("series", []))
    return out


def week_report(client: KhoaTideClient, code: str, start: date, days: int = 7,
                offset: int = 0, sample: bool = False) -> dict:
    """start부터 days일치 리포트 묶음."""
    name = None
    reports = []
    for i in range(days):
        d = start + timedelta(days=i)
        try:
            rep = day_report(client, code, d, offset=offset, sample=sample)
            reports.append(rep)
        except Exception as e:  # noqa
            reports.append({"date": d.isoformat(), "weekday": _WD[d.weekday()],
                            "error": str(e)})
    name = station_name(code)
    return {"station": {"code": code, "name": name}, "days": reports}
