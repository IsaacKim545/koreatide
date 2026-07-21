#!/usr/bin/env python
"""관측소별 조차 기준선 생성 — data/station_ranges.json.

왜 필요한가
-----------
조차(만조−간조)의 절대값은 관측소마다 스케일이 완전히 다릅니다.
인천은 대사리에 900cm를 넘지만, 동해안(묵호·속초)은 사리여도 30cm 남짓입니다.
따라서 "600cm 이상이면 사리" 같은 고정 기준을 전국에 쓰면
  - 서해 관측소는 늘 '큰 조차'
  - 동해 관측소는 영원히 '작은 조차'
로 굳어집니다. 관측소마다 자기 기준(소조차~대조차 범위)을 갖고,
그 안에서 오늘이 어디쯤인지를 상대 위치로 판정해야 맞습니다.

수집 방법
---------
계절 변동(춘·추분 대사리)까지 담기 위해 1·4·7·10월에서 각 10일씩,
관측소당 40일치 예측조위를 받아 일별 조차를 계산합니다.
표본에서 p10을 소조차(neap), p90을 대조차(spring) 기준으로 삼습니다.
(최소·최대 대신 백분위를 쓰는 이유: 이상치 한 날에 기준선이 끌려가지 않도록)

사용:
    python scripts/build_ranges.py              # 전체 관측소
    python scripts/build_ranges.py --year 2026  # 표본 연도 지정
    python scripts/build_ranges.py --only DT_0001,DT_0006
    python scripts/build_ranges.py --resume     # 기존 결과에 이어서(중단 후 재개)

호출량: 관측소 55곳 × 40일 = 약 2,200회. 캐시(data/tide_cache)가 재사용되므로
재실행 시에는 훨씬 적게 호출합니다. 일일 한도에 걸리면 --resume 으로 이어가세요.
"""
import argparse
import json
import os
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tide.stations import STATIONS  # noqa: E402
from src.tide.khoa import KhoaTideClient, find_extrema  # noqa: E402
from src.tide.keyconf import load_service_key, get_api_url  # noqa: E402

# 계절 4구간 × 10일. 달 주기(약 29.5일)와 어긋나게 배치해 사리·조금이 고루 걸리도록 함.
SEASON_STARTS = [(1, 5), (4, 3), (7, 8), (10, 6)]
WINDOW_DAYS = 10

# 표본이 이보다 적으면 기준선을 신뢰할 수 없다고 보고 제외
MIN_SAMPLES = 12


def sample_dates(year: int):
    """수집 대상 날짜 목록."""
    out = []
    for month, day in SEASON_STARTS:
        start = date(year, month, day)
        out += [start + timedelta(days=i) for i in range(WINDOW_DAYS)]
    return out


def percentile(sorted_vals, q: float) -> float:
    """선형보간 백분위. q는 0~1."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def daily_range(client, code: str, d: date) -> float:
    """그날의 조차(cm). 극값을 못 잡으면 시계열 최대−최소로 대체."""
    ymd = d.strftime("%Y%m%d")
    # margin=False: 기준선 통계에는 자정경계 보정이 불필요(호출 1/3로 감소)
    _meta, series = client.fetch_series(code, ymd, min_interval=30)
    if not series:
        raise RuntimeError("빈 시계열")

    ex = find_extrema(series)
    highs = [e["level"] for e in ex if e["type"] == "고조"]
    lows = [e["level"] for e in ex if e["type"] == "저조"]
    if highs and lows:
        return max(highs) - min(lows)

    levels = [p for _t, p, _o in series]
    return max(levels) - min(levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=date.today().year - 1,
                    help="표본 연도 (기본: 작년 — 한 해가 온전히 존재하므로)")
    ap.add_argument("--only", default="", help="관측소 코드 쉼표 구분")
    ap.add_argument("--resume", action="store_true",
                    help="기존 station_ranges.json 에 있는 관측소는 건너뜀")
    ap.add_argument("--sleep", type=float, default=0.1,
                    help="호출 간 대기(초). API 부하 조절")
    args = ap.parse_args()

    key = load_service_key()
    if not key:
        print("[오류] 서비스키가 필요합니다. 먼저 저장하세요:\n"
              "       python scripts/tide.py --set-key <키>")
        sys.exit(1)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(root, "data", "station_ranges.json")

    result = {}
    if args.resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(f"이어서 진행: 기존 {len(result)}곳 유지\n")

    targets = dict(STATIONS)
    if args.only:
        codes = {c.strip() for c in args.only.split(",") if c.strip()}
        targets = {c: n for c, n in STATIONS.items() if c in codes}

    client = KhoaTideClient(service_key=key, base_url=get_api_url())
    dates = sample_dates(args.year)
    print(f"표본: {args.year}년 {len(dates)}일 × 관측소 {len(targets)}곳\n")

    skipped = []
    for i, (code, name) in enumerate(sorted(targets.items(), key=lambda x: x[1]), 1):
        if args.resume and code in result:
            print(f"[{i:>2}/{len(targets)}] {name:<12} 건너뜀(이미 있음)")
            continue

        ranges = []
        errors = 0
        for d in dates:
            try:
                ranges.append(daily_range(client, code, d))
            except Exception:  # noqa
                errors += 1
            if args.sleep:
                time.sleep(args.sleep)

        if len(ranges) < MIN_SAMPLES:
            print(f"[{i:>2}/{len(targets)}] {name:<12} 표본 부족({len(ranges)}일) — 제외")
            skipped.append(name)
            continue

        ranges.sort()
        neap = percentile(ranges, 0.10)
        spring = percentile(ranges, 0.90)
        median = percentile(ranges, 0.50)

        # 소조차와 대조차가 거의 같으면(조차 변동이 없는 곳) 상대 판정이 무의미
        if spring - neap < 5:
            print(f"[{i:>2}/{len(targets)}] {name:<12} 조차 변동 미미"
                  f"({neap:.0f}~{spring:.0f}cm) — 제외")
            skipped.append(name)
            continue

        result[code] = {
            "name": name,
            "neap": round(neap),        # 소조차 기준 (p10)
            "spring": round(spring),    # 대조차 기준 (p90)
            "median": round(median),
            "min": round(ranges[0]),
            "max": round(ranges[-1]),
            "samples": len(ranges),
        }
        note = f" (실패 {errors}일)" if errors else ""
        print(f"[{i:>2}/{len(targets)}] {name:<12} "
              f"소조 {neap:>4.0f} / 중앙 {median:>4.0f} / 대조 {spring:>4.0f} cm"
              f"  n={len(ranges)}{note}")

        # 중간 저장 — 한도 초과로 중단돼도 여기까지는 남음
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {out_path} ({len(result)}곳)")
    if skipped:
        print(f"제외: {', '.join(skipped)}")
        print("  → 제외된 곳은 기존 고정 기준(600/200cm)으로 자동 대체됩니다.")


if __name__ == "__main__":
    main()
