#!/usr/bin/env python
"""물때 조회 CLI — 대한민국 전국 조위관측소 만조/간조 + 물때.

국립해양조사원(KHOA) 예측조위로 만조/간조 시각과 조차를 계산합니다.

사전 준비: 무료 서비스키 발급 (공공데이터포털 data.go.kr 또는 KHOA 바다누리)
    set KHOA_API_KEY=발급받은키           (Windows)
    export KHOA_API_KEY=발급받은키          (Linux/Mac)

사용 예:
    python scripts/tide.py --list                       # 관측소 목록
    python scripts/tide.py --station 인천 --date 2026-07-15
    python scripts/tide.py --station DT_0005            # 부산, 오늘
    python scripts/tide.py --station 인천 --sample      # 키 없이 샘플(인천) 테스트
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tide.stations import load_stations, find_station  # noqa: E402
from src.tide.khoa import KhoaTideClient  # noqa: E402
from src.tide.mulddae import get_mulddae  # noqa: E402
from src.tide.baseline import range_label  # noqa: E402
from src.tide.keyconf import load_service_key, save_service_key, get_api_url  # noqa: E402

HL_KO = {"고조": "만조", "저조": "간조"}


def print_stations():
    st = load_stations()
    print(f"전국 조위관측소 {len(st)}곳:\n")
    items = sorted(st.items(), key=lambda x: x[1])
    for i in range(0, len(items), 3):
        row = items[i:i + 3]
        print("  " + "".join(f"{name:<12}({code})  " for code, name in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", help="관측소 이름(예: 인천) 또는 코드(예: DT_0001)")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (기본: 오늘)")
    ap.add_argument("--key", default=None, help="KHOA 서비스키 (없으면 env/설정파일 사용)")
    ap.add_argument("--set-key", default=None, help="서비스키를 저장하고 종료(config/khoa_key.txt)")
    ap.add_argument("--api-url", default=None, help="API 엔드포인트 override (data.go.kr 등)")
    ap.add_argument("--min", type=int, default=10, help="예측조위 시간간격(분), 기본 10")
    ap.add_argument("--offset", type=int, default=0, help="물때 지역보정(일), 기본 0(서해)")
    ap.add_argument("--days", type=int, default=1, help="주간 물때표: 시작일부터 N일 (예: 7)")
    ap.add_argument("--sample", action="store_true", help="키 없이 샘플(인천) 데이터 사용")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--debug", action="store_true", help="요청 URL(키 마스킹) 출력")
    ap.add_argument("--list", action="store_true", help="관측소 목록 출력")
    args = ap.parse_args()

    if args.set_key:
        path = save_service_key(args.set_key)
        print(f"서비스키를 저장했습니다: {path}\n이제 --sample 없이 전국 조회가 됩니다.")
        return

    if args.list or not args.station:
        print_stations()
        if not args.station:
            print("\n예: python scripts/tide.py --station 인천 --date 2026-07-15")
        return

    code = find_station(args.station)
    if not code:
        print(f"관측소를 찾을 수 없습니다: '{args.station}'  (--list 로 목록 확인)")
        sys.exit(1)

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    try:
        yyyymmdd = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError:
        print(f"날짜 형식 오류(YYYY-MM-DD): {date}")
        sys.exit(1)

    key = load_service_key(args.key)
    sample = args.sample or not key
    if sample and code != "DT_0001":
        print("[안내] 서비스키가 없어 샘플 모드입니다. 샘플은 인천(DT_0001)만 제공됩니다.\n"
              "       전국 조회: 무료 키 발급 후  python scripts/tide.py --set-key 발급키")

    client = KhoaTideClient(service_key=key, base_url=get_api_url(args.api_url),
                            debug=args.debug)

    # 주간(여러 날) 모드
    if args.days and args.days > 1:
        from src.tide.report import week_report
        start = datetime.strptime(date, "%Y-%m-%d").date()
        wk = week_report(client, code, start, days=args.days,
                         offset=args.offset, sample=sample)
        print(f"\n== {wk['station']['name']} ({code}) / {date}부터 {args.days}일 ==")
        for r in wk["days"]:
            if "error" in r:
                print(f" {r['date']}({r['weekday']})  조회 실패: {r['error'][:40]}")
                continue
            mul = f"{r['mulddae']['name']}" if r["mulddae"] else "-"
            ex = "  ".join(f"{e['type']}{e['time']}" for e in r["extrema"])
            print(f" {r['date']}({r['weekday']}) {mul:>4} 조차{r['range_cm']:>4}cm | {ex}")
        return

    try:
        day = client.get_day(code, yyyymmdd, min_interval=args.min,
                             use_cache=not args.no_cache, sample=sample)
    except Exception as e:  # noqa
        print(f"[오류] 조회 실패: {e}")
        sys.exit(1)

    name = day["meta"].get("name", code)
    extrema = day["extrema"]

    # 헤더 + 물때
    line = f"== {name} ({code}) / {date} =="
    print("\n" + line)
    mul = get_mulddae(datetime.strptime(date, "%Y-%m-%d"), offset=args.offset)
    if mul:
        ly, lm, ld = mul["lunar"]
        print(f"물때: {mul['name']} ({mul['phase']})  |  음력 {lm}.{ld}")
    else:
        print("물때: (korean_lunar_calendar 미설치 → 물때 이름 생략)")
    print("-" * len(line))

    if not extrema:
        print(" 만조/간조를 찾지 못했습니다.")
        return
    highs = [e["level"] for e in extrema if e["type"] == "고조"]
    lows = [e["level"] for e in extrema if e["type"] == "저조"]
    for e in extrema:
        ko = HL_KO.get(e["type"], e["type"])
        print(f" {ko}  {e['time'].strftime('%H:%M')}   {e['level']:>6.0f} cm")
    print("-" * len(line))
    if highs and lows:
        rng = max(highs) - min(lows)
        print(f" 최대 조차: {rng:.0f} cm ({range_label(rng, code)})")


if __name__ == "__main__":
    main()
