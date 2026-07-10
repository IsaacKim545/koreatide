#!/usr/bin/env python
"""조위관측소 목록 관리 — 내장 목록을 data/stations.json 으로 내보내기.

사용 예:
    python scripts/tide_stations.py --export      # data/stations.json 생성
    python scripts/tide_stations.py --list        # 콘솔 출력

향후 KHOA가 관측소를 추가하면, 바다누리 API 조회 페이지의 관측소 목록을 참고해
data/stations.json 을 직접 갱신하면 됩니다(엔진은 json이 있으면 우선 사용).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tide.stations import STATIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="data/stations.json 생성")
    ap.add_argument("--list", action="store_true", help="목록 출력")
    args = ap.parse_args()

    if args.export or not args.list:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data", "stations.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(STATIONS, f, ensure_ascii=False, indent=2)
        print(f"저장: {path} ({len(STATIONS)}곳)")

    if args.list:
        for code, name in sorted(STATIONS.items(), key=lambda x: x[1]):
            print(f"  {name:<14} {code}")


if __name__ == "__main__":
    main()
