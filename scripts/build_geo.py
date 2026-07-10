#!/usr/bin/env python
"""지점 좌표 캐시 생성 — 지도(웹)용.

각 조위관측소의 위도/경도를 KHOA API에서 한 번씩 받아 data/stations_geo.json 에
저장합니다. 지도에서 마커를 찍는 데 사용됩니다. (서비스키 필요, 1회 실행)

사용:
    python scripts/build_geo.py
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tide.stations import STATIONS  # noqa: E402
from src.tide.khoa import KhoaTideClient  # noqa: E402
from src.tide.keyconf import load_service_key, get_api_url  # noqa: E402


def main():
    key = load_service_key()
    if not key:
        print("[오류] 서비스키가 필요합니다. 먼저 저장하세요:\n"
              "       python scripts/tide.py --set-key <키>")
        sys.exit(1)
    client = KhoaTideClient(service_key=key, base_url=get_api_url())
    today = date.today().strftime("%Y%m%d")

    geo = {}
    ok = 0
    for code, name in STATIONS.items():
        try:
            meta, _ = client.fetch_series(code, today, min_interval=60)
            lat = float(meta.get("lat"))
            lon = float(meta.get("lot"))
            geo[code] = {"name": name, "lat": lat, "lon": lon}
            ok += 1
            print(f"  {name:<14} {code}  ({lat:.4f}, {lon:.4f})")
        except Exception as e:  # noqa
            print(f"  {name:<14} {code}  실패: {str(e)[:50]}")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "data", "stations_geo.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out} ({ok}/{len(STATIONS)}곳)")


if __name__ == "__main__":
    main()
