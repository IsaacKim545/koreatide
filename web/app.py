#!/usr/bin/env python
"""물때 홈페이지 — Flask 백엔드.

정적 프론트(templates/index.html)와 JSON API를 제공합니다. KHOA 서비스키는
서버 측(config/khoa_key.txt 또는 KHOA_API_KEY)에서만 쓰이고 브라우저엔 노출되지 않습니다.

실행:
    pip install flask
    python web/app.py
    → http://127.0.0.1:5000

API:
    GET /api/stations                     → [{"code","name"}, ...]
    GET /api/tide?station=인천&date=2026-07-16&days=7&offset=0
"""
import json
import os
import sys
from datetime import date as _date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template  # noqa: E402

from src.tide.stations import load_stations, find_station  # noqa: E402
from src.tide.khoa import KhoaTideClient  # noqa: E402
from src.tide.report import week_report  # noqa: E402
from src.tide.keyconf import load_service_key, get_api_url  # noqa: E402

app = Flask(__name__)


def _client():
    key = load_service_key()
    return KhoaTideClient(service_key=key, base_url=get_api_url()), (not key)


@app.route("/")
def index():
    return render_template("index.html")


def _load_geo():
    """data/stations_geo.json (build_geo.py 산출물)이 있으면 좌표를 반환."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "stations_geo.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa
            return {}
    return {}


@app.route("/api/stations")
def api_stations():
    st = load_stations()
    geo = _load_geo()
    out = []
    for c, n in sorted(st.items(), key=lambda x: x[1]):
        item = {"code": c, "name": n}
        if c in geo:
            item["lat"] = geo[c].get("lat")
            item["lon"] = geo[c].get("lon")
        out.append(item)
    return jsonify(out)


@app.route("/api/tide")
def api_tide():
    station = (request.args.get("station") or "").strip()
    date_s = (request.args.get("date") or "").strip()
    try:
        days = max(1, min(14, int(request.args.get("days", 1))))
    except ValueError:
        days = 1
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    code = find_station(station)
    if not code:
        return jsonify({"error": f"관측소를 찾을 수 없습니다: '{station}'"}), 400

    try:
        start = datetime.strptime(date_s, "%Y-%m-%d").date() if date_s else _date.today()
    except ValueError:
        return jsonify({"error": f"날짜 형식 오류(YYYY-MM-DD): {date_s}"}), 400

    client, sample = _client()
    if sample and code != "DT_0001":
        return jsonify({"error": "서비스키가 설정되지 않았습니다. 인천만 조회 가능합니다. "
                        "config/khoa_key.txt 에 키를 저장하거나 KHOA_API_KEY 를 설정하세요.",
                        "sample": True}), 403
    try:
        wk = week_report(client, code, start, days=days, offset=offset, sample=sample)
    except Exception as e:  # noqa
        return jsonify({"error": str(e)}), 500
    wk["sample"] = sample
    return jsonify(wk)


def _lan_ip():
    """이 PC의 LAN IP 추정 (다른 기기에서 접속할 주소)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))       # 실제 전송 없음, 로컬 IP 확인용
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa
        return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # HOST 환경변수로 바인딩 제어. 기본 0.0.0.0 → 같은 와이파이의 다른 기기에서도 접속 가능.
    host = os.environ.get("HOST", "0.0.0.0")
    ip = _lan_ip()
    print("=" * 52)
    print(" 물때 홈페이지 실행 중")
    print(f"   이 PC:      http://localhost:{port}")
    if host == "0.0.0.0":
        print(f"   다른 기기:  http://{ip}:{port}   (같은 와이파이)")
        print("   * 다른 기기에서 안 되면 Windows 방화벽에서 Python 허용 필요")
    print("=" * 52)
    app.run(host=host, port=port, debug=True)
