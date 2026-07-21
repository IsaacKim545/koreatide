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
import re
import sys
from datetime import date as _date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template, redirect  # noqa: E402

from src.tide.stations import load_stations, find_station  # noqa: E402
from src.tide.khoa import KhoaTideClient  # noqa: E402
from src.tide.report import week_report  # noqa: E402
from src.tide.keyconf import load_service_key, get_api_url  # noqa: E402

app = Flask(__name__)


@app.before_request
def _force_https():
    # Render 등 프록시 뒤에서는 원래 접속 프로토콜이 X-Forwarded-Proto 헤더에 담김.
    # http로 들어오면 동일 URL의 https로 301 이동 (없으면 https로 간주 → 로컬은 영향 없음).
    if request.headers.get("X-Forwarded-Proto", "https") == "http":
        return redirect(request.url.replace("http://", "https://", 1), code=301)


@app.after_request
def _no_cache(resp):
    # HTML/JSON을 브라우저가 오래 캐시하지 않도록 → 재배포 시 즉시 반영
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    # 브라우저가 이후 이 도메인을 항상 HTTPS로 접속하도록(1년). HTTP 접속 자체를 없앰.
    resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


def _client():
    key = load_service_key()
    return KhoaTideClient(service_key=key, base_url=get_api_url()), (not key)


@app.route("/")
def index():
    # 구글 애널리틱스 측정 ID / 네이버 사이트 인증 코드 / 애드센스 게시자 ID를
    # 환경변수로 주입(없으면 미삽입).
    ga_id = os.environ.get("GA_MEASUREMENT_ID", "")
    naver_verification = os.environ.get("NAVER_SITE_VERIFICATION", "")
    adsense_client = os.environ.get("ADSENSE_CLIENT", "")  # 예: ca-pub-1234567890123456
    return render_template("index.html", ga_id=ga_id,
                           naver_verification=naver_verification,
                           adsense_client=adsense_client)


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


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/robots.txt")
def robots():
    # 절충안: 일반 검색봇 + AI '검색/답변' 봇은 허용(방문 유입), AI '학습' 봇은 차단.
    lines = ["# 기본: 모든 크롤러 허용", "User-agent: *", "Allow: /", ""]

    # AI 검색·답변 봇: 인용·링크로 방문을 보내므로 명시 허용
    allow_ai = ["Google-Extended", "OAI-SearchBot", "ChatGPT-User",
                "PerplexityBot", "Perplexity-User"]
    lines.append("# AI 검색·답변 봇: 허용(인용·링크로 방문 유입)")
    for bot in allow_ai:
        lines += [f"User-agent: {bot}", "Allow: /", ""]

    # AI 학습(대량 스크래핑) 봇: 콘텐츠 학습 이용 거부
    block_ai = ["GPTBot", "CCBot", "ClaudeBot", "anthropic-ai",
                "Bytespider", "Applebot-Extended"]
    lines.append("# AI 학습 봇: 차단(콘텐츠 학습 이용 거부)")
    for bot in block_ai:
        lines += [f"User-agent: {bot}", "Disallow: /", ""]

    lines += ["Sitemap: https://koreatide.com/sitemap.xml", ""]
    return app.response_class("\n".join(lines), mimetype="text/plain")


@app.route("/ads.txt")
def ads_txt():
    # 구글 애드센스 ads.txt. ADSENSE_CLIENT(ca-pub-XXXX)가 설정된 경우에만 게시.
    # 형식: google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0
    client = os.environ.get("ADSENSE_CLIENT", "").strip()
    pub = client.replace("ca-", "", 1) if client.startswith("ca-pub-") else client
    if not pub.startswith("pub-"):
        return app.response_class("", mimetype="text/plain", status=404)
    line = f"google.com, {pub}, DIRECT, f08c47fec0942fa0\n"
    return app.response_class(line, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    today = _date.today().isoformat()
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           '  <url><loc>https://koreatide.com/</loc>'
           f'<lastmod>{today}</lastmod><changefreq>daily</changefreq>'
           '<priority>1.0</priority></url>\n'
           '  <url><loc>https://koreatide.com/privacy</loc>'
           '<changefreq>yearly</changefreq><priority>0.3</priority></url>\n'
           '</urlset>\n')
    return app.response_class(xml, mimetype="application/xml")


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
        # 여러 날 조회는 자정경계 보정을 꺼 API 호출 수를 줄임(속도↑). 하루 조회만 정밀.
        wk = week_report(client, code, start, days=days, offset=offset,
                         sample=sample, margin=(days == 1))
    except Exception as e:  # noqa
        return jsonify({"error": str(e)}), 500
    wk["sample"] = sample
    return jsonify(wk)


# ── 이전 도메인 소유자(워드프레스 블로그)의 잔존 URL 처리 ────────────────────
# 구글이 예전 색인 기록을 보고 계속 재크롤링하는 경로들.
# 404("일시적으로 없음")보다 410 Gone("영구 삭제")을 주면 색인에서 더 빨리 빠짐.
_LEGACY_PATTERNS = [
    re.compile(p) for p in (
        r"^/wp-(content|includes|admin|login|json|cron)\b",   # 워드프레스 내부 경로
        r"^/xmlrpc\.php$",
        r"^/(category|tag|author|archives)/",                 # 분류 아카이브
        r"^/\d{4}/\d{2}(/|$)",                                # 날짜 아카이브 /2015/03/
        r"(^|/)feed/?$",                                      # RSS
        r"^/[a-z0-9]+(?:-[a-z0-9]+){3,}/?$",                  # 긴 영문 하이픈 슬러그
    )
]


def _is_legacy_path(path: str) -> bool:
    """이전 블로그 잔재로 보이는 경로인지 판정."""
    if path.startswith(("/api/", "/static/")):
        return False
    return any(rx.search(path) for rx in _LEGACY_PATTERNS)


@app.errorhandler(404)
def _handle_missing(_e):
    path = request.path

    # API는 사람이 아니라 프로그램이 부르므로 JSON으로 응답
    if path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404

    if _is_legacy_path(path):
        return render_template(
            "error.html", code=410,
            heading="삭제된 페이지입니다",
            message="찾으시는 주소의 페이지는 영구적으로 삭제되었습니다.",
            note="이 도메인은 현재 전국 만조·간조·물때 정보 서비스로 운영되고 있습니다."
        ), 410

    return render_template(
        "error.html", code=404,
        heading="페이지를 찾을 수 없습니다",
        message="주소가 잘못되었거나 페이지가 이동되었을 수 있습니다.",
        note=None
    ), 404


@app.errorhandler(500)
def _handle_error(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "internal server error"}), 500
    return render_template(
        "error.html", code=500,
        heading="일시적인 오류가 발생했습니다",
        message="잠시 후 다시 시도해 주세요.",
        note=None
    ), 500


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
