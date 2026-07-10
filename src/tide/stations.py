"""KHOA 조위관측소 코드 ↔ 이름 (전국 55개, 공식 목록).

출처: 국립해양조사원 바다누리 오픈API '조위관측소 실측·예측 조위' 관측소 목록.
scripts/tide_stations.py 로 최신 목록을 다시 받아 data/stations.json 로 갱신할 수 있습니다.
"""
from __future__ import annotations

import json
import os
from typing import Optional

# 공식 관측소 코드 → 이름
STATIONS = {
    "DT_0063": "가덕도", "DT_0032": "강화대교", "DT_0031": "거문도", "DT_0029": "거제도",
    "DT_0026": "고흥발포", "DT_0049": "광양", "DT_0042": "교본초", "DT_0018": "군산",
    "DT_0017": "대산", "DT_0065": "덕적도", "DT_0057": "동해항", "DT_0062": "마산",
    "DT_0023": "모슬포", "DT_0007": "목포", "DT_0006": "묵호", "DT_0025": "보령",
    "DT_0005": "부산", "DT_0056": "부산항신항", "DT_0061": "삼천포", "DT_0094": "서거차도",
    "DT_0010": "서귀포", "DT_0051": "서천마량", "DT_0022": "성산포", "DT_0093": "소무의도",
    "DT_0012": "속초", "IE_0061": "신안가거초", "DT_0008": "안산", "DT_0067": "안흥",
    "DT_0037": "어청도", "DT_0016": "여수", "DT_0092": "여호항", "DT_0003": "영광",
    "DT_0044": "영종대교", "DT_0043": "영흥도", "IE_0062": "옹진소청초", "DT_0027": "완도",
    "DT_0013": "울릉도", "DT_0020": "울산", "DT_0068": "위도",
    "IE_0060": "이어도", "DT_0001": "인천", "DT_0052": "인천송도", "DT_0024": "장항",
    "DT_0004": "제주", "DT_0028": "진도", "DT_0021": "추자도", "DT_0050": "태안",
    "DT_0014": "통영", "DT_0002": "평택", "DT_0091": "포항",
    "DT_0066": "향화도", "DT_0011": "후포", "DT_0035": "흑산도",
}


def _project_stations_json() -> str:
    # src/tide/stations.py -> 프로젝트 루트/data/stations.json
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "data", "stations.json")


def load_stations() -> dict:
    """data/stations.json 이 있으면 그걸, 없으면 내장 STATIONS 를 반환."""
    path = _project_stations_json()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except Exception:  # noqa
            pass
    return dict(STATIONS)


def station_name(code: str) -> str:
    return load_stations().get(code, code)


def find_station(query: str) -> Optional[str]:
    """이름(부분일치) 또는 코드로 관측소 코드를 찾음. 없으면 None."""
    q = query.strip()
    st = load_stations()
    if q in st:                       # 코드 직접
        return q
    up = q.upper()
    if up in st:
        return up
    # 정확한 이름 우선
    for code, name in st.items():
        if name == q:
            return code
    # 부분 일치
    for code, name in st.items():
        if q in name:
            return code
    return None
