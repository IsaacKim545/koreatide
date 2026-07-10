"""대한민국 조석(물때) 계산·조회 엔진.

국립해양조사원(KHOA) 바다누리 OpenAPI의 '조위관측소 실측·예측 조위'(SV_AP_02_009)를
사용해 예측 조위 곡선을 받고, 극값에서 만조/간조를 추출하며, 음력 기반 물때를 계산합니다.
"""
from .stations import STATIONS, find_station, station_name
from .khoa import KhoaTideClient, find_extrema
from .mulddae import get_mulddae

__all__ = ["STATIONS", "find_station", "station_name",
           "KhoaTideClient", "find_extrema", "get_mulddae"]
