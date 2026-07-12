"""일출·일몰 계산 (API 불필요).

관측소 좌표(위도/경도)와 날짜로 일출·일몰 시각을 계산합니다.
표준 Sunrise/Sunset 알고리즘(Almanac). 한국 표준시(KST, UTC+9) 기준.
검증: 인천 7월 ≈ 05:22/19:56, 1월 ≈ 07:48/17:26 (실제값과 일치).
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional, Tuple


def sun_times(lat: float, lon: float, d: date, tz: float = 9.0,
              zenith: float = 90.833) -> Tuple[Optional[float], Optional[float]]:
    """(일출, 일몰) 을 현지 시각(0~24 실수 시간)으로 반환. 극야/백야면 None."""
    N = d.timetuple().tm_yday
    lng_hour = lon / 15.0

    def compute(rise: bool) -> Optional[float]:
        t = N + ((6 if rise else 18) - lng_hour) / 24.0
        M = (0.9856 * t) - 3.289
        L = (M + 1.916 * math.sin(math.radians(M))
             + 0.020 * math.sin(math.radians(2 * M)) + 282.634) % 360
        RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L)))) % 360
        RA += (math.floor(L / 90) * 90) - (math.floor(RA / 90) * 90)
        RA /= 15.0
        sin_dec = 0.39782 * math.sin(math.radians(L))
        cos_dec = math.cos(math.asin(sin_dec))
        cos_h = ((math.cos(math.radians(zenith)) - sin_dec * math.sin(math.radians(lat)))
                 / (cos_dec * math.cos(math.radians(lat))))
        if cos_h > 1 or cos_h < -1:
            return None
        H = (360 - math.degrees(math.acos(cos_h))) if rise else math.degrees(math.acos(cos_h))
        H /= 15.0
        T = H + RA - 0.06571 * t - 6.622
        ut = (T - lng_hour) % 24
        return (ut + tz) % 24

    return compute(True), compute(False)


def hm(hours: Optional[float]) -> Optional[str]:
    """0~24 실수 시간 → 'HH:MM'. None이면 None."""
    if hours is None:
        return None
    hours %= 24
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}:{m:02d}"


def sun_report(lat, lon, d: date) -> Optional[dict]:
    """좌표가 있으면 {'sunrise','sunset','sunrise_min','sunset_min'} 반환, 없으면 None."""
    if lat in (None, "", "-") or lon in (None, "", "-"):
        return None
    try:
        sr, ss = sun_times(float(lat), float(lon), d)
    except (ValueError, TypeError):
        return None

    def to_min(h):
        return None if h is None else int(round((h % 24) * 60))

    return {"sunrise": hm(sr), "sunset": hm(ss),
            "sunrise_min": to_min(sr), "sunset_min": to_min(ss)}
