"""KHOA 바다누리 OpenAPI 클라이언트 — 조위관측소 실측·예측 조위.

API: GetSurveyTideLevelApiService (SV_AP_02_009)
엔드포인트: https://www.khoa.go.kr/oceandata/odmiapi/GetSurveyTideLevelApiService.do
응답(JSON): body.items.item[] = {obsvtrNm, lat, lot, obsrvnDt, bscTdlvHgt(실측cm), tdlvHgt(예측cm)}

serviceKey는 공공데이터포털(data.go.kr) 또는 KHOA 바다누리에서 무료 발급.
serviceKey 없이 sample=True 로 호출하면 인천(DT_0001) 샘플 데이터를 받아 테스트할 수 있습니다.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import List, Optional, Tuple

# 인증용 엔드포인트 — 공공데이터포털(apis.data.go.kr)에서 발급한 serviceKey 사용
BASE_URL = "https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService"
# 샘플(미리보기) 엔드포인트 — KHOA 내부, serviceKey 없이 isSample=Y로 인천만 조회
SAMPLE_URL = "https://www.khoa.go.kr/oceandata/odmiapi/GetSurveyTideLevelApiService.do"

# (datetime, 예측조위cm, 실측조위cm|None)
Sample = Tuple[datetime, float, Optional[float]]


def _parse_dt(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"알 수 없는 날짜 형식: {s}")


def _extract_items(payload: dict) -> list:
    """body.items.item 을 유연하게 추출 (list 또는 단일 dict)."""
    body = payload.get("body") or payload.get("response", {}).get("body") or {}
    items = body.get("items") or {}
    if isinstance(items, dict):
        item = items.get("item", [])
    else:
        item = items
    if isinstance(item, dict):
        item = [item]
    return item or []


class KhoaTideClient:
    def __init__(self, service_key: Optional[str] = None,
                 base_url: str = BASE_URL, cache_dir: Optional[str] = None,
                 timeout: int = 20, debug: bool = False):
        self.service_key = service_key
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        if cache_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(here, "..", "..", "data", "tide_cache")
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    def _build_url(self, obs_code: str, date: str, min_interval: int,
                   num_rows: int, page: int, sample: bool) -> str:
        params = {
            "obsCode": obs_code,
            "reqDate": date,
            "type": "json",
            "numOfRows": str(num_rows),
            "pageNo": str(page),
        }
        if min_interval and min_interval > 1:
            params["min"] = str(min_interval)
        if sample or not self.service_key:
            # 샘플은 항상 .do(미리보기) 엔드포인트 + isSample=Y (인천만)
            base = SAMPLE_URL
            params["isSample"] = "Y"
            prefix = ""
        else:
            # 인증 호출은 .do 없는 엔드포인트(self.base_url) + serviceKey
            base = self.base_url
            # serviceKey 인코딩 처리:
            #  - 이미 %인코딩된 'Encoding' 키(예: %2B%2F%3D)면 그대로 사용
            #  - 원문 'Decoding' 키(+ / = 포함)면 반드시 퍼센트 인코딩해야 서버가 인식
            key = self.service_key.strip()
            if "%" not in key:
                key = urllib.parse.quote(key, safe="")
            prefix = "serviceKey=" + key + "&"
        return f"{base}?{prefix}{urllib.parse.urlencode(params)}"

    @staticmethod
    def _mask_url(url: str) -> str:
        import re
        return re.sub(r"(serviceKey=)[^&]+", r"\1***", url)

    def _cache_path(self, obs_code: str, date: str, min_interval: int) -> str:
        return os.path.join(self.cache_dir, f"{obs_code}_{date}_m{min_interval}.json")

    # ------------------------------------------------------------------
    def fetch_series(self, obs_code: str, date: str, min_interval: int = 10,
                     use_cache: bool = True, sample: bool = False):
        """예측 조위 시계열을 반환. date는 'YYYYMMDD'.

        반환: (station_meta: dict, series: List[Sample])
        station_meta = {"name","lat","lot"}
        """
        cache_p = self._cache_path(obs_code, date, min_interval)
        if use_cache and os.path.exists(cache_p):
            with open(cache_p, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["meta"], [
                (_parse_dt(t), p, o) for t, p, o in cached["series"]]

        # 하루치 전체를 확보하기 위해 페이지를 순회(최대 300행/페이지).
        NUM = 300
        meta = None
        all_items: list = []
        for page in range(1, 12):        # 안전 상한 (300*11 > 1440분)
            url = self._build_url(obs_code, date, min_interval, num_rows=NUM,
                                  page=page, sample=sample)
            if self.debug:
                print(f"[debug] GET {self._mask_url(url)}")
            req = urllib.request.Request(url, headers={"User-Agent": "tide-client/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                snippet = raw.strip().replace("\n", " ")[:200]
                if "Unauthorized" in raw or "SERVICE KEY" in raw.upper() or \
                   "REGISTERED" in raw.upper():
                    raise RuntimeError(
                        "인증 실패: 서비스키가 등록되지 않았거나 이 API에 활용신청이 "
                        "안 된 키입니다. 공공데이터포털에서 해당 API 활용신청 상태를 확인하세요.\n"
                        f"  응답: {snippet}")
                raise RuntimeError("응답이 JSON이 아닙니다(엔드포인트/파라미터 확인). "
                                   f"응답 일부: {snippet}")

            header = payload.get("header") or payload.get("response", {}).get("header", {})
            rcode = header.get("resultCode")
            if rcode not in (None, "00", "0"):
                msg = header.get("resultMsg", "")
                hint = ""
                if str(rcode) in ("11", "30", "31"):
                    hint = ("\n  → 서비스키 문제일 수 있습니다. 확인하세요:\n"
                            "    1) 키를 저장했는지: python scripts/tide.py --set-key <키>\n"
                            "    2) data.go.kr 키는 'Decoding' 키를 넣어보세요(엔진이 자동 인코딩)\n"
                            "    3) data.go.kr 키면 엔드포인트가 다를 수 있음: --api-url <요청주소>\n"
                            "    4) --debug 로 실제 요청 URL 확인")
                raise RuntimeError(f"KHOA API 오류 (resultCode={rcode}): {msg}{hint}")

            items = _extract_items(payload)
            if not items:
                if page == 1:
                    raise RuntimeError("응답에 데이터가 없습니다. "
                                       "관측소코드/날짜/서비스키를 확인하세요.")
                break
            if meta is None:
                meta = {"name": items[0].get("obsvtrNm", obs_code),
                        "lat": items[0].get("lat"), "lot": items[0].get("lot")}
            all_items.extend(items)
            total = payload.get("body", {}).get("totalCount")
            if len(items) < NUM or (total and len(all_items) >= int(total)):
                break

        series: List[Sample] = []
        for it in all_items:
            dt = _parse_dt(str(it["obsrvnDt"]))
            pred = float(it["tdlvHgt"])
            obs = it.get("bscTdlvHgt")
            obs = float(obs) if obs not in (None, "", "-") else None
            series.append((dt, pred, obs))
        series.sort(key=lambda x: x[0])

        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_p, "w", encoding="utf-8") as f:
                json.dump({"meta": meta,
                           "series": [[dt.strftime("%Y-%m-%d %H:%M"), p, o]
                                      for dt, p, o in series]},
                          f, ensure_ascii=False)
        return meta, series

    def get_day(self, obs_code: str, date: str, min_interval: int = 10,
                use_cache: bool = True, sample: bool = False,
                margin: bool = True) -> dict:
        """하루치 만조/간조(극값)와 시계열을 반환.

        margin=True: 자정 경계에 걸치는 만조/간조를 놓치지 않도록 전날·다음날을
        함께 받아 곡선을 이어 붙인 뒤, 극값 중 대상일에 속하는 것만 남깁니다.
        (전/다음날 조회는 캐시되어 재사용됩니다. sample 모드는 당일만 가능해 margin off.)
        """
        from datetime import datetime as _dt, timedelta as _td
        center = _dt.strptime(date, "%Y%m%d").date()

        dates = [date]
        if margin and not sample:
            dates = [(center - _td(days=1)).strftime("%Y%m%d"), date,
                     (center + _td(days=1)).strftime("%Y%m%d")]

        meta = None
        full: List[Sample] = []
        for dd in dates:
            try:
                m, s = self.fetch_series(obs_code, dd, min_interval, use_cache, sample)
            except Exception:
                if dd == date:
                    raise          # 대상일 실패는 그대로 오류
                continue           # 이웃 날짜 실패는 무시(경계 보정만 약화)
            meta = meta or m
            full.extend(s)
        full.sort(key=lambda x: x[0])

        extrema = find_extrema(full)
        # 대상일에 속하는 극값만 유지
        day_ex = [e for e in extrema if e["time"].date() == center]
        # 대상일 시계열만 추려 반환(시각화/디버깅용)
        day_series = [s for s in full if s[0].date() == center]
        return {"obs_code": obs_code, "date": date, "meta": meta or {},
                "extrema": day_ex, "series": day_series}


# ---------------------------------------------------------------------------
# 만조/간조 추출: 예측 조위 곡선의 국소 극대(고조)·극소(저조)
# ---------------------------------------------------------------------------
def find_extrema(series: List[Sample], window: int = 6) -> List[dict]:
    """series=[(dt, pred_cm, obs_cm)] → [{'type':'고조'/'저조','time':dt,'level':cm}].

    반일주조(하루 2고조·2저조) 예측 조위의 '내부 전환점'만 만조/간조로 잡습니다.
    - 하루 경계(첫/마지막 표본)는 전환점이 아니므로 제외.
    - 반경 window 안에서 좌우 모두보다 크면(작으면) 고조(저조).
    - 같은 종류가 연달아 나오면 더 극단인 값으로 병합(평탄부 중복 제거).
    window: 좌우 비교 반경(표본 수). 10분 간격이면 6 = ±60분.
    """
    n = len(series)
    if n < 3:
        return []
    levels = [p for _, p, _ in series]
    times = [t for t, _, _ in series]
    out: List[dict] = []
    for i in range(1, n - 1):                      # 경계 제외
        v = levels[i]
        left = levels[max(0, i - window):i]
        right = levels[i + 1:i + window + 1]
        if not left or not right:
            continue
        is_max = v >= max(left) and v >= max(right) and (v > left[0] or v > right[-1])
        is_min = v <= min(left) and v <= min(right) and (v < left[0] or v < right[-1])
        if is_max:
            kind = "고조"
        elif is_min:
            kind = "저조"
        else:
            continue
        if out and out[-1]["type"] == kind:        # 평탄부 병합
            if (kind == "고조" and v > out[-1]["level"]) or \
               (kind == "저조" and v < out[-1]["level"]):
                out[-1] = {"type": kind, "time": times[i], "level": v}
            continue
        out.append({"type": kind, "time": times[i], "level": v})
    return out
