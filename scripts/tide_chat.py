#!/usr/bin/env python
"""물때 대화 도우미 — 자연어로 물때를 물어보세요.

"내일 인천 물때 알려줘", "부산 7월 15일 만조 시간" 처럼 물으면
지점·날짜를 해석해 국립해양조사원 예측조위로 정확히 답합니다.
(질의 해석은 규칙 기반 NLU, 값은 KHOA 공식 데이터)

사용:
    python scripts/tide_chat.py                       # 대화 모드
    python scripts/tide_chat.py --ask "내일 인천 물때"   # 한 번만
    python scripts/tide_chat.py --sample              # 키 없이(인천) 테스트

서비스키: 환경변수 KHOA_API_KEY 또는 --key. 없으면 인천 샘플만 가능.
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tide.nlu import parse_station, parse_date_opt  # noqa: E402
from src.tide.stations import station_name, load_stations  # noqa: E402
from src.tide.khoa import KhoaTideClient  # noqa: E402
from src.tide.mulddae import get_mulddae, range_phase  # noqa: E402
from src.tide.keyconf import load_service_key, save_service_key, get_api_url  # noqa: E402

HL_KO = {"고조": "만조", "저조": "간조"}
_WD = ["월", "화", "수", "목", "금", "토", "일"]


ASK_STATION = ("어느 지점의 물때인지 알려주세요. 예: '내일 인천 물때' "
               "(전체 목록은 '목록'이라고 입력)")


def render(client, code, d, sample=False, offset=0) -> str:
    """(관측소코드, 날짜) → 답변 문자열. 조회+포맷만 담당(문맥 판단은 호출부)."""
    ymd = d.strftime("%Y%m%d")
    name = station_name(code)
    if sample and code != "DT_0001":
        return (f"서비스키가 없어 '인천'만 조회할 수 있어요. '{name}' 조회는 "
                f"무료 키 발급 후 가능합니다.")
    try:
        day = client.get_day(code, ymd, sample=sample)
    except Exception as e:  # noqa
        return f"'{name}' {d:%Y-%m-%d} 조회 중 문제가 생겼어요: {e}"

    name = day["meta"].get("name", name)
    ex = day["extrema"]
    if not ex:
        return f"{name} {d:%Y-%m-%d} 만조/간조를 계산하지 못했어요."

    highs = [e for e in ex if e["type"] == "고조"]
    lows = [e for e in ex if e["type"] == "저조"]

    def fmt(lst):
        return ", ".join(f"{e['time']:%H:%M}({e['level']:.0f}cm)" for e in lst)

    wd = _WD[d.weekday()]
    lines = [f"[{name}] {d:%Y-%m-%d}({wd})"]
    mul = get_mulddae(datetime(d.year, d.month, d.day), offset=offset)
    if mul:
        lines.append(f"물때: {mul['name']} ({mul['phase']}) · 음력 {mul['lunar'][1]}.{mul['lunar'][2]}")
    if highs:
        lines.append(f"만조: {fmt(highs)}")
    if lows:
        lines.append(f"간조: {fmt(lows)}")
    if highs and lows:
        rng = max(h["level"] for h in highs) - min(l["level"] for l in lows)
        lines.append(f"최대 조차: {rng:.0f}cm ({range_phase(rng)})")
    return "\n".join(lines)


def resolve_query(text, last_code, pending_date, today):
    """질의에서 (관측소코드, 사용날짜, 새 pending_date, 되묻기여부)를 결정.

    날짜 문맥 규칙(예측 가능하도록):
    - 날짜는 이번 문장에 명시된 값을 최우선.
    - 명시가 없으면, '지점 되묻기' 직후에만 그때 저장한 pending_date를 사용.
    - 그 외에는 오늘. (오래된 날짜가 엉뚱한 질문에 새어들지 않게 함)
    """
    explicit = parse_date_opt(text, today)
    code = parse_station(text) or last_code
    eff_date = explicit or pending_date or today
    if not code:
        # 지점을 못 찾음 → 되묻고, 이번에 준 날짜만 다음 턴을 위해 보관
        return None, eff_date, explicit, True
    return code, eff_date, None, False   # 완결된 질의 → pending 소비/초기화


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", default=None, help="한 번만 질문하고 종료")
    ap.add_argument("--key", default=None)
    ap.add_argument("--set-key", default=None, help="서비스키를 저장하고 종료")
    ap.add_argument("--api-url", default=None, help="API 엔드포인트 override")
    ap.add_argument("--sample", action="store_true", help="키 없이 인천 샘플")
    ap.add_argument("--offset", type=int, default=0, help="물때 지역보정(일)")
    ap.add_argument("--debug", action="store_true", help="요청 URL(키 마스킹) 출력")
    args = ap.parse_args()

    if args.set_key:
        path = save_service_key(args.set_key)
        print(f"서비스키를 저장했습니다: {path}")
        return

    key = load_service_key(args.key)
    sample = args.sample or not key
    client = KhoaTideClient(service_key=key, base_url=get_api_url(args.api_url),
                            debug=args.debug)

    from datetime import datetime as _dt
    if args.ask:
        today = _dt.now().date()
        code, d, _, ask = resolve_query(args.ask, None, None, today)
        print(ASK_STATION if ask else render(client, code, d, sample, args.offset))
        return

    print("물때 도우미입니다. 예: '내일 인천 물때 알려줘'  (종료: /exit, 지점목록: 목록)")
    if sample:
        print("[안내] 서비스키가 없어 인천 샘플만 조회됩니다. 전국은 KHOA_API_KEY 설정 후.\n")
    last = None            # 직전 관측소 코드 (지점 생략 후속질문용)
    pending_date = None    # '지점 되묻기' 흐름에서만 이어받을 날짜
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q in ("목록", "관측소", "list"):
            st = load_stations()
            print("  " + ", ".join(sorted(st.values())) + "\n")
            continue
        if q in ("초기화", "/reset"):
            last, pending_date = None, None
            print("[대화 문맥 초기화]\n")
            continue
        today = _dt.now().date()
        code, d, pending_date, ask = resolve_query(q, last, pending_date, today)
        if ask:
            print(ASK_STATION + "\n")
            continue
        last = code
        print(render(client, code, d, sample, args.offset) + "\n")


if __name__ == "__main__":
    main()
