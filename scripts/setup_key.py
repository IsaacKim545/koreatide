#!/usr/bin/env python
"""KHOA 서비스키 입력·검증 — config/khoa_key.txt.

`tide.py --set-key <키>` 와 같은 일을 하지만 두 가지가 다릅니다.
  1) 키를 화면에 찍지 않고 받습니다(getpass). 명령 기록(history)에도 남지 않습니다.
     --set-key 방식은 키가 셸 기록 파일에 평문으로 남아 유출 위험이 있습니다.
  2) 저장 전에 실제 API를 한 번 호출해 키가 동작하는지 확인합니다.
     data.go.kr 키는 '활용신청 승인' 전이거나 Encoding/Decoding 키를 잘못 고르면
     조용히 실패하는데, 그걸 저장 시점에 바로 잡아줍니다.

사용:
    python scripts/setup_key.py            # 입력 → 검증 → 저장
    python scripts/setup_key.py --show     # 현재 저장된 키 상태만 확인
    python scripts/setup_key.py --no-verify   # 검증 건너뛰고 저장
    python scripts/setup_key.py --key <키>    # 비대화식(자동화용, 기록 남음 주의)

키 발급: 공공데이터포털(data.go.kr) → '조위관측소 실측·예측 조위' 활용신청 → 승인 후
        마이페이지에서 일반 인증키 확인. Decoding 키를 넣으면 됩니다.
"""
import argparse
import getpass
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tide.keyconf import (  # noqa: E402
    load_service_key, save_service_key, get_api_url, _key_file,
)
from src.tide.khoa import KhoaTideClient  # noqa: E402

TEST_CODE = "DT_0001"      # 인천 — 항상 존재하는 관측소


def mask(key: str) -> str:
    """키를 앞뒤 일부만 남기고 가림."""
    k = key.strip()
    if len(k) <= 12:
        return "*" * len(k)
    return f"{k[:6]}…{k[-4:]}  (총 {len(k)}자)"


def show_status():
    path = os.path.abspath(_key_file())
    env = os.environ.get("KHOA_API_KEY")
    key = load_service_key()

    print(f"설정 파일: {path}")
    print(f"  존재 여부: {'있음' if os.path.exists(path) else '없음'}")
    if env:
        print(f"환경변수 KHOA_API_KEY: 설정됨 — {mask(env)}")
        print("  * 환경변수가 설정 파일보다 우선합니다.")
    else:
        print("환경변수 KHOA_API_KEY: 없음")
    print(f"엔드포인트: {get_api_url()}")
    print()
    if key:
        print(f"현재 사용될 키: {mask(key)}")
    else:
        print("현재 사용될 키: 없음 → 샘플 모드(인천만 조회 가능)")
    return key


def verify(key: str) -> bool:
    """실제 API를 한 번 호출해 키가 동작하는지 확인."""
    client = KhoaTideClient(service_key=key, base_url=get_api_url())
    # 어제 날짜 — 오늘치가 아직 안 올라온 경우를 피함
    ymd = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    print(f"\n검증 중… (인천 {ymd} 조회)")
    try:
        # use_cache=False: 캐시된 옛 응답이 아니라 실제 인증을 확인해야 함
        meta, series = client.fetch_series(TEST_CODE, ymd, min_interval=60,
                                           use_cache=False)
    except Exception as e:  # noqa
        print(f"  실패: {str(e)[:300]}")
        print("\n확인해 보세요:")
        print("  1) 공공데이터포털에서 해당 API '활용신청'이 승인됐는지")
        print("     (승인 전에는 키가 있어도 인증 실패합니다)")
        print("  2) Encoding 키가 아니라 Decoding 키를 넣었는지")
        print("  3) 승인 직후라면 반영까지 최대 1시간 걸릴 수 있습니다")
        return False

    if not series:
        print("  실패: 응답은 왔지만 데이터가 비어 있습니다.")
        return False

    name = (meta or {}).get("name", TEST_CODE)
    print(f"  성공: {name} {len(series)}개 관측값 수신")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="현재 키 상태만 확인하고 종료")
    ap.add_argument("--key", default=None,
                    help="비대화식 입력(자동화용). 셸 기록에 남으니 주의")
    ap.add_argument("--no-verify", action="store_true", help="API 검증 건너뛰기")
    args = ap.parse_args()

    print("=" * 56)
    print(" KHOA 서비스키 설정")
    print("=" * 56)
    existing = show_status()

    if args.show:
        return

    print("-" * 56)
    if existing and not args.key:
        ans = input("이미 키가 있습니다. 새 키로 교체할까요? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("취소했습니다. 기존 키를 그대로 씁니다.")
            return

    if args.key:
        key = args.key.strip()
    else:
        print("발급받은 서비스키를 붙여넣으세요. (입력 내용은 화면에 보이지 않습니다)")
        try:
            key = getpass.getpass("키: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n취소했습니다.")
            return

    if not key:
        print("[오류] 빈 값입니다. 저장하지 않았습니다.")
        sys.exit(1)

    # 흔한 실수: 따옴표째 붙여넣기
    if key[0] == key[-1] and key[0] in ("'", '"') and len(key) > 1:
        key = key[1:-1].strip()
        print("  (앞뒤 따옴표를 제거했습니다)")

    if " " in key:
        print("[경고] 키에 공백이 있습니다. 붙여넣기가 잘렸을 수 있습니다.")

    print(f"입력된 키: {mask(key)}")

    if not args.no_verify:
        if not verify(key):
            ans = input("\n검증에 실패했습니다. 그래도 저장할까요? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("저장하지 않았습니다.")
                sys.exit(1)

    path = save_service_key(key)
    print(f"\n저장 완료: {os.path.abspath(path)}")
    print("  * 이 파일은 .gitignore에 포함되어 커밋되지 않습니다.")
    print("\n다음 단계:")
    print("  python scripts/tide.py --station 인천        # 조회 테스트")
    print("  python scripts/build_ranges.py               # 관측소별 기준선 생성")
    print("\n서버(Render)는 이 파일이 아니라 환경변수를 씁니다:")
    print("  Environment → KHOA_API_KEY = <같은 키>")


if __name__ == "__main__":
    main()
