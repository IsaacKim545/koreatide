# 물때 홈페이지 배포 가이드 — Render + Porkbun (koreatide.com)

Flask 앱을 Render(PaaS)에 올리고 `koreatide.com` 도메인을 연결합니다. HTTPS는 Render가
자동 발급합니다. 서비스키는 서버 환경변수로만 주입하고, 코드/깃에는 넣지 않습니다.

준비물: GitHub 계정, Render 계정(render.com, 무료), Porkbun에서 산 `koreatide.com`.

---

## 1) 커밋 전 점검

- `data/stations_geo.json` 이 있어야 지도가 서버에서도 동작합니다. 없으면 먼저 생성:
  ```
  python scripts\build_geo.py
  ```
- `config/khoa_key.txt`(개인 키)는 `.gitignore`에 있어 **커밋되지 않습니다**. 서버에는
  아래 6단계에서 환경변수로 넣습니다. (절대 깃에 올리지 마세요.)

## 2) GitHub에 올리기

프로젝트 폴더(E:\LLM)에서:
```
git init
git add .
git commit -m "물때 홈페이지"
```
GitHub에서 새 저장소(예: `koreatide`)를 만든 뒤, 안내에 나온 대로:
```
git remote add origin https://github.com/<사용자명>/koreatide.git
git branch -M main
git push -u origin main
```

## 3) Render에 웹 서비스 만들기

1. render.com 로그인 → **New +** → **Web Service**
2. 방금 만든 GitHub 저장소를 연결(Connect)
3. 저장소에 `render.yaml`이 있어 대부분 자동 인식됩니다. 수동 설정이 필요하면:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements-web.txt`
   - **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 60`
   - **Plan**: Free (또는 Starter)

## 4) 서비스키 환경변수 설정 (중요)

Render 서비스 → **Environment** → **Add Environment Variable**:
- Key: `KHOA_API_KEY`
- Value: (발급받은 KHOA 서비스키 — Decoding 키)

저장하면 앱이 이 값을 자동으로 사용합니다(코드의 keyconf가 환경변수를 우선 사용).

**(선택) 구글 애널리틱스**: GA4 측정 ID가 있으면 환경변수 하나 더 추가하세요.
- Key: `GA_MEASUREMENT_ID`
- Value: `G-XXXXXXXXXX` (GA4 속성 → 데이터 스트림에서 확인)

설정하면 페이지에 gtag가 자동 삽입되고, '조회' 시 `query_tide` 이벤트도 전송됩니다.
(설정 안 하면 애널리틱스 코드는 삽입되지 않습니다.)

## 5) 배포 & 임시주소 확인

**Deploy** 후 몇 분 기다리면 `https://koreatide.onrender.com` 같은 임시 주소가 생깁니다.
접속해서 지점 조회·지도·그래프가 정상인지 확인하세요.
(무료 플랜은 일정 시간 미사용 시 잠들어 첫 접속이 ~30초 느릴 수 있습니다. Starter로
올리면 상시 가동됩니다.)

## 6) koreatide.com 연결

1. Render 서비스 → **Settings** → **Custom Domains** → **Add Custom Domain**
   - `koreatide.com` 과 `www.koreatide.com` 둘 다 추가
2. Render가 **연결에 필요한 DNS 레코드**를 보여줍니다(아래는 일반적인 형태, 화면에 뜬 값을 그대로 쓰세요):
   - 최상위(`koreatide.com`): **A 레코드** → Render가 알려주는 IP (예: 216.24.57.x)
     - Porkbun이 apex CNAME을 막으면 A 레코드 사용. Porkbun은 **ALIAS**도 지원하니, Render가
       CNAME 타깃(`koreatide.onrender.com`)을 주면 `ALIAS @ → koreatide.onrender.com` 로 넣어도 됩니다.
   - `www`: **CNAME** → `koreatide.onrender.com`
3. **Porkbun**: 도메인 관리 → **DNS** 에서 위 레코드를 추가/수정
   - 기존 Porkbun 기본 파킹 레코드(A @ 등)는 지우거나 위 값으로 교체
4. 저장 후 DNS 전파(수분~수시간) 뒤, Render가 **SSL 인증서를 자동 발급**합니다.
   Custom Domains 화면에 "Verified" + 자물쇠가 뜨면 완료.

이제 `https://koreatide.com` 으로 접속됩니다.

---

## 참고

- **Railway**로 하려면: 저장소 연결 → `Procfile` 자동 인식 → Variables에 `KHOA_API_KEY`
  추가 → Settings에서 커스텀 도메인 연결(Porkbun DNS에 CNAME). 흐름은 Render와 동일합니다.
- **재배포**: GitHub `main`에 push하면 Render가 자동으로 다시 배포합니다.
- **보안**: 키는 항상 환경변수로. `debug`는 프로덕션에서 꺼진 상태로 동작합니다(gunicorn 사용).
- **비용**: 도메인(연 ~$11) + Render Free($0, 콜드스타트 있음) 또는 Starter(월 ~$7, 상시가동).
- 조회 캐시(`data/tide_cache/`)는 서버 재배포 시 초기화되지만, 캐시일 뿐 문제 없습니다.
