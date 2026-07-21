---
name: project-tide-engine
description: E:\KoreaTide 프로젝트의 물때(조석) 조회 엔진 — KHOA API 기반 전국 만조/간조·물때
metadata:
  type: project
---

E:\KoreaTide에 대한민국 물때(조석) 조회 엔진을 추가함(2026-07-10). 사용자가 "물때 알려주는 AI"를 원했고, LLM 예측은 부정확하므로 공식 데이터 기반 결정론적 엔진으로 구현하기로 함(사용자 선택: 정확한 계산·조회 엔진 / 전국 주요 지점 / 데이터 없음→공식 출처).

데이터 출처: 국립해양조사원(KHOA) 바다누리 OpenAPI, API명 GetSurveyTideLevelApiService (SV_AP_02_009, 조위관측소 실측·예측 조위).
엔드포인트(중요): 인증용은 공공데이터포털 https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService (serviceKey 사용). KHOA 자체 https://www.khoa.go.kr/oceandata/odmiapi/GetSurveyTideLevelApiService.do 는 샘플(isSample=Y) 전용 — serviceKey 인증 안 됨(resultCode=11의 원인이었음). 브라우저로 직접 확인: .do+키→11, apis.data.go.kr+더미키→Unauthorized(정상 엔드포인트), .do 없는 khoa 주소→404.
파라미터: serviceKey, obsCode(DT_0001 등), reqDate(YYYYMMDD), type=json, min(간격분), numOfRows(≤300), pageNo. serviceKey 없이 isSample=Y로 인천 샘플 테스트 가능.
응답 JSON: header.resultCode("00"=정상), body.items.item[] = {obsvtrNm, lat, lot, obsrvnDt, bscTdlvHgt(실측cm), tdlvHgt(예측cm)}. 1분 간격 1440건/일.

구현: src/tide/(stations.py=전국 55개 관측소 코드↔이름 공식목록, khoa.py=API클라이언트+페이징+캐시+만조/간조 극값추출 find_extrema, mulddae.py=음력 서해식 7물때 계산), scripts/tide.py(CLI), scripts/tide_stations.py, data/stations.json, data/tide_cache/.

만조/간조는 예측조위 곡선의 내부 전환점(극대=고조/만조, 극소=저조/간조)으로 추출(하루 경계 artifact 제외). 물때 이름은 음력일 기반 서해 7물때식(관례, offset 보정 가능), 조차(cm)는 객관적 지표. 음력변환은 korean_lunar_calendar(선택).

검증: 실제 KHOA JSON 파싱, 극값추출(4개 만조/간조 교대), 물때표(음력1→7물,8→조금 등), 55개 관측소 로드/검색 모두 통과. 라이브 API는 키 필요해 사용자 환경에서 테스트 예정(--sample로 키 없이 인천 확인 가능).

자연어 인터페이스(2026-07-10 추가): src/tide/nlu.py(규칙기반 파서 — 관측소명/코드 + 날짜: 오늘·내일·모레·글피·N일뒤·M월D일·ISO·요일 추출, 긴이름 우선매칭), scripts/tide_chat.py(자연어 질의→엔진→한국어 답변 REPL, --ask 원샷). LLM 대신 규칙기반(정확). NLU 9케이스 + 답변 포맷 검증 통과.

멀티턴 문맥: parse_date_opt(날짜 없으면 None). '초기화'/'/reset'로 리셋.

문맥 재설계(2차 수정): (1) parse_station에 접미사(신항/대교/항/도) 제거 매칭 추가 — '거제'→거제도, '가덕'→가덕도, '영종'→영종대교, '동해'→동해항. 단 핵심어 2자 이상만(완도/진도/위도는 전체매칭 유지). (2) 날짜 leak 수정 — last_date 상시유지 폐기. tide_chat이 resolve_query+render로 분리; pending_date는 '지점 되묻기' 직후에만 사용하고 완결질의는 pending 소비. 즉 날짜는 명시>되묻기pending>오늘 순, 오래된 날짜가 새 질문에 새지 않음. (사용자 사례: "내일 거제"실패→"삼천포"가 내일로 나오던 버그 해결). 검증 완료.

인증키 관리(2026-07-10): src/tide/keyconf.py — 키 우선순위(인자>env KHOA_API_KEY>파일 config/khoa_key.txt), save/load, 엔드포인트 override(KHOA_API_URL/--api-url, data.go.kr 대응). CLI에 --set-key/--api-url 추가. 인증 요청 URL 검증됨: GetSurveyTideLevelApiService.do?serviceKey=<키>&obsCode=&reqDate=&type=json&numOfRows=300&pageNo=&min=. .gitignore로 khoa_key.txt/캐시/runs 제외.

버그수정(2026-07-10): (a) resultCode=11 원인은 serviceKey 인코딩 — _build_url에서 %없는 Decoding키는 quote(safe='')로 퍼센트인코딩, %있는 Encoding키는 그대로. (b) --debug로 마스킹된 요청 URL 출력. (c) get_day에 margin=True: 전날/대상/다음날 조회 이어붙여 find_extrema 후 대상일 극값만 필터 → 자정 경계 만조/간조 복원(검증됨). 에러메시지에 키/엔드포인트 힌트 추가. 첫 성공(인천 당일)은 실은 sample 모드였음.

홈페이지/확장(2026-07-10): 사용자가 Flask 기반 정식 홈페이지(로컬 우선) 선택. 추가: src/tide/advice.py(낚시·갯벌 적기: 사리/조금·조차·간조시각→갯벌 드러남·물돌이·안전경고), src/tide/report.py(day_report/week_report 구조화 dict, CLI·웹 공용), scripts/tide.py --days N(주간표), web/app.py(Flask: /api/stations, /api/tide?station=&date=&days=; 키는 서버측만), web/templates/index.html(지점 datalist+날짜+기간 선택, 일별 카드=물때뱃지/만조간조칩/조언). 실행: pip install flask; python web/app.py → 127.0.0.1:5000. advice/주간집계 순수검증 통과.

기능확장(2026-07-10, Starter플랜): 구글애널리틱스(app.py가 GA_MEASUREMENT_ID env→템플릿 주입, gtag+query_tide 이벤트), 즐겨찾기(localStorage tide_favs, ☆토글+칩), 여행자가이드(접이식 details: 갯벌/해수욕/해루질/안전 팁), 조위그래프 개선(그라디언트 채움, y축 cm라벨, 오늘 '지금' 세로선+현재수위 점). 검증: Jinja 안전(GA블록만), 그래프좌표/보간/즐겨찾기 로직 통과.

런칭 후 작업 로그(2026-07-12):
- SEO: 구글 서치콘솔(DNS TXT 소유확인)+사이트맵 제출+색인요청 완료. 네이버 서치어드바이저 소유확인(HTML태그, NAVER_SITE_VERIFICATION env) 완료. robots.txt/sitemap.xml 라우트 추가(app.py). GA4 연동됨(측정ID G-X515R5BWDV, GA_MEASUREMENT_ID env).
- 공유: OG/트위터 메타태그, favicon(web/static/favicon.svg), og.png(web/static/, 1200x630 파도·달 배너) — 카카오톡 리치 미리보기 확인됨. theme-color.
- 기능: 일출·일몰(src/tide/sun.py, API불필요 계산, report에 sun, 그래프 밤음영+카드 🌅🌇). 개인정보처리방침(/privacy, 문의 itdasoft1@gmail.com). 디자인 폴리시(배경 그라디언트·카드 그림자·모바일 반응형).
- robots.txt 절충안: AI 검색봇(Google-Extended·OAI-SearchBot·PerplexityBot 등) 허용, AI 학습봇(GPTBot·CCBot·ClaudeBot·anthropic-ai·Bytespider·Applebot-Extended) 차단. 일반 검색봇은 그대로 허용.
- HTTP→HTTPS 강제+HSTS(app.py). no-cache 헤더(재배포 즉시반영).
- 대표 도메인 통일: apex(koreatide.com)를 canonical로. Render Custom Domains에서 www 삭제→apex 추가 방식으로 primary를 apex로 전환. koreatide.com 직접 200 확인, www→apex 리다이렉트는 전파 마무리 중. 코드 태그(canonical/og/sitemap)는 이미 apex 기준이라 변경 없음.
- 옛 도메인(한류 KoreaTide) 검색결과: 내 도메인엔 옛 페이지 없음(site:검색 확인). 타사이트(Sur.ly/FB/Pinterest)는 제거 불가—시간이 지나며 밀려남. 구글 AI개요에 이제 내 사이트 노출됨.
- 보류: 광고(애드센스, 원래 3번 작업). 개인정보처리방침·트래픽 갖춰지면 추후 신청 예정.

배포 완료(2026-07-10): koreatide.com (Porkbun 도메인 + Render PaaS) 라이브. www=CNAME, 루트@=ALIAS/A(216.24.57.1)→onrender. KHOA_API_KEY는 Render 환경변수. 주간조회는 margin=False로 API호출 절감+gunicorn threads4/timeout120(무료플랜 콜드스타트/타임아웃 대응). 갯벌/낚시 태그는 사용자 요청으로 제거(사리/조금/중간물 태그만 유지). 배포 반영 안 될 때 원인=브라우저 캐시(Ctrl+Shift+R).

배포(2026-07-10): 사용자가 Porkbun에서 koreatide.com 구매 결정, Render(PaaS)에 배포하기로. 준비파일: web/__init__.py(패키지화), wsgi.py(from web.app import app), requirements-web.txt(flask/gunicorn/korean_lunar_calendar만 — torch 불필요, 웹런타임 torch-free 확인됨), render.yaml(빌드/스타트/KHOA_API_KEY env sync:false), Procfile, DEPLOY.md(GitHub→Render→env키→Porkbun DNS 커스텀도메인 단계별). 키는 KHOA_API_KEY 환경변수로 주입(keyconf가 우선사용), config/khoa_key.txt는 gitignore. stations_geo.json은 커밋해야 서버 지도 동작.

그래프·지도(2026-07-10): (A) 조위 곡선 — report.day_report에 다운샘플 series(~48pt) 추가, index.html이 SVG로 곡선+만조/간조 마커 렌더(검증됨). (B) 지도 — scripts/build_geo.py가 API로 55지점 lat/lon을 data/stations_geo.json에 캐시(1회, 키필요), app.py /api/stations가 좌표 포함, index.html에 Leaflet 지도+마커클릭 조회(좌표없으면 목록만). 배포(C)는 사용자가 도메인으로 진행 예정.

주의: KHOA 관측소 코드 전체는 브라우저로 공식 목록 select option value에서 추출한 것(권위본). 55개. 코드 수정 후 실행중인 tide/web 프로세스는 재시작해야 반영됨(파이썬 모듈 캐시). E:\KoreaTide는 bash 마운트 불가 — 검증은 로직 인라인 재현으로.
