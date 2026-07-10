# 10B LLM — 학습 코드베이스

10B 파라미터 규모의 decoder-only 트랜스포머(Llama 계열)를 밑바닥부터 사전학습하기 위한 PyTorch 코드베이스입니다.
클라우드 멀티 GPU(A100/H100) 환경에서 **FSDP + bf16 + activation checkpointing**으로 학습하도록 설계되었습니다.

## 아키텍처

- Decoder-only Transformer (pre-norm)
- **RMSNorm** 정규화
- **RoPE** (Rotary Position Embedding)
- **GQA** (Grouped-Query Attention)
- **SwiGLU** FFN
- **SDPA / FlashAttention** (PyTorch `scaled_dot_product_attention`)
- Weight tying 옵션, tied/untied lm_head 지원

기본 `10b` 설정 기준 ≈ 10.5B 파라미터 (hidden 4096 / layers 48 / heads 32 / kv_heads 8 / intermediate 14336 / vocab 32000).

## 디렉터리 구조

```
LLM/
├── README.md
├── requirements.txt
├── configs/
│   ├── small.json      # ~110M, 코드 검증/디버깅용
│   └── 10b.json        # ~10.5B, 본 학습용
├── src/
│   ├── __init__.py
│   ├── config.py       # ModelConfig / TrainConfig 데이터클래스
│   ├── model.py        # Transformer 구현
│   ├── tokenizer.py    # BPE 토크나이저 학습/로드 래퍼
│   ├── data.py         # 토큰화, memmap, packed dataset/dataloader
│   ├── trainer.py      # FSDP 학습 루프
│   └── utils.py        # 분산/체크포인트/로깅 유틸
│   ├── serve.py        # 배치 추론 엔진 + 동적 배처
│   └── utils.py        # 분산/체크포인트/로깅 유틸
└── scripts/
    ├── train_tokenizer.py   # BPE 학습
    ├── prepare_data.py      # jsonl/txt → 토큰 memmap (.bin)
    ├── train.py             # 학습 진입점 (torchrun)
    ├── generate.py          # 단일 추론/샘플링 (KV 캐시)
    ├── serve.py             # 배치 추론 HTTP 서버
    ├── export.py            # 체크포인트 → 단일 state_dict
    ├── smoke_test.py        # 엔드투엔드 스모크 테스트
    └── launch.sh            # 멀티노드 torchrun 예시
```

## 주요 기능

- **KV 캐시 추론**: `generate()`가 prefill + 증분 디코딩, EOS 조기 종료 지원.
- **배치 추론 서버**: left-padding + attention mask로 서로 다른 길이 프롬프트를
  한 배치로 처리. 동적 배처가 동시 요청을 모아 throughput을 높입니다. (`scripts/serve.py`)
- **분산 sharded 체크포인트**: `torch.distributed.checkpoint`(DCP)로 각 rank가
  자기 샤드를 병렬 저장. 모델+옵티마이저+step을 모두 담아 학습 재개 가능하며,
  다른 world_size로도 재개(resharding)됩니다. `final` 시 배포용 consolidated
  `.pt`도 함께 생성.
- **데이터**: 비겹침 청크 + `DistributedSampler` 셔플/샤딩(기본) 또는 무작위 윈도우.
- **W&B 로깅 + MFU** 추정.

## 추론 서버

```bash
python scripts/serve.py --ckpt runs/10b/ckpt_final_full.pt --tokenizer tokenizer/ --port 8000

curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' \
    -d '{"prompt": "옛날 옛적에", "max_new_tokens": 128, "temperature": 0.8, "top_k": 50}'
```

## 체크포인트에서 재개

```bash
# sharded 디렉터리에서 재개 (모델+옵티마이저+step 복원)
torchrun --nproc_per_node=8 scripts/train.py --config configs/10b.json \
    --data data/train.bin --out runs/10b --resume runs/10b/ckpt_step10000
```

## 물때(조석) 조회 — 대한민국 전국

국립해양조사원(KHOA) 예측조위로 **만조/간조 시각·조차·물때**를 계산합니다. (LLM이 아니라
공식 데이터를 쓰는 정확한 엔진입니다.) 전국 55개 조위관측소를 지원합니다.

```bat
REM 1) 관측소 목록 보기
python scripts\tide.py --list

REM 2) 서비스키 없이 바로 테스트 (샘플, 인천만)
python scripts\tide.py --station 인천 --sample

REM 3) 인증키 저장 (한 번만; config\khoa_key.txt 에 보관)
python scripts\tide.py --set-key 발급받은서비스키

REM 4) 이후로는 전국 어디든 바로 조회
python scripts\tide.py --station 부산 --date 2026-07-15
```

**인증키 사용 방법 (3가지, 우선순위 순)**

1. 명령 인자: `--key 발급키`
2. 환경변수: `set KHOA_API_KEY=발급키`
3. 저장 파일: `python scripts\tide.py --set-key 발급키` → `config\khoa_key.txt` (한 번 저장하면 계속 사용)

- 인증 엔드포인트(기본): `https://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService`
  (공공데이터포털에서 "국립해양조사원 조위관측소 실측·예측 조위" 활용신청 후 받은 serviceKey 사용)
- KHOA 자체 `...odmiapi/...do` 주소는 **샘플 전용**이라 serviceKey 인증이 안 됩니다(엔진은
  `--sample`일 때만 이 주소를 씁니다).
- 다른 요청주소가 필요하면 `--api-url <요청주소>` 또는 `set KHOA_API_URL=...` 로 바꿉니다.
- 키가 안 될 때(`resultCode=11/30`): data.go.kr 키는 "Encoding"과 "Decoding" 두 형태가
  있습니다. 엔진은 원문(Decoding) 키를 자동으로 퍼센트 인코딩하므로 **Decoding 키를 넣어보세요**.
  `--debug` 로 실제 요청 URL(키는 마스킹)을 확인할 수 있습니다.
- 자정 경계 보정: 만조/간조가 자정 근처에 걸쳐도 놓치지 않도록 전날·다음날을 함께 조회해
  이어 붙인 뒤 대상일 극값만 남깁니다(이웃 날짜는 캐시됨). `get_day(margin=False)`로 끌 수 있습니다.
- `config\khoa_key.txt` 는 개인 키라 `.gitignore` 에 포함되어 공유되지 않습니다.

출력 예:

```
== 인천 (DT_0001) / 2026-07-15 ==
물때: 7물 (사리(대조))  |  음력 6.1
--------------------------------
 간조  03:12       45 cm
 만조  09:20      812 cm
 간조  15:40      120 cm
 만조  21:55      760 cm
--------------------------------
 최대 조차: 767 cm (큰 조차)
```

- **서비스키 발급(무료)**: 공공데이터포털(data.go.kr)에서 "국립해양조사원_조위관측소
  실측·예측 조위" 활용신청, 또는 KHOA 바다누리(khoa.go.kr) 오픈API 신청.
- **물때 이름**은 서해안 7물때식 관례이며 지역차가 있어 `--offset` 으로 보정합니다.
  조수 세기의 객관적 지표인 **조차(cm)** 는 예측조위로 정확히 계산됩니다.
- 음력 물때 이름 표시는 `pip install korean_lunar_calendar` 가 필요합니다(없어도
  만조/간조·조차는 정상 출력).
- 조회 결과는 `data/tide_cache/` 에 캐시되어 재조회·오프라인에 재사용됩니다.

### 물때 홈페이지 (웹)

지점·날짜를 골라 **주간 물때표 + 낚시/갯벌 적기 안내**를 브라우저로 봅니다. 서비스키는
서버(config/khoa_key.txt 또는 KHOA_API_KEY)에서만 쓰이고 브라우저엔 노출되지 않습니다.

```bat
pip install flask
python web\app.py
REM → 브라우저에서 http://127.0.0.1:5000
```

기능: 일별 **조위 곡선 그래프**(만조/간조 마커), 주간 물때표, 낚시/갯벌 조언.
구조: `web/app.py`(Flask 백엔드 + JSON API), `web/templates/index.html`(프론트).
API: `GET /api/stations`, `GET /api/tide?station=인천&date=2026-07-16&days=7`.

**지도에서 지점 선택**을 쓰려면 지점 좌표를 한 번 캐시하세요(서비스키 필요):

```bat
python scripts\build_geo.py       REM data\stations_geo.json 생성 (1회)
```

이후 웹 상단 '🗺 지도' 버튼으로 전국 관측소를 지도에서 클릭해 조회할 수 있습니다.
(좌표 캐시가 없으면 지도 대신 목록/검색만 사용됩니다.)

### 주간 물때표 (CLI)

```bat
python scripts\tide.py --station 인천 --date 2026-07-16 --days 7
```

### 자연어로 물어보기 (물때 도우미)

지점·날짜를 문장에서 알아듣고 답합니다(규칙 기반 해석 + KHOA 공식 데이터):

```bat
python scripts\tide_chat.py --sample --ask "내일 인천 물때 알려줘"
python scripts\tide_chat.py                 REM 대화 모드 (전국, KHOA_API_KEY 설정 시)
```

예: "부산 7월 15일 만조 시간", "모레 목포", "3일 뒤 통영", "토요일 강화대교" 등을 이해합니다.
답변 예:

```
[인천] 2026-07-15(수)
물때: 7물 (사리(대조)) · 음력 6.1
만조: 09:20(812cm), 21:55(760cm)
간조: 03:12(45cm), 15:40(120cm)
최대 조차: 767cm (매우 큰 조차(사리급))
```

## 가장 빠른 시작 (Windows, 클릭 한 번)

전체 파이프라인(토크나이저 → 데이터 → 사전학습 → SFT → 대화)을 샘플 데이터로 한 번에 돌려봅니다.
CPU만으로도 몇 분이면 끝나며, "대화 가능한" 토이 모델까지 만들어집니다
(데이터·파라미터가 아주 작아 답변 품질은 낮습니다. 파이프라인 검증용):

```bat
pip install -r requirements.txt
run_demo.bat
```

마지막 단계에서 채팅 프롬프트가 뜨면 대화를 입력해 보세요 (`/exit`로 종료).

### Windows 주의사항

- **주석/줄바꿈**: cmd에서는 `#` 주석과 `\` 줄바꿈이 동작하지 않습니다. 명령은 한 줄로 입력하세요.
- **단일 GPU/CPU**: `torchrun`이 필요 없습니다. 그냥 `python scripts\train.py ...`로 실행하세요.
- **멀티 GPU에서 torchrun 오류**(`use_libuv ... without libuv support`): 실행 전 아래를 설정하세요.
  ```bat
  set USE_LIBUV=0
  torchrun --nproc_per_node=8 scripts\train.py --config configs\10b.json --data data\train.bin --out runs\10b
  ```
- **curl 예시**는 bash용입니다. Windows에서는 PowerShell의 `Invoke-RestMethod`를 쓰거나 큰따옴표로 바꾸세요.

## 빠른 시작 (수동, 단계별)

```bash
pip install -r requirements.txt

# 1) 토크나이저 학습 (코퍼스 일부로)
python scripts/train_tokenizer.py --input data/corpus.jsonl --vocab-size 32000 --out tokenizer/

# 2) 데이터 토큰화 → .bin memmap
python scripts/prepare_data.py --input data/corpus.jsonl --tokenizer tokenizer/ \
    --out data/train.bin --seq-len 4096

# 3) 소형 모델로 파이프라인 검증 (단일 GPU/CPU)
python scripts/train.py --config configs/small.json --data data/train.bin --max-steps 20

# 4) 본 학습 (8xGPU 단일 노드)
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/10b.json --data data/train.bin --out runs/10b

# 멀티노드는 scripts/launch.sh 참고
```

## 규모 감각 (10B, 대략)

- 파라미터: ~10.5B → bf16 가중치만 ~21GB
- 옵티마이저(AdamW fp32 m,v) + fp32 마스터: 파라미터당 약 16B → ~168GB, FSDP로 샤딩
- 토큰 예산: Chinchilla 최적 ≈ 20 tokens/param ≈ 200B+ 토큰
- 참고: 64×H100로 200B 토큰 학습 시 수 일~수 주 규모. 실제 비용/시간은 처리량에 크게 의존합니다.

자세한 하이퍼파라미터는 `configs/10b.json` 및 각 소스 파일의 docstring 참고.
