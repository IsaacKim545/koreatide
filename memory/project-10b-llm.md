---
name: project-10b-llm
description: E:\LLM 프로젝트 — 10B 파라미터 LLM을 밑바닥부터 사전학습하는 PyTorch 코드베이스
metadata:
  type: project
---

E:\LLM 프로젝트는 100억(10B) 파라미터 decoder-only 트랜스포머(Llama 계열)를 밑바닥부터 사전학습하는 것이 목표.

사용자 선택(2026-07-10): 전체 코드베이스 구축 방향, 인프라는 클라우드 멀티 GPU(A100/H100), FSDP 분산학습 상정.

구축된 구조: src/(config, model, tokenizer, data, trainer, utils) + scripts/(train_tokenizer, prepare_data, train, generate, export, smoke_test, launch.sh) + configs/(10b.json ≈10.7B, small.json ≈100M).
아키텍처: RMSNorm + RoPE + GQA(32/8 heads) + SwiGLU, SDPA(flash), FSDP full-shard + bf16 + activation checkpointing.

추가 기능(2026-07-10): (1) KV 캐시 증분 디코딩 — model.py에 KVCache 클래스, generate에 use_cache/eos_id, greedy 동등성 테스트. (2) 데이터 셔플링 개선 — ChunkedDataset(비겹침+DistributedSampler 셔플/샤딩, 기본) + RandomWindowDataset, make_dataloader mode 인자. (3) W&B 로깅 — trainer에 _init_wandb/_wandb_log, MFU 추정(_mfu/_peak_flops), train.py --wandb-project/--wandb-run.

검증 상태: 컴파일 통과(trainer/config/utils/tokenizer/data 등), 파라미터 수/LR/RoPE/GQA/청크인덱싱/KV concat/샤딩 순수로직 검증 완료. torch 실제 forward/backward + KV캐시 동등성은 샌드박스가 526MB 휠 다운로드를 못 끝내 미실행 — 사용자가 `python scripts/smoke_test.py`로 직접 확인.

추가 기능2(2026-07-10): (4) 분산 sharded 체크포인트 — trainer.py에 DCP(torch.distributed.checkpoint) 저장/로드, 모델+옵티마이저+step 재개, resharding 지원, final 시 consolidated .pt(ckpt_final_full.pt)도 생성, 비FSDP 폴백. (5) 배치 추론 서버 — src/serve.py(InferenceEngine: left-pad+attention mask 배치 생성, DynamicBatcher) + scripts/serve.py(stdlib HTTP, POST /generate). model.py에 attn_mask 파라미터 3곳 추가.

검증2: serve/scripts.serve/trainer 컴파일 OK, 배치 마스크·체크포인트 구조 순수로직 검증 OK. model.py/generate.py는 마운트 절단으로 mount 컴파일은 실패하나 실제 파일은 Read로 정상 확인.

추가 기능3(2026-07-10): (6) ResumableLoader(data.py) — 결정적·샤딩·정확한 위치복원 데이터로더, 체크포인트에 data_state 저장(trainer). (7) 모니터링 — trainer가 metrics.jsonl 기록, scripts/dashboard.py가 Chart.js 자체포함 HTML 생성. (8) 대화형 AI 전체 스택: tokenizer에 CHAT_TOKENS(<|system|>,<|user|>,<|assistant|>,<|eot|>), src/chat.py(ChatTemplate assistant-only loss 마스킹 + SFTDataset), scripts/finetune.py(SFT, init_from으로 베이스가중치 로드), scripts/chat.py(대화 REPL, KV캐시, eot 정지). config에 init_from 추가.

데모: configs/demo.json(초소형 CPU용), run_demo.bat(Windows 원클릭: 토크나이저→데이터→사전학습→SFT→대화), data/sample_corpus.jsonl, data/sample_chat.jsonl.

사용자 환경(중요): Windows, Python 3.14, torch 2.13.0 설치 완료(작동 확인 — train.py가 실제 임포트/모델생성/파라미터카운트까지 정상 실행됨. 마운트 절단은 순수 마운트 아티팩트였음이 입증). 남은 이슈는 코드 아님: (a) 데이터 파일 부재(run_demo.bat로 해결), (b) cmd에서 #주석·\줄바꿈 미동작, (c) 멀티GPU torchrun libuv 오류 → `set USE_LIBUV=0` 필요, 단일 GPU/CPU는 torchrun 없이 python 직접 실행.

주의: bash 쪽 E:\LLM 마운트가 큰 파일 쓰기를 지연/절단하는 인프라 이슈 있음(실제 디스크 파일은 Read 툴 기준 정상). 향후 bash로 E:\LLM 파일 검증 시 outputs 마운트 경유 권장.
