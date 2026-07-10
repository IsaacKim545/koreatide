#!/usr/bin/env bash
# 멀티노드 FSDP 학습 실행 예시 (torchrun / Slurm 참고용).
set -euo pipefail

# ---- 단일 노드, 8 GPU ----
# torchrun --standalone --nproc_per_node=8 \
#   scripts/train.py --config configs/10b.json \
#   --data data/train.bin --val data/val.bin --out runs/10b

# ---- 멀티노드 (예: 8노드 × 8 GPU = 64 GPU) ----
# 각 노드에서 아래 환경변수를 세팅하고 실행 (또는 Slurm srun 사용).
: "${NNODES:=8}"
: "${GPUS_PER_NODE:=8}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${CONFIG:=configs/10b.json}"
: "${DATA:=data/train.bin}"
: "${VAL:=data/val.bin}"
: "${OUT:=runs/10b}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py \
  --config "${CONFIG}" \
  --data "${DATA}" \
  --val "${VAL}" \
  --out "${OUT}"

# ---- Slurm 예시 (sbatch 스크립트 내부) ----
# #SBATCH --nodes=8
# #SBATCH --ntasks-per-node=8
# #SBATCH --gpus-per-node=8
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
#   --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=29500 \
#   scripts/train.py --config configs/10b.json --data data/train.bin --out runs/10b
