#!/usr/bin/env bash
set -euo pipefail

SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/teacher_cache}"
DATASET="${DATASET:-data/the_cauldron_yes_no_vsr_token1000_img512_parquet}"
MODEL="${MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "${OUTPUT_DIR}/logs"

local_files_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_files_args+=(--local-files-only)
fi

uv run python scripts/cache_smolvlm_yes_no_teacher.py \
  --dataset "${DATASET}" \
  --image-source student-512 \
  --model "${MODEL}" \
  --output "${OUTPUT_DIR}/smolvlm_yes_no_vsr_token1000_img512.shard$(printf '%03d' "${SHARD_INDEX}").jsonl" \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${SHARD_INDEX}" \
  --device "${DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --top-k 10 \
  --temperature 1.0 \
  --resume \
  "${local_files_args[@]}"
