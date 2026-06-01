#!/usr/bin/env bash
set -euo pipefail

# One process can saturate CPU image decoding through DataLoader workers.
# Run separate alpha jobs concurrently only when each job has its own GPU.
RUN_MODE="${RUN_MODE:-default}"
case "${RUN_MODE}" in
  default)
    CONFIG_NAME="student_baseline"
    ;;
  local)
    CONFIG_NAME="student_baseline_local"
    ;;
  *)
    echo "RUN_MODE must be 'default' or 'local'." >&2
    exit 2
    ;;
esac

overrides=()
if [[ -n "${NUM_WORKERS:-}" ]]; then
  overrides+=("data.num_workers=${NUM_WORKERS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  overrides+=("data.batch_size=${BATCH_SIZE}")
fi
if [[ -n "${MAX_EPOCHS:-}" ]]; then
  overrides+=("trainer.max_epochs=${MAX_EPOCHS}")
fi

for ALPHA in 0 0.5 1; do
  uv run python scripts/train_student_baseline.py --config-name "${CONFIG_NAME}" \
    "distillation.alpha=${ALPHA}" \
    "${overrides[@]}" \
    "$@"
done
