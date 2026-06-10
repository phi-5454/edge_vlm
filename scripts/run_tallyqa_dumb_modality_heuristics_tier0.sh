#!/usr/bin/env bash
set -euo pipefail

TIER_FILE="${TIER_FILE:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum/tier_0_acc_ge_0p60_n_ge_1000/prompt_classes.txt}"
RUNS="${RUNS:-all}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PATIENCE="${PATIENCE:-6}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
TEACHER_CACHE="${TEACHER_CACHE:-artifacts/teacher_cache/composite_ece_temp_smol1p1_frcnn2p2_beta12p968_tallyqa_target_mobilenet224.jsonl}"
SAMPLING_CURRICULUM="${SAMPLING_CURRICULUM:-}"
SAMPLING_CURRICULUM_STEPS="${SAMPLING_CURRICULUM_STEPS:-1500}"
_POSITIONAL_INDEX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier-file)
      TIER_FILE="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --prefetch-factor)
      PREFETCH_FACTOR="$2"
      shift 2
      ;;
    --persistent-workers)
      PERSISTENT_WORKERS="$2"
      shift 2
      ;;
    --pin-memory)
      PIN_MEMORY="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --check-val-every-n-epoch)
      CHECK_VAL_EVERY_N_EPOCH="$2"
      shift 2
      ;;
    --teacher-cache)
      TEACHER_CACHE="$2"
      shift 2
      ;;
    --sampling-curriculum)
      SAMPLING_CURRICULUM="$2"
      shift 2
      ;;
    --sampling-curriculum-steps)
      SAMPLING_CURRICULUM_STEPS="$2"
      shift 2
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
    *)
      if [[ "${_POSITIONAL_INDEX}" == "0" ]]; then
        TIER_FILE="$1"
      elif [[ "${_POSITIONAL_INDEX}" == "1" ]]; then
        RUNS="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        exit 2
      fi
      _POSITIONAL_INDEX=$((_POSITIONAL_INDEX + 1))
      shift
      ;;
  esac
done

make_sampling_curriculum() {
  if [[ -n "${SAMPLING_CURRICULUM}" ]]; then
    echo "${SAMPLING_CURRICULUM}"
    return 0
  fi
  local schedule_path
  local first_epoch_size
  schedule_path="$(mktemp /tmp/tallyqa_sampling_curriculum.XXXXXX.json)"
  first_epoch_size=$((SAMPLING_CURRICULUM_STEPS * BATCH_SIZE))
  printf '[\n' > "${schedule_path}"
  printf '  {\n' >> "${schedule_path}"
  printf '    "start_epoch": 1,\n' >> "${schedule_path}"
  printf '    "train_sampling": "prompt_class_tempered",\n' >> "${schedule_path}"
  printf '    "prompt_class_sampling_temperature": 0.5,\n' >> "${schedule_path}"
  printf '    "train_epoch_size": %s\n' "${first_epoch_size}" >> "${schedule_path}"
  printf '  },\n' >> "${schedule_path}"
  printf '  {\n' >> "${schedule_path}"
  printf '    "start_epoch": 2,\n' >> "${schedule_path}"
  printf '    "train_sampling": "natural",\n' >> "${schedule_path}"
  printf '    "train_epoch_size": null\n' >> "${schedule_path}"
  printf '  }\n' >> "${schedule_path}"
  printf ']\n' >> "${schedule_path}"
  echo "${schedule_path}"
}

SAMPLING_CURRICULUM_PATH="$(make_sampling_curriculum)"

COMMON_OVERRIDES=(
  "trainer.max_epochs=${MAX_EPOCHS}"
  "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
  "trainer.early_stopping.enabled=true"
  "trainer.early_stopping.patience=${PATIENCE}"
  "trainer.early_stopping.monitor=val/prompt_class_output_weighted_mae"
  "trainer.early_stopping.mode=min"
  "trainer.gradient_clip_val=1.0"
  "trainer.gradient_clip_algorithm=norm"
  "trainer.reload_dataloaders_every_n_epochs=1"
  "data.batch_size=${BATCH_SIZE}"
  "data.num_workers=${NUM_WORKERS}"
  "data.prefetch_factor=${PREFETCH_FACTOR}"
  "data.persistent_workers=${PERSISTENT_WORKERS}"
  "data.pin_memory=${PIN_MEMORY}"
  "data.prompt_class_names_file=${TIER_FILE}"
  "data.shuffle_train=true"
  "data.require_teacher_cache=true"
  "data.teacher_probability_temperature=1.0"
  "data.train_sampling=prompt_class_tempered"
  "data.prompt_class_sampling_temperature=0.5"
  "data.train_epoch_size=43626"
  "data.curriculum_schedule=${SAMPLING_CURRICULUM_PATH}"
  "paths.teacher_cache=${TEACHER_CACHE}"
  "model.image_backbone=mobilenet_v3_small"
  "model.image_feature_cutoff=auto"
  "model.image_token_mode=spatial"
  "model.fusion_mode=transformer"
  "model.fusion_depth=4"
  "model.fusion_heads=4"
  "model.fusion_mlp_ratio=4"
  "model.freeze_image_features=true"
  "model.use_prompt_identity=true"
  "model.use_image_positional_embeddings=true"
  "model.image_position_tokens=196"
  "model.dropout=0.1"
  "optimizer.weight_decay=0.01"
  "optimizer.warmup_steps=1000"
  "optimizer.warmup_start_learning_rate=0.0001"
  "distillation.class_weight_mode=balanced"
  "distillation.alpha=1.0"
  "distillation.beta=0.25"
  "distillation.target_distribution=local_soft"
  "distillation.local_soft_sigma=0.5"
  "distillation.local_soft_radius=1"
  "validation_plots.enabled=true"
  "validation_plots.samples=4"
  "validation_plots.every_n_epochs=1"
  "wandb.watch.enabled=true"
  "wandb.watch.log=all"
  "wandb.watch.log_freq=100"
)

run_selected() {
  local run_id="$1"
  local run_spec
  if [[ "${RUNS}" == "all" || "${RUNS}" == "*" ]]; then
    return 0
  fi
  IFS=',' read -ra run_specs <<< "${RUNS}"
  for run_spec in "${run_specs[@]}"; do
    run_spec="${run_spec//[[:space:]]/}"
    if [[ "${run_spec}" == "${run_id}" || "${run_spec}" == "${run_id#0}" ]]; then
      return 0
    fi
  done
  return 1
}

run_one() {
  local run_id="$1"
  local run_name="$2"
  shift 2
  if ! run_selected "${run_id}"; then
    echo "Skipping run ${run_id}: ${run_name}"
    return 0
  fi
  echo "Running ${run_id}: ${run_name}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: uv run python scripts/train_tallyqa_student.py --config-name tallyqa_student experiment.run_name=${run_name} ..."
    return 0
  fi
  uv run python scripts/train_tallyqa_student.py \
    --config-name tallyqa_student \
    "experiment.run_name=${run_name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

echo "Selected RUNS=${RUNS}"

# 00: prompt-only dumb baseline. Image encoder still runs, but image tokens are zeroed before fusion.
run_one "00" "tallyqa-tier0-heuristic-prompt-only-zero-image-tokens" \
  "model.zero_image_tokens=true" \
  "model.zero_query_token=false"

# 01: image-only dumb baseline. Prompt/query token is zeroed before fusion.
run_one "01" "tallyqa-tier0-heuristic-image-only-zero-query-token" \
  "model.zero_image_tokens=false" \
  "model.zero_query_token=true"
