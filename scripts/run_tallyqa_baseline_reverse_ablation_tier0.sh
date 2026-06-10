#!/usr/bin/env bash
set -euo pipefail

TIER_FILE="${TIER_FILE:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum/tier_0_acc_ge_0p60_n_ge_1000/prompt_classes.txt}"
TIER_ROOT="${TIER_ROOT:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum}"
TIER_1_FILE="${TIER_1_FILE:-${TIER_ROOT}/tier_1_acc_ge_0p60_n_ge_500/prompt_classes.txt}"
TIER_2_FILE="${TIER_2_FILE:-${TIER_ROOT}/tier_2_acc_ge_0p60/prompt_classes.txt}"
TIER_3_FILE="${TIER_3_FILE:-${TIER_ROOT}/tier_3_acc_ge_0p55/prompt_classes.txt}"
TIER_4_FILE="${TIER_4_FILE:-${TIER_ROOT}/tier_4_acc_ge_0p40/prompt_classes.txt}"
TIER_FUSION_MODE="${TIER_FUSION_MODE:-transformer}"
RUNS="${RUNS:-all}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PATIENCE="${PATIENCE:-5}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
TEACHER_CACHE="${TEACHER_CACHE:-artifacts/teacher_cache/composite_ece_temp_smol1p1_frcnn2p2_beta12p968_tallyqa_target_mobilenet224.jsonl}"
SAMPLING_DECAY_STEPS="${SAMPLING_DECAY_STEPS:-2000}"
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
    --tier-fusion-mode)
      TIER_FUSION_MODE="$2"
      shift 2
      ;;
    --sampling-decay-steps)
      SAMPLING_DECAY_STEPS="$2"
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

COMMON_OVERRIDES=(
  "trainer.max_epochs=${MAX_EPOCHS}"
  "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}"
  "trainer.early_stopping.enabled=true"
  "trainer.early_stopping.patience=${PATIENCE}"
  "trainer.early_stopping.monitor=val/prompt_class_output_weighted_mae"
  "trainer.early_stopping.mode=min"
  "trainer.gradient_clip_val=1.0"
  "trainer.gradient_clip_algorithm=norm"
  "data.batch_size=${BATCH_SIZE}"
  "data.num_workers=${NUM_WORKERS}"
  "data.prefetch_factor=${PREFETCH_FACTOR}"
  "data.persistent_workers=${PERSISTENT_WORKERS}"
  "data.pin_memory=${PIN_MEMORY}"
  "data.prompt_class_names_file=${TIER_FILE}"
  "data.shuffle_train=true"
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
  "distillation.class_weight_mode=balanced"
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
  local range_start
  local range_end
  if [[ "${RUNS}" == "all" || "${RUNS}" == "*" ]]; then
    return 0
  fi
  IFS=',' read -ra run_specs <<< "${RUNS}"
  for run_spec in "${run_specs[@]}"; do
    run_spec="${run_spec//[[:space:]]/}"
    if [[ -z "${run_spec}" ]]; then
      continue
    fi
    if [[ "${run_spec}" == "${run_id}" || "${run_spec}" == "${run_id#0}" ]]; then
      return 0
    fi
    if [[ "${run_spec}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      range_start=$((10#${BASH_REMATCH[1]}))
      range_end=$((10#${BASH_REMATCH[2]}))
      if (( 10#${run_id} >= range_start && 10#${run_id} <= range_end )); then
        return 0
      fi
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

run_large_soft_teacher() {
  local run_id="$1"
  local run_name="$2"
  local prompt_file="$3"
  local fusion_mode="$4"
  shift 4
  run_one "${run_id}" "${run_name}" \
    "data.require_teacher_cache=true" \
    "paths.teacher_cache=${TEACHER_CACHE}" \
    "data.teacher_probability_temperature=1.0" \
    "data.prompt_class_names_file=${prompt_file}" \
    "data.train_sampling=prompt_class_tempered" \
    "data.prompt_class_sampling_temperature=0.0" \
    "data.prompt_class_sampling_end_temperature=0.25" \
    "data.prompt_class_sampling_decay_steps=null" \
    "data.prompt_class_sampling_ramp_start_step=2000" \
    "data.train_epoch_size=null" \
    "data.curriculum_schedule=null" \
    "trainer.reload_dataloaders_every_n_epochs=1" \
    "distillation.alpha=1.0" \
    "distillation.beta=0.25" \
    "distillation.target_distribution=local_soft" \
    "distillation.local_soft_sigma=0.5" \
    "distillation.local_soft_radius=1" \
    "model.image_backbone=mobilenet_v3_large" \
    "model.fusion_mode=${fusion_mode}" \
    "model.dropout=0.1" \
    "optimizer.weight_decay=0.01" \
    "optimizer.lr_schedule=warmup_plateau_decay" \
    "optimizer.lr_decay_start_step=1500" \
    "optimizer.lr_final_learning_rate=0.0001" \
    "optimizer.warmup_steps=1000" \
    "optimizer.warmup_start_learning_rate=0.0001" \
    "$@"
}

echo "Selected RUNS=${RUNS}"

# 00: minimal baseline. Frozen MobileNetV3-small + transformer fusion, hard labels only.
run_one "00" "tallyqa-tier0-00-small-frozen-hard-no-reg" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=natural" \
  "data.train_epoch_size=null" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.0" \
  "optimizer.weight_decay=0.0" \
  "optimizer.lr_schedule=none" \
  "optimizer.warmup_steps=0" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 01: add dropout.
run_one "01" "tallyqa-tier0-01-plus-dropout" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=natural" \
  "data.train_epoch_size=null" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.0" \
  "optimizer.lr_schedule=none" \
  "optimizer.warmup_steps=0" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 02: add weight decay.
run_one "02" "tallyqa-tier0-02-plus-weight-decay" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=natural" \
  "data.train_epoch_size=null" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=none" \
  "optimizer.warmup_steps=0" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 03: add the existing LR warmup schedule.
run_one "03" "tallyqa-tier0-03-plus-lr-warmup" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=natural" \
  "data.train_epoch_size=null" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 04: replace warmup-only with warmup, plateau, then linear decay to warmup-start LR.
run_one "04" "tallyqa-tier0-04-plus-warmup-plateau-decay" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=natural" \
  "data.train_epoch_size=null" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 05: add sqrt prompt-class sampling. Epoch size is held near the natural tier size.
run_one "05" "tallyqa-tier0-05-plus-p025-prompt-sampling" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.25" \
  "data.train_epoch_size=null" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 06: replace fixed sqrt sampling with a step-based interpolation to natural sampling.
# Temperature is recomputed per epoch from the active dataset size and batch size.
run_one "06" "tallyqa-tier0-06-plus-sampling-curriculum" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 07: add local soft hard-label targets.
run_one "07" "tallyqa-tier0-07-plus-local-soft-targets" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 08: add composite-teacher KL on top of local soft targets.
run_one "08" "tallyqa-tier0-08-plus-composite-teacher-kl" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 09: same teacher KL as 08, but keep hard targets instead of local soft targets.
run_one "09" "tallyqa-tier0-09-composite-teacher-kl-hard-targets" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 10: same as 08, but switch the image encoder to MobileNetV3-large.
run_large_soft_teacher \
  "10" \
  "tallyqa-tier0-10-large-backbone-full-baseline" \
  "${TIER_FILE}" \
  "transformer"

# 11: same as 06, but ramp prompt sampling from natural after step 2000 to p=0.25 at train end.
run_one "11" "tallyqa-tier0-11-sampling-p025-ramp-after-2000" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 12: same as 08, with FiLM after small MobileNet's last 28x28 and 14x14x40 blocks.
run_one "12" "tallyqa-tier0-12-small-film-28-last-14-last-soft-teacher" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.image_film_at='3,6'" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 13: same as 10, with FiLM after large MobileNet's last 28x28 and 14x14x80 blocks.
run_one "13" "tallyqa-tier0-13-large-film-28-last-14x14x80-soft-teacher" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.image_backbone=mobilenet_v3_large" \
  "model.image_film_at='6,10'" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 14: same as 10, but replace token fusion with prompt-FiLM image-only MLP fusion.
run_large_soft_teacher \
  "14" \
  "tallyqa-tier0-14-large-predepthwise-film-mlp" \
  "${TIER_FILE}" \
  "film_mlp" \
  "model.image_film_at='7,11'" \
  "model.image_film_position=pre_depthwise" \
  "model.use_prompt_identity=false" \
  "model.use_image_positional_embeddings=false"

# 15: same as 10, but unfreeze the MobileNet image encoder at the configured lower image LR.
run_large_soft_teacher \
  "15" \
  "tallyqa-tier0-15-large-transformer-unfrozen-image" \
  "${TIER_FILE}" \
  "transformer" \
  "model.freeze_image_features=false" \
  "optimizer.image_learning_rate_scale=0.1"

# 16: same as 08, but unfreeze the MobileNetV3-small image encoder at 0.1x LR.
run_one "16" "tallyqa-tier0-16-small-transformer-unfrozen-image" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.0" \
  "data.prompt_class_sampling_end_temperature=0.25" \
  "data.prompt_class_sampling_decay_steps=null" \
  "data.prompt_class_sampling_ramp_start_step=2000" \
  "data.train_epoch_size=null" \
  "data.curriculum_schedule=null" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.dropout=0.1" \
  "model.freeze_image_features=false" \
  "optimizer.image_learning_rate_scale=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.lr_schedule=warmup_plateau_decay" \
  "optimizer.lr_decay_start_step=1500" \
  "optimizer.lr_final_learning_rate=0.0001" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 17-20: same recipe as 10, but grow the prompt-class tier.
# Set --tier-fusion-mode concat_mlp after run 14 if the MLP wins.
run_large_soft_teacher \
  "17" \
  "tallyqa-tier1-17-large-${TIER_FUSION_MODE}-soft-teacher" \
  "${TIER_1_FILE}" \
  "${TIER_FUSION_MODE}"

run_large_soft_teacher \
  "18" \
  "tallyqa-tier2-18-large-${TIER_FUSION_MODE}-soft-teacher" \
  "${TIER_2_FILE}" \
  "${TIER_FUSION_MODE}"

run_large_soft_teacher \
  "19" \
  "tallyqa-tier3-19-large-${TIER_FUSION_MODE}-soft-teacher" \
  "${TIER_3_FILE}" \
  "${TIER_FUSION_MODE}"

run_large_soft_teacher \
  "20" \
  "tallyqa-tier4-20-large-${TIER_FUSION_MODE}-soft-teacher" \
  "${TIER_4_FILE}" \
  "${TIER_FUSION_MODE}"
