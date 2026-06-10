#!/usr/bin/env bash
set -euo pipefail

TIER_FILE="${1:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum/tier_0_acc_ge_0p60_n_ge_1000/prompt_classes.txt}"
RUNS="${RUNS:-${2:-all}}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-5}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
TEACHER_CACHE="${TEACHER_CACHE:-artifacts/teacher_cache/composite_ece_temp_smol1p1_frcnn2p2_beta12p968_tallyqa_target_mobilenet224.jsonl}"
SAMPLING_CURRICULUM="${SAMPLING_CURRICULUM:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum/sqrt_to_natural_sampling_curriculum_approx1500steps_bs32.json}"

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
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 04: add sqrt prompt-class sampling. Epoch size is held near the natural tier size.
run_one "04" "tallyqa-tier0-04-plus-sqrt-prompt-sampling" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.5" \
  "data.train_epoch_size=43626" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 05: replace fixed sqrt sampling with a short sampling curriculum.
# The checked-in schedule is epoch-based: sqrt sampling for epoch 1, then natural sampling.
# With Tier 0 and BATCH_SIZE=32, epoch 1 is roughly 1,500 optimizer steps.
run_one "05" "tallyqa-tier0-05-plus-sampling-curriculum" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.5" \
  "data.train_epoch_size=43626" \
  "data.curriculum_schedule=${SAMPLING_CURRICULUM}" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=hard" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 06: add local soft hard-label targets.
run_one "06" "tallyqa-tier0-06-plus-local-soft-targets" \
  "data.require_teacher_cache=false" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.5" \
  "data.train_epoch_size=43626" \
  "data.curriculum_schedule=${SAMPLING_CURRICULUM}" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.0" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 07: add composite-teacher KL on top of local soft targets.
run_one "07" "tallyqa-tier0-07-plus-composite-teacher-kl" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.5" \
  "data.train_epoch_size=43626" \
  "data.curriculum_schedule=${SAMPLING_CURRICULUM}" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"

# 08: same as 07, but switch the image encoder to MobileNetV3-large.
run_one "08" "tallyqa-tier0-08-large-backbone-full-baseline" \
  "data.require_teacher_cache=true" \
  "paths.teacher_cache=${TEACHER_CACHE}" \
  "data.teacher_probability_temperature=1.0" \
  "data.train_sampling=prompt_class_tempered" \
  "data.prompt_class_sampling_temperature=0.5" \
  "data.train_epoch_size=43626" \
  "data.curriculum_schedule=${SAMPLING_CURRICULUM}" \
  "trainer.reload_dataloaders_every_n_epochs=1" \
  "distillation.alpha=1.0" \
  "distillation.beta=0.25" \
  "distillation.target_distribution=local_soft" \
  "distillation.local_soft_sigma=0.5" \
  "distillation.local_soft_radius=1" \
  "model.image_backbone=mobilenet_v3_large" \
  "model.dropout=0.1" \
  "optimizer.weight_decay=0.01" \
  "optimizer.warmup_steps=1000" \
  "optimizer.warmup_start_learning_rate=0.0001"
