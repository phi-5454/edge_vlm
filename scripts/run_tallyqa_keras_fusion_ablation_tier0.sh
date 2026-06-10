#!/usr/bin/env bash
set -euo pipefail

TIER_FILE="${TIER_FILE:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum/tier_0_acc_ge_0p60_n_ge_1000/prompt_classes.txt}"
TEACHER_CACHE="${TEACHER_CACHE:-artifacts/teacher_cache/composite_ece_temp_smol1p1_frcnn2p2_beta12p968_tallyqa_target_mobilenet224.jsonl}"
RUNS="${RUNS:-all}"
DRY_RUN="${DRY_RUN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-6}"
BACKBONE="${BACKBONE:-mobilenet_v3_small}"
FUSION_DIM="${FUSION_DIM:-128}"
FUSION_DEPTH="${FUSION_DEPTH:-4}"
FUSION_HEADS="${FUSION_HEADS:-4}"
FUSION_MLP_RATIO="${FUSION_MLP_RATIO:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier-file)
      TIER_FILE="$2"
      shift 2
      ;;
    --teacher-cache)
      TEACHER_CACHE="$2"
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
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --backbone)
      BACKBONE="$2"
      shift 2
      ;;
    --fusion-dim)
      FUSION_DIM="$2"
      shift 2
      ;;
    --fusion-depth)
      FUSION_DEPTH="$2"
      shift 2
      ;;
    --fusion-heads)
      FUSION_HEADS="$2"
      shift 2
      ;;
    --fusion-mlp-ratio)
      FUSION_MLP_RATIO="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      exit 2
      ;;
  esac
done

run_selected() {
  local run_id="$1"
  local run_spec
  if [[ "${RUNS}" == "all" || "${RUNS}" == "*" ]]; then
    return 0
  fi
  IFS=',' read -ra run_specs <<< "${RUNS}"
  for run_spec in "${run_specs[@]}"; do
    run_spec="${run_spec//[[:space:]]/}"
    if [[ "${run_spec}" == "${run_id}" ]]; then
      return 0
    fi
  done
  return 1
}

run_one() {
  local run_id="$1"
  local run_name="$2"
  local fusion_mode="$3"
  shift 3
  if ! run_selected "${run_id}"; then
    echo "Skipping run ${run_id}: ${run_name}"
    return 0
  fi
  echo "Running ${run_id}: ${run_name}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: uv run python scripts/train_tallyqa_keras_student.py --config-name tallyqa_keras_student experiment.run_name=${run_name} keras_model.fusion_mode=${fusion_mode} ..."
    return 0
  fi
  uv run python scripts/train_tallyqa_keras_student.py \
    --config-name tallyqa_keras_student \
    "experiment.run_name=${run_name}" \
    "paths.teacher_cache=${TEACHER_CACHE}" \
    "data.require_teacher_cache=true" \
    "data.missing_teacher_policy=filter" \
    "data.prompt_class_names_file=${TIER_FILE}" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.shuffle_train=true" \
    "data.train_sampling=prompt_class_tempered" \
    "data.prompt_class_sampling_temperature=0.0" \
    "data.prompt_class_sampling_end_temperature=0.25" \
    "data.prompt_class_sampling_decay_steps=null" \
    "data.prompt_class_sampling_ramp_start_step=2000" \
    "data.train_epoch_size=null" \
    "data.image_preprocessing=mobilenet_v3_external" \
    "model.freeze_embeddings=true" \
    "model.freeze_image_features=true" \
    "model.image_pretrained=true" \
    "distillation.alpha=1.0" \
    "distillation.beta=0.25" \
    "distillation.temperature=2.0" \
    "distillation.class_weight_mode=balanced" \
    "distillation.kl_class_weights=null" \
    "distillation.target_distribution=local_soft" \
    "distillation.local_soft_sigma=0.5" \
    "distillation.local_soft_radius=1" \
    "keras_model.architecture=current_student" \
    "keras_model.image_backbone=${BACKBONE}" \
    "keras_model.image_feature_cutoff=auto" \
    "keras_model.include_mobilenet_preprocessing=false" \
    "keras_model.fusion_mode=${fusion_mode}" \
    "keras_model.fusion_dim=${FUSION_DIM}" \
    "keras_model.fusion_depth=${FUSION_DEPTH}" \
    "keras_model.fusion_heads=${FUSION_HEADS}" \
    "keras_model.fusion_mlp_ratio=${FUSION_MLP_RATIO}" \
    "keras_model.image_film_at=null" \
    "keras_model.dropout=0.1" \
    "keras_model.use_prompt_identity=true" \
    "keras_model.use_image_positional_embeddings=true" \
    "keras_model.visualkeras.enabled=true" \
    "optimizer.learning_rate=${LEARNING_RATE}" \
    "optimizer.weight_decay=${WEIGHT_DECAY}" \
    "optimizer.lr_schedule=warmup_plateau_decay" \
    "optimizer.lr_decay_start_step=1500" \
    "optimizer.lr_final_learning_rate=0.0001" \
    "optimizer.warmup_steps=1000" \
    "optimizer.warmup_start_learning_rate=0.0001" \
    "trainer.max_epochs=${MAX_EPOCHS}" \
    "trainer.log_every_n_steps=25" \
    "trainer.early_stopping.enabled=true" \
    "trainer.early_stopping.monitor=val/class_weighted_mae" \
    "trainer.early_stopping.mode=min" \
    "trainer.early_stopping.patience=${PATIENCE}" \
    "export.export_tflite=false" \
    "export.quantization.mode=none" \
    "$@"
}

echo "Selected RUNS=${RUNS}"

run_one "mlp" "tallyqa-keras-tier0-current-mlp-float" "mlp"
run_one "film_mlp" "tallyqa-keras-tier0-current-film-mlp-float" "film_mlp" \
  "keras_model.image_film_at=image_tokens" \
  "keras_model.use_prompt_identity=false" \
  "keras_model.use_image_positional_embeddings=false"
run_one "normformer" "tallyqa-keras-tier0-current-normformer-float" "normformer"
