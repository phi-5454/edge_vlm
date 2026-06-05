#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="tallyqa-local-single"
CONFIG_NAME="tallyqa_student_local"
MAX_EPOCHS="3"
#LIMIT_TRAIN_BATCHES="100"
#LIMIT_VAL_BATCHES="20"
#LIMIT_TEST_BATCHES="20"
LIMIT_TRAIN_BATCHES="10000000000"
LIMIT_VAL_BATCHES="2000000000"
LIMIT_TEST_BATCHES="200000000"
CHECK_VAL_EVERY_N_EPOCH="1"
RELOAD_DATALOADERS_EVERY_N_EPOCHS="0"
EARLY_STOPPING_ENABLED="true"
EARLY_STOPPING_PATIENCE="3"
EARLY_STOPPING_MONITOR="val/loss"
EARLY_STOPPING_MODE="min"
EARLY_STOPPING_MIN_DELTA="0.0"
GRADIENT_CLIP_VAL="1.0"
GRADIENT_CLIP_ALGORITHM="norm"
BATCH_SIZE="16"
SHUFFLE_TRAIN="true"
TRAIN_EXAMPLE_LIMIT="null"
TRAIN_SAMPLING="natural"
PROMPT_CLASS_SAMPLING_TEMPERATURE="0.5"
TRAIN_EPOCH_SIZE="null"
IMAGE_LEARNING_RATE_SCALE="0.1"
CLASS_WEIGHTS="null"
BALANCED_LOSS="true"
PROMPT_CLASS_FILTER_CSV="null"
MIN_PROMPT_ACCURACY="null"
PROMPT_CLASS_NAMES="null"
PROMPT_CLASS_NAMES_FILE="null"
CURRICULUM_SCHEDULE="null"
DISTILLATION_ALPHA="1.0"
DISTILLATION_BETA="0.5"
TARGET_DISTRIBUTION="hard"
LOCAL_SOFT_SIGMA="1.0"
LOCAL_SOFT_RADIUS="1"
IMAGE_BACKBONE="mobilenet_v3_large"
IMAGE_FEATURE_CUTOFF="auto"
IMAGE_FILM_AT="null"
IMAGE_TOKEN_MODE="spatial"
FUSION_MODE="transformer"
FUSION_DEPTH="4"
FUSION_HEADS="4"
FUSION_MLP_RATIO="2"
FREEZE_IMAGE_FEATURES="false"
USE_PROMPT_IDENTITY="true"
USE_IMAGE_POSITIONAL_EMBEDDINGS="true"
IMAGE_POSITION_TOKENS="196"
WANDB_WATCH_ENABLED="true"
WANDB_WATCH_LOG="all"
WANDB_WATCH_LOG_FREQ="100"
VALIDATION_PLOTS_ENABLED="true"
VALIDATION_PLOTS_SAMPLES="4"
VALIDATION_PLOTS_EVERY_N_EPOCHS="1"
REQUIRE_TEACHER_CACHE="true"

usage() {
  cat <<'EOF'
Usage: scripts/run_tallyqa_student_local_once.sh [options]

Options:
  --run-name NAME
  --config-name tallyqa_student|tallyqa_student_local
  --max-epochs N
  --limit-train-batches N|null
  --limit-val-batches N|null
  --limit-test-batches N|null
  --check-val-every-n-epoch N
  --reload-dataloaders-every-n-epochs N
  --early-stopping-patience N
  --early-stopping-monitor METRIC
  --early-stopping-mode min|max
  --early-stopping-min-delta FLOAT
  --no-early-stopping
  --gradient-clip-val FLOAT|null
  --gradient-clip-algorithm norm|value
  --batch-size N
  --no-train-shuffle
  --train-shuffle
  --train-example-limit N|null
  --train-sampling natural|prompt_class_tempered
  --prompt-class-sampling-temperature FLOAT
  --train-epoch-size N|null
  --image-learning-rate-scale FLOAT
  --class-weights '[w0,w1,w2,w3,w4,w5]'
  --balanced-loss
  --unbalanced-loss
  --prompt-class-filter-csv PATH|null
  --min-prompt-accuracy FLOAT|null
  --prompt-class-names NAME[,NAME]
  --prompt-class-names-file PATH|null
  --curriculum-schedule PATH|null
  --distillation-alpha FLOAT
  --distillation-beta FLOAT
  --target-distribution hard|local_soft
  --local-soft-sigma FLOAT
  --local-soft-radius N
  --image-backbone mobilenet_v3_large|mobilenet_v3_small
  --image-feature-cutoff auto|none|N
  --image-film-at none|N
  --image-token-mode spatial|pooled
  --fusion-mode transformer|concat_mlp
  --fusion-depth N
  --fusion-heads N
  --fusion-mlp-ratio N
  --freeze-image-features
  --train-image-features
  --no-prompt-identity
  --no-image-positional-embeddings
  --image-position-tokens N
  --wandb-watch-log all|gradients|parameters
  --wandb-watch-log-freq N
  --no-wandb-watch
  --validation-plot-samples N
  --validation-plot-every-n-epochs N
  --no-validation-plots
  --no-teacher-cache
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --limit-train-batches)
      LIMIT_TRAIN_BATCHES="$2"
      shift 2
      ;;
    --limit-val-batches)
      LIMIT_VAL_BATCHES="$2"
      shift 2
      ;;
    --limit-test-batches)
      LIMIT_TEST_BATCHES="$2"
      shift 2
      ;;
    --check-val-every-n-epoch)
      CHECK_VAL_EVERY_N_EPOCH="$2"
      shift 2
      ;;
    --reload-dataloaders-every-n-epochs)
      RELOAD_DATALOADERS_EVERY_N_EPOCHS="$2"
      shift 2
      ;;
    --early-stopping-patience)
      EARLY_STOPPING_PATIENCE="$2"
      shift 2
      ;;
    --early-stopping-monitor)
      EARLY_STOPPING_MONITOR="$2"
      shift 2
      ;;
    --early-stopping-mode)
      EARLY_STOPPING_MODE="$2"
      shift 2
      ;;
    --early-stopping-min-delta)
      EARLY_STOPPING_MIN_DELTA="$2"
      shift 2
      ;;
    --no-early-stopping)
      EARLY_STOPPING_ENABLED="false"
      shift
      ;;
    --gradient-clip-val)
      GRADIENT_CLIP_VAL="$2"
      shift 2
      ;;
    --gradient-clip-algorithm)
      GRADIENT_CLIP_ALGORITHM="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --no-train-shuffle)
      SHUFFLE_TRAIN="false"
      shift
      ;;
    --train-shuffle)
      SHUFFLE_TRAIN="true"
      shift
      ;;
    --train-example-limit)
      TRAIN_EXAMPLE_LIMIT="$2"
      shift 2
      ;;
    --train-sampling)
      TRAIN_SAMPLING="$2"
      shift 2
      ;;
    --prompt-class-sampling-temperature)
      PROMPT_CLASS_SAMPLING_TEMPERATURE="$2"
      shift 2
      ;;
    --train-epoch-size)
      TRAIN_EPOCH_SIZE="$2"
      shift 2
      ;;
    --image-learning-rate-scale)
      IMAGE_LEARNING_RATE_SCALE="$2"
      shift 2
      ;;
    --class-weights)
      CLASS_WEIGHTS="$2"
      shift 2
      ;;
    --balanced-loss)
      BALANCED_LOSS="true"
      shift
      ;;
    --unbalanced-loss)
      BALANCED_LOSS="false"
      shift
      ;;
    --prompt-class-filter-csv)
      PROMPT_CLASS_FILTER_CSV="$2"
      shift 2
      ;;
    --min-prompt-accuracy)
      MIN_PROMPT_ACCURACY="$2"
      shift 2
      ;;
    --prompt-class-names)
      PROMPT_CLASS_NAMES="$2"
      shift 2
      ;;
    --prompt-class-names-file)
      PROMPT_CLASS_NAMES_FILE="$2"
      shift 2
      ;;
    --curriculum-schedule)
      CURRICULUM_SCHEDULE="$2"
      if [[ "${RELOAD_DATALOADERS_EVERY_N_EPOCHS}" == "0" && "${CURRICULUM_SCHEDULE}" != "null" ]]; then
        RELOAD_DATALOADERS_EVERY_N_EPOCHS="1"
      fi
      shift 2
      ;;
    --distillation-alpha)
      DISTILLATION_ALPHA="$2"
      shift 2
      ;;
    --distillation-beta)
      DISTILLATION_BETA="$2"
      shift 2
      ;;
    --target-distribution)
      TARGET_DISTRIBUTION="$2"
      shift 2
      ;;
    --local-soft-sigma)
      LOCAL_SOFT_SIGMA="$2"
      shift 2
      ;;
    --local-soft-radius)
      LOCAL_SOFT_RADIUS="$2"
      shift 2
      ;;
    --image-backbone)
      IMAGE_BACKBONE="$2"
      shift 2
      ;;
    --image-feature-cutoff)
      IMAGE_FEATURE_CUTOFF="$2"
      shift 2
      ;;
    --image-film-at)
      IMAGE_FILM_AT="$2"
      shift 2
      ;;
    --image-token-mode)
      IMAGE_TOKEN_MODE="$2"
      shift 2
      ;;
    --fusion-mode)
      FUSION_MODE="$2"
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
    --freeze-image-features)
      FREEZE_IMAGE_FEATURES="true"
      shift
      ;;
    --train-image-features)
      FREEZE_IMAGE_FEATURES="false"
      shift
      ;;
    --no-prompt-identity)
      USE_PROMPT_IDENTITY="false"
      shift
      ;;
    --no-image-positional-embeddings)
      USE_IMAGE_POSITIONAL_EMBEDDINGS="false"
      shift
      ;;
    --image-position-tokens)
      IMAGE_POSITION_TOKENS="$2"
      shift 2
      ;;
    --wandb-watch-log)
      WANDB_WATCH_LOG="$2"
      shift 2
      ;;
    --wandb-watch-log-freq)
      WANDB_WATCH_LOG_FREQ="$2"
      shift 2
      ;;
    --no-wandb-watch)
      WANDB_WATCH_ENABLED="false"
      shift
      ;;
    --validation-plot-samples)
      VALIDATION_PLOTS_SAMPLES="$2"
      shift 2
      ;;
    --validation-plot-every-n-epochs)
      VALIDATION_PLOTS_EVERY_N_EPOCHS="$2"
      shift 2
      ;;
    --no-validation-plots)
      VALIDATION_PLOTS_ENABLED="false"
      shift
      ;;
    --no-teacher-cache)
      REQUIRE_TEACHER_CACHE="false"
      DISTILLATION_BETA="0.0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${CLASS_WEIGHTS}" != "null" && "${BALANCED_LOSS}" == "true" ]]; then
  echo "Use either --class-weights or --balanced-loss, not both." >&2
  exit 2
fi

if [[ "${BALANCED_LOSS}" == "true" ]]; then
  CLASS_WEIGHT_MODE="balanced"
else
  CLASS_WEIGHT_MODE="null"
fi

uv run python scripts/train_tallyqa_student.py \
  --config-name "${CONFIG_NAME}" \
  "experiment.run_name=${RUN_NAME}" \
  "trainer.max_epochs=${MAX_EPOCHS}" \
  "trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}" \
  "trainer.limit_val_batches=${LIMIT_VAL_BATCHES}" \
  "trainer.limit_test_batches=${LIMIT_TEST_BATCHES}" \
  "trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
  "trainer.reload_dataloaders_every_n_epochs=${RELOAD_DATALOADERS_EVERY_N_EPOCHS}" \
  "trainer.early_stopping.enabled=${EARLY_STOPPING_ENABLED}" \
  "trainer.early_stopping.patience=${EARLY_STOPPING_PATIENCE}" \
  "trainer.early_stopping.monitor=${EARLY_STOPPING_MONITOR}" \
  "trainer.early_stopping.mode=${EARLY_STOPPING_MODE}" \
  "trainer.early_stopping.min_delta=${EARLY_STOPPING_MIN_DELTA}" \
  "trainer.gradient_clip_val=${GRADIENT_CLIP_VAL}" \
  "trainer.gradient_clip_algorithm=${GRADIENT_CLIP_ALGORITHM}" \
  "optimizer.image_learning_rate_scale=${IMAGE_LEARNING_RATE_SCALE}" \
  "data.batch_size=${BATCH_SIZE}" \
  "data.shuffle_train=${SHUFFLE_TRAIN}" \
  "data.train_example_limit=${TRAIN_EXAMPLE_LIMIT}" \
  "data.train_sampling=${TRAIN_SAMPLING}" \
  "data.prompt_class_sampling_temperature=${PROMPT_CLASS_SAMPLING_TEMPERATURE}" \
  "data.train_epoch_size=${TRAIN_EPOCH_SIZE}" \
  "distillation.class_weights=${CLASS_WEIGHTS}" \
  "distillation.class_weight_mode=${CLASS_WEIGHT_MODE}" \
  "distillation.alpha=${DISTILLATION_ALPHA}" \
  "distillation.beta=${DISTILLATION_BETA}" \
  "distillation.target_distribution=${TARGET_DISTRIBUTION}" \
  "distillation.local_soft_sigma=${LOCAL_SOFT_SIGMA}" \
  "distillation.local_soft_radius=${LOCAL_SOFT_RADIUS}" \
  "model.image_backbone=${IMAGE_BACKBONE}" \
  "model.image_feature_cutoff=${IMAGE_FEATURE_CUTOFF}" \
  "model.image_film_at=${IMAGE_FILM_AT}" \
  "model.image_token_mode=${IMAGE_TOKEN_MODE}" \
  "model.fusion_mode=${FUSION_MODE}" \
  "model.fusion_depth=${FUSION_DEPTH}" \
  "model.fusion_heads=${FUSION_HEADS}" \
  "model.fusion_mlp_ratio=${FUSION_MLP_RATIO}" \
  "model.freeze_image_features=${FREEZE_IMAGE_FEATURES}" \
  "model.use_prompt_identity=${USE_PROMPT_IDENTITY}" \
  "model.use_image_positional_embeddings=${USE_IMAGE_POSITIONAL_EMBEDDINGS}" \
  "model.image_position_tokens=${IMAGE_POSITION_TOKENS}" \
  "wandb.watch.enabled=${WANDB_WATCH_ENABLED}" \
  "wandb.watch.log=${WANDB_WATCH_LOG}" \
  "wandb.watch.log_freq=${WANDB_WATCH_LOG_FREQ}" \
  "validation_plots.enabled=${VALIDATION_PLOTS_ENABLED}" \
  "validation_plots.samples=${VALIDATION_PLOTS_SAMPLES}" \
  "validation_plots.every_n_epochs=${VALIDATION_PLOTS_EVERY_N_EPOCHS}" \
  "data.require_teacher_cache=${REQUIRE_TEACHER_CACHE}" \
  "data.prompt_class_filter_csv=${PROMPT_CLASS_FILTER_CSV}" \
  "data.min_prompt_accuracy=${MIN_PROMPT_ACCURACY}" \
  "data.prompt_class_names=${PROMPT_CLASS_NAMES}" \
  "data.prompt_class_names_file=${PROMPT_CLASS_NAMES_FILE}" \
  "data.curriculum_schedule=${CURRICULUM_SCHEDULE}"
