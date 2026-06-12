#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-data/tallyqa_cauldron_target_mobilenet224_letterbox}"
DATASET_OUTPUT="${DATASET_OUTPUT:-data/max78000_tallyqa_count_fold2_56}"
PROMPT_CLASS_NAMES_FILE="${PROMPT_CLASS_NAMES_FILE:-}"
TIERED_CURRICULUM_DIR="${TIERED_CURRICULUM_DIR:-}"
DATASET_TIER="${DATASET_TIER:-}"
AI8X_TRAINING="${AI8X_TRAINING:-../MAX78000/ai8x-training}"
AI8X_TRAINING_REPO="${AI8X_TRAINING_REPO:-https://github.com/analogdevicesinc/ai8x-training.git}"
RUN_NAME="${RUN_NAME:-tallyqa_count_mbv3small}"
MODEL_NAME="${MODEL_NAME:-ai85tallyqambv3smallcount}"
DATASET_NAME="${DATASET_NAME:-tallyqa_count_fold2_56}"
MODEL_INPUT_CHANNELS="${MODEL_INPUT_CHANNELS:-12}"
QAT_POLICY="${QAT_POLICY:-policies/qat_policy_tallyqa_count.yaml}"
SCHEDULE="${SCHEDULE:-policies/schedule-tallyqa-count.yaml}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
OPTIMIZER="${OPTIMIZER:-Adam}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
PRINT_FREQ="${PRINT_FREQ:-100}"
WORKERS="${WORKERS:-0}"
SEED="${SEED:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
MATERIALIZE="${MATERIALIZE:-1}"
STAGE="${STAGE:-1}"
SETUP_AI8X_ENV="${SETUP_AI8X_ENV:-1}"
TRAIN="${TRAIN:-1}"
MODEL_REPORT="${MODEL_REPORT:-1}"
WANDB_UPLOAD_CKPT="${WANDB_UPLOAD_CKPT:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-vlm-micro}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-../wandb_api_key.env}"
FORCE="${FORCE:-0}"
CLONE_AI8X="${CLONE_AI8X:-0}"
DRY_RUN="${DRY_RUN:-0}"
KERAS_TIER0_QAT_COMPARISON="${KERAS_TIER0_QAT_COMPARISON:-0}"
TORCH_REVERSE_ABLATION_COMPARISON="${TORCH_REVERSE_ABLATION_COMPARISON:-0}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/max78000/tallyqa_training}"
TEACHER_CACHE="${TEACHER_CACHE:-}"
TEACHER_PROBABILITY_TEMPERATURE="${TEACHER_PROBABILITY_TEMPERATURE:-1.0}"
MISSING_TEACHER_POLICY="${MISSING_TEACHER_POLICY:-filter}"
DISTILLATION_ALPHA="${DISTILLATION_ALPHA:-1.0}"
DISTILLATION_BETA="${DISTILLATION_BETA:-0.0}"
DISTILLATION_TEMPERATURE="${DISTILLATION_TEMPERATURE:-2.0}"

DATASET_OUTPUT_SET=0
TIERED_CURRICULUM_DIR_SET=0
DATASET_TIER_SET=0
RUN_NAME_SET=0
MODEL_NAME_SET=0
DATASET_NAME_SET=0
MODEL_INPUT_CHANNELS_SET=0
QAT_POLICY_SET=0
SCHEDULE_SET=0
EPOCHS_SET=0
BATCH_SIZE_SET=0
LEARNING_RATE_SET=0
OPTIMIZER_SET=0
WEIGHT_DECAY_SET=0
PRINT_FREQ_SET=0
TEACHER_CACHE_SET=0
DISTILLATION_BETA_SET=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_max78000_tallyqa_colab.sh [options]

Runs the current MAX78000 TallyQA count training scaffold from edge_vlm, while
keeping the ADI ai8x-training checkout external to this repo.

Common options:
  --source DATASET             Source TallyQA target dataset.
  --dataset-output DIR         Materialized MAX78000 dataset output.
  --prompt-class-names-file FILE
                               Optional prompt class subset. Enables 0/1/2/3/4/5+ general count mode.
  --tiered-curriculum-dir DIR   Directory containing tier_*/prompt_classes.txt.
  --dataset-tier NAME           Train on one tier under --dataset-output. If missing, all tiers
                                are materialized from --tiered-curriculum-dir first.
  --ai8x-training DIR          Path to sibling ai8x-training checkout.
  --clone-ai8x                 Clone ai8x-training if --ai8x-training is absent.
  --run-name NAME              ADI training run name.
  --model NAME                 ADI model factory name.
  --dataset-name NAME          ADI dataset registry name.
  --model-input-channels N     Input channels for the architecture report.
  --epochs N
  --batch-size N
  --lr FLOAT
  --optimizer NAME
  --weight-decay FLOAT
  --workers N                  DataLoader workers. Defaults to 0 for Colab stability.
  --seed N
  --teacher-cache FILE        Optional TallyQA teacher cache JSONL for cached KL distillation.
  --teacher-probability-temperature FLOAT
  --missing-teacher-policy keep|filter
  --distillation-alpha FLOAT
  --distillation-beta FLOAT
  --distillation-temperature FLOAT
  --extra-train-args STRING    Extra raw args appended to train.py.
  --skip-materialize
  --skip-stage
  --skip-ai8x-env
  --skip-model-report
  --skip-wandb-checkpoint-upload
  --skip-train
  --wandb-project NAME
  --wandb-entity NAME
  --wandb-mode online|offline|disabled
  --wandb-env-file FILE
  --force
  --dry-run
  --torch-reverse-ablation-comparison
                               Apply MAX settings closest to the long-history PyTorch
                               reverse-ablation winner recipe (14pu / tier growth):
                               tiered dataset, large-minimal MAX model, 30 epochs,
                               batch 256, Adam lr 1e-3, wd 1e-2, QAT from epoch 0,
                               30-epoch decay schedule, print every 25.
  --keras-tier0-qat-comparison
                               Apply MAX settings closest to the Keras tier0 MLP QAT run:
                               30 epochs, batch 256, Adam lr 1e-3, wd 1e-2,
                               QAT from epoch 0, 30-epoch decay schedule, print every 25.

Colab shape:
  cd /content/edge_vlm
  bash scripts/run_max78000_tallyqa_colab.sh --clone-ai8x --force

Local shape:
  cd /home/younes/Courses/ETH/ML_Micro/edge_vlm
  bash scripts/run_max78000_tallyqa_colab.sh --force
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_DATASET="$2"
      shift 2
      ;;
    --dataset-output)
      DATASET_OUTPUT="$2"
      DATASET_OUTPUT_SET=1
      shift 2
      ;;
    --prompt-class-names-file)
      PROMPT_CLASS_NAMES_FILE="$2"
      shift 2
      ;;
    --tiered-curriculum-dir)
      TIERED_CURRICULUM_DIR="$2"
      TIERED_CURRICULUM_DIR_SET=1
      shift 2
      ;;
    --dataset-tier)
      DATASET_TIER="$2"
      DATASET_TIER_SET=1
      shift 2
      ;;
    --ai8x-training)
      AI8X_TRAINING="$2"
      shift 2
      ;;
    --ai8x-training-repo)
      AI8X_TRAINING_REPO="$2"
      shift 2
      ;;
    --clone-ai8x)
      CLONE_AI8X=1
      shift
      ;;
    --run-name)
      RUN_NAME="$2"
      RUN_NAME_SET=1
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      MODEL_NAME_SET=1
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      DATASET_NAME_SET=1
      shift 2
      ;;
    --model-input-channels)
      MODEL_INPUT_CHANNELS="$2"
      MODEL_INPUT_CHANNELS_SET=1
      shift 2
      ;;
    --qat-policy)
      QAT_POLICY="$2"
      QAT_POLICY_SET=1
      shift 2
      ;;
    --schedule)
      SCHEDULE="$2"
      SCHEDULE_SET=1
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      EPOCHS_SET=1
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      BATCH_SIZE_SET=1
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      LEARNING_RATE_SET=1
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      OPTIMIZER_SET=1
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      WEIGHT_DECAY_SET=1
      shift 2
      ;;
    --print-freq)
      PRINT_FREQ="$2"
      PRINT_FREQ_SET=1
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --teacher-cache)
      TEACHER_CACHE="$2"
      TEACHER_CACHE_SET=1
      shift 2
      ;;
    --teacher-probability-temperature)
      TEACHER_PROBABILITY_TEMPERATURE="$2"
      shift 2
      ;;
    --missing-teacher-policy)
      MISSING_TEACHER_POLICY="$2"
      shift 2
      ;;
    --distillation-alpha)
      DISTILLATION_ALPHA="$2"
      shift 2
      ;;
    --distillation-beta)
      DISTILLATION_BETA="$2"
      DISTILLATION_BETA_SET=1
      shift 2
      ;;
    --distillation-temperature)
      DISTILLATION_TEMPERATURE="$2"
      shift 2
      ;;
    --extra-train-args)
      EXTRA_TRAIN_ARGS="$2"
      shift 2
      ;;
    --report-dir)
      REPORT_DIR="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb-env-file)
      WANDB_ENV_FILE="$2"
      shift 2
      ;;
    --skip-materialize)
      MATERIALIZE=0
      shift
      ;;
    --skip-stage)
      STAGE=0
      shift
      ;;
    --skip-ai8x-env)
      SETUP_AI8X_ENV=0
      shift
      ;;
    --skip-model-report)
      MODEL_REPORT=0
      shift
      ;;
    --skip-wandb-checkpoint-upload)
      WANDB_UPLOAD_CKPT=0
      shift
      ;;
    --skip-train)
      TRAIN=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --torch-reverse-ablation-comparison)
      TORCH_REVERSE_ABLATION_COMPARISON=1
      shift
      ;;
    --keras-tier0-qat-comparison)
      KERAS_TIER0_QAT_COMPARISON=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      echo "Unexpected positional argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

torch_reverse_enabled=0
keras_comparison_enabled=0
if [[ "${TORCH_REVERSE_ABLATION_COMPARISON}" == "1" || "${TORCH_REVERSE_ABLATION_COMPARISON}" == "true" ]]; then
  torch_reverse_enabled=1
fi
if [[ "${KERAS_TIER0_QAT_COMPARISON}" == "1" || "${KERAS_TIER0_QAT_COMPARISON}" == "true" ]]; then
  keras_comparison_enabled=1
fi
if [[ "${torch_reverse_enabled}" == "1" && "${keras_comparison_enabled}" == "1" ]]; then
  echo "Use either --torch-reverse-ablation-comparison or --keras-tier0-qat-comparison, not both." >&2
  exit 2
fi

if [[ "${torch_reverse_enabled}" == "1" ]]; then
  if [[ "${RUN_NAME_SET}" == "0" ]]; then
    RUN_NAME="max78000-tier0-reverse-ablation-14pu-approx-qat"
  fi
  if [[ "${MODEL_NAME_SET}" == "0" ]]; then
    MODEL_NAME="ai85tallyqambv3largepromptfilmcount"
  fi
  if [[ "${DATASET_NAME_SET}" == "0" ]]; then
    DATASET_NAME="tallyqa_count_fold2_56_prompt_embed576"
  fi
  if [[ "${MODEL_INPUT_CHANNELS_SET}" == "0" ]]; then
    MODEL_INPUT_CHANNELS=588
  fi
  if [[ "${DATASET_OUTPUT_SET}" == "0" ]]; then
    DATASET_OUTPUT="data/max78000_tallyqa_tiers_fold2_56"
  fi
  if [[ "${TIERED_CURRICULUM_DIR_SET}" == "0" ]]; then
    TIERED_CURRICULUM_DIR="artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum"
  fi
  if [[ "${DATASET_TIER_SET}" == "0" ]]; then
    DATASET_TIER="tier_0_acc_ge_0p60_n_ge_1000"
  fi
  if [[ "${QAT_POLICY_SET}" == "0" ]]; then
    QAT_POLICY="policies/qat_policy_tallyqa_count_from_start.yaml"
  fi
  if [[ "${SCHEDULE_SET}" == "0" ]]; then
    SCHEDULE="policies/schedule-tallyqa-count-30ep.yaml"
  fi
  if [[ "${EPOCHS_SET}" == "0" ]]; then
    EPOCHS=30
  fi
  if [[ "${BATCH_SIZE_SET}" == "0" ]]; then
    BATCH_SIZE=256
  fi
  if [[ "${LEARNING_RATE_SET}" == "0" ]]; then
    LEARNING_RATE=0.001
  fi
  if [[ "${OPTIMIZER_SET}" == "0" ]]; then
    OPTIMIZER=Adam
  fi
  if [[ "${WEIGHT_DECAY_SET}" == "0" ]]; then
    WEIGHT_DECAY=0.01
  fi
  if [[ "${PRINT_FREQ_SET}" == "0" ]]; then
    PRINT_FREQ=25
  fi
  if [[ "${TEACHER_CACHE_SET}" == "0" ]]; then
    TEACHER_CACHE="artifacts/teacher_cache/composite_ece_temp_smol1p1_frcnn2p2_beta12p968_tallyqa_target_mobilenet224.jsonl"
  fi
  if [[ "${DISTILLATION_BETA_SET}" == "0" ]]; then
    DISTILLATION_BETA=0.25
  fi
fi

if [[ "${keras_comparison_enabled}" == "1" ]]; then
  if [[ "${RUN_NAME_SET}" == "0" ]]; then
    RUN_NAME="max78000-tier0-keras-comparison-qat"
  fi
  if [[ "${QAT_POLICY_SET}" == "0" ]]; then
    QAT_POLICY="policies/qat_policy_tallyqa_count_from_start.yaml"
  fi
  if [[ "${SCHEDULE_SET}" == "0" ]]; then
    SCHEDULE="policies/schedule-tallyqa-count-30ep.yaml"
  fi
  if [[ "${EPOCHS_SET}" == "0" ]]; then
    EPOCHS=30
  fi
  if [[ "${BATCH_SIZE_SET}" == "0" ]]; then
    BATCH_SIZE=256
  fi
  if [[ "${LEARNING_RATE_SET}" == "0" ]]; then
    LEARNING_RATE=0.001
  fi
  if [[ "${OPTIMIZER_SET}" == "0" ]]; then
    OPTIMIZER=Adam
  fi
  if [[ "${WEIGHT_DECAY_SET}" == "0" ]]; then
    WEIGHT_DECAY=0.01
  fi
  if [[ "${PRINT_FREQ_SET}" == "0" ]]; then
    PRINT_FREQ=25
  fi
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

run_shell() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    bash -lc "$*"
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_cmd uv

if [[ -n "${PROMPT_CLASS_NAMES_FILE}" && -n "${TIERED_CURRICULUM_DIR}" ]]; then
  echo "Use either --prompt-class-names-file or --tiered-curriculum-dir, not both." >&2
  exit 2
fi

if [[ -n "${TIERED_CURRICULUM_DIR}" && -z "${DATASET_TIER}" ]] \
  && [[ "${TRAIN}" == "1" || "${TRAIN}" == "true" ]]; then
  echo "--tiered-curriculum-dir materializes all tiers; pass --dataset-tier to choose one for training." >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DATASET}" ]]; then
  echo "Source dataset not found: ${SOURCE_DATASET}" >&2
  echo "Copy it into place first, for example from Google Drive in Colab." >&2
  exit 1
fi

if [[ ! -d "${AI8X_TRAINING}" ]]; then
  if [[ "${CLONE_AI8X}" == "1" || "${CLONE_AI8X}" == "true" ]]; then
    require_cmd git
    run_cmd mkdir -p "$(dirname "${AI8X_TRAINING}")"
    run_cmd git clone "${AI8X_TRAINING_REPO}" "${AI8X_TRAINING}"
  else
    echo "ai8x-training checkout not found: ${AI8X_TRAINING}" >&2
    echo "Pass --clone-ai8x in Colab, or provide --ai8x-training /path/to/ai8x-training." >&2
    exit 1
  fi
fi

if [[ -d "${AI8X_TRAINING}/.git" ]] && {
  [[ "${CLONE_AI8X}" == "1" || "${CLONE_AI8X}" == "true" ]] ||
  [[ ! -f "${AI8X_TRAINING}/distiller/distiller/__init__.py" ]]
}; then
  require_cmd git
  run_cmd git -C "${AI8X_TRAINING}" submodule update --init --recursive
fi

selected_dataset_output="${DATASET_OUTPUT}"
if [[ -n "${DATASET_TIER}" ]]; then
  selected_dataset_output="${DATASET_OUTPUT}/${DATASET_TIER}"
fi

needs_materialize=0
if [[ -z "${DATASET_TIER}" ]] && [[ "${MATERIALIZE}" == "1" || "${MATERIALIZE}" == "true" ]]; then
  needs_materialize=1
fi
if [[ -n "${DATASET_TIER}" && ! -f "${selected_dataset_output}/manifest.jsonl" ]]; then
  if [[ -z "${TIERED_CURRICULUM_DIR}" ]]; then
    echo "Materialized tier missing: ${selected_dataset_output}/manifest.jsonl" >&2
    echo "Pass --tiered-curriculum-dir so the missing tier datasets can be materialized." >&2
    exit 2
  fi
  needs_materialize=1
fi
if [[ -n "${DATASET_TIER}" && -n "${TIERED_CURRICULUM_DIR}" ]] \
  && [[ "${MATERIALIZE}" == "1" || "${MATERIALIZE}" == "true" ]] \
  && [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
  needs_materialize=1
fi

if [[ "${needs_materialize}" == "1" ]]; then
  materialize_args=(
    uv run python scripts/materialize_max78000_tallyqa_dataset.py
    --source "${SOURCE_DATASET}"
    --output "${DATASET_OUTPUT}"
    --seed "${SEED}"
  )
  if [[ -n "${TIERED_CURRICULUM_DIR}" ]]; then
    materialize_args+=(--tiered-curriculum-dir "${TIERED_CURRICULUM_DIR}")
  elif [[ -n "${PROMPT_CLASS_NAMES_FILE}" ]]; then
    materialize_args+=(--prompt-class-names-file "${PROMPT_CLASS_NAMES_FILE}")
  fi
  if [[ -n "${TEACHER_CACHE}" ]]; then
    materialize_args+=(
      --teacher-cache "${TEACHER_CACHE}"
      --teacher-probability-temperature "${TEACHER_PROBABILITY_TEMPERATURE}"
      --missing-teacher-policy "${MISSING_TEACHER_POLICY}"
    )
  fi
  if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
    materialize_args+=(--force)
  fi
  run_cmd "${materialize_args[@]}"
fi

if [[ ! -f "${selected_dataset_output}/manifest.jsonl" ]]; then
  echo "Materialized dataset missing: ${selected_dataset_output}/manifest.jsonl" >&2
  exit 1
fi

if python - "${DISTILLATION_BETA}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
then
  if ! SELECTED_MANIFEST="${selected_dataset_output}/manifest.jsonl" python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["SELECTED_MANIFEST"])
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        raise SystemExit(0 if "teacher_probs" in row else 1)
raise SystemExit(1)
PY
  then
    echo "Distillation is enabled (DISTILLATION_BETA=${DISTILLATION_BETA}), but ${selected_dataset_output}/manifest.jsonl has no teacher_probs." >&2
    echo "Re-materialize with --teacher-cache or pass --force. To train without cached-teacher KL, set --distillation-beta 0." >&2
    exit 2
  fi
fi

if [[ "${STAGE}" == "1" || "${STAGE}" == "true" ]]; then
  stage_args=(
    uv run python scripts/stage_max78000_tallyqa_pipeline.py
    --ai8x-training "${AI8X_TRAINING}"
  )
  if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
    stage_args+=(--force)
  fi
  run_cmd "${stage_args[@]}"
  run_cmd uv run python scripts/patch_max78000_ai8x_distillation.py --ai8x-training "${AI8X_TRAINING}"
fi

ai8x_abs="$(cd "${AI8X_TRAINING}" && pwd)"
data_abs="$(cd "$(dirname "${selected_dataset_output}")" && pwd)/$(basename "${selected_dataset_output}")"
report_abs="$(mkdir -p "${REPORT_DIR}/${RUN_NAME}" && cd "${REPORT_DIR}/${RUN_NAME}" && pwd)"
distiller_pythonpath="${ai8x_abs}/distiller"

if [[ "${MODEL_REPORT}" == "1" || "${MODEL_REPORT}" == "true" ]]; then
  model_report_args=(
    uv run python scripts/report_max78000_tallyqa_model.py
    --ai8x-training "${ai8x_abs}"
    --output-dir "${report_abs}/model"
    --factory "${MODEL_NAME}"
    --input-channels "${MODEL_INPUT_CHANNELS}"
    --num-classes 6
  )
  echo "+ ${model_report_args[*]}"
  if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    if ! "${model_report_args[@]}"; then
      echo "Warning: MAX78000 model report failed; continuing with training." >&2
    fi
  fi
fi

if [[ "${SETUP_AI8X_ENV}" == "1" || "${SETUP_AI8X_ENV}" == "true" ]]; then
  run_shell "cd '${ai8x_abs}' && uv venv --python 3.11 --clear --seed .venv"
  ai8x_python="${ai8x_abs}/.venv/bin/python"
  req_tmp="${ai8x_abs}/.edge_vlm_filtered_requirements"
  run_cmd mkdir -p "${req_tmp}"
  python3 - "${ai8x_abs}/requirements-base.txt" "${req_tmp}/requirements-base.txt" \
    "${ai8x_abs}/requirements-datasets.txt" "${req_tmp}/requirements-datasets.txt" <<'PY'
import sys
from pathlib import Path

for src, dst in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    lines = []
    for line in Path(src).read_text().splitlines():
        if line.strip().lower().startswith("pyffmpeg=="):
            lines.append(f"# {line}  # filtered by edge_vlm wrapper: stale pin is unavailable on PyPI")
        else:
            lines.append(line)
    Path(dst).write_text("\n".join(lines) + "\n")
PY
  run_shell "cd '${ai8x_abs}' && uv pip install --python '${ai8x_python}' 'setuptools<81' wheel"
  run_shell "cd '${ai8x_abs}' && uv pip install --python '${ai8x_python}' --no-build-isolation-package visdom -r '${req_tmp}/requirements-base.txt' -r '${req_tmp}/requirements-datasets.txt' pycocotools==2.0.8"
  "${ai8x_python}" - <<'PY'
import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_paths()["purelib"])
site_packages.mkdir(parents=True, exist_ok=True)
stub_path = site_packages / "pyffmpeg.py"
stub_path.write_text(
    '"""edge_vlm compatibility stub for ai8x Kinetics import-time registration."""\n\n'
    "class FFmpeg:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise ImportError(\n"
    "            'pyffmpeg is unavailable because ai8x-training pins a stale package; '\n"
    "            'the Kinetics dataset is not supported in this edge_vlm environment.'\n"
    "        )\n",
    encoding="utf-8",
)
print(f"Wrote {stub_path}")
PY
  if [[ -f "${ai8x_abs}/distiller/setup.py" || -f "${ai8x_abs}/distiller/pyproject.toml" ]]; then
    run_shell "cd '${ai8x_abs}' && uv pip install --python '${ai8x_python}' -e distiller --config-settings editable_mode=strict"
  elif [[ -d "${ai8x_abs}/distiller" ]]; then
    echo "Warning: ${ai8x_abs}/distiller has no setup.py or pyproject.toml; continuing without editable distiller install." >&2
    "${ai8x_python}" - "${distiller_pythonpath}" <<'PY'
import sys
import sysconfig
from pathlib import Path

distiller_path = Path(sys.argv[1]).resolve()
inner_init = distiller_path / "distiller" / "__init__.py"
outer_init = distiller_path / "__init__.py"
if inner_init.exists():
    outer_init.write_text(
        "from pathlib import Path\n"
        "inner = Path(__file__).resolve().parent / 'distiller'\n"
        "__path__ = [str(inner)]\n"
        "exec((inner / '__init__.py').read_text(encoding='utf-8'), globals())\n",
        encoding="utf-8",
    )
    print(f"Wrote {outer_init} -> {inner_init}")
else:
    print(f"Warning: expected Distiller package source is missing: {inner_init}", file=sys.stderr)
site_packages = Path(sysconfig.get_paths()["purelib"])
site_packages.mkdir(parents=True, exist_ok=True)
pth_path = site_packages / "edge_vlm_distiller.pth"
pth_path.write_text(f"{distiller_path}\n", encoding="utf-8")
print(f"Wrote {pth_path} -> {distiller_path}")
PY
  else
    echo "Warning: ${ai8x_abs}/distiller is missing; continuing without editable distiller install." >&2
  fi
fi

train_args=(
  .venv/bin/python train.py
  --deterministic
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --optimizer "${OPTIMIZER}"
  --lr "${LEARNING_RATE}"
  --wd "${WEIGHT_DECAY}"
  --model "${MODEL_NAME}"
  --use-bias
  --dataset "${DATASET_NAME}"
  --data "${data_abs}"
  --device MAX78000
  --qat-policy "${QAT_POLICY}"
  --compress "${SCHEDULE}"
  --validation-split 0
  --print-freq "${PRINT_FREQ}"
  --workers "${WORKERS}"
  --name "${RUN_NAME}"
)

if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_TRAIN_ARGS} )
  train_args+=("${extra_args[@]}")
fi

manifest_path="${report_abs}/run_manifest.json"
python3 - "$manifest_path" \
  "$repo_root" \
  "$ai8x_abs" \
  "$distiller_pythonpath" \
  "$data_abs" \
  "$RUN_NAME" \
  "$MODEL_NAME" \
  "$DATASET_NAME" \
  "$MODEL_INPUT_CHANNELS" \
  "$QAT_POLICY" \
  "$SCHEDULE" \
  "$EPOCHS" \
  "$BATCH_SIZE" \
  "$LEARNING_RATE" \
  "$OPTIMIZER" \
  "$WEIGHT_DECAY" \
  "$WORKERS" \
  "$TEACHER_CACHE" \
  "$TEACHER_PROBABILITY_TEMPERATURE" \
  "$MISSING_TEACHER_POLICY" \
  "$DISTILLATION_ALPHA" \
  "$DISTILLATION_BETA" \
  "$DISTILLATION_TEMPERATURE" \
  "$KERAS_TIER0_QAT_COMPARISON" \
  "$TORCH_REVERSE_ABLATION_COMPARISON" \
  "${PROMPT_CLASS_NAMES_FILE}" \
  "${TIERED_CURRICULUM_DIR}" \
  "${DATASET_TIER}" \
  "$TRAIN" \
  "${train_args[@]}" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    repo_root,
    ai8x_training,
    distiller_pythonpath,
    data_dir,
    run_name,
    model_name,
    dataset_name,
    model_input_channels,
    qat_policy,
    schedule,
    epochs,
    batch_size,
    learning_rate,
    optimizer,
    weight_decay,
    workers,
    teacher_cache,
    teacher_probability_temperature,
    missing_teacher_policy,
    distillation_alpha,
    distillation_beta,
    distillation_temperature,
    keras_tier0_qat_comparison,
    torch_reverse_ablation_comparison,
    prompt_class_names_file,
    tiered_curriculum_dir,
    dataset_tier,
    train_enabled,
    *train_command,
) = sys.argv[1:]

def git_head(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_name": run_name,
    "repo_root": repo_root,
    "edge_vlm_git_head": git_head(repo_root),
    "ai8x_training": ai8x_training,
    "ai8x_training_git_head": git_head(ai8x_training),
    "distiller_pythonpath": distiller_pythonpath,
    "data_dir": data_dir,
    "prompt_class_names_file": prompt_class_names_file or None,
    "tiered_curriculum_dir": tiered_curriculum_dir or None,
    "dataset_tier": dataset_tier or None,
    "model_name": model_name,
    "dataset_name": dataset_name,
    "model_input_channels": int(model_input_channels),
    "qat_policy": qat_policy,
    "schedule": schedule,
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "learning_rate": float(learning_rate),
    "optimizer": optimizer,
    "weight_decay": float(weight_decay),
    "workers": int(workers),
    "teacher_cache": teacher_cache or None,
    "teacher_probability_temperature": float(teacher_probability_temperature),
    "missing_teacher_policy": missing_teacher_policy,
    "distillation": {
        "alpha": float(distillation_alpha),
        "beta": float(distillation_beta),
        "temperature": float(distillation_temperature),
        "source": "cached teacher_probs in materialized manifest",
    },
    "wandb_outputs": {
        "post_eval_output_dir": str(Path(manifest_path).parent / "wandb_eval"),
        "post_eval_samples": 4,
        "post_eval_batch_size": int(batch_size),
        "plots": [
            "val_plots/confusion_matrix",
            "val_plots/image_encoding",
            "test_plots/confusion_matrix",
            "test_plots/image_encoding",
        ],
    },
    "keras_tier0_qat_comparison": keras_tier0_qat_comparison in {"1", "true", "True"},
    "torch_reverse_ablation_comparison": torch_reverse_ablation_comparison
    in {"1", "true", "True"},
    "comparison_recipe": None,
    "train_enabled": train_enabled in {"1", "true", "True"},
    "train_command": train_command,
}
if payload["torch_reverse_ablation_comparison"]:
    payload["comparison_recipe"] = {
        "source_script": "scripts/run_tallyqa_baseline_reverse_ablation_tier0.sh",
        "source_run_family": "14pu winner recipe and tier-growth runs 17-20",
        "matched": {
            "tiered_prompt_class_filter": bool(dataset_tier),
            "input_source": (
                "full letterbox TallyQA target tensors folded to 56x56x12, "
                "plus 576-d precomputed prompt embedding planes"
            ),
            "precomputed_prompt_embeddings": (
                "artifacts/models/tallyqa_smolvlm_prompt_embeddings_letterbox.pt"
            ),
            "model_family": model_name,
            "prompt_embedding_conditioning": dataset_name,
            "prompt_feature_conditioning": (
                "trainable additive prompt-embedding bias after 28x28 and 14x14 reductions"
            ),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": optimizer,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "qat": qat_policy,
            "lr_schedule": schedule,
        },
        "not_matched_yet": [
            "original query_projection Dense/GELU/LayerNorm block",
            "full scale-and-shift FiLM gamma path",
            "original prompt_patch_mlp concat + 1d conv head",
            "local-soft target distribution",
            "prompt-class tempered sampler with epoch reload",
            "separate image-backbone learning-rate scale",
            "Lightning early stopping on prompt_class_output_weighted_mae",
        ],
        "max_wandb_trace": {
            "post_training_checkpoint_eval": True,
            "validation_test_confusion_matrices": True,
            "unique_image_examples": True,
            "head_map_visualization": "14x14x1 forward_features output",
        },
    }
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {manifest_path}")
PY

if [[ "${TRAIN}" == "1" || "${TRAIN}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "Would launch ADI ai8x-training run: ${RUN_NAME}"
    if [[ "${WANDB_MODE}" == "disabled" ]]; then
      printf '+ cd %q &&' "${ai8x_abs}"
      printf ' PYTHONPATH=%q' "${distiller_pythonpath}:${PYTHONPATH:-}"
      printf ' %q' "${train_args[@]}"
      printf ' 2>&1 | tee %q\n' "${report_abs}/train.log"
    else
      printf '+ uv run python scripts/run_max78000_ai8x_with_wandb.py --cwd %q --log-file %q --project %q --run-name %q --job-type max78000-training --mode %q --manifest %q --model-report-dir %q --distiller-pythonpath %q --checkpoint-root %q --checkpoint-run-name %q --post-eval-output-dir %q --post-eval-samples 4 --post-eval-batch-size %q --' \
        "${ai8x_abs}" "${report_abs}/train.log" "${WANDB_PROJECT}" "${RUN_NAME}" "${WANDB_MODE}" "${manifest_path}" "${report_abs}/model" "${distiller_pythonpath}" "${ai8x_abs}/logs" "${RUN_NAME}" "${report_abs}/wandb_eval" "${BATCH_SIZE}"
      printf ' %q' "${train_args[@]}"
      printf '\n'
    fi
  else
    echo "Launching ADI ai8x-training run: ${RUN_NAME}"
    if [[ "${WANDB_MODE}" == "disabled" ]]; then
      (
        cd "${ai8x_abs}"
        export PYTHONPATH="${distiller_pythonpath}:${PYTHONPATH:-}"
        export EDGE_VLM_DISTILLATION_ALPHA="${DISTILLATION_ALPHA}"
        export EDGE_VLM_DISTILLATION_BETA="${DISTILLATION_BETA}"
        export EDGE_VLM_DISTILLATION_TEMPERATURE="${DISTILLATION_TEMPERATURE}"
        "${train_args[@]}"
      ) 2>&1 | tee "${report_abs}/train.log"
    else
      wandb_train_args=(
        uv run python scripts/run_max78000_ai8x_with_wandb.py
        --cwd "${ai8x_abs}"
        --log-file "${report_abs}/train.log"
        --project "${WANDB_PROJECT}"
        --run-name "${RUN_NAME}"
        --job-type "max78000-training"
        --mode "${WANDB_MODE}"
        --manifest "${manifest_path}"
        --model-report-dir "${report_abs}/model"
        --distiller-pythonpath "${distiller_pythonpath}"
        --checkpoint-root "${ai8x_abs}/logs"
        --checkpoint-run-name "${RUN_NAME}"
        --post-eval-output-dir "${report_abs}/wandb_eval"
        --post-eval-samples 4
        --post-eval-batch-size "${BATCH_SIZE}"
      )
      if [[ -n "${WANDB_ENTITY}" ]]; then
        wandb_train_args+=(--entity "${WANDB_ENTITY}")
      fi
      if [[ -f "${WANDB_ENV_FILE}" ]]; then
        wandb_train_args+=(--env-file "${WANDB_ENV_FILE}")
      fi
      export EDGE_VLM_DISTILLATION_ALPHA="${DISTILLATION_ALPHA}"
      export EDGE_VLM_DISTILLATION_BETA="${DISTILLATION_BETA}"
      export EDGE_VLM_DISTILLATION_TEMPERATURE="${DISTILLATION_TEMPERATURE}"
      wandb_train_args+=(-- "${train_args[@]}")
      echo "+ ${wandb_train_args[*]}"
      "${wandb_train_args[@]}"
    fi
  fi
else
  echo "Skipping training. Command that would run:"
  printf 'cd %q &&' "${ai8x_abs}"
  printf ' %q' "${train_args[@]}"
  printf '\n'
fi

if [[ "${TRAIN}" == "1" || "${TRAIN}" == "true" ]] && [[ "${WANDB_MODE}" == "disabled" ]] && [[ "${WANDB_UPLOAD_CKPT}" == "1" || "${WANDB_UPLOAD_CKPT}" == "true" ]]; then
  chosen_ckpt="$(
    python3 - "${ai8x_abs}" "${RUN_NAME}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
logs = root / "logs"
candidates: list[Path] = []
if logs.exists():
    candidates.extend(logs.rglob(f"{run_name}_best.pth.tar"))
    candidates.extend(logs.rglob("best.pth.tar"))
    candidates.extend(logs.rglob("*_best.pth.tar"))
deduped = sorted({path.resolve() for path in candidates if path.is_file()}, key=lambda p: p.stat().st_mtime, reverse=True)
print(deduped[0] if deduped else "")
PY
  )"
  if [[ -n "${chosen_ckpt}" ]]; then
    upload_args=(
      uv run python scripts/upload_wandb_artifact.py
      --project "${WANDB_PROJECT}"
      --run-name "${RUN_NAME}-max78000-checkpoint-upload"
      --job-type "max78000-checkpoint-upload"
      --artifact-name "${RUN_NAME}-chosen-test-checkpoint"
      --artifact-type "model-checkpoint"
      --alias "best"
      --alias "test-evaluated"
      --file "${chosen_ckpt}"
      --file "${manifest_path}"
      --file "${report_abs}/train.log"
      --mode "${WANDB_MODE}"
    )
    if [[ -n "${WANDB_ENTITY}" ]]; then
      upload_args+=(--entity "${WANDB_ENTITY}")
    fi
    if [[ -f "${WANDB_ENV_FILE}" ]]; then
      upload_args+=(--env-file "${WANDB_ENV_FILE}")
    fi
    echo "+ ${upload_args[*]}"
    if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
      if ! "${upload_args[@]}"; then
        echo "Warning: W&B checkpoint artifact upload failed for ${chosen_ckpt}." >&2
      fi
    fi
  else
    echo "Warning: no ADI best checkpoint found under ${ai8x_abs}/logs; skipping W&B checkpoint artifact upload." >&2
  fi
fi

echo "MAX78000 wrapper complete."
echo "Report directory: ${report_abs}"
