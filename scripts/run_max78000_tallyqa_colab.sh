#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-data/tallyqa_cauldron_target_mobilenet224_letterbox}"
DATASET_OUTPUT="${DATASET_OUTPUT:-data/max78000_tallyqa_count_fold2_56}"
PROMPT_CLASS_NAMES_FILE="${PROMPT_CLASS_NAMES_FILE:-}"
AI8X_TRAINING="${AI8X_TRAINING:-../MAX78000/ai8x-training}"
AI8X_TRAINING_REPO="${AI8X_TRAINING_REPO:-https://github.com/analogdevicesinc/ai8x-training.git}"
RUN_NAME="${RUN_NAME:-tallyqa_count_mbv3small}"
MODEL_NAME="${MODEL_NAME:-ai85tallyqambv3smallcount}"
DATASET_NAME="${DATASET_NAME:-tallyqa_count_fold2_56}"
QAT_POLICY="${QAT_POLICY:-policies/qat_policy_tallyqa_count.yaml}"
SCHEDULE="${SCHEDULE:-policies/schedule-tallyqa-count.yaml}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
OPTIMIZER="${OPTIMIZER:-Adam}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
PRINT_FREQ="${PRINT_FREQ:-100}"
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
REPORT_DIR="${REPORT_DIR:-artifacts/reports/max78000/tallyqa_training}"

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
  --ai8x-training DIR          Path to sibling ai8x-training checkout.
  --clone-ai8x                 Clone ai8x-training if --ai8x-training is absent.
  --run-name NAME              ADI training run name.
  --epochs N
  --batch-size N
  --lr FLOAT
  --optimizer NAME
  --weight-decay FLOAT
  --seed N
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
      shift 2
      ;;
    --prompt-class-names-file)
      PROMPT_CLASS_NAMES_FILE="$2"
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
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --qat-policy)
      QAT_POLICY="$2"
      shift 2
      ;;
    --schedule)
      SCHEDULE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --print-freq)
      PRINT_FREQ="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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

if [[ "${MATERIALIZE}" == "1" || "${MATERIALIZE}" == "true" ]]; then
  materialize_args=(
    uv run python scripts/materialize_max78000_tallyqa_dataset.py
    --source "${SOURCE_DATASET}"
    --output "${DATASET_OUTPUT}"
    --seed "${SEED}"
  )
  if [[ -n "${PROMPT_CLASS_NAMES_FILE}" ]]; then
    materialize_args+=(--prompt-class-names-file "${PROMPT_CLASS_NAMES_FILE}")
  fi
  if [[ "${FORCE}" == "1" || "${FORCE}" == "true" ]]; then
    materialize_args+=(--force)
  fi
  run_cmd "${materialize_args[@]}"
elif [[ ! -f "${DATASET_OUTPUT}/manifest.jsonl" ]]; then
  echo "Materialized dataset missing: ${DATASET_OUTPUT}/manifest.jsonl" >&2
  exit 1
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
fi

ai8x_abs="$(cd "${AI8X_TRAINING}" && pwd)"
data_abs="$(cd "$(dirname "${DATASET_OUTPUT}")" && pwd)/$(basename "${DATASET_OUTPUT}")"
report_abs="$(mkdir -p "${REPORT_DIR}/${RUN_NAME}" && cd "${REPORT_DIR}/${RUN_NAME}" && pwd)"
distiller_pythonpath="${ai8x_abs}/distiller"

if [[ "${MODEL_REPORT}" == "1" || "${MODEL_REPORT}" == "true" ]]; then
  model_report_args=(
    uv run python scripts/report_max78000_tallyqa_model.py
    --ai8x-training "${ai8x_abs}"
    --output-dir "${report_abs}/model"
    --factory "${MODEL_NAME}"
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
  "$QAT_POLICY" \
  "$SCHEDULE" \
  "$EPOCHS" \
  "$BATCH_SIZE" \
  "$LEARNING_RATE" \
  "$OPTIMIZER" \
  "$WEIGHT_DECAY" \
  "${PROMPT_CLASS_NAMES_FILE}" \
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
    qat_policy,
    schedule,
    epochs,
    batch_size,
    learning_rate,
    optimizer,
    weight_decay,
    prompt_class_names_file,
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
    "model_name": model_name,
    "dataset_name": dataset_name,
    "qat_policy": qat_policy,
    "schedule": schedule,
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "learning_rate": float(learning_rate),
    "optimizer": optimizer,
    "weight_decay": float(weight_decay),
    "train_enabled": train_enabled in {"1", "true", "True"},
    "train_command": train_command,
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {manifest_path}")
PY

if [[ "${TRAIN}" == "1" || "${TRAIN}" == "true" ]]; then
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "Would launch ADI ai8x-training run: ${RUN_NAME}"
    printf '+ cd %q &&' "${ai8x_abs}"
    printf ' PYTHONPATH=%q' "${distiller_pythonpath}:${PYTHONPATH:-}"
    printf ' %q' "${train_args[@]}"
    printf ' 2>&1 | tee %q\n' "${report_abs}/train.log"
  else
    echo "Launching ADI ai8x-training run: ${RUN_NAME}"
    (
      cd "${ai8x_abs}"
      export PYTHONPATH="${distiller_pythonpath}:${PYTHONPATH:-}"
      .venv/bin/python - <<'PY'
import sys
import distiller

print("Python path head:", sys.path[:5])
print(
    "Resolved distiller:",
    getattr(distiller, "__file__", None),
    list(getattr(distiller, "__path__", [])),
)
from distiller import apputils, model_summaries

print("Resolved distiller imports:", apputils.__name__, model_summaries.__name__)
PY
      "${train_args[@]}"
    ) 2>&1 | tee "${report_abs}/train.log"
  fi
else
  echo "Skipping training. Command that would run:"
  printf 'cd %q &&' "${ai8x_abs}"
  printf ' %q' "${train_args[@]}"
  printf '\n'
fi

if [[ "${TRAIN}" == "1" || "${TRAIN}" == "true" ]] && [[ "${WANDB_UPLOAD_CKPT}" == "1" || "${WANDB_UPLOAD_CKPT}" == "true" ]]; then
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
