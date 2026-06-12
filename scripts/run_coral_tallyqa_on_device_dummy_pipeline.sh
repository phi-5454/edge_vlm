#!/usr/bin/env bash
set -euo pipefail

CORALMICRO="../coralmicro"
MODEL="artifacts/reports/coral/edgetpu_compiler/prompt_patch_mlp_static_prompt_minimalistic_large_compile_probe_docker/ptq/model_int8_edgetpu.tflite"
PROMPT_LOOKUP_MANIFEST="artifacts/exports/coral/prompt_embedding_lookup/prompt_embedding_lookup_manifest.json"
DATASET="data/tallyqa_cauldron_target_mobilenet224_letterbox"
RUN_NAME="coral_micro_tallyqa_untrained_prompt_patch_mlp_dummy"
PORT="/dev/ttyACM0"
BAUD="115200"
SERIAL_TIMEOUT_S="120"
READY_TIMEOUT_S=""
RX_READY_TIMEOUT_S=""
RESULT_TIMEOUT_S=""
POST_FLASH_DELAY_S="8"
PAYLOAD_CHUNK_SIZE="512"
PAYLOAD_CHUNK_DELAY_S="0.0005"
MAX_EXAMPLES="128"
ANSWER_MIN="0"
ANSWER_MAX="5"
COLLAPSE_AT="5"
FLASHTOOL_PYTHON=""
SKIP_STAGE=0
SKIP_BUILD=0
SKIP_FLASH=0
SKIP_CACHE=0
SKIP_EDA=0
FORCE=0
DRY_RUN=0
SUDO_FLASH=0
DEBUG_PROTOCOL=0

usage() {
  cat <<'EOF'
Usage: scripts/run_coral_tallyqa_on_device_dummy_pipeline.sh [options]

Runs the Coral Micro TallyQA benchmark path from staging through cache EDA.
Defaults target the untrained, packed prompt_patch_mlp EdgeTPU probe.

Options:
  --coralmicro PATH       Coral Micro SDK checkout (default: ../coralmicro)
  --model PATH            Compiled EdgeTPU .tflite to stage
  --prompt-lookup-manifest PATH
                          Prompt lookup manifest matching staged firmware header
  --dataset PATH          TallyQA target dataset
  --run-name NAME         Artifact/run stem
  --port PATH             Serial device (default: /dev/ttyACM0)
  --baud INT              Serial baud (default: 115200)
  --serial-timeout-s SEC  Seconds to wait for each serial protocol line (default: 120)
  --ready-timeout-s SEC   Seconds to wait for board READY
  --rx-ready-timeout-s SEC
                          Seconds to wait for RX_READY after sending JSON header
  --result-timeout-s SEC  Seconds to wait for RESULT after image payload
  --post-flash-delay-s SEC
                          Seconds to wait after flashing before serial cache (default: 8)
  --payload-chunk-size INT
                          Bytes per image payload serial write (default: 512)
  --payload-chunk-delay-s SEC
                          Delay between image payload chunks (default: 0.0005)
  --max-examples INT      Dataset examples to stream (default: 128)
  --answer-min INT        Minimum output class (default: 0)
  --answer-max INT        Maximum output class (default: 5)
  --collapse-at INT       Collapse labels >= n into final bucket (default: 5)
  --flashtool-python PATH Python executable for Coral flashtool
  --skip-stage            Do not stage app/model into SDK
  --skip-build            Do not run SDK build.sh
  --skip-flash            Do not flash board app
  --skip-cache            Do not run serial dataset sweep
  --skip-eda              Do not run cache EDA
  --sudo-flash            Run only the Coral flashtool command through sudo
  --debug-protocol        Print host-side protocol milestones during cache
  --force                 Overwrite staged files and cache output
  --dry-run               Print commands without executing them
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coralmicro) CORALMICRO="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt-lookup-manifest) PROMPT_LOOKUP_MANIFEST="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --baud) BAUD="$2"; shift 2 ;;
    --serial-timeout-s) SERIAL_TIMEOUT_S="$2"; shift 2 ;;
    --ready-timeout-s) READY_TIMEOUT_S="$2"; shift 2 ;;
    --rx-ready-timeout-s) RX_READY_TIMEOUT_S="$2"; shift 2 ;;
    --result-timeout-s) RESULT_TIMEOUT_S="$2"; shift 2 ;;
    --post-flash-delay-s) POST_FLASH_DELAY_S="$2"; shift 2 ;;
    --payload-chunk-size) PAYLOAD_CHUNK_SIZE="$2"; shift 2 ;;
    --payload-chunk-delay-s) PAYLOAD_CHUNK_DELAY_S="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --answer-min) ANSWER_MIN="$2"; shift 2 ;;
    --answer-max) ANSWER_MAX="$2"; shift 2 ;;
    --collapse-at) COLLAPSE_AT="$2"; shift 2 ;;
    --flashtool-python) FLASHTOOL_PYTHON="$2"; shift 2 ;;
    --skip-stage) SKIP_STAGE=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-flash) SKIP_FLASH=1; shift ;;
    --skip-cache) SKIP_CACHE=1; shift ;;
    --skip-eda) SKIP_EDA=1; shift ;;
    --sudo-flash) SUDO_FLASH=1; shift ;;
    --debug-protocol) DEBUG_PROTOCOL=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

CACHE="artifacts/teacher_cache/${RUN_NAME}.jsonl"
RAW_LOG="artifacts/profiles/coral/${RUN_NAME}_serial.log"
EDA_DIR="artifacts/reports/coral/on_device_benchmark/${RUN_NAME}"

if [[ -z "${FLASHTOOL_PYTHON}" ]]; then
  if [[ -x "${CORALMICRO}/.venv/bin/python" ]]; then
    FLASHTOOL_PYTHON="$(cd "${CORALMICRO}" && pwd)/.venv/bin/python"
  else
    FLASHTOOL_PYTHON="python3"
  fi
elif [[ "${FLASHTOOL_PYTHON}" != /* ]]; then
  FLASHTOOL_PYTHON="$(pwd)/${FLASHTOOL_PYTHON}"
fi

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

if [[ "${SKIP_STAGE}" == "0" ]]; then
  stage_cmd=(
    uv run python scripts/coral_micro_stage_tallyqa_benchmark_app.py
    --coralmicro "${CORALMICRO}"
    --model "${MODEL}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    stage_cmd+=(--force)
  fi
  run_cmd "${stage_cmd[@]}"
fi

if [[ "${SKIP_BUILD}" == "0" ]]; then
  run_cmd bash -lc "cd '${CORALMICRO}' && bash build.sh"
fi

if [[ "${SKIP_FLASH}" == "0" ]]; then
  if [[ "${SUDO_FLASH}" == "1" ]]; then
    run_cmd sudo env PATH="${PATH}" "${FLASHTOOL_PYTHON}" "${CORALMICRO}/scripts/flashtool.py" -e vlm_micro_tallyqa_benchmark_serial
  else
    run_cmd bash -lc "cd '${CORALMICRO}' && '${FLASHTOOL_PYTHON}' scripts/flashtool.py -e vlm_micro_tallyqa_benchmark_serial"
  fi
  if [[ "${POST_FLASH_DELAY_S}" != "0" ]]; then
    run_cmd sleep "${POST_FLASH_DELAY_S}"
  fi
fi

if [[ "${SKIP_CACHE}" == "0" ]]; then
  cache_cmd=(
    uv run python scripts/cache_coral_micro_tallyqa_teacher.py
    --port "${PORT}"
    --baud "${BAUD}"
    --serial-timeout-s "${SERIAL_TIMEOUT_S}"
    --payload-chunk-size "${PAYLOAD_CHUNK_SIZE}"
    --payload-chunk-delay-s "${PAYLOAD_CHUNK_DELAY_S}"
    --dataset "${DATASET}"
    --output "${CACHE}"
    --model-name "${RUN_NAME}"
    --prompt-lookup-manifest "${PROMPT_LOOKUP_MANIFEST}"
    --max-examples "${MAX_EXAMPLES}"
    --answer-min "${ANSWER_MIN}"
    --answer-max "${ANSWER_MAX}"
    --collapse-at "${COLLAPSE_AT}"
    --raw-log "${RAW_LOG}"
  )
  if [[ -n "${READY_TIMEOUT_S}" ]]; then
    cache_cmd+=(--ready-timeout-s "${READY_TIMEOUT_S}")
  fi
  if [[ -n "${RX_READY_TIMEOUT_S}" ]]; then
    cache_cmd+=(--rx-ready-timeout-s "${RX_READY_TIMEOUT_S}")
  fi
  if [[ -n "${RESULT_TIMEOUT_S}" ]]; then
    cache_cmd+=(--result-timeout-s "${RESULT_TIMEOUT_S}")
  fi
  if [[ "${DEBUG_PROTOCOL}" == "1" ]]; then
    cache_cmd+=(--debug-protocol)
  fi
  if [[ "${FORCE}" == "1" ]]; then
    cache_cmd+=(--force)
  else
    cache_cmd+=(--resume)
  fi
  run_cmd "${cache_cmd[@]}"
fi

if [[ "${SKIP_EDA}" == "0" ]]; then
  run_cmd uv run python scripts/run_coral_micro_tallyqa_cache_eda.py \
    --cache "${CACHE}" \
    --dataset "${DATASET}" \
    --output-dir "${EDA_DIR}" \
    --title "${RUN_NAME}" \
    --answer-min "${ANSWER_MIN}" \
    --answer-max "${ANSWER_MAX}" \
    --collapse-at "${COLLAPSE_AT}"
fi

cat <<EOF

Pipeline artifacts:
  cache:    ${CACHE}
  manifest: ${CACHE%.jsonl}.manifest.json
  raw log:  ${RAW_LOG}
  EDA:      ${EDA_DIR}
EOF
