#!/usr/bin/env bash
set -euo pipefail

CORALMICRO="../coralmicro"
PORT="auto"
BAUD="115200"
ITERATIONS="100"
TIMEOUT_S="240"
POST_FLASH_DELAY_S="8"
FORCE=0
SKIP_STAGE=0
SKIP_BUILD=0
SKIP_FLASH=0
SKIP_CAPTURE=0
SUDO_FLASH=0
DRY_RUN=0
FLASHTOOL_PYTHON=""
MODEL_KIND="auto"
ECHO_RAW=0

APP="vlm_micro_tallyqa_benchmark_serial"
PROMPT_LOOKUP_HEADER="artifacts/exports/coral/prompt_embedding_lookup/tallyqa_prompt_embedding_lookup.h"

MODELS=()
RUN_NAMES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_coral_micro_selftest_latency.sh [options]

Runs the Coral Micro on-board seeded self-test latency path:
  stage model -> build benchmark target -> flash -> capture 100 measured inferences.

By default it runs three models:
  - untrained raw-prompt dummy
  - SSD Lite MobileDet COCO
  - SSD MobileNetV2 COCO17

Options:
  --coralmicro PATH       Coral Micro SDK checkout (default: ../coralmicro)
  --model PATH            Run one model instead of the default set
  --run-name NAME         Required with --model
  --port PORT             Serial port, or auto (default: auto)
  --baud INT              Serial baud (default: 115200)
  --iterations INT        Measured self-test iterations to require (default: 100)
  --timeout-s SEC         Capture timeout per model (default: 240)
  --post-flash-delay-s SEC
                          Sleep after flashing before capture (default: 8)
  --flashtool-python PATH Python used for flashtool; defaults to SDK .venv/bin/python
  --model-kind KIND      auto, tallyqa, or detection (default: auto)
  --prompt-lookup-header PATH
                          Quantized prompt lookup header for TallyQA models
                          (default: full lookup table)
  --echo-raw              Print raw non-protocol serial lines during capture
  --sudo-flash            Run flashtool through sudo env PATH=...
  --skip-stage
  --skip-build
  --skip-flash
  --skip-capture
  --force                 Overwrite reports/logs and staged SDK files
  --dry-run               Print commands without executing
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coralmicro) CORALMICRO="$2"; shift 2 ;;
    --model) MODELS+=("$2"); shift 2 ;;
    --run-name) RUN_NAMES+=("$2"); shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --baud) BAUD="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
    --post-flash-delay-s) POST_FLASH_DELAY_S="$2"; shift 2 ;;
    --flashtool-python) FLASHTOOL_PYTHON="$2"; shift 2 ;;
    --model-kind) MODEL_KIND="$2"; shift 2 ;;
    --prompt-lookup-header) PROMPT_LOOKUP_HEADER="$2"; shift 2 ;;
    --echo-raw) ECHO_RAW=1; shift ;;
    --sudo-flash) SUDO_FLASH=1; shift ;;
    --skip-stage) SKIP_STAGE=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-flash) SKIP_FLASH=1; shift ;;
    --skip-capture) SKIP_CAPTURE=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  MODELS=(
    "artifacts/reports/coral/app_test_dummy/current_raw_prompt_embedding_contract_dummy/model_int8_edgetpu.tflite"
    "artifacts/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
    "artifacts/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
  )
  RUN_NAMES=(
    "untrained_raw_prompt_dummy"
    "ssdlite_mobiledet_coco_qat_postprocess"
    "tf2_ssd_mobilenet_v2_coco17_ptq"
  )
fi

if [[ "${#MODELS[@]}" -ne "${#RUN_NAMES[@]}" ]]; then
  echo "Expected the same number of --model and --run-name values." >&2
  exit 2
fi

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

run_shell() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    bash -lc "$*"
  fi
}

flash_app() {
  local app_dir="${CORALMICRO}/build/examples/${APP}"
  local elf="${app_dir}/${APP}"
  local stripped="${app_dir}/${APP}.stripped"

  if [[ "${DRY_RUN}" == "0" && ! -e "${elf}" && ! -e "${stripped}" ]]; then
    echo "No flashable ELF found for ${APP} under ${app_dir}." >&2
    echo "Expected either ${elf} or ${stripped} after build." >&2
    exit 1
  fi

  local flash_cmd=()
  if [[ -e "${elf}" || "${DRY_RUN}" == "1" ]]; then
    flash_cmd=("${FLASHTOOL_PYTHON}" "${CORALMICRO}/scripts/flashtool.py" -e "${APP}")
  else
    flash_cmd=(
      "${FLASHTOOL_PYTHON}" "${CORALMICRO}/scripts/flashtool.py"
      --elf_path "${stripped}"
      --data_dir "${app_dir}"
    )
  fi

  if [[ "${SUDO_FLASH}" == "1" ]]; then
    run_cmd sudo env PATH="${PATH}" "${flash_cmd[@]}"
  else
    run_cmd "${flash_cmd[@]}"
  fi
}

for index in "${!MODELS[@]}"; do
  model="${MODELS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  output="artifacts/reports/coral/selftest/${run_name}.json"
  raw_log="artifacts/profiles/coral/${run_name}_serial.log"

  echo
  echo "=== Coral Micro self-test: ${run_name} ==="
  echo "Model: ${model}"

  if [[ "${SKIP_STAGE}" == "0" ]]; then
    stage_cmd=(
      uv run python scripts/coral_micro_stage_tallyqa_benchmark_app.py
      --coralmicro "${CORALMICRO}"
      --model "${model}"
      --prompt-lookup-header "${PROMPT_LOOKUP_HEADER}"
      --model-kind "${MODEL_KIND}"
    )
    if [[ "${FORCE}" == "1" ]]; then
      stage_cmd+=(--force)
    fi
    run_cmd "${stage_cmd[@]}"
  fi

  if [[ "${SKIP_BUILD}" == "0" ]]; then
    run_shell "cd '${CORALMICRO}' && bash build.sh -c && make -C build -j \"\$(nproc)\" '${APP}'"
  fi

  if [[ "${SKIP_FLASH}" == "0" ]]; then
    flash_app
    if [[ "${POST_FLASH_DELAY_S}" != "0" ]]; then
      run_cmd sleep "${POST_FLASH_DELAY_S}"
    fi
  fi

  if [[ "${SKIP_CAPTURE}" == "0" ]]; then
    capture_cmd=(
      uv run python scripts/capture_coral_micro_selftest.py
      --port "${PORT}"
      --baud "${BAUD}"
      --model-name "${run_name}"
      --output "${output}"
      --raw-log "${raw_log}"
      --min-measured-iterations "${ITERATIONS}"
      --timeout-s "${TIMEOUT_S}"
    )
    if [[ "${FORCE}" == "1" ]]; then
      capture_cmd+=(--force)
    fi
    if [[ "${ECHO_RAW}" == "1" ]]; then
      capture_cmd+=(--echo-raw)
    fi
    run_cmd "${capture_cmd[@]}"
  fi

  echo "Report: ${output}"
  echo "Raw log: ${raw_log}"
done
