#!/usr/bin/env bash
set -euo pipefail

TIER_ROOT="${TIER_ROOT:-artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum}"
TIERS="${TIERS:-0,1,2,3,4}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_tallyqa_keras_fusion_ablation_tiers.sh [--tiers LIST] [ablation options...]

Runs scripts/run_tallyqa_keras_fusion_ablation_tier0.sh once per tiered prompt
set. By default it runs tiers 0..4 and skips any run whose results JSON already
exists, so rerunning the same command is suitable after a Colab interruption.

Tier selectors:
  --tiers 0,1,2,3,4      Comma-separated tier ids to run.
  --tier-root DIR        Directory containing tier_* prompt class folders.
  --no-skip-completed    Re-run even if a run already has a results JSON.

All other arguments are forwarded to run_tallyqa_keras_fusion_ablation_tier0.sh.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiers)
      TIERS="$2"
      shift 2
      ;;
    --tier-root)
      TIER_ROOT="$2"
      shift 2
      ;;
    --skip-completed)
      SKIP_COMPLETED=1
      shift
      ;;
    --no-skip-completed|--force-rerun)
      SKIP_COMPLETED=0
      FORWARD_ARGS+=("--force-rerun")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

tier_file() {
  case "$1" in
    0) printf '%s\n' "${TIER_ROOT}/tier_0_acc_ge_0p60_n_ge_1000/prompt_classes.txt" ;;
    1) printf '%s\n' "${TIER_ROOT}/tier_1_acc_ge_0p60_n_ge_500/prompt_classes.txt" ;;
    2) printf '%s\n' "${TIER_ROOT}/tier_2_acc_ge_0p60/prompt_classes.txt" ;;
    3) printf '%s\n' "${TIER_ROOT}/tier_3_acc_ge_0p55/prompt_classes.txt" ;;
    4) printf '%s\n' "${TIER_ROOT}/tier_4_acc_ge_0p40/prompt_classes.txt" ;;
    *)
      echo "Unknown tier id: $1" >&2
      return 2
      ;;
  esac
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

IFS=',' read -ra tier_ids <<< "${TIERS}"
for tier_id in "${tier_ids[@]}"; do
  tier_id="${tier_id//[[:space:]]/}"
  if [[ -z "${tier_id}" ]]; then
    continue
  fi
  prompt_file="$(tier_file "${tier_id}")"
  if [[ ! -f "${prompt_file}" ]]; then
    echo "Tier ${tier_id} prompt file not found: ${prompt_file}" >&2
    exit 1
  fi
  echo "=== Running Keras fusion ablation for tier ${tier_id}: ${prompt_file} ==="
  cmd=(
    bash scripts/run_tallyqa_keras_fusion_ablation_tier0.sh
    --tier-file "${prompt_file}"
    --tier-label "tier${tier_id}"
    "${FORWARD_ARGS[@]}"
  )
  if [[ "${SKIP_COMPLETED}" == "1" || "${SKIP_COMPLETED}" == "true" ]]; then
    cmd+=(--skip-completed)
  fi
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
done
