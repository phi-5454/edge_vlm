#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-edge-vlm-edgetpu-compiler:latest}"
REPRESENTATIVE_SAMPLES="${REPRESENTATIVE_SAMPLES:-128}"
RUNS="${RUNS:-mlp,normformer}"

usage() {
  cat <<'EOF'
Usage: scripts/run_edgetpu_skeleton_compile_probes.sh [options]

Options:
  --image NAME                  Docker image tag to build/use.
  --representative-samples N    Representative samples for int8 export.
  --runs LIST                   Comma-separated list: mlp,normformer,fusion_mlp,fusion_normformer,fusion_normformer_no_softmax.
  --no-build                    Do not build the Docker image first.
  -h, --help                    Show this help.

Environment overrides:
  IMAGE=edge-vlm-edgetpu-compiler:latest
  REPRESENTATIVE_SAMPLES=128
  RUNS=mlp,normformer
EOF
}

BUILD_IMAGE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --representative-samples)
      REPRESENTATIVE_SAMPLES="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --no-build)
      BUILD_IMAGE=0
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

if [[ "${BUILD_IMAGE}" == "1" ]]; then
  docker build -f docker/edgetpu-compiler.Dockerfile -t "${IMAGE}" .
fi

IFS=',' read -r -a RUN_LIST <<< "${RUNS}"
for run in "${RUN_LIST[@]}"; do
  case "${run}" in
    mlp)
      uv run python scripts/compile_tallyqa_keras_skeleton_edgetpu.py \
        --run-name mlp_relu_default_compile_probe \
        --fusion-mode mlp \
        --activation relu \
        --representative-samples "${REPRESENTATIVE_SAMPLES}" \
        --compiler-container-image "${IMAGE}"
      ;;
    normformer)
      uv run python scripts/compile_tallyqa_keras_skeleton_edgetpu.py \
        --run-name normformer_relu_default_compile_probe \
        --skeleton-kind current_student \
        --fusion-mode normformer \
        --attention-impl static \
        --activation relu \
        --representative-samples "${REPRESENTATIVE_SAMPLES}" \
        --compiler-container-image "${IMAGE}"
      ;;
    fusion_mlp)
      uv run python scripts/compile_tallyqa_keras_skeleton_edgetpu.py \
        --run-name fusion_only_mlp_relu_compile_probe \
        --skeleton-kind fusion_only \
        --fusion-mode mlp \
        --activation relu \
        --representative-samples "${REPRESENTATIVE_SAMPLES}" \
        --compiler-container-image "${IMAGE}"
      ;;
    fusion_normformer)
      uv run python scripts/compile_tallyqa_keras_skeleton_edgetpu.py \
        --run-name fusion_only_normformer_relu_compile_probe \
        --skeleton-kind fusion_only \
        --fusion-mode normformer \
        --attention-impl static \
        --attention-normalization softmax \
        --activation relu \
        --representative-samples "${REPRESENTATIVE_SAMPLES}" \
        --compiler-container-image "${IMAGE}"
      ;;
    fusion_normformer_no_softmax)
      uv run python scripts/compile_tallyqa_keras_skeleton_edgetpu.py \
        --run-name fusion_only_normformer_no_softmax_relu_compile_probe \
        --skeleton-kind fusion_only \
        --fusion-mode normformer \
        --attention-impl static \
        --attention-normalization none \
        --activation relu \
        --representative-samples "${REPRESENTATIVE_SAMPLES}" \
        --compiler-container-image "${IMAGE}"
      ;;
    *)
      echo "Unknown run '${run}'. Expected one of: mlp,normformer,fusion_mlp,fusion_normformer,fusion_normformer_no_softmax." >&2
      exit 2
      ;;
  esac
done
