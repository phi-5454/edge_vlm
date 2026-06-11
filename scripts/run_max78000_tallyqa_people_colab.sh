#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Warning: run_max78000_tallyqa_people_colab.sh is deprecated; use run_max78000_tallyqa_colab.sh." >&2
exec "${script_dir}/run_max78000_tallyqa_colab.sh" "$@"
