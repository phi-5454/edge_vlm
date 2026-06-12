# Coral TallyQA On-Device Benchmark

This benchmark path is separate from the visual demo. The visual demo is for
interactive viewing; this app is for deterministic dataset sweeps with latency
and accuracy artifacts.

## Components

- Board app: `coral_micro/tallyqa_benchmark_serial/`
- Stage app/model into the Coral SDK:
  `scripts/coral_micro_stage_tallyqa_benchmark_app.py`
- Host-side dataset sweep and cache writer:
  `scripts/cache_coral_micro_tallyqa_teacher.py`
- One-command dummy pipeline:
  `scripts/run_coral_tallyqa_on_device_dummy_pipeline.sh`
- Coral cache EDA wrapper:
  `scripts/run_coral_micro_tallyqa_cache_eda.py`
- Standard example visualization:
  `scripts/visualize_tallyqa_teacher_logits.py`

## One-Command Dummy Pipeline

This defaults to the untrained packed prompt-patch-MLP EdgeTPU probe:

```bash
bash scripts/run_coral_tallyqa_on_device_dummy_pipeline.sh \
  --port /dev/ttyACM0 \
  --max-examples 128 \
  --force
```

For command assembly only:

```bash
bash scripts/run_coral_tallyqa_on_device_dummy_pipeline.sh \
  --dry-run \
  --skip-build \
  --skip-flash \
  --skip-cache \
  --skip-eda \
  --force
```

Useful partial runs:

```bash
# Stage only.
bash scripts/run_coral_tallyqa_on_device_dummy_pipeline.sh \
  --skip-build --skip-flash --skip-cache --skip-eda --force

# Board already flashed: sweep + EDA only.
bash scripts/run_coral_tallyqa_on_device_dummy_pipeline.sh \
  --skip-stage --skip-build --skip-flash \
  --port /dev/ttyACM0 \
  --max-examples 128 \
  --force
```

## Serial Protocol

The host sends one image at a time. For two-input models with an image tensor
and a prompt-embedding tensor, the host also sends a `prompt_id`. The image
bytes are still the only binary payload; the board fills the prompt tensor from
the staged quantized lookup table.

```text
VLM_MICRO_INPUT {"dataset_index":123,"image_index":45,"prompt_id":7,"bytes":150528}\n
<150528 raw NHWC uint8 bytes>
```

The board replies with:

```text
VLM_MICRO_RESULT {"dataset_index":123,...,"invoke_us":...,"outputs":[...]}
```

The board also prints:

```text
VLM_MICRO_READY {...}
VLM_MICRO_ERROR {...}
```

All benchmark tooling filters these prefixes so normal boot logs can remain in
the raw serial capture.

`VLM_MICRO_READY` includes memory and tensor diagnostics:

- `tensor_arena_bytes`
- `arena_used_bytes`
- `arena_recorded_used_bytes`
- `arena_recorded_requested_bytes`
- `arena_recorded_alloc_count`
- `image_input_index`
- `prompt_input_index`; `-1` means image-only/static-prompt model
- `prompt_lookup_count`
- `prompt_lookup_dim`
- `inputs`
- `outputs`
- `recorded_allocations`

The host cache script copies these into the cache manifest under
`board_memory`.

## Stage, Build, Flash

From this repository:

```bash
uv run python scripts/coral_micro_stage_tallyqa_benchmark_app.py \
  --coralmicro ../coralmicro \
  --model artifacts/reports/coral/edgetpu_compiler/prompt_patch_mlp_static_prompt_minimalistic_large_compile_probe_docker/ptq/model_int8_edgetpu.tflite \
  --prompt-lookup-header artifacts/exports/coral/prompt_embedding_lookup/tallyqa_prompt_embedding_lookup.h \
  --force
```

Then from the Coral SDK root:

```bash
cd ../coralmicro
bash build.sh
if [[ -e build/examples/vlm_micro_tallyqa_benchmark_serial/vlm_micro_tallyqa_benchmark_serial ]]; then
  .venv/bin/python scripts/flashtool.py -e vlm_micro_tallyqa_benchmark_serial
else
  .venv/bin/python scripts/flashtool.py \
    --elf_path build/examples/vlm_micro_tallyqa_benchmark_serial/vlm_micro_tallyqa_benchmark_serial.stripped \
    --data_dir build/examples/vlm_micro_tallyqa_benchmark_serial
fi
```

## Dataset Sweep

Back in this repository, run a smoke sweep:

```bash
uv run python scripts/cache_coral_micro_tallyqa_teacher.py \
  --port /dev/ttyACM0 \
  --dataset data/tallyqa_cauldron_target_mobilenet224_letterbox \
  --prompt-lookup-manifest artifacts/exports/coral/prompt_embedding_lookup/prompt_embedding_lookup_manifest.json \
  --output artifacts/teacher_cache/coral_micro_tallyqa_prompt_patch_mlp_smoke128.jsonl \
  --model-name coral_micro_prompt_patch_mlp_large_edgetpu \
  --max-examples 128 \
  --force \
  --raw-log artifacts/profiles/coral/tallyqa_prompt_patch_mlp_smoke128_serial.log
```

The script writes:

- `artifacts/teacher_cache/*.jsonl`: standard TallyQA teacher-cache records
- `artifacts/teacher_cache/*.manifest.json`: accuracy, confusion, latency
- `artifacts/profiles/coral/*_serial.log`: raw board serial output

## Example Figure

The resulting cache can be visualized with the standard teacher-cache figure:

```bash
uv run python scripts/visualize_tallyqa_teacher_logits.py \
  --dataset data/tallyqa_cauldron_target_mobilenet224_letterbox \
  --cache artifacts/teacher_cache/coral_micro_tallyqa_prompt_patch_mlp_smoke128.jsonl \
  --output artifacts/reports/coral/on_device_benchmark/tallyqa_prompt_patch_mlp_smoke128/example_predictions.png \
  --count 12 \
  --cols 3 \
  --answer-max 5 \
  --collapse-at 5
```

The same cache can be passed to the existing teacher EDA scripts for confusion
matrices, prompt-class accuracy plots, and comparisons against local TFLite or
reference detector caches.

Or run the Coral cache EDA wrapper:

```bash
uv run python scripts/run_coral_micro_tallyqa_cache_eda.py \
  --cache artifacts/teacher_cache/coral_micro_tallyqa_prompt_patch_mlp_smoke128.jsonl \
  --dataset data/tallyqa_cauldron_target_mobilenet224_letterbox \
  --output-dir artifacts/reports/coral/on_device_benchmark/tallyqa_prompt_patch_mlp_smoke128 \
  --title coral_micro_prompt_patch_mlp_smoke128 \
  --answer-max 5 \
  --collapse-at 5
```

This writes standard accuracy plots, confusion matrix, example prediction grid,
and a latency summary plot when the cache manifest contains board timing.

## Future Reference Detectors

The board app emits generic output tensors rather than only count logits. The
current host cache script interprets one output tensor as count-class logits.
For SSD/FOMO/reference detectors, add a host-side parser that converts the
emitted output tensors into count candidates while preserving the same JSONL
cache schema and board timing fields.
