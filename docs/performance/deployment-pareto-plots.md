# Deployment Pareto Plots

This report path prepares three cross-model deployment plots:

| Panel | X | Y | Bubble |
| --- | --- | --- | --- |
| A | latency per inference, ms | accuracy | deployed model size |
| B | runtime RAM / tensor arena bytes | accuracy | deployed model size |
| C | energy per inference, mJ | accuracy | latency |

Use one accuracy metric consistently across all rows. For TallyQA runs, prefer
`prompt_class_output_weighted_accuracy` when available because it balances both
prompt classes and count-output classes.

## Input Table

Start from:

```bash
artifacts/reports/deployment_pareto/model_metrics.template.csv
```

Fill a measured table such as:

```bash
artifacts/reports/deployment_pareto/model_metrics.csv
```

Columns:

| Column | Meaning |
| --- | --- |
| `model_id` | Stable machine name. |
| `display_name` | Short label shown in plots. |
| `target` | One of `pytorch`, `keras`, `tflite`, `coral`, `max78000`, `teacher`, `baseline`, or similar. |
| `family` | Color grouping, for example `student`, `vlm`, `detector`, `coral`, `max78000`, `baseline`. |
| `variant` | Float/PTQ/QAT/int8/device/etc. |
| `accuracy` | Numeric accuracy to plot. |
| `accuracy_metric` | Name of the metric used, for example `prompt_class_output_weighted_accuracy`. |
| `latency_ms` | Per-inference latency in ms. Exclude data loading unless explicitly comparing full pipelines. |
| `model_size_bytes` | Deployed artifact byte size. |
| `runtime_ram_bytes` | Peak runtime RAM, tensor arena used bytes, or activation memory. Record the source. |
| `energy_mj` | Energy per inference. Prefer measured energy. Leave blank until measured. |
| `power_mw` | Average measured power during inference window, if available. |
| `*_source` | File, W&B run, compiler report, or serial log backing the number. |
| `notes` | Any caveats, such as CPU fallback or synthetic/example values. |

## Known Sources

Keras/PyTorch accuracy:

- `artifacts/reports/tallyqa_keras_student/<run>/results.json`
- `artifacts/reports/tallyqa_student/<run>_results.json`
- W&B run summaries for the same keys.

TFLite/PTQ accuracy:

- `test_quantized/*` metrics in the Keras W&B run.
- `artifacts/reports/tallyqa_keras_student/<run>/results.json` when quantized
  testing was enabled.

Coral latency/RAM:

- `artifacts/teacher_cache/<coral-cache>.manifest.json`
- `artifacts/reports/coral/on_device_benchmark/<run>/tables/latency_records.csv`
- `artifacts/reports/coral/on_device_benchmark/<run>/figures/latency_histograms.png`
- Board-ready fields: `arena_used_bytes`, `arena_recorded_used_bytes`,
  `tensor_arena_bytes`.

Coral model size:

- Edge TPU model file, usually:
  `artifacts/reports/coral/edgetpu_compiler/<run>/ptq/model_int8_edgetpu.tflite`

MAX78000 profile:

- `scripts/summarize_max78000_profile.py` normalized summaries.
- `artifacts/reports/max78000/...`
- ADI logs from `ai8xize.py` for hardware ops/MACCs, memory, cycles, and
  optional energy.

Power/energy:

- Prefer measured energy per inference from a board power harness.
- If only average power is available:
  `energy_mj = power_mw * latency_ms / 1000`.
- Record the measurement setup in `energy_source` or a linked performance note.

## Generate Plots

```bash
uv run python scripts/plot_deployment_pareto.py \
  --input artifacts/reports/deployment_pareto/model_metrics.csv \
  --output-dir artifacts/reports/deployment_pareto \
  --combined
```

Outputs:

- `figures/a_latency_accuracy_model_size.png`
- `figures/b_ram_accuracy_model_size.png`
- `figures/c_energy_accuracy_latency.png`
- `figures/abc_deployment_pareto.png`
- `model_metrics.normalized.csv`
- `manifest.json`

If `energy_mj` is blank for all rows, panel C is written with a "No complete
rows" placeholder. This is intentional, so the report structure is stable before
power measurements exist.
