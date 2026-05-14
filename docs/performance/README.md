# Performance Tracking

Use this directory for benchmark summaries and links to raw artifacts.

## Metrics To Track

- Host training: loss, throughput, GPU/CPU memory, samples per second.
- Host inference: latency distribution, generated tokens per second, peak memory.
- Target inference: end-to-end latency, accelerator utilization, SRAM/flash use,
  energy per inference when available.
- Model quality: task-level validation metrics and qualitative failure cases.

## Artifact Convention

Raw profiling output should be append-only JSONL under `artifacts/profiles/`.
Summaries that influence design choices should be copied here as markdown tables
and referenced from `docs/decisions/`.

Suggested filename pattern:

`YYYY-MM-DD_<platform>_<model>_<change>.md`
