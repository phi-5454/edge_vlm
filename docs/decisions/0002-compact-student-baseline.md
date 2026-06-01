# 0002 Compact Student Baseline

Date: 2026-06-02

## Status

Proposed

## Context

The first distilled student needs a reproducible reference point before
board-specific graph constraints are applied. CPU image loading is expected to
be a material training bottleneck on many-core GPU hosts.

## Decision

Train a binary yes/no baseline from compact prompts and padded 512x512 image
parquet data. Prompts use the teacher tokenizer IDs after the final boilerplate
line is removed. Copy only the teacher embedding rows used by retained prompts,
plus one padding row.

Use MobileNet-V3-small pooled image features, 128-dimensional query and image
projections, and a small MobileViT-like local-convolution/global-attention fusion
stack. Split 70/10/20 by hashed image identity to prevent prompt-level leakage.

The default distillation objective is:

`(1 - alpha) * MSE(student_logit, teacher_yes_minus_no_logit) + alpha * BCE(student_logit, hard_label)`

An optional temperature-scaled soft BCE mode is exposed for comparison.

AdamW uses a per-step linear learning-rate warmup from `0.0001` to `0.001`
over the first 1,000 optimizer steps, then remains at `0.001`.

Interrupted teacher caches are supported. By default, training filters to the
cache-covered prompts for every alpha value, including the hard-label-only
baseline, so sweep results remain comparable. A hard-label-only run can opt into
all prompts with `data.missing_teacher_policy=keep`. An invalid trailing JSONL
record is ignored as an interrupted write; malformed records elsewhere fail.

## Evidence

- Dataset: `data/the_cauldron_yes_no_vsr_token1000_img512_parquet`
- Training entrypoint: `scripts/train_student_baseline.py`
- Alpha sweep: `scripts/run_student_baseline_alpha_sweep.sh`
- Architecture reports: `artifacts/reports/student_baseline/`
- Local resource smoke report: `artifacts/reports/student_baseline/local_resource_smoke.json`
- W&B project: `vlm-micro`

## Consequences

The baseline is intentionally not yet board-shaped. The parquet loader uses
worker-local row-group and decoded-image LRU caches, prefetching, persistent
workers, optional pinned memory, and image-grouped training batches to reuse
decoded tensors across prompts. Block-level shuffling preserves enough parquet
locality to avoid globally random row-group reads. Architecture reports and
checkpoints remain separate for each alpha value.

The generated image parquet has roughly 354 MB row groups. Each worker keeps one
row group and its own decoded-image cache, while each prefetched 224x224 float32
batch consumes additional memory. The default profile remains tuned for a larger
training host. For a 16 GB CPU host, use the `student_baseline_local` profile:
no worker processes, batch size eight, an eight-image decoded tensor cache, and
no pinned memory. This was sized for the observed local machine: four physical
cores / eight threads, 16 GB RAM, and no NVIDIA GPU. Local runs use a distinct
run name and W&B tag. Run the local sweep with:

`RUN_MODE=local scripts/run_student_baseline_alpha_sweep.sh`

Worker count, batch size, cache sizes, prefetching, and pinned memory remain
explicit tuning parameters for measured host-specific optimization.
