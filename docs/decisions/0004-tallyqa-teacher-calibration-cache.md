# TallyQA Teacher Calibration Cache

Date: 2026-06-04

## Context

Full TallyQA teacher caching is expensive for larger SmolVLM variants. Before
running a full 2.2B cache, we need a smaller set that can test whether the
larger teacher materially improves accuracy over the cached 256M teacher.

## Decision

Add a teacher-cache `calibration` run mode that selects a deterministic balanced
subset across student prompt classes and collapsed output classes. The cache
schema remains identical to the full cache, so downstream accuracy plots and
student distillation loading continue to work.

Default calibration settings:

- `calibration_examples = 4096`
- `calibration_seed = 20260604`
- `calibration_collapse_at = 5`

Add a paired cache comparison script that intersects records by `dataset_index`
and reports overall, by-prompt, and by-output accuracy deltas, including an
approximate continuity-corrected McNemar statistic for the paired overall
comparison.

## Rationale

A contiguous `--max-examples` subset would overrepresent the dataset head. The
balanced calibration subset better covers the prompt and count classes that
matter for deciding whether the larger teacher is worth the full cache cost.

## Consequences

The calibration set is deterministic and reproducible from the manifest
selection metadata. Comparisons against the existing 256M full cache are made on
the intersecting calibration indices, avoiding unpaired accuracy comparisons.
