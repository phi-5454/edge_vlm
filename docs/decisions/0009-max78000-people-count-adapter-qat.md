# 0009. MAX78000 People-Count Adapter and QAT-First Training

Date: 2026-06-09

## Status

Accepted

## Context

The first MAX78000 demo target is a prompt-free people-count classifier derived
from the TallyQA target dataset. The model predicts five positive-count classes:
`1`, `2`, `3`, `4`, and `5+`. Since this is intended for MAX78000 synthesis, the
training path should use ADI's native `ai8x-training` dataset/model discovery and
QAT policy rather than a generic PyTorch export path.

## Decision

Keep the canonical source files in this repository and stage them into the
adjacent ADI checkout when training:

- `max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py`
- `max78000/ai8x_training/datasets/tallyqa_people.py`
- `max78000/ai8x_training/policies/qat_policy_tallyqa_people.yaml`
- `max78000/ai8x_training/policies/schedule-tallyqa-people.yaml`

Materialize a MAX78000-specific manifest at
`data/max78000_tallyqa_people_count_224/` before training. The materialized view
filters TallyQA to `people`, drops zero-count examples because the current head
has no zero class, collapses answers above five into `5+`, and records the
deterministic image-hash split.

Use QAT from the start of this training route:

- ADI QAT policy: start at epoch 10, 8-bit weights, `shift_quantile: 0.995`.
- LR schedule: 50 epoch starter schedule with milestones at 20, 35, and 45.

## Evidence

Materialization command:

```bash
uv run python scripts/materialize_max78000_people_dataset.py --force
```

Generated split counts:

```text
train: 19620
val: 2744
test: 5816
```

Generated label counts:

```text
1: 5257
2: 9656
3: 4823
4: 2877
5+: 5567
```

Adapter smoke test:

```text
train records: 19620
sample tensor: (3, 224, 224)
sample normalized range: [-128.0, 126.0]
```

Static checks:

```bash
uv run --extra dev ruff check \
  max78000/ai8x_training/datasets/tallyqa_people.py \
  scripts/materialize_max78000_people_dataset.py \
  scripts/stage_max78000_people_pipeline.py \
  pyproject.toml
```

## Consequences

The ADI training environment does not need to parse the full Cauldron/TallyQA
pipeline for ordinary training. It only consumes a manifest, source image memmap,
and metadata. Changing the class contract, such as adding a zero-count class,
requires regenerating the materialized manifest and changing the model head.
