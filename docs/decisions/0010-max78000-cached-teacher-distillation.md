# 0010. MAX78000 Cached-Teacher Distillation and W&B Trace Parity

Date: 2026-06-12

## Status

Accepted

## Context

The MAX78000 TallyQA route now trains a prompt-conditioned count classifier from
the materialized tiered TallyQA dataset. The original PyTorch/Keras training
routes use cached teacher outputs, richer W&B traces, checkpoints, confusion
matrices, example predictions, and image-encoding visualizations. The ADI
`ai8x-training` loop expects each dataset item to return `(input, target)`, so
passing full teacher tensors requires a small compatibility layer.

## Decision

Materialize cached teacher probabilities directly into the MAX78000 manifest as
`teacher_probs`. The dataset adapter returns a packed target tensor when these
probabilities exist:

```text
[hard_label, teacher_p0, teacher_p1, teacher_p2, teacher_p3, teacher_p4, teacher_p5]
```

The wrapper stages the dataset/model files and then applies a reproducible patch
to the adjacent ADI `train.py`. The patch splits packed targets immediately
before loss/metrics:

- hard labels go to cross-entropy, accuracy, confusion, and sample helpers.
- teacher probabilities go to a KL term.
- objective is `alpha * CE + beta * KL(student / T, teacher) * T^2`.

The Colab wrapper fails early when `distillation_beta > 0` but the selected
manifest does not contain `teacher_probs`. This avoids accidentally training a
plain CE run from a stale materialized dataset.

The default MAX comparison preset runs 20 total epochs. Epochs 0 through 9 are
plain float training, and QAT starts at epoch 10 via
`qat_policy_tallyqa_count.yaml`. The 20-epoch LR schedule decays at epochs 10,
16, and 19.

The W&B wrapper now performs a post-training checkpoint evaluation and saves the
MAX analogue of the original training outputs: model report files, train log,
run manifest, best checkpoint, validation/test metrics, confusion matrices, and
unique-image examples with the `14x14` head map plus prediction bars.

The distillation patch also exposes `CE Loss`, `KL Loss`, and
`Distillation Loss` as normal ai8x meters. The W&B stdout parser therefore logs
the same loss decomposition used by the original TallyQA routes instead of only
seeing the collapsed objective loss.

The wrapper promotes the static model report to W&B panels, uploads all
available MAX checkpoint files as a checkpoint artifact, logs the selected best
checkpoint separately with the post-evaluation outputs, and records
best-checkpoint parameter histograms. It also saves/logs separate
`prediction_examples.png` plots in addition to the `14x14` image-encoding
plots for validation and test.

## Evidence

Primary command shape:

```bash
MPLBACKEND=Agg bash scripts/run_max78000_tallyqa_colab.sh \
  --torch-reverse-ablation-comparison \
  --workers 0 \
  --wandb-env-file /content/wandb_api_key.env \
  --force \
  --clone-ai8x
```

Manual distillation flags:

```bash
--teacher-cache artifacts/teacher_cache/composite_ece_temp_smol1p1_frcnn2p2_beta12p968_tallyqa_target_mobilenet224.jsonl
--distillation-alpha 1.0
--distillation-beta 0.25
--distillation-temperature 2.0
```

Static checks used during implementation:

```bash
python -m py_compile \
  scripts/materialize_max78000_tallyqa_dataset.py \
  scripts/patch_max78000_ai8x_distillation.py \
  scripts/evaluate_max78000_tallyqa_wandb_outputs.py \
  scripts/run_max78000_ai8x_with_wandb.py \
  max78000/ai8x_training/datasets/tallyqa_count.py
bash -n scripts/run_max78000_tallyqa_colab.sh
git diff --check
```

Dataset probe confirmed that a distillation-enabled sample returns input shape
`(588, 56, 56)` and a packed target shape `(7,)` whose teacher probabilities sum
to one.

## Consequences

The MAX route can consume surrogate teacher tensors without changing ADI's
dataset/loader interface. Distillation remains opt-in through `beta`; a zero beta
uses the hard-label path even if cached probabilities are present. Post-training
plots are checkpoint-based rather than callback-based, so they cover the final
chosen checkpoint and do not add per-epoch plotting overhead. Parameter
histograms are checkpoint-based as well; live gradient histograms are not
available through the unmodified ADI training loop.
