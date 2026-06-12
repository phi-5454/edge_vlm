# 0011. Faithful Coral QAT Prompt Input Contract

Date: 2026-06-12

## Status

Accepted

## Context

Keras QAT runs using prompt token IDs kept prompt embedding lookup, mask-aware
pooling, prompt LayerNorm, and prompt tensor routing inside the model graph.
Full-integer TFLite quantizes those runtime pieces even though TensorFlow Model
Optimization does not fake-quantize several of them. The resulting Keras QAT
simulation and deployed TFLite graph can diverge before any board-specific
effects.

## Decision

Support a deployment-faithful Keras path with a two-input model contract:

- `images`: MobileNet-preprocessed image tensor.
- `prompt_embedding`: pre-pooled raw prompt embedding vector.

The prompt lookup and pooling happen outside the model. The same prompt vector is
then calibrated and quantized as a real TFLite input. For this path, prompt
projection LayerNorm is disabled by default so the strict QAT preflight does not
accept a graph that trains through float normalization and deploys an int8
lowered variant.

Use `--raw-prompt-embedding` in the tier-0 Keras wrapper to select this contract.
Existing token-ID runs remain available for comparison.

## Evidence

Static checks after adding the raw prompt input path:

```bash
uv run python -m py_compile scripts/train_tallyqa_keras_student.py
bash -n scripts/run_tallyqa_keras_fusion_ablation_tier0.sh
```

The Keras training script also exports a smoke report comparing the QAT-simulated
Keras graph against the exported TFLite graph on held-out batches:

```text
artifacts/reports/tallyqa_keras_student/<run>/tflite_smoke_compare/<mode>_simulated_vs_tflite.json
```

## Consequences

Raw-prompt runs are not weight-compatible with old token-ID checkpoints for the
prompt lookup/pooling part of the graph. Fine-tuning should start from a
checkpoint trained with the same `raw_embedding` input contract, or load by name
with skipped mismatches only as a diagnostic.

The QAT-vs-TFLite smoke comparison is the acceptance gate before spending time
on long training or board tests.
