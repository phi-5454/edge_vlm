# 0007 Keras Tflite Prior Student

Date: 2026-06-09

## Status

Proposed

## Context

The PyTorch TallyQA student is useful for distillation experiments, but its
MobileNetV3 plus transformer-style fusion path contains patterns that are risky
or unsupported for a conservative TFLite/Edge TPU deployment path.

We need a parallel Keras pipeline that keeps the same dataset splits, prompt
artifacts, teacher-cache distillation objective, and W&B logging style while
making quantization mode an explicit experimental variable.

## Decision

Add `scripts/train_tallyqa_keras_student.py` and
`conf/tallyqa_keras_student.yaml`.

The Keras student uses a TFLite-prior architecture: compact prompt embedding,
masked prompt mean, Conv2D/DepthwiseConv2D image tower, global average pooling,
concat fusion, and Dense logits. It intentionally avoids the PyTorch
transformer fusion, LayerNorm, GELU, and spatial-token/positional-embedding
path.

The Keras deployable model does not expose `attention_mask` as an input. Prompt
padding is handled by `Embedding(mask_zero=True)` and mask-aware pooling.

Quantization is controlled by `export.quantization.mode`:

- `none`: train/export float TFLite only.
- `ptq`: train float Keras, then representative-dataset post-training
  quantization.
- `qat`: wrap the Keras student with TensorFlow Model Optimization before
  training, then export a quantized TFLite artifact.

QAT is selective: only float Conv2D, DepthwiseConv2D, and Dense layers are
annotated. The integer prompt token input and embedding/gather path are left
unannotated because fake-quantizing integer token IDs is invalid.

## Evidence

- Training script: `scripts/train_tallyqa_keras_student.py`
- Config: `conf/tallyqa_keras_student.yaml`
- Architecture/compatibility report:
  `artifacts/reports/tallyqa_keras_student/*_architecture.json`
- Result report: `artifacts/reports/tallyqa_keras_student/*_results.json`
- Exports: `artifacts/exports/coral/keras_tallyqa_*.tflite`

## Consequences

The Keras path is not intended to be weight-compatible with the current PyTorch
checkpoint. It is a deployment-shaped student for comparison and later board
export.

QAT depends on `tensorflow-model-optimization` compatibility with the local
TensorFlow/Keras environment. The PTQ path has fewer dependency risks and should
be the first baseline.
