# Keras TFLite Student Pipeline

This runbook mirrors the TallyQA student distillation objective in a
Keras/TFLite-prior pipeline.

## Architecture

The Keras model is intentionally not a line-by-line port of the PyTorch student.
It keeps the same inputs and distillation targets, but replaces risky deployment
patterns with conservative TFLite-friendly blocks.

| PyTorch student component | Keras/TFLite-prior handling |
| --- | --- |
| MobileNetV3 feature stack | Small Conv2D/DepthwiseConv2D image tower. |
| GELU | ReLU. |
| LayerNorm | Avoided; BatchNorm is used in conv blocks and folded by conversion. |
| Transformer fusion | Prompt/image concat plus Dense fusion. |
| Spatial image tokens + positional embeddings | GlobalAveragePooling2D image vector. |
| Runtime prompt token embedding | Kept for parity, but not QAT-annotated. Future deployment should freeze prompts or precompute prompt vectors. |
| Attention mask input | Removed from the Keras deployable model. Padding is handled with `Embedding(mask_zero=True)` and mask-aware pooling. |

The script writes these substitutions into the architecture report under
`pytorch_student_steps_not_mirrored_for_tflite`.

## PTQ Baseline

```bash
uv run --extra coral python scripts/train_tallyqa_keras_student.py \
  export.quantization.mode=ptq \
  experiment.run_name=keras-tallyqa-ptq
```

Outputs:

- `artifacts/reports/tallyqa_keras_student/keras-tallyqa-ptq_architecture.json`
- `artifacts/reports/tallyqa_keras_student/keras-tallyqa-ptq_results.json`
- `artifacts/exports/coral/keras_tallyqa_float.tflite`
- `artifacts/exports/coral/keras_tallyqa_quantized.tflite`

## QAT Run

Install the optional QAT extra first:

```bash
uv sync --extra coral --extra qat --extra dev
```

Then run:

```bash
uv run --extra coral --extra qat python scripts/train_tallyqa_keras_student.py \
  export.quantization.mode=qat \
  experiment.run_name=keras-tallyqa-qat
```

QAT uses TensorFlow Model Optimization with legacy `tf_keras`
(`TF_USE_LEGACY_KERAS=1`). The script selectively QAT-annotates float
`Conv2D`, `DepthwiseConv2D`, and `Dense` layers. It does not QAT-annotate the
integer prompt-token input, embedding/gather, or prompt pooling path because
`tfmot.quantize_model()` tries to fake-quantize integer inputs if the whole
model is wrapped.

If `tensorflow-model-optimization` is incompatible with the local
TensorFlow/Keras version, use PTQ as the baseline and record the QAT failure in
the run report before changing package versions.

## Fast Offline Smoke

```bash
uv run --extra coral python scripts/train_tallyqa_keras_student.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=1 \
  trainer.limit_test_batches=1 \
  data.batch_size=2 \
  data.train_example_limit=2 \
  data.tensor_cache_size=2 \
  distillation.class_weight_mode=null \
  wandb.mode=offline \
  export.export_tflite=false \
  experiment.run_name=keras-smoke-test
```

This validates the loader, Keras training loop, W&B offline logging, and reports
without exporting TFLite.

## Metrics

The Keras loop uses `tqdm` progress bars for epoch, train-batch, validation
metric, and test metric progress. W&B logs mirror the scalar/report/checkpoint
shape of `scripts/train_tallyqa_student.py` where the Keras backend exposes the
same information.

W&B scalar keys include:

- train-time step metrics every `trainer.log_every_n_steps`:
  `train/loss_step`, `train/ce_loss_step`, `train/ce_loss_unweighted_step`,
  `train/kl_loss_step`, `train/accuracy_step`, `train/mae_step`,
  `train/within_1_accuracy_step`, `train/lr`
- `train/loss`, `train/ce_loss`, `train/kl_loss`
- `train/ce_loss_unweighted`, `train/accuracy`, `train/mae`,
  `train/within_1_accuracy`
- `val/loss`, `val/ce_loss`, `val/kl_loss`
- `val/ce_loss_unweighted`, `val/accuracy`, `val/mae`,
  `val/within_1_accuracy`
- `val/class_weighted_accuracy`, `val/class_weighted_mae`,
  `val/class_weighted_within_1_accuracy`
- `test/accuracy`, `test/mae`, `test/within_1_accuracy`
- `test/class_weighted_accuracy`, `test/class_weighted_mae`,
  `test/class_weighted_within_1_accuracy`

W&B artifact files include the architecture report, result JSON, and best Keras
weights. Validation and test confusion matrices are logged as W&B images. The
Lightning-specific `wandb.watch` graph/gradient logging and validation image
activation plots are not mirrored in this Keras path.
