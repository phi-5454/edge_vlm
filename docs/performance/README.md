# Performance Reports

Keep raw logs under `artifacts/profiles/` and summarize conclusions here.

Runbooks:

- [Coral Detection Bringup](coral-detection-bringup.md): TFLite object detector
  from model artifact to Coral camera inference and serial capture.
- [Keras TFLite Student Pipeline](keras-tflite-student.md): Keras distillation
  path with explicit PTQ versus QAT comparison.

Minimum comparison table for each serious experiment:

| Stage | Accuracy | Size | Latency | Energy/Power | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32 | | | | | |
| PyTorch simulated quant | | | | | |
| Coral TFLite int8 | | | | | |
| Coral Edge TPU on board | | | | | |
| MAX78000 simulated | | | | | |
| MAX78000 on board | | | | | |

Always include the artifact hash, toolchain commit/version, and exact command
used to generate each row.
