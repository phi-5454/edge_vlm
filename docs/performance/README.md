# Performance Reports

Keep raw logs under `artifacts/profiles/` and summarize conclusions here.

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
