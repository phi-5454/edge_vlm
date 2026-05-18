# Repository Guidelines

## Project Goal

This repository is the central control plane for developing, training,
deploying, profiling, and analyzing compact VLM-style models on two targets:

- Google Coral Dev Board Micro, using TensorFlow Lite Micro and, where possible,
  a fully quantized Edge TPU compiled model.
- MAX78000FTHR / MAX78000-class boards, using ADI's PyTorch-native
  `ai8x-training` and `ai8x-synthesis` flow.

The model may start from a shared PyTorch/Lightning implementation, but every
deployment step must preserve enough metadata to answer: what changed, where was
precision lost, how much did the model shrink, which ops moved to hardware, and
what happened to accuracy, latency, memory, and energy.

## Adjacent Toolchains

Expected sibling directories:

- `../coralmicro`: Coral Dev Board Micro SDK and examples.
- `../MAX78000/ai8x-training`: ADI training flow.
- `../MAX78000/ai8x-synthesis`: ADI synthesis / `ai8xize.py` flow.
- `../ai8x-training`: alternate top-level clone; prefer the `../MAX78000/*`
  pair unless the user says otherwise.

Do not vendor these SDKs into this repository. Treat them as external toolchains
and record the commit/version used in each experiment report.

## Required Experiment Discipline

Every non-trivial architecture, quantization, deployment, or profiling change
must leave a trace in one of:

- W&B run metadata.
- A JSON/JSONL artifact under `artifacts/`.
- A decision record under `docs/decisions/`.
- A performance summary under `docs/performance/`.

Do not make unsupported performance claims. Tie claims to a W&B run, a generated
report, a serial log, compiler output, or a board measurement.

## Development Commands

- `uv sync --extra dev`: install Python dependencies.
- `uv run vlm-micro record-decision decision.slug=<slug>`: create a decision
  record.
- `uv run vlm-micro artifact-report artifacts/models/model.pt`: summarize file
  sizes and hashes for exported artifacts.
- `uv run --extra dev pytest`: run tests.
- `uv run --extra dev ruff check .`: run lint checks.

W&B credentials should live outside the repo, for example in
`../wandb_api_key.env`. Never commit API keys, board serial logs containing
secrets, or large raw datasets.

## Board-Specific Warnings

For Coral, keep the deployable graph Edge-TPU-friendly from the beginning:
fully quantized 8-bit TFLite, static tensor shapes, constant parameters, and
supported ops. Unsupported ops cause CPU fallback; on Dev Board Micro that
fallback must also be supported by TensorFlow Lite Micro and fit memory.

For MAX78000, design with hardware layout constraints in mind. The ADI flow is
not a generic ONNX/TFLite backend. Prefer `ai8x.py` layer patterns, maintain the
synthesis YAML alongside checkpoints, and verify with simulated `-8` evaluation
before expecting generated C to work on hardware.

## Coding Style

Use Python 3.11 to stay close to the MAX78000 documented environment. Keep board
integration code isolated under `src/vlm_micro/boards/`. Shared model/training
code must not import board SDKs directly.

Prefer structured config and structured reports over ad hoc shell output. If a
script parses compiler or serial output, retain the raw log and emit a normalized
JSON summary.
