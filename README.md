# VLM on Microcontrollers

Central workspace for a compact VLM development pipeline targeting:

- Google Coral Dev Board Micro via Keras/TensorFlow Lite/TFLM/Edge TPU.
- MAX78000FTHR via ADI `ai8x-training` and `ai8x-synthesis`.

The repository is intentionally the experiment/control layer, not a clone of the
board SDKs. The adjacent SDKs stay in sibling directories and are referenced by
configuration.

## Initial Setup

```bash
uv sync --extra dev
```

Use W&B through an env file outside the repo:

```bash
printf 'WANDB_API_KEY=...\n' > ../wandb_api_key.env
```

## Current Layout

- `conf/`: Hydra configuration for common training and board-specific pipelines.
- `src/vlm_micro/`: small project utilities and board adapter placeholders.
- `docs/decisions/`: design records.
- `docs/performance/`: benchmark summaries and board comparison notes.
- `docs/references/`: local notes from SDK and official documentation.
- `artifacts/`: generated models, exports, reports, profiles, and runs.

## Intended Flow

1. Train and evaluate the central PyTorch model with Lightning, Hydra, and W&B.
2. Export a traceable baseline artifact with architecture, metric, size, and hash
   metadata.
3. Coral path: translate/rebuild the deployable model in Keras, run TF/QAT or
   representative-dataset PTQ, convert to full-integer TFLite, compile with
   `edgetpu_compiler`, then run through a `coralmicro` app.
4. MAX78000 path: keep the deployable model in the ADI PyTorch layer subset,
   run QAT/PTQ through `ai8x-training` / `ai8x-synthesis`, generate C with
   `ai8xize.py`, then flash/profile with the MSDK flow.
5. Record simulated and real measurements in normalized reports before comparing
   board results.

## First Useful Commands

Create a design record:

```bash
uv run vlm-micro record-decision decision.slug=initial-pipeline
```

Summarize exported artifacts:

```bash
uv run vlm-micro artifact-report artifacts/models/example.pt artifacts/exports/example.tflite
```

## Main Warnings

Coral and MAX78000 are not interchangeable deployment backends. A model that is
easy to train in PyTorch may be awkward or impossible on both boards. Keep a
small hardware-shaped student model separate from any large teacher model.

The Coral path should be validated with Edge TPU compiler logs early. The
compiler can map unsupported tails to CPU, and Dev Board Micro CPU fallback is
limited by TFLM support and memory.

The MAX78000 path should be validated with ADI-compatible layers early. Memory,
processor allocation, output shifts, and YAML/checkpoint consistency are central
parts of the model design, not late packaging details.
