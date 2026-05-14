# edge-vlm

Scaffold for training, profiling, and deploying SmolVLM on a CNN-accelerated
microcontroller platform. The project emphasizes reproducible measurements and
explicit reasoning for design choices.

## Setup

```bash
uv sync --extra dev
```

W&B authentication is loaded from `../wandb_api_key.env` by default. Keep that
file outside version control and define `WANDB_API_KEY=...` inside it.

## Train

The default config fine-tunes SmolVLM against The Cauldron with Lightning and
logs to W&B. Runtime configuration is handled by Hydra from `conf/config.yaml`.

```bash
uv run edge-vlm command=train
```

Use Hydra overrides for local changes:

```bash
uv run edge-vlm command=train train.tracking.offline=true train.data.max_samples=16
```

## Profile

```bash
uv run edge-vlm command=profile profile.steps=20
```

Use `docs/performance/` for benchmark summaries and keep raw JSONL outputs under
`artifacts/profiles/`.

## Record Design Reasoning

```bash
uv run edge-vlm command=decision decision.slug=quantization-baseline
```

Decision records live in `docs/decisions/` and should reference relevant W&B
runs, profiling logs, and benchmark summaries.
