# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 project for SmolVLM training, profiling, and deployment
research on CNN-accelerated microcontrollers. Source modules live directly under
`src/`, Hydra configuration lives in `conf/`, tests live in `tests/`, and design
records live in `docs/decisions/`. Keep raw profiling artifacts under
`artifacts/profiles/` and summarize meaningful results in `docs/performance/`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: install runtime and development dependencies.
- `uv run edge-vlm command=train`: run the Lightning/W&B training scaffold using
  `conf/config.yaml`.
- `uv run edge-vlm command=profile profile.steps=20`: run a local inference
  profiling pass.
- `uv run edge-vlm command=decision decision.slug=quantization-baseline`: create
  a design decision note.
- `uv run --extra dev python -m pytest`: run tests with the managed interpreter.
- `uv run --extra dev ruff check .`: run lint checks.

W&B credentials are loaded from `../wandb_api_key.env` through
`train.tracking.env_file`. Never copy API keys into tracked files or logs.

## Coding Style & Naming Conventions

Use Python 3.12 features where they simplify code without reducing clarity.
Follow PEP 8, use 4-space indentation, and keep functions small and explicit.
Name modules and packages with lowercase snake_case, functions and variables
with snake_case, classes with PascalCase, and constants with UPPER_SNAKE_CASE.

Prefer type hints for public functions and module boundaries. Keep side effects
inside CLI or training entry points so modules remain easy to test.

## Testing Guidelines

Use `pytest` for tests. Place test files under `tests/` and name them
`test_<module>.py`. Test functions should describe behavior, for example
`test_train_config_from_hydra`.

When adding new behavior, include tests for the normal path and at least one
relevant edge case. For code that interacts with files, models, or external
services, prefer fixtures and mocks over network-dependent tests.

## Commit & Pull Request Guidelines

This repository has no commit history yet, so there is no established local
commit convention. Use concise, imperative commit messages such as
`Add profiling scaffold` or `Document Cauldron training config`.

Pull requests should include a short summary, the commands used to verify the
change, and any relevant issue links. For user-facing behavior, include sample
output or screenshots when helpful. Keep PRs focused on one logical change.

## Agent-Specific Instructions

Before editing, inspect existing files and preserve user changes. Any design or
performance claim should reference a W&B run, profiling JSONL, benchmark summary,
or decision record. Update this guide when project structure, tooling, or
commands change.
