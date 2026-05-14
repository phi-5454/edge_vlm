# Project Scaffold

Date: 2026-05-14

## Context

The project targets SmolVLM deployment on a CNN-accelerated microcontroller
platform. Design changes must be tied to measurements, because model quality,
latency, memory, and deployment constraints will trade off against one another.

## Decision

Use `uv` for runtime management, Hydra for configuration, Lightning for training
structure, W&B for experiment tracking, JSONL files for low-friction profiling
logs, and markdown decision records for the reasoning behind architecture and
optimization choices.

## Evidence

- Training scaffold: `src/training/`
- Profiling scaffold: `src/profiling/`
- Default Hydra config: `conf/config.yaml`

## Consequences

Every experiment should produce both machine-readable metrics and a short human
record when it informs a design choice. This adds small documentation overhead
but prevents performance claims from becoming disconnected from the code.
