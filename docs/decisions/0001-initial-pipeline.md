# 0001 Initial Pipeline Split

Date: 2026-05-18

## Status

Accepted

## Decision

Use a shared PyTorch/Lightning/Hydra/W&B training layer, then branch into two
explicit deployment stacks:

- Coral Micro: rebuild or translate the deployable student into Keras, then use
  TensorFlow Lite integer quantization and Edge TPU compilation.
- MAX78000: keep a hardware-shaped PyTorch model compatible with ADI `ai8x.py`,
  QAT/PTQ, YAML synthesis, and generated C.

## Rationale

The boards reward different graph shapes and tooling. A single "export anywhere"
artifact would hide the exact places where accuracy, size, and hardware mapping
change. Separate deployment branches make precision loss and unsupported
operation fallback visible.

## Consequences

The central model code must stay small and portable, and the project needs
architecture parity checks between PyTorch and Keras for Coral. MAX78000 models
need early constraints from `ai8x-training` rather than late conversion.
