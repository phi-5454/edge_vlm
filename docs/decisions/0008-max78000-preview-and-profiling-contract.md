# 0008 MAX78000 Preview And Profiling Contract

Date: 2026-06-09

## Status

Accepted

## Context

The Coral bring-up has a browser preview path for live camera detections. The
MAX78000 path needs a similar operator view, but the ADI flow should also expose
more detailed hardware mapping information than the Coral TFLite path: hardware
MACCs, software fallback ops, processor placement, weight/bias memory use,
latency estimates, and eventually measured energy.

## Decision

Use a serial-first MAX78000 preview contract. Board firmware emits
newline-delimited JSON containing a small preview image, prediction/detection
outputs, and timing fields. The host runs
`scripts/max78000_preview_server.py`, which reads serial JSON, logs normalized
JSONL under `artifacts/profiles/max78000/`, and serves a browser preview with
boxes overlaid on the latest frame.

Use `scripts/summarize_max78000_profile.py` to combine ADI `ai8xize.py` logs,
generated project file sizes/hashes, and board serial timing into
`artifacts/profiles/max78000/report.json`.

Document the end-to-end flow in
`docs/performance/max78000-preview-and-profiling.md`, starting from
ADI-compatible training and simulated 8-bit evaluation, then quantization,
`ai8xize.py` C generation, firmware integration, flashing, host preview, and
profiling summary.

## Consequences

- The host preview can be developed before the board wrapper is complete.
- Serial preview is intentionally low-frame-rate; it is for bring-up and
  debugging, not production video transport.
- Generator estimates and board measurements stay separate in the report.
- Future firmware should add processor map and data memory allocation fields if
  they are not already recoverable from generated logs.
