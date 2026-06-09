# 0006 Coral Tflite Detection Bringup

Date: 2026-06-09

## Status

Proposed

## Context

The repository now contains an Edge TPU-compiled TFLite object detector at
`artifacts/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite`. Before
training board-shaped VLM students, we need a generic TFLite-on-board testing
path that starts from a compiled model artifact, runs camera inference, and
captures normalized host-side outputs.

## Decision

Use a serial-first Coral Micro app for bring-up. The app captures frames from
the onboard camera, invokes the Edge TPU detector, and prints newline-delimited
JSON records with a stable `VLM_MICRO_DETECTION` prefix.

Keep the app source in this repository under `coral_micro/`, then stage it into
the adjacent `../coralmicro` SDK with a script. Host capture writes both the raw
serial log and normalized JSONL/report files under `artifacts/profiles/coral/`.

## Evidence

- Model artifact:
  `artifacts/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite`
- App source: `coral_micro/detect_objects_serial/`
- Staging script: `scripts/coral_micro_stage_detection_app.py`
- Host capture script: `scripts/capture_coral_detection_serial.py`

## Consequences

This path validates the board, camera, Edge TPU runtime, tensor arena sizing,
and host logging independently of the training pipeline. It reports detections
and latency, but not dataset mAP; mAP requires a separate image/annotation
replay path rather than live camera frames.
