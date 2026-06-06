# External Models

This directory is for local checkouts, downloaded checkpoints, converted
weights, and generated caches for reference counting models. Large files in this
directory are ignored by git.

Planned reference models:

- YOLO11n, using COCO detection classes and the TallyQA prompt-to-COCO mapping.
- EfficientDet-Lite, using COCO detection classes and the same mapping.
- Edge Impulse FOMO, evaluated as a small-device-oriented counting reference.
- Clip-count, used as a stronger external teacher and, later, as a source of
  density-map distillation targets.

Keep any upstream repositories or checkpoints here rather than vendoring them
into `src/` or `scripts/`.
