# Counting Reference Models and Density Distillation

Date: 2026-06-06

## Context

SmolVLM-256M and SmolVLM2-2.2B are weak counting teachers on TallyQA. Dedicated
detection/counting models provide a more meaningful reference point for how much
performance is lost when moving toward compact deployable students.

The normalized TallyQA prompts are largely COCO-mappable: the current coverage
artifact reports `80/107` prompt classes as single COCO label or alias matches,
and `82/107` when grouped prompts such as `animals` and `vehicles` are included.
This makes COCO detectors useful as reference baselines, while leaving the
remaining prompts to open-vocabulary or counting-specific models.

External checkpoints and upstream repositories should stay out of git under
`external-models/` or the existing local `external_models/` directory.

## Decision

Use these models as the initial literature/reference baselines:

- YOLO11n for a small modern COCO detector baseline.
- EfficientDet-Lite for an efficient detector family with deployment relevance.
- Edge Impulse FOMO for a microcontroller-oriented object localization/counting
  reference.
- Clip-count as a stronger external counting teacher.

Evaluate each baseline through the same cache/report interface where possible:

- exact count accuracy,
- within-1 accuracy,
- collapsed `5+` accuracy,
- confusion matrix,
- per-prompt-class accuracy.

For COCO detectors, use an explicit prompt-to-COCO mapping and record unmapped
prompts. For grouped prompts such as `animals`, count the union of the mapped
COCO classes and keep that mapping in the cache metadata.

For future smaller students, add a second distillation path from Clip-count
density maps. The student should be evaluated both on final count quality and on
how well any intermediate spatial representation matches the teacher density
signal.

## Rationale

The detector baselines separate two questions:

- whether poor counting accuracy is caused by the dataset/modeling task itself,
- whether our compact student is losing performance relative to small but
  established detection models.

Clip-count is a better fit for supervision than VLM numeric logits because it
can provide spatial density information, not just a final count. That makes it
useful for training compact visual encoders to preserve count-relevant local
evidence as model size decreases.

## Consequences

Baseline reports must identify:

- model name and checkpoint,
- prompt mapping used,
- thresholding/post-processing settings,
- covered and skipped prompt classes,
- exact command or W&B run/artifact used.

External model code and checkpoints remain local-only unless explicitly packaged
as small metadata artifacts.
