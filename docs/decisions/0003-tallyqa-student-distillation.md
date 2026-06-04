# TallyQA Student Distillation Loss

Date: 2026-06-04

## Context

The TallyQA target dataset uses one-word student prompts and collapsed answer
classes `0, 1, 2, 3, 4, 5+`. The cached SmolVLM teacher provides soft numeric
answer probabilities over `0..15`, which are collapsed into the same six
classes for student training.

The current collapsed teacher accuracy is useful but imperfect, so the hard
TallyQA answer should remain a strong anchor during early student runs.

## Decision

Use a six-logit classifier and train with:

```text
loss = alpha * CE(student_logits, hard_target)
     + beta * T^2 * KL(teacher_probs || student_probs_T)
```

Initial defaults:

- `alpha = 1.0`
- `beta = 0.5`
- `temperature = 2.0`

## Rationale

This follows the standard knowledge distillation form: supervised
cross-entropy on ground truth plus temperature-scaled KL divergence from the
teacher distribution to the student distribution. Independent `alpha` and
`beta` weights are equivalent to the common `(1 - lambda)` / `lambda`
parameterization, but make it easier to keep CE at a stable scale while tuning
the teacher contribution.

The teacher is not strong enough yet to dominate the target signal, so `beta`
starts below the CE weight. Reasonable sweeps are `beta in {0.0, 0.25, 0.5,
1.0}` at fixed `alpha = 1.0`, with higher `beta` more attractive if the larger
teacher cache improves per-class accuracy.

## Consequences

Student checkpoints report CE, KL, and combined loss separately, so later runs
can diagnose whether the teacher term improves validation accuracy or simply
pulls the student toward teacher mistakes.
