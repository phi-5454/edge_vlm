# Tiny VLM Distillation Architecture

## Goal

Replace full SmolVLM backpropagation with a small student model optimized for
fast inference and microcontroller deployment. SmolVLM remains a frozen teacher
that supplies soft targets and intermediate representations.

## Student Model

The student is a dual-encoder VLM with late fusion:

- Vision encoder: MobileNetV3-Small feature trunk with global average pooling.
- Text encoder: learned token and position embeddings followed by a 2-layer
  Transformer encoder.
- Projection heads: image and text embeddings are projected to a shared 128-d
  space.
- Fusion block: concatenated image/text embeddings pass through a small MLP.
- Distillation head: fused features are projected to the teacher target
  dimension for hidden-state matching.
- Optional task head: a compact classification head can be attached for fixed
  answer sets or deployment-specific intents.

This is not a generative decoder. That is intentional: small size and fast
inference are the priority. The first deployment target should be ranking,
classification, retrieval, or constrained-answer VQA rather than open-ended text
generation.

## Distillation Losses

Use a weighted mix of:

- Teacher embedding loss: cosine or MSE loss between student `teacher_embeds`
  and a frozen SmolVLM pooled representation.
- Contrastive loss: align image/text pairs in the shared projection space.
- Task loss: cross-entropy or KL divergence for labeled or teacher-generated
  answer classes.

## Size Tracking

Log both loaded uncompressed size and target quantized size. For this student,
the expected deployment path is int8 first, then int4 or structured pruning if
accuracy margins allow it.
