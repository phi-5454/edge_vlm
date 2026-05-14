import torch

from config import DistillConfig
from models.tiny_vlm import TinyVLM, TinyVlmConfig
from training.callbacks import model_size_metrics
from training.datamodule import TeacherCacheDataset
from training.distill import TinyVlmDistillationModule


def test_tiny_vlm_forward_shapes() -> None:
    config = TinyVlmConfig(
        vocab_size=128,
        max_text_tokens=16,
        text_width=32,
        text_layers=1,
        text_heads=4,
        projection_dim=32,
        fusion_hidden_dim=64,
        teacher_dim=48,
        pretrained_vision=False,
    )
    model = TinyVLM(config)

    outputs = model(
        pixel_values=torch.randn(2, 3, 224, 224),
        input_ids=torch.randint(0, config.vocab_size, (2, 12)),
        attention_mask=torch.ones(2, 12, dtype=torch.long),
    )

    assert outputs["image_embeds"].shape == (2, 32)
    assert outputs["text_embeds"].shape == (2, 32)
    assert outputs["fused_embeds"].shape == (2, 64)
    assert outputs["teacher_embeds"].shape == (2, 48)


def test_tiny_vlm_is_smaller_than_teacher_target() -> None:
    model = TinyVLM(TinyVlmConfig(pretrained_vision=False))

    metrics = model_size_metrics(model, target_quantized_bits=8)

    assert metrics["model/parameters"] < 10_000_000


def test_distillation_student_uses_first_image_from_processor_batch() -> None:
    config = DistillConfig()
    config.student.pretrained_vision = False
    module = TinyVlmDistillationModule(config)
    pixel_values = torch.randn(1, 17, 3, 512, 512)

    student_pixel_values = module._student_pixel_values(pixel_values)

    assert student_pixel_values.shape == (1, 3, 512, 512)


def test_teacher_cache_dataset_attaches_embeddings() -> None:
    dataset = [{"text": "a"}, {"text": "b"}]
    embeddings = torch.randn(2, 4)

    cached_dataset = TeacherCacheDataset(dataset, embeddings)

    assert cached_dataset[1]["text"] == "b"
    assert torch.equal(cached_dataset[1]["teacher_embedding"], embeddings[1])


def test_distillation_uses_cached_teacher_targets() -> None:
    config = DistillConfig()
    config.student.pretrained_vision = False
    module = TinyVlmDistillationModule(config)
    cached_targets = torch.randn(2, 8)

    targets = module._teacher_targets({"teacher_embedding": cached_targets})

    assert torch.equal(targets, cached_targets)
