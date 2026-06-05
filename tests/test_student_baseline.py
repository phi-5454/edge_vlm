from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from vlm_micro.student.data import (
    CompactVocabulary,
    ImageGroupedBatchSampler,
    ParquetImageStore,
    StudentDataModule,
    StudentDataset,
    collate_student_batch,
    load_teacher_targets,
    split_for_image,
)
from vlm_micro.student.lightning import StudentBaselineModule
from vlm_micro.student.model import StudentBaseline


def _fusion_input_shape(model: StudentBaseline, image_size: int = 224) -> tuple[int, ...]:
    shapes: list[tuple[int, ...]] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        shapes.append(tuple(inputs[0].shape))

    handle = model.fusion[0].register_forward_hook(hook)
    try:
        model(
            torch.tensor([[1, 2]]),
            torch.tensor([[True, True]]),
            torch.randn(1, 3, image_size, image_size),
        )
    finally:
        handle.remove()
    return shapes[0]


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), (100, 120, 140)).save(output, format="JPEG")
    return output.getvalue()


def test_compact_vocabulary_reserves_padding_row() -> None:
    vocabulary = CompactVocabulary.from_rows([{"student_token_ids": [9, 4, 9]}])

    assert vocabulary.teacher_token_ids == (4, 9)
    assert vocabulary.size == 3
    assert vocabulary.remap([9, 4]) == [2, 1]


def test_image_group_split_is_deterministic() -> None:
    assert split_for_image("vqav2:42", 42) == split_for_image("vqav2:42", 42)


def test_image_grouped_sampler_keeps_repeated_images_adjacent() -> None:
    rows = [
        {"student_token_ids": [1], "student_image_id": "a", "answer": "yes"},
        {"student_token_ids": [1], "student_image_id": "b", "answer": "yes"},
        {"student_token_ids": [1], "student_image_id": "a", "answer": "yes"},
    ]
    dataset = StudentDataset(rows, [0, 1, 2], CompactVocabulary.from_rows(rows), {}, None)  # type: ignore[arg-type]

    flattened = [
        item
        for batch in ImageGroupedBatchSampler(dataset, batch_size=3, seed=42, shuffle_block_size=2)
        for item in batch
    ]

    assert abs(flattened.index(0) - flattened.index(2)) == 1


def test_parquet_dataset_and_model_forward(tmp_path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist([{"student_image_id": "clevr:0", "image_bytes": _jpeg_bytes()}]),
        tmp_path / "images.parquet",
    )
    rows = [
        {
            "student_token_ids": [4, 9],
            "student_image_id": "clevr:0",
            "answer": "yes",
        }
    ]
    vocabulary = CompactVocabulary.from_rows(rows)
    dataset = StudentDataset(
        rows=rows,
        indices=[0],
        vocabulary=vocabulary,
        teacher_targets={0: 1.5},
        image_store=ParquetImageStore(tmp_path / "images.parquet", 1, 1, 64),
    )
    batch = collate_student_batch([dataset[0]])
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
    )

    logits = model(batch["token_ids"], batch["attention_mask"], batch["images"])

    assert logits.shape == (1,)


def test_student_baseline_defaults_to_mobilenet_v3_large() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
    )

    assert model.image_backbone_name == "mobilenet_v3_large"
    assert model.image_feature_cutoff == 13
    assert model.image_token_mode == "spatial"
    assert model.prompt_identity is not None
    assert model.image_position_embeddings is not None
    assert model.image_position_embeddings.shape == (1, 196, 16)
    assert model.image_projection[0].in_features == 112
    assert _fusion_input_shape(model) == (1, 197, 16)


def test_student_baseline_can_still_select_mobilenet_v3_small() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
        image_backbone="mobilenet_v3_small",
    )

    assert model.image_backbone_name == "mobilenet_v3_small"
    assert model.image_feature_cutoff == 9
    assert model.image_token_mode == "spatial"
    assert model.image_projection[0].in_features == 48
    assert _fusion_input_shape(model) == (1, 197, 16)


def test_student_baseline_can_still_use_full_mobilenet_features() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
        image_feature_cutoff=None,
    )

    assert model.image_feature_cutoff is None
    assert model.image_projection[0].in_features == 960


def test_student_baseline_can_still_use_pooled_image_token() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
        image_token_mode="pooled",
    )

    assert _fusion_input_shape(model) == (1, 2, 16)


def test_student_baseline_can_freeze_image_features_only() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
        freeze_image_features=True,
    )

    assert all(not parameter.requires_grad for parameter in model.image_features.parameters())
    assert model.prompt_identity is not None
    assert model.prompt_identity.requires_grad
    assert model.image_position_embeddings is not None
    assert model.image_position_embeddings.requires_grad
    assert all(parameter.requires_grad for parameter in model.image_projection.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_student_baseline_can_disable_prompt_and_image_position_embeddings() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
        use_prompt_identity=False,
        use_image_positional_embeddings=False,
    )

    assert model.prompt_identity is None
    assert model.image_position_embeddings is None
    assert _fusion_input_shape(model) == (1, 197, 16)


def test_student_baseline_can_apply_prompt_film_before_112_channel_blocks() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
        image_backbone="mobilenet_v3_large",
        image_feature_cutoff=13,
        image_film_at=10,
        freeze_image_features=True,
        num_outputs=6,
    )

    assert model.image_film_at == 10
    assert model.image_film is not None
    assert model.image_film.channels == 80
    assert all(not parameter.requires_grad for parameter in model.image_features.parameters())
    assert all(parameter.requires_grad for parameter in model.image_film.parameters())
    assert torch.count_nonzero(model.image_film.to_scale_shift.weight) == 0
    assert torch.count_nonzero(model.image_film.to_scale_shift.bias) == 0
    assert _fusion_input_shape(model) == (1, 197, 16)


def test_student_baseline_concat_mlp_fusion_has_no_transformer_blocks() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=4,
        fusion_heads=4,
        fusion_mode="concat_mlp",
        use_prompt_identity=False,
        use_image_positional_embeddings=False,
        num_outputs=6,
    )
    logits = model(
        torch.tensor([[1, 2]]),
        torch.tensor([[True, True]]),
        torch.randn(1, 3, 224, 224),
    )

    assert isinstance(model.fusion, torch.nn.Identity)
    assert model.prompt_identity is None
    assert model.image_position_embeddings is None
    assert logits.shape == (1, 6)


def test_alpha_one_does_not_require_teacher_targets() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
    )
    module = StudentBaselineModule(model, alpha=1, learning_rate=1e-3, weight_decay=0)
    batch = {
        "token_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[True, True]]),
        "images": torch.randn(1, 3, 64, 64),
        "labels": torch.tensor([1.0]),
        "teacher_logits": torch.tensor([float("nan")]),
    }

    assert torch.isfinite(module._shared_step(batch, "train"))


def test_optimizer_warms_up_learning_rate_per_step() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(2, 16),
        image_pretrained=False,
        query_dim=16,
        image_dim=16,
        fusion_dim=16,
        fusion_depth=1,
        fusion_heads=4,
    )
    module = StudentBaselineModule(
        model,
        alpha=1,
        learning_rate=1e-3,
        warmup_start_learning_rate=1e-4,
        warmup_steps=1000,
        weight_decay=0,
    )

    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler_config = configured["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]

    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert scheduler_config["interval"] == "step"
    for _ in range(1000):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_teacher_cache_ignores_only_truncated_trailing_record(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.jsonl"
    cache.write_text(
        json.dumps(
            {
                "dataset_index": 4,
                "teacher_logits": {"standalone": {"yes_minus_no_logit": 1.25}},
            }
        )
        + "\n"
        + '{"dataset_index": 5,',
        encoding="utf-8",
    )

    targets = load_teacher_targets(cache)

    assert targets == {4: 1.25}


def test_data_module_filters_rows_missing_teacher_targets(tmp_path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_subset": "clevr",
                    "original_index": index,
                    "qa_index": 0,
                    "answer": "yes",
                    "student_prompt": f"prompt {index}",
                    "student_token_ids": [index + 1],
                    "student_image_id": f"clevr:{index}",
                }
                for index in range(3)
            ]
        ),
        tmp_path / "combined.parquet",
    )
    cache = tmp_path / "teacher.jsonl"
    cache.write_text(
        json.dumps(
            {
                "dataset_index": 1,
                "teacher_logits": {"standalone": {"yes_minus_no_logit": -0.5}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    data = StudentDataModule(
        tmp_path,
        cache,
        batch_size=1,
        num_workers=0,
        image_size=64,
        seed=42,
        missing_teacher_policy="filter",
    )

    assert sum(data.split_sizes().values()) == 1
    assert sum(data.full_split_sizes().values()) == 3
    assert data.cache_coverage()["covered_fraction"] == 1 / 3
    assert data.hparams.dataset_root == str(tmp_path)
    assert data.hparams.teacher_cache == str(cache)
    metadata = tmp_path / "datamodule_hparams.pt"
    torch.save(dict(data.hparams), metadata)
    assert torch.load(metadata, weights_only=True)["dataset_root"] == str(tmp_path)
