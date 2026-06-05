from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from vlm_micro.student.data import (
    TallyQAStudentDataset,
    TallyQAStudentDataModule,
    Uint8MemmapImageStore,
    collapse_count,
    collate_tallyqa_student_batch,
    load_tallyqa_teacher_targets,
)
from vlm_micro.student.lightning import MulticlassTotals, TallyQAStudentModule
from vlm_micro.student.model import StudentBaseline


def _write_tallyqa_fixture(root: Path) -> Path:
    examples = [
        {
            "example_id": "ex0",
            "source_subset": "tallyqa",
            "source_row_index": 0,
            "qa_index": 0,
            "answer": 2,
            "student_prompt": "chairs",
            "item": "chairs",
            "item_class_id": 0,
            "image_id": "img0",
            "image_index": 0,
        },
        {
            "example_id": "ex1",
            "source_subset": "tallyqa",
            "source_row_index": 1,
            "qa_index": 0,
            "answer": 7,
            "student_prompt": "people",
            "item": "people",
            "item_class_id": 1,
            "image_id": "img1",
            "image_index": 1,
        },
    ]
    pq.write_table(pa.Table.from_pylist(examples), root / "examples.parquet")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "image": {
                    "shape": [2, 3, 224, 224],
                    "tensor_file": "images.uint8.bin",
                    "index_file": "images.index.jsonl",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    images = np.memmap(root / "images.uint8.bin", dtype=np.uint8, mode="w+", shape=(2, 3, 224, 224))
    images[0] = 128
    images[1] = 64
    images.flush()
    prompt_path = root / "prompt_embeddings.pt"
    torch.save(
        {
            "prompt_token_ids": torch.tensor([[1, 0], [2, 3]]),
            "prompt_attention_mask": torch.tensor([[True, False], [True, True]]),
            "embedding_rows": torch.randn(3, 8),
            "prompt_classes": [],
        },
        prompt_path,
    )
    return prompt_path


def test_tallyqa_teacher_targets_collapse_5_plus(tmp_path: Path) -> None:
    cache = tmp_path / "teacher.jsonl"
    cache.write_text(
        json.dumps(
            {
                "dataset_index": 0,
                "teacher_logits": {
                    "numeric_answer_candidates": [
                        {"answer": 0, "candidate_probability": 0.1},
                        {"answer": 5, "candidate_probability": 0.2},
                        {"answer": 8, "candidate_probability": 0.3},
                        {"answer": 15, "candidate_probability": 0.4},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    targets = load_tallyqa_teacher_targets(cache, num_classes=6, collapse_at=5)

    assert targets[0].tolist() == pytest.approx([0.1, 0.0, 0.0, 0.0, 0.0, 0.9])


def test_tallyqa_datamodule_loads_examples_and_uint8_images(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
    )
    assert sum(data.full_split_sizes().values()) == 2
    dataset = TallyQAStudentDataset(
        rows=data.rows,
        indices=[0, 1],
        prompt_token_ids=data.prompt_token_ids,
        prompt_attention_mask=data.prompt_attention_mask,
        teacher_targets=data.teacher_targets,
        image_store=Uint8MemmapImageStore(tmp_path, tensor_cache_size=2),
    )
    batch = collate_tallyqa_student_batch([dataset[0], dataset[1]])

    assert batch["token_ids"].shape == (2, 2)
    assert batch["images"].shape == (2, 3, 224, 224)
    assert batch["labels"].tolist() == [collapse_count(2), collapse_count(7)]
    assert batch["student_prompts"] == ["chairs", "people"]
    assert torch.isfinite(batch["images"]).all()


def test_tallyqa_datamodule_filters_prompt_classes_by_accuracy(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    filter_csv = tmp_path / "prompt_accuracy.csv"
    filter_csv.write_text(
        "student_prompt,item_class_id,count,correct,accuracy\n"
        "chairs,0,10,6,0.6\n"
        "people,1,10,4,0.4\n",
        encoding="utf-8",
    )

    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        prompt_class_filter_csv=filter_csv,
        min_prompt_accuracy=0.5,
    )

    assert sum(data.split_sizes().values()) == 1
    active_index = next(index for indices in data.indices.values() for index in indices)
    assert data.rows[active_index]["student_prompt"] == "chairs"


def test_tallyqa_datamodule_filters_prompt_classes_by_name(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        prompt_class_names="people",
    )

    assert sum(data.split_sizes().values()) == 1
    active_index = next(index for indices in data.indices.values() for index in indices)
    assert data.rows[active_index]["student_prompt"] == "people"


def test_tallyqa_datamodule_filters_prompt_classes_by_name_file(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("# comment\npeople\n", encoding="utf-8")
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        prompt_class_names_file=prompt_file,
    )

    assert sum(data.split_sizes().values()) == 1
    active_index = next(index for indices in data.indices.values() for index in indices)
    assert data.rows[active_index]["student_prompt"] == "people"


def test_tallyqa_datamodule_curriculum_changes_train_prompts_by_epoch(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    examples = [
        {
            "example_id": "ex0",
            "source_subset": "tallyqa",
            "source_row_index": 0,
            "qa_index": 0,
            "answer": 2,
            "student_prompt": "chairs",
            "item": "chairs",
            "item_class_id": 0,
            "image_id": "img0",
            "image_index": 0,
        },
        {
            "example_id": "ex1",
            "source_subset": "tallyqa",
            "source_row_index": 1,
            "qa_index": 0,
            "answer": 7,
            "student_prompt": "people",
            "item": "people",
            "item_class_id": 1,
            "image_id": "img2",
            "image_index": 1,
        },
    ]
    pq.write_table(pa.Table.from_pylist(examples), tmp_path / "examples.parquet")
    people_file = tmp_path / "people.txt"
    chairs_file = tmp_path / "chairs.txt"
    schedule_file = tmp_path / "schedule.json"
    people_file.write_text("people\n", encoding="utf-8")
    chairs_file.write_text("chairs\n", encoding="utf-8")
    schedule_file.write_text(
        json.dumps(
            [
                {"start_epoch": 1, "prompt_class_names_file": str(people_file)},
                {"start_epoch": 3, "prompt_class_names_file": str(chairs_file)},
            ]
        ),
        encoding="utf-8",
    )
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        curriculum_schedule=schedule_file,
    )

    assert {data.rows[index]["student_prompt"] for index in data._train_indices()} == {"people"}
    data.set_train_epoch(2)
    assert {data.rows[index]["student_prompt"] for index in data._train_indices()} == {"chairs"}


def test_tallyqa_datamodule_reports_active_label_counts(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
    )

    counts = data.label_counts("train")

    assert sum(counts.values()) == data.split_sizes()["train"]
    assert counts[collapse_count(2)] + counts[collapse_count(7)] == data.split_sizes()["train"]


def test_tallyqa_train_loader_can_disable_shuffle_for_overfit_debug(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=1,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        group_train_by_image=False,
        shuffle_train=False,
    )

    first = next(iter(data.train_dataloader()))["dataset_index"].tolist()
    second = next(iter(data.train_dataloader()))["dataset_index"].tolist()

    assert first == second


def test_tallyqa_train_example_limit_fixes_training_pool(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    examples = [
        {
            "example_id": f"ex{index}",
            "source_subset": "tallyqa",
            "source_row_index": index,
            "qa_index": 0,
            "answer": index,
            "student_prompt": "chairs" if index % 2 == 0 else "people",
            "item": "chairs" if index % 2 == 0 else "people",
            "item_class_id": index % 2,
            "image_id": image_id,
            "image_index": index % 2,
        }
        for index, image_id in enumerate(["img0", "img2", "img3", "img4"])
    ]
    pq.write_table(pa.Table.from_pylist(examples), tmp_path / "examples.parquet")
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        group_train_by_image=False,
        shuffle_train=True,
        train_example_limit=2,
    )

    assert data.full_split_sizes()["train"] == 4
    assert data.split_sizes()["train"] == 2
    assert data.train_dataloader().dataset.indices == [0, 1]


def test_tallyqa_prompt_class_tempered_sampler_uses_inverse_sqrt_counts(tmp_path: Path) -> None:
    prompt_path = _write_tallyqa_fixture(tmp_path)
    examples = [
        {
            "example_id": f"ex{index}",
            "source_subset": "tallyqa",
            "source_row_index": index,
            "qa_index": 0,
            "answer": index,
            "student_prompt": prompt,
            "item": prompt,
            "item_class_id": class_id,
            "image_id": image_id,
            "image_index": index % 2,
        }
        for index, (image_id, prompt, class_id) in enumerate(
            [
                ("img0", "chairs", 0),
                ("img2", "chairs", 0),
                ("img3", "chairs", 0),
                ("img4", "people", 1),
            ]
        )
    ]
    pq.write_table(pa.Table.from_pylist(examples), tmp_path / "examples.parquet")
    data = TallyQAStudentDataModule(
        dataset_root=tmp_path,
        prompt_embeddings=prompt_path,
        teacher_cache=None,
        batch_size=2,
        num_workers=0,
        seed=42,
        missing_teacher_policy="keep",
        tensor_cache_size=2,
        group_train_by_image=True,
        shuffle_train=True,
        train_sampling="prompt_class_tempered",
        prompt_class_sampling_temperature=0.5,
    )
    dataset = data.train_dataloader().dataset
    weights = data._prompt_class_sampling_weights(dataset)

    assert weights.tolist() == pytest.approx([3**-0.5, 3**-0.5, 3**-0.5, 1.0])


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits[: token_ids.shape[0]]


def test_tallyqa_student_module_multiclass_loss() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(3, 8),
        image_pretrained=False,
        image_backbone="mobilenet_v3_small",
        query_dim=8,
        image_dim=8,
        fusion_dim=8,
        fusion_depth=1,
        fusion_heads=4,
        num_outputs=6,
    )
    module = TallyQAStudentModule(
        model=model,
        alpha=1.0,
        beta=0.5,
        learning_rate=1e-3,
        weight_decay=0,
        temperature=2.0,
    )
    batch = {
        "token_ids": torch.tensor([[1, 0], [2, 3]]),
        "attention_mask": torch.tensor([[True, False], [True, True]]),
        "images": torch.randn(2, 3, 64, 64),
        "labels": torch.tensor([2, 5]),
        "teacher_probs": torch.softmax(torch.randn(2, 6), dim=1),
    }

    loss = module._shared_step(batch, "train")

    assert torch.isfinite(loss)


def test_tallyqa_student_module_class_weights_downweight_hard_loss() -> None:
    module = TallyQAStudentModule(
        model=FixedLogitModel(torch.zeros(2, 6)),
        alpha=1.0,
        beta=0.0,
        learning_rate=1e-3,
        weight_decay=0,
        class_weights=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )
    batch = {
        "token_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.tensor([[True], [True]]),
        "images": torch.randn(2, 3, 64, 64),
        "labels": torch.tensor([0, 5]),
        "teacher_probs": torch.softmax(torch.randn(2, 6), dim=1),
    }

    loss = module._shared_step(batch, "train")

    assert float(loss) == pytest.approx(float(torch.log(torch.tensor(6.0)) / 2))


def test_tallyqa_student_module_local_soft_targets_are_centered_on_label() -> None:
    module = TallyQAStudentModule(
        model=FixedLogitModel(torch.zeros(1, 6)),
        alpha=1.0,
        beta=0.0,
        learning_rate=1e-3,
        weight_decay=0,
        target_distribution="local_soft",
        local_soft_sigma=1.0,
        local_soft_radius=1,
    )

    targets = module._target_distribution(torch.tensor([2]), num_classes=6)
    expected = torch.tensor([[0.0, 0.2741, 0.4519, 0.2741, 0.0, 0.0]])

    assert torch.allclose(targets, expected, atol=1e-4)
    assert float(targets.sum()) == pytest.approx(1.0)


def test_tallyqa_student_optimizer_uses_lower_image_learning_rate() -> None:
    model = StudentBaseline(
        embedding_rows=torch.randn(3, 8),
        image_pretrained=False,
        image_backbone="mobilenet_v3_small",
        query_dim=8,
        image_dim=8,
        fusion_dim=8,
        fusion_depth=1,
        fusion_heads=4,
        image_feature_cutoff="auto",
        num_outputs=6,
    )
    module = TallyQAStudentModule(
        model=model,
        alpha=1.0,
        beta=0.0,
        learning_rate=1e-3,
        weight_decay=0,
        image_learning_rate_scale=0.1,
    )

    optimizer = module.configure_optimizers()
    groups = {group["name"]: group for group in optimizer.param_groups}

    assert set(groups) == {"main", "image_features"}
    assert groups["main"]["lr"] == pytest.approx(1e-3)
    assert groups["image_features"]["lr"] == pytest.approx(1e-4)


def test_tallyqa_student_module_accumulates_validation_confusion() -> None:
    module = TallyQAStudentModule(
        model=FixedLogitModel(
            torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0],
                    [0.0, 4.0, 0.0],
                ]
            )
        ),
        alpha=1.0,
        beta=0.0,
        learning_rate=1e-3,
        weight_decay=0,
    )
    batch = {
        "token_ids": torch.tensor([[1], [2], [3]]),
        "attention_mask": torch.tensor([[True], [True], [True]]),
        "images": torch.randn(3, 3, 64, 64),
        "labels": torch.tensor([0, 1, 1]),
        "teacher_probs": torch.softmax(torch.randn(3, 3), dim=1),
    }

    module._shared_step(batch, "val")

    assert module._confusions["val"].tolist() == [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]


def test_multiclass_totals_report_within_one_accuracy() -> None:
    totals = MulticlassTotals()

    totals.update(
        logits=torch.tensor(
            [
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ]
        ),
        labels=torch.tensor([1, 3, 1]),
    )

    assert totals.metrics()["accuracy"] == pytest.approx(1 / 3)
    assert totals.metrics()["within_1_accuracy"] == pytest.approx(2 / 3)
