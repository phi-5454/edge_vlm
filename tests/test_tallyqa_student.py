from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from vlm_micro.student.data import (
    TallyQAStudentDataset,
    TallyQAStudentDataModule,
    Uint8MemmapImageStore,
    collapse_count,
    collate_tallyqa_student_batch,
    load_tallyqa_teacher_targets,
)
from vlm_micro.student.lightning import TallyQAStudentModule
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
    assert torch.isfinite(batch["images"]).all()


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
