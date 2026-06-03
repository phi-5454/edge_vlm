from __future__ import annotations

import torch

from scripts.precompute_tallyqa_prompt_embeddings import (
    compact_prompt_tensors,
    masked_mean_prompt_embeddings,
)


def test_compact_prompt_tensors_reserve_padding_row() -> None:
    rows = [
        {"class_id": 0, "item": "a", "teacher_token_ids": [10, 20]},
        {"class_id": 1, "item": "b", "teacher_token_ids": [20]},
    ]

    token_ids, attention_mask, enriched = compact_prompt_tensors(
        rows,
        teacher_to_compact={10: 1, 20: 2},
        max_length=2,
    )

    assert token_ids.tolist() == [[1, 2], [2, 0]]
    assert attention_mask.tolist() == [[True, True], [True, False]]
    assert enriched[0]["compact_token_ids"] == [1, 2]
    assert enriched[1]["compact_token_ids"] == [2]


def test_masked_mean_prompt_embeddings_ignore_padding() -> None:
    embedding_rows = torch.tensor(
        [
            [2.0, 4.0],
            [10.0, 20.0],
        ]
    )
    token_ids = torch.tensor([[1, 2], [2, 0]])
    attention_mask = torch.tensor([[True, True], [True, False]])

    pooled = masked_mean_prompt_embeddings(embedding_rows, token_ids, attention_mask)

    assert pooled.tolist() == [[6.0, 12.0], [10.0, 20.0]]
