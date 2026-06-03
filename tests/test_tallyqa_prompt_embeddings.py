from __future__ import annotations

import torch

from scripts.cache_smolvlm_tallyqa_teacher import safe_text_pad_token_id, validate_input_ids
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


def test_safe_text_pad_token_id_prefers_valid_text_config_pad() -> None:
    class Config:
        pad_token_id = 128002

        class text_config:
            pad_token_id = 2

    class Model:
        config = Config()

        @staticmethod
        def get_input_embeddings() -> torch.nn.Embedding:
            return torch.nn.Embedding(49280, 8)

    assert safe_text_pad_token_id(Model()) == 2


def test_validate_input_ids_rejects_out_of_range_ids() -> None:
    class Model:
        @staticmethod
        def get_input_embeddings() -> torch.nn.Embedding:
            return torch.nn.Embedding(10, 4)

    try:
        validate_input_ids(Model(), torch.tensor([[0, 9, 10]]), "test")
    except ValueError as error:
        assert "outside embedding range" in str(error)
    else:
        raise AssertionError("Expected out-of-range input IDs to fail validation.")
