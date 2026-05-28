from __future__ import annotations

from collections import Counter

from vlm_micro.cauldron import (
    _length_coverage,
    _normalized_word_frequencies,
    _prompt_constructibility_coverage,
    filter_yes_no_texts,
    iter_qa_pairs,
    normalize_tokens,
    yes_no_answer,
)


def test_normalize_tokens_lowercases_and_keeps_simple_apostrophes() -> None:
    assert normalize_tokens("What's the COLOR? Net-like, 2x!") == [
        "what's",
        "the",
        "color",
        "net",
        "like",
        "2x",
    ]


def test_iter_qa_pairs_reads_cauldron_user_assistant_schema() -> None:
    sample = {
        "texts": [
            {"user": "Question one?", "assistant": "Answer one.", "source": "test"},
            {"user": "Question two?", "assistant": "Answer two.", "source": "test"},
        ]
    }

    assert list(iter_qa_pairs(sample)) == [
        ("Question one?", "Answer one."),
        ("Question two?", "Answer two."),
    ]


def test_normalized_word_frequencies_use_token_total() -> None:
    assert _normalized_word_frequencies(Counter({"yes": 3, "no": 1}), top_k=2) == [
        {"word": "yes", "count": 3, "frequency": 0.75},
        {"word": "no", "count": 1, "frequency": 0.25},
    ]


def test_length_coverage_is_cumulative_by_budget() -> None:
    assert _length_coverage([3, 1, 3, 2]) == [
        {"max_length": 1, "prompts": 1, "covered_prompts": 1, "prompt_fraction": 0.25},
        {"max_length": 2, "prompts": 1, "covered_prompts": 2, "prompt_fraction": 0.5},
        {"max_length": 3, "prompts": 2, "covered_prompts": 4, "prompt_fraction": 1.0},
    ]


def test_prompt_constructibility_coverage_uses_top_vocabulary_set() -> None:
    rows = _prompt_constructibility_coverage(
        Counter({"a": 3, "b": 2, "c": 1}),
        [{"a"}, {"a", "b"}, {"c"}],
        [1, 2],
    )

    assert rows == [
        {
            "vocab_items": 1,
            "vocab_fraction": 1 / 3,
            "token_occurrence_fraction": 3 / 6,
            "constructible_prompt_fraction": 1 / 3,
        },
        {
            "vocab_items": 2,
            "vocab_fraction": 2 / 3,
            "token_occurrence_fraction": 5 / 6,
            "constructible_prompt_fraction": 2 / 3,
        },
        {
            "vocab_items": 3,
            "vocab_fraction": 1.0,
            "token_occurrence_fraction": 1.0,
            "constructible_prompt_fraction": 1.0,
        },
    ]


def test_yes_no_answer_requires_one_normalized_token() -> None:
    assert yes_no_answer("Yes.") == "yes"
    assert yes_no_answer("no") == "no"
    assert yes_no_answer("Answer: yes") is None
    assert yes_no_answer("maybe") is None


def test_filter_yes_no_texts_keeps_binary_answers_and_normalizes() -> None:
    texts = [
        {"user": "Is it red?", "assistant": "Yes.", "source": "unit"},
        {"user": "What color?", "assistant": "Red.", "source": "unit"},
        {"user": "Is it blue?", "assistant": "No!", "source": "unit"},
    ]

    assert filter_yes_no_texts(texts) == [
        {"user": "Is it red?", "assistant": "yes", "source": "unit"},
        {"user": "Is it blue?", "assistant": "no", "source": "unit"},
    ]
