from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class CauldronSpec:
    datasets: list[str]
    optional_reasoning: list[str]


def load_cauldron_spec(path: str | Path) -> CauldronSpec:
    cfg = OmegaConf.load(path)
    return CauldronSpec(
        datasets=[str(item) for item in cfg.get("datasets", [])],
        optional_reasoning=[str(item) for item in cfg.get("optional_reasoning", [])],
    )


def selected_subsets(spec: CauldronSpec, include_optional_reasoning: bool) -> list[str]:
    subsets = list(spec.datasets)
    if include_optional_reasoning:
        subsets.extend(spec.optional_reasoning)
    return subsets


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def iter_qa_pairs(sample: dict[str, Any]) -> Iterable[tuple[str, str]]:
    texts = sample.get("texts")
    if isinstance(texts, list):
        for message in texts:
            if not isinstance(message, dict):
                continue
            user = message.get("user")
            assistant = message.get("assistant")
            if isinstance(user, str) and isinstance(assistant, str):
                yield user.strip(), assistant.strip()
        return

    question = sample.get("question") or sample.get("query") or sample.get("prompt")
    answer = sample.get("answer")
    if isinstance(question, str) and isinstance(answer, str):
        yield question.strip(), answer.strip()


def yes_no_answer(answer: str) -> str | None:
    tokens = normalize_tokens(answer)
    if len(tokens) != 1 or tokens[0] not in {"yes", "no"}:
        return None
    return tokens[0]


def filter_yes_no_texts(texts: Any) -> list[dict[str, str]]:
    if not isinstance(texts, list):
        return []

    filtered: list[dict[str, str]] = []
    for message in texts:
        if not isinstance(message, dict):
            continue
        user = message.get("user")
        assistant = message.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            continue
        label = yes_no_answer(assistant)
        if label is None:
            continue
        row = {"user": user.strip(), "assistant": label}
        source = message.get("source")
        if isinstance(source, str):
            row["source"] = source
        filtered.append(row)
    return filtered


def _configured_subsets(cfg: DictConfig, spec: CauldronSpec) -> list[str]:
    explicit = cfg.get("subsets", [])
    if explicit:
        return [str(item) for item in explicit]
    return selected_subsets(spec, bool(cfg.include_optional_reasoning))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _write_streaming_jsonl(cfg: DictConfig, subset: str, output: Path) -> int:
    tmp_output = output.with_suffix(".jsonl.tmp")
    dataset = load_dataset(
        str(cfg.dataset_name),
        subset,
        split=str(cfg.split),
        streaming=True,
    )
    max_examples = cfg.streaming_max_examples
    max_examples_int = None if max_examples is None else int(max_examples)
    rows = 0
    with tmp_output.open("w", encoding="utf-8") as handle:
        for sample in dataset:
            handle.write(json.dumps(_json_safe(sample), ensure_ascii=False) + "\n")
            rows += 1
            if rows % 1000 == 0:
                print(f"{subset}: streamed {rows} rows", flush=True)
            if max_examples_int is not None and rows >= max_examples_int:
                break
    tmp_output.rename(output)
    return rows


def _download_raw_parquet(cfg: DictConfig, subset: str) -> list[str]:
    raw_root = Path(str(cfg.local_root)) / "_parquet" / subset
    raw_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    prefix = f"{subset}/"
    files = [
        path
        for path in api.list_repo_files(str(cfg.dataset_name), repo_type="dataset")
        if path.startswith(prefix) and path.endswith(".parquet") and Path(path).name.startswith(str(cfg.split))
    ]
    if not files:
        raise FileNotFoundError(f"No parquet files found for {cfg.dataset_name}/{subset}:{cfg.split}")

    local_files: list[str] = []
    for index, repo_file in enumerate(files, start=1):
        print(f"{subset}: downloading parquet {index}/{len(files)} {repo_file}", flush=True)
        local_files.append(
            hf_hub_download(
                repo_id=str(cfg.dataset_name),
                repo_type="dataset",
                filename=repo_file,
                local_dir=raw_root.parent,
            )
        )
    return local_files


def _download_subset_parquet_as_hf(cfg: DictConfig, subset: str, output: Path) -> int:
    local_files = _download_raw_parquet(cfg, subset)

    dataset = load_dataset("parquet", data_files=local_files, split="train")
    tmp_output = output.parent / f".{output.name}.tmp"
    if tmp_output.exists():
        shutil.rmtree(tmp_output)
    dataset.save_to_disk(tmp_output)
    if output.exists():
        shutil.rmtree(output)
    tmp_output.rename(output)
    return len(dataset)


def cache_cauldron_datasets(cfg: DictConfig) -> list[Path]:
    spec = load_cauldron_spec(str(cfg.spec))
    subsets = _configured_subsets(cfg, spec)
    root = Path(str(cfg.local_root))
    root.mkdir(parents=True, exist_ok=True)
    cache_format = str(cfg.cache_format)

    outputs: list[Path] = []
    manifest: dict[str, Any] = {
        "dataset_name": str(cfg.dataset_name),
        "split": str(cfg.split),
        "spec": str(cfg.spec),
        "include_optional_reasoning": bool(cfg.include_optional_reasoning),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subsets": {},
    }
    for subset in subsets:
        output = root / f"{subset}.jsonl" if cache_format == "jsonl" else root / subset
        if output.exists() and not bool(cfg.force_download):
            manifest["subsets"][subset] = {"path": str(output), "status": "already_present"}
            outputs.append(output)
            continue

        if cache_format == "jsonl":
            print(f"Streaming {cfg.dataset_name}/{subset}:{cfg.split} -> {output}", flush=True)
            rows = _write_streaming_jsonl(cfg, subset, output)
            manifest["subsets"][subset] = {
                "path": str(output),
                "status": "streamed",
                "format": "jsonl",
                "num_rows": rows,
            }
            outputs.append(output)
            continue

        if cache_format == "parquet-hf":
            print(f"Downloading parquet {cfg.dataset_name}/{subset}:{cfg.split} -> {output}", flush=True)
            rows = _download_subset_parquet_as_hf(cfg, subset, output)
            manifest["subsets"][subset] = {
                "path": str(output),
                "status": "downloaded",
                "format": "parquet-hf",
                "num_rows": rows,
            }
            outputs.append(output)
            continue

        if cache_format == "parquet":
            print(f"Downloading raw parquet {cfg.dataset_name}/{subset}:{cfg.split}", flush=True)
            files = _download_raw_parquet(cfg, subset)
            output = root / "_parquet" / subset
            manifest["subsets"][subset] = {
                "path": str(output),
                "status": "downloaded",
                "format": "parquet",
                "num_files": len(files),
            }
            outputs.append(output)
            continue

        print(f"Downloading {cfg.dataset_name}/{subset}:{cfg.split} -> {output}", flush=True)
        tmp_output = root / f".{subset}.tmp"
        if tmp_output.exists():
            shutil.rmtree(tmp_output)
        dataset = load_dataset(
            str(cfg.dataset_name),
            subset,
            split=str(cfg.split),
            streaming=False,
        )
        dataset.save_to_disk(tmp_output)
        if output.exists():
            shutil.rmtree(output)
        tmp_output.rename(output)
        manifest["subsets"][subset] = {
            "path": str(output),
            "status": "downloaded",
            "format": "hf",
            "num_rows": len(dataset),
        }
        outputs.append(output)

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return outputs


def _length_hist(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    sorted_lengths = sorted(lengths)
    return {
        "count": len(lengths),
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": sum(lengths) / len(lengths),
        "p50": sorted_lengths[int(0.50 * (len(sorted_lengths) - 1))],
        "p90": sorted_lengths[int(0.90 * (len(sorted_lengths) - 1))],
        "p95": sorted_lengths[int(0.95 * (len(sorted_lengths) - 1))],
        "p99": sorted_lengths[int(0.99 * (len(sorted_lengths) - 1))],
    }


def _coverage(
    counter: Counter[str],
    text_token_sets: list[set[str]],
    coverage_points: list[int],
) -> list[dict[str, float | int]]:
    total_tokens = counter.total()
    vocab_size = len(counter)
    if total_tokens == 0 or vocab_size == 0:
        return []

    ranked_words = [word for word, _ in counter.most_common()]
    covered_tokens = 0
    selected_words: set[str] = set()
    point_set = {point for point in coverage_points if point > 0}
    point_set.add(vocab_size)
    rows = []
    for rank, word in enumerate(ranked_words, start=1):
        selected_words.add(word)
        covered_tokens += counter[word]
        if rank not in point_set and rank != vocab_size:
            continue
        constructible = sum(1 for token_set in text_token_sets if token_set <= selected_words)
        rows.append(
            {
                "top_words": rank,
                "vocab_fraction": rank / vocab_size,
                "token_fraction": covered_tokens / total_tokens,
                "constructible_text_fraction": constructible / len(text_token_sets)
                if text_token_sets
                else 0.0,
            }
        )
    return rows


def _prompt_constructibility_coverage(
    counter: Counter[Any],
    prompt_token_sets: list[set[Any]],
    coverage_points: list[int],
) -> list[dict[str, float | int]]:
    total_tokens = counter.total()
    vocab_size = len(counter)
    if total_tokens == 0 or vocab_size == 0:
        return []

    ranked_items = [item for item, _ in counter.most_common()]
    rank_by_item = {item: rank for rank, item in enumerate(ranked_items, start=1)}
    cumulative_tokens_by_rank: dict[int, int] = {}
    covered_tokens = 0
    for rank, item in enumerate(ranked_items, start=1):
        covered_tokens += counter[item]
        cumulative_tokens_by_rank[rank] = covered_tokens

    required_rank_counts: Counter[int] = Counter()
    for prompt_set in prompt_token_sets:
        if not prompt_set:
            continue
        required_rank_counts.update([max(rank_by_item[item] for item in prompt_set)])

    point_set = {point for point in coverage_points if point > 0}
    point_set.update(required_rank_counts)
    point_set.add(vocab_size)
    rows = []
    constructible = 0
    for rank in sorted(point for point in point_set if point <= vocab_size):
        constructible += sum(count for required_rank, count in required_rank_counts.items() if required_rank == rank)
        rows.append(
            {
                "vocab_items": rank,
                "vocab_fraction": rank / vocab_size,
                "token_occurrence_fraction": cumulative_tokens_by_rank[rank] / total_tokens,
                "constructible_prompt_fraction": constructible / len(prompt_token_sets)
                if prompt_token_sets
                else 0.0,
            }
        )
    return rows


def _plot_length_hist(lengths: list[int], title: str, output: Path) -> None:
    plt.figure(figsize=(8, 5))
    upper = max(lengths) if lengths else 1
    bins = range(0, min(upper, 50) + 2)
    plt.hist([min(value, 50) for value in lengths], bins=bins, color="#3266a8", edgecolor="white")
    plt.title(title)
    plt.xlabel("Normalized word count, clipped at 50")
    plt.ylabel("Texts")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _plot_top_words(counter: Counter[str], title: str, output: Path, top_k: int) -> None:
    rows = counter.most_common(top_k)
    labels = [word for word, _ in rows]
    values = [count for _, count in rows]
    plt.figure(figsize=(max(10, math.ceil(top_k * 0.22)), 5))
    plt.bar(labels, values, color="#4f8f66")
    plt.title(title)
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _normalized_word_frequencies(counter: Counter[str], top_k: int) -> list[dict[str, float | int | str]]:
    total = counter.total()
    if total == 0:
        return []
    return [
        {"word": word, "count": count, "frequency": count / total}
        for word, count in counter.most_common(top_k)
    ]


def _plot_normalized_word_frequencies(
    counter: Counter[str],
    title: str,
    output: Path,
    top_k: int,
) -> None:
    rows = _normalized_word_frequencies(counter, top_k)
    labels = [str(row["word"]) for row in rows]
    values = [100 * float(row["frequency"]) for row in rows]
    plt.figure(figsize=(max(10, math.ceil(top_k * 0.22)), 5))
    plt.bar(labels, values, color="#8a5f2d")
    plt.title(title)
    plt.xlabel("Word")
    plt.ylabel("Normalized frequency (% of tokens)")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _plot_coverage(rows: list[dict[str, float | int]], title: str, output: Path) -> None:
    plt.figure(figsize=(8, 5))
    x = [100 * float(row["vocab_fraction"]) for row in rows]
    token_y = [100 * float(row["token_fraction"]) for row in rows]
    text_y = [100 * float(row["constructible_text_fraction"]) for row in rows]
    plt.plot(x, token_y, marker="o", label="token occurrence coverage")
    plt.plot(x, text_y, marker="o", label="fully constructible texts")
    plt.title(title)
    plt.xlabel("Top vocabulary covered (%)")
    plt.ylabel("Coverage (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _plot_prompt_constructibility_coverage(
    rows: list[dict[str, float | int]],
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    if rows:
        x = [int(row["vocab_items"]) for row in rows]
        y = [100 * float(row["constructible_prompt_fraction"]) for row in rows]
        plt.plot(x, y, marker="o", color="#7b4fa3")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Prompts constructible (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _length_coverage(lengths: list[int]) -> list[dict[str, float | int]]:
    if not lengths:
        return []
    total = len(lengths)
    counts = Counter(lengths)
    covered = 0
    rows = []
    for length in sorted(counts):
        covered += counts[length]
        rows.append(
            {
                "max_length": length,
                "prompts": counts[length],
                "covered_prompts": covered,
                "prompt_fraction": covered / total,
            }
        )
    return rows


def _plot_length_coverage(
    lengths: list[int],
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    rows = _length_coverage(lengths)
    plt.figure(figsize=(8, 5))
    if rows:
        x = [int(row["max_length"]) for row in rows]
        y = [100 * float(row["prompt_fraction"]) for row in rows]
        plt.step(x, y, where="post", color="#7b4fa3")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Prompts covered (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _coerce_points(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(item) for item in value]
    return []


def _analyze_dataset(
    dataset: Dataset,
    max_examples: int | None,
    prompt_tokenizer: Any | None = None,
) -> dict[str, Any]:
    if "texts" in dataset.column_names:
        dataset = dataset.select_columns(["texts"])
    question_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    question_sets: list[set[str]] = []
    answer_sets: list[set[str]] = []
    yes_no_answer_counts: Counter[str] = Counter()
    yes_no_prompt_word_lengths: list[int] = []
    yes_no_prompt_token_lengths: list[int] = []
    yes_no_prompt_word_counter: Counter[str] = Counter()
    yes_no_prompt_token_counter: Counter[int] = Counter()
    yes_no_prompt_word_sets: list[set[str]] = []
    yes_no_prompt_token_sets: list[set[int]] = []
    examples = 0
    qa_pairs = 0

    limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    for index in range(limit):
        examples += 1
        for question, answer in iter_qa_pairs(dataset[index]):
            q_tokens = normalize_tokens(question)
            a_tokens = normalize_tokens(answer)
            if not q_tokens and not a_tokens:
                continue
            qa_pairs += 1
            question_counter.update(q_tokens)
            answer_counter.update(a_tokens)
            question_lengths.append(len(q_tokens))
            answer_lengths.append(len(a_tokens))
            question_sets.append(set(q_tokens))
            answer_sets.append(set(a_tokens))
            label = yes_no_answer(answer)
            if label is not None:
                yes_no_answer_counts.update([label])
                yes_no_prompt_word_lengths.append(len(q_tokens))
                yes_no_prompt_word_counter.update(q_tokens)
                yes_no_prompt_word_sets.append(set(q_tokens))
                if prompt_tokenizer is not None:
                    prompt_token_ids = prompt_tokenizer(question, add_special_tokens=False)["input_ids"]
                    yes_no_prompt_token_counter.update(prompt_token_ids)
                    yes_no_prompt_token_lengths.append(len(prompt_token_ids))
                    yes_no_prompt_token_sets.append(set(prompt_token_ids))

    return {
        "examples": examples,
        "qa_pairs": qa_pairs,
        "questions": {
            "counter": question_counter,
            "lengths": question_lengths,
            "sets": question_sets,
        },
        "answers": {
            "counter": answer_counter,
            "lengths": answer_lengths,
            "sets": answer_sets,
        },
        "yes_no_prompts": {
            "answer_counts": yes_no_answer_counts,
            "word_lengths": yes_no_prompt_word_lengths,
            "token_lengths": yes_no_prompt_token_lengths,
            "word_counter": yes_no_prompt_word_counter,
            "token_counter": yes_no_prompt_token_counter,
            "word_sets": yes_no_prompt_word_sets,
            "token_sets": yes_no_prompt_token_sets,
        },
    }


def _iter_jsonl(path: Path, max_examples: int | None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_examples is not None and index >= max_examples:
                break
            yield json.loads(line)


def _analyze_jsonl(
    path: Path,
    max_examples: int | None,
    prompt_tokenizer: Any | None = None,
) -> dict[str, Any]:
    question_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    question_sets: list[set[str]] = []
    answer_sets: list[set[str]] = []
    yes_no_answer_counts: Counter[str] = Counter()
    yes_no_prompt_word_lengths: list[int] = []
    yes_no_prompt_token_lengths: list[int] = []
    yes_no_prompt_word_counter: Counter[str] = Counter()
    yes_no_prompt_token_counter: Counter[int] = Counter()
    yes_no_prompt_word_sets: list[set[str]] = []
    yes_no_prompt_token_sets: list[set[int]] = []
    examples = 0
    qa_pairs = 0

    for sample in _iter_jsonl(path, max_examples):
        examples += 1
        for question, answer in iter_qa_pairs(sample):
            q_tokens = normalize_tokens(question)
            a_tokens = normalize_tokens(answer)
            if not q_tokens and not a_tokens:
                continue
            qa_pairs += 1
            question_counter.update(q_tokens)
            answer_counter.update(a_tokens)
            question_lengths.append(len(q_tokens))
            answer_lengths.append(len(a_tokens))
            question_sets.append(set(q_tokens))
            answer_sets.append(set(a_tokens))
            label = yes_no_answer(answer)
            if label is not None:
                yes_no_answer_counts.update([label])
                yes_no_prompt_word_lengths.append(len(q_tokens))
                yes_no_prompt_word_counter.update(q_tokens)
                yes_no_prompt_word_sets.append(set(q_tokens))
                if prompt_tokenizer is not None:
                    prompt_token_ids = prompt_tokenizer(question, add_special_tokens=False)["input_ids"]
                    yes_no_prompt_token_counter.update(prompt_token_ids)
                    yes_no_prompt_token_lengths.append(len(prompt_token_ids))
                    yes_no_prompt_token_sets.append(set(prompt_token_ids))

    return {
        "examples": examples,
        "qa_pairs": qa_pairs,
        "questions": {
            "counter": question_counter,
            "lengths": question_lengths,
            "sets": question_sets,
        },
        "answers": {
            "counter": answer_counter,
            "lengths": answer_lengths,
            "sets": answer_sets,
        },
        "yes_no_prompts": {
            "answer_counts": yes_no_answer_counts,
            "word_lengths": yes_no_prompt_word_lengths,
            "token_lengths": yes_no_prompt_token_lengths,
            "word_counter": yes_no_prompt_word_counter,
            "token_counter": yes_no_prompt_token_counter,
            "word_sets": yes_no_prompt_word_sets,
            "token_sets": yes_no_prompt_token_sets,
        },
    }


def _analyze_parquet(
    path: Path,
    max_examples: int | None,
    prompt_tokenizer: Any | None = None,
) -> dict[str, Any]:
    question_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    question_sets: list[set[str]] = []
    answer_sets: list[set[str]] = []
    yes_no_answer_counts: Counter[str] = Counter()
    yes_no_prompt_word_lengths: list[int] = []
    yes_no_prompt_token_lengths: list[int] = []
    yes_no_prompt_word_counter: Counter[str] = Counter()
    yes_no_prompt_token_counter: Counter[int] = Counter()
    yes_no_prompt_word_sets: list[set[str]] = []
    yes_no_prompt_token_sets: list[set[int]] = []
    examples = 0
    qa_pairs = 0

    for parquet_file in sorted(path.glob("*.parquet")):
        table = pq.read_table(parquet_file, columns=["texts"])
        for row in table.to_pylist():
            if max_examples is not None and examples >= max_examples:
                break
            examples += 1
            for question, answer in iter_qa_pairs(row):
                q_tokens = normalize_tokens(question)
                a_tokens = normalize_tokens(answer)
                if not q_tokens and not a_tokens:
                    continue
                qa_pairs += 1
                question_counter.update(q_tokens)
                answer_counter.update(a_tokens)
                question_lengths.append(len(q_tokens))
                answer_lengths.append(len(a_tokens))
                question_sets.append(set(q_tokens))
                answer_sets.append(set(a_tokens))
                label = yes_no_answer(answer)
                if label is not None:
                    yes_no_answer_counts.update([label])
                    yes_no_prompt_word_lengths.append(len(q_tokens))
                    yes_no_prompt_word_counter.update(q_tokens)
                    yes_no_prompt_word_sets.append(set(q_tokens))
                    if prompt_tokenizer is not None:
                        prompt_token_ids = prompt_tokenizer(question, add_special_tokens=False)["input_ids"]
                        yes_no_prompt_token_counter.update(prompt_token_ids)
                        yes_no_prompt_token_lengths.append(len(prompt_token_ids))
                        yes_no_prompt_token_sets.append(set(prompt_token_ids))
        if max_examples is not None and examples >= max_examples:
            break

    return {
        "examples": examples,
        "qa_pairs": qa_pairs,
        "questions": {
            "counter": question_counter,
            "lengths": question_lengths,
            "sets": question_sets,
        },
        "answers": {
            "counter": answer_counter,
            "lengths": answer_lengths,
            "sets": answer_sets,
        },
        "yes_no_prompts": {
            "answer_counts": yes_no_answer_counts,
            "word_lengths": yes_no_prompt_word_lengths,
            "token_lengths": yes_no_prompt_token_lengths,
            "word_counter": yes_no_prompt_word_counter,
            "token_counter": yes_no_prompt_token_counter,
            "word_sets": yes_no_prompt_word_sets,
            "token_sets": yes_no_prompt_token_sets,
        },
    }


def _load_local_analysis(
    root: Path,
    subset: str,
    max_examples: int | None,
    prompt_tokenizer: Any | None = None,
) -> dict[str, Any] | None:
    hf_path = root / subset
    if hf_path.exists():
        return _analyze_dataset(load_from_disk(hf_path), max_examples, prompt_tokenizer)
    parquet_path = root / "_parquet" / subset
    if parquet_path.exists():
        return _analyze_parquet(parquet_path, max_examples, prompt_tokenizer)
    jsonl_path = root / f"{subset}.jsonl"
    if jsonl_path.exists():
        return _analyze_jsonl(jsonl_path, max_examples, prompt_tokenizer)
    return None


def _load_local_dataset(root: Path, subset: str) -> Dataset | None:
    hf_path = root / subset
    if hf_path.exists():
        return load_from_disk(hf_path)
    jsonl_path = root / f"{subset}.jsonl"
    if jsonl_path.exists():
        return load_dataset("json", data_files=str(jsonl_path), split="train")
    parquet_path = root / "_parquet" / subset
    if parquet_path.exists():
        files = sorted(str(path) for path in parquet_path.glob("*.parquet"))
        if files:
            return load_dataset("parquet", data_files=files, split="train")
    return None


def _make_yes_no_mapper(subset: str) -> Any:
    def mapper(sample: dict[str, Any], index: int) -> dict[str, Any]:
        texts = filter_yes_no_texts(sample.get("texts"))
        return {
            "texts": texts,
            "source_subset": subset,
            "original_index": index,
            "yes_no_count": len(texts),
        }

    return mapper


def build_cauldron_yes_no_dataset(cache_cfg: DictConfig, yes_no_cfg: DictConfig) -> Path:
    spec = load_cauldron_spec(str(cache_cfg.spec))
    subsets = _configured_subsets(yes_no_cfg, spec)
    if not bool(yes_no_cfg.include_optional_reasoning):
        subsets = [subset for subset in subsets if subset not in spec.optional_reasoning]

    source_root = Path(str(yes_no_cfg.source_root))
    output_root = Path(str(yes_no_cfg.output_root))
    report_path = Path(str(yes_no_cfg.report))
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    max_examples = yes_no_cfg.max_examples_per_subset
    max_examples_int = None if max_examples is None else int(max_examples)
    force = bool(yes_no_cfg.force)

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "build-cauldron-yes-no",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "subsets_requested": subsets,
        "subsets_missing": [],
        "subsets": {},
        "normalization": "answer is kept only when regex tokenization yields exactly one token: yes or no",
        "reasoning": (
            "This compact Cauldron derivative keeps only binary one-word answers "
            "for efficient VLM distillation and microcontroller-oriented experiments."
        ),
    }

    for subset in subsets:
        output = output_root / subset
        if output.exists() and not force:
            dataset = load_from_disk(output)
            yes_counts = Counter()
            for row in dataset:
                for message in row["texts"]:
                    yes_counts.update([message["assistant"]])
            summary["subsets"][subset] = {
                "path": str(output),
                "status": "already_present",
                "source_rows": None,
                "rows": len(dataset),
                "yes_no_pairs": sum(yes_counts.values()),
                "answers": dict(yes_counts),
            }
            print(
                f"{subset}: already present with {len(dataset)} rows / "
                f"{sum(yes_counts.values())} yes-no pairs",
                flush=True,
            )
            continue

        dataset = _load_local_dataset(source_root, subset)
        if dataset is None:
            if bool(yes_no_cfg.skip_missing):
                summary["subsets_missing"].append(subset)
                continue
            raise FileNotFoundError(f"Missing local Cauldron subset for {subset} under {source_root}")

        if max_examples_int is not None:
            dataset = dataset.select(range(min(len(dataset), max_examples_int)))

        if "texts" in dataset.column_names:
            dataset = dataset.select_columns(["images", "texts"] if "images" in dataset.column_names else ["texts"])

        mapped = dataset.map(
            _make_yes_no_mapper(subset),
            with_indices=True,
            desc=f"{subset}: keeping yes/no answers",
        )
        filtered = mapped.filter(
            lambda row: int(row["yes_no_count"]) > 0,
            desc=f"{subset}: dropping non yes/no rows",
        )

        tmp_output = output_root / f".{subset}.tmp"
        if tmp_output.exists():
            shutil.rmtree(tmp_output)
        filtered.save_to_disk(tmp_output)
        if output.exists():
            shutil.rmtree(output)
        tmp_output.rename(output)

        yes_counts = Counter()
        for row in filtered:
            for message in row["texts"]:
                yes_counts.update([message["assistant"]])
        summary["subsets"][subset] = {
            "path": str(output),
            "status": "written",
            "source_rows": len(dataset),
            "rows": len(filtered),
            "yes_no_pairs": sum(yes_counts.values()),
            "answers": dict(yes_counts),
        }
        print(
            f"{subset}: kept {len(filtered)} rows / {sum(yes_counts.values())} yes-no pairs",
            flush=True,
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def run_cauldron_eda(cache_cfg: DictConfig, eda_cfg: DictConfig) -> Path:
    spec = load_cauldron_spec(str(cache_cfg.spec))
    subsets = _configured_subsets(cache_cfg, spec)
    if not bool(eda_cfg.include_optional_reasoning):
        subsets = [subset for subset in subsets if subset not in spec.optional_reasoning]
    root = Path(str(eda_cfg.local_root))
    output_dir = Path(str(eda_cfg.output_dir))
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    max_examples = eda_cfg.max_examples_per_subset
    max_examples_int = None if max_examples is None else int(max_examples)
    coverage_points = _coerce_points(eda_cfg.coverage_points)
    top_k = int(eda_cfg.top_k_words)
    prompt_tokenizer_name = str(eda_cfg.prompt_coverage_tokenizer_name)
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        prompt_tokenizer_name,
        local_files_only=bool(eda_cfg.prompt_coverage_tokenizer_local_files_only),
        trust_remote_code=bool(eda_cfg.prompt_coverage_tokenizer_trust_remote_code),
    )

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "eda-cauldron",
        "dataset_root": str(root),
        "subsets_requested": subsets,
        "subsets_missing": [],
        "subsets": {},
        "combined": {},
        "normalization": "lowercase plus regex [a-z0-9]+(?:'[a-z0-9]+)?",
        "prompt_coverage_tokenizer": prompt_tokenizer_name,
        "reasoning": (
            "This EDA measures how compact the normalized question and answer "
            "vocabularies are before choosing tokenizer/vocabulary constraints "
            "for a microcontroller-sized VLM student."
        ),
    }

    combined_questions: Counter[str] = Counter()
    combined_answers: Counter[str] = Counter()
    combined_question_lengths: list[int] = []
    combined_answer_lengths: list[int] = []
    combined_question_sets: list[set[str]] = []
    combined_answer_sets: list[set[str]] = []
    combined_yes_no_answer_counts: Counter[str] = Counter()
    combined_yes_no_prompt_word_lengths: list[int] = []
    combined_yes_no_prompt_token_lengths: list[int] = []
    combined_yes_no_prompt_word_counter: Counter[str] = Counter()
    combined_yes_no_prompt_token_counter: Counter[int] = Counter()
    combined_yes_no_prompt_word_sets: list[set[str]] = []
    combined_yes_no_prompt_token_sets: list[set[int]] = []

    for subset in subsets:
        analysis = _load_local_analysis(root, subset, max_examples_int, prompt_tokenizer)
        if analysis is None:
            if bool(eda_cfg.skip_missing):
                summary["subsets_missing"].append(subset)
                continue
            raise FileNotFoundError(
                f"Missing local dataset {root / subset} or {root / (subset + '.jsonl')}. "
                "Run `uv run vlm-micro cache-cauldron` first."
            )
        q = analysis["questions"]
        a = analysis["answers"]
        yn = analysis["yes_no_prompts"]
        q_coverage = _coverage(q["counter"], q["sets"], coverage_points)
        a_coverage = _coverage(a["counter"], a["sets"], coverage_points)
        yn_word_coverage = _prompt_constructibility_coverage(
            yn["word_counter"],
            yn["word_sets"],
            coverage_points,
        )
        yn_token_coverage = _prompt_constructibility_coverage(
            yn["token_counter"],
            yn["token_sets"],
            coverage_points,
        )

        subset_dir = figures_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        _plot_length_hist(q["lengths"], f"{subset}: question length", subset_dir / "question_lengths.png")
        _plot_length_hist(a["lengths"], f"{subset}: answer length", subset_dir / "answer_lengths.png")
        _plot_top_words(q["counter"], f"{subset}: question words", subset_dir / "question_top_words.png", top_k)
        _plot_top_words(a["counter"], f"{subset}: answer words", subset_dir / "answer_top_words.png", top_k)
        _plot_normalized_word_frequencies(
            q["counter"],
            f"{subset}: query normalized word frequency",
            subset_dir / "query_word_frequency_normalized.png",
            top_k,
        )
        _plot_normalized_word_frequencies(
            a["counter"],
            f"{subset}: answer normalized word frequency",
            subset_dir / "answer_word_frequency_normalized.png",
            top_k,
        )
        _plot_coverage(q_coverage, f"{subset}: question coverage", subset_dir / "question_coverage.png")
        _plot_coverage(a_coverage, f"{subset}: answer coverage", subset_dir / "answer_coverage.png")
        _plot_length_coverage(
            yn["word_lengths"],
            f"{subset}: yes/no prompt word length coverage",
            "Normalized words in prompt",
            subset_dir / "yes_no_prompt_word_length_coverage.png",
        )
        _plot_length_coverage(
            yn["token_lengths"],
            f"{subset}: yes/no prompt token length coverage",
            f"{prompt_tokenizer_name} tokens in prompt",
            subset_dir / "yes_no_prompt_token_length_coverage.png",
        )
        _plot_prompt_constructibility_coverage(
            yn_word_coverage,
            f"{subset}: yes/no prompt word-set coverage",
            "Top normalized prompt words in set",
            subset_dir / "yes_no_prompt_word_coverage.png",
        )
        _plot_prompt_constructibility_coverage(
            yn_token_coverage,
            f"{subset}: yes/no prompt token-set coverage",
            f"Top {prompt_tokenizer_name} prompt token IDs in set",
            subset_dir / "yes_no_prompt_token_coverage.png",
        )

        summary["subsets"][subset] = {
            "examples": analysis["examples"],
            "qa_pairs": analysis["qa_pairs"],
            "questions": {
                "lengths": _length_hist(q["lengths"]),
                "vocab_size": len(q["counter"]),
                "total_tokens": q["counter"].total(),
                "top_words": q["counter"].most_common(top_k),
                "normalized_word_frequencies": _normalized_word_frequencies(q["counter"], top_k),
                "coverage": q_coverage,
            },
            "answers": {
                "lengths": _length_hist(a["lengths"]),
                "vocab_size": len(a["counter"]),
                "total_tokens": a["counter"].total(),
                "top_words": a["counter"].most_common(top_k),
                "normalized_word_frequencies": _normalized_word_frequencies(a["counter"], top_k),
                "coverage": a_coverage,
            },
            "yes_no_prompts": {
                "answers": dict(yn["answer_counts"]),
                "word_lengths": _length_hist(yn["word_lengths"]),
                "word_length_coverage": _length_coverage(yn["word_lengths"]),
                "word_coverage": yn_word_coverage,
                "token_lengths": _length_hist(yn["token_lengths"]),
                "token_length_coverage": _length_coverage(yn["token_lengths"]),
                "token_coverage": yn_token_coverage,
            },
        }

        combined_questions.update(q["counter"])
        combined_answers.update(a["counter"])
        combined_question_lengths.extend(q["lengths"])
        combined_answer_lengths.extend(a["lengths"])
        combined_question_sets.extend(q["sets"])
        combined_answer_sets.extend(a["sets"])
        combined_yes_no_answer_counts.update(yn["answer_counts"])
        combined_yes_no_prompt_word_lengths.extend(yn["word_lengths"])
        combined_yes_no_prompt_token_lengths.extend(yn["token_lengths"])
        combined_yes_no_prompt_word_counter.update(yn["word_counter"])
        combined_yes_no_prompt_token_counter.update(yn["token_counter"])
        combined_yes_no_prompt_word_sets.extend(yn["word_sets"])
        combined_yes_no_prompt_token_sets.extend(yn["token_sets"])

    combined_q_coverage = _coverage(combined_questions, combined_question_sets, coverage_points)
    combined_a_coverage = _coverage(combined_answers, combined_answer_sets, coverage_points)
    combined_yn_word_coverage = _prompt_constructibility_coverage(
        combined_yes_no_prompt_word_counter,
        combined_yes_no_prompt_word_sets,
        coverage_points,
    )
    combined_yn_token_coverage = _prompt_constructibility_coverage(
        combined_yes_no_prompt_token_counter,
        combined_yes_no_prompt_token_sets,
        coverage_points,
    )
    _plot_length_hist(combined_question_lengths, "combined: question length", figures_dir / "combined_question_lengths.png")
    _plot_length_hist(combined_answer_lengths, "combined: answer length", figures_dir / "combined_answer_lengths.png")
    _plot_top_words(combined_questions, "combined: question words", figures_dir / "combined_question_top_words.png", top_k)
    _plot_top_words(combined_answers, "combined: answer words", figures_dir / "combined_answer_top_words.png", top_k)
    _plot_normalized_word_frequencies(
        combined_questions,
        "combined: query normalized word frequency",
        figures_dir / "combined_query_word_frequency_normalized.png",
        top_k,
    )
    _plot_normalized_word_frequencies(
        combined_answers,
        "combined: answer normalized word frequency",
        figures_dir / "combined_answer_word_frequency_normalized.png",
        top_k,
    )
    _plot_coverage(combined_q_coverage, "combined: question coverage", figures_dir / "combined_question_coverage.png")
    _plot_coverage(combined_a_coverage, "combined: answer coverage", figures_dir / "combined_answer_coverage.png")
    _plot_length_coverage(
        combined_yes_no_prompt_word_lengths,
        "combined: yes/no prompt word length coverage",
        "Normalized words in prompt",
        figures_dir / "combined_yes_no_prompt_word_length_coverage.png",
    )
    _plot_length_coverage(
        combined_yes_no_prompt_token_lengths,
        "combined: yes/no prompt token length coverage",
        f"{prompt_tokenizer_name} tokens in prompt",
        figures_dir / "combined_yes_no_prompt_token_length_coverage.png",
    )
    _plot_prompt_constructibility_coverage(
        combined_yn_word_coverage,
        "combined: yes/no prompt word-set coverage",
        "Top normalized prompt words in set",
        figures_dir / "combined_yes_no_prompt_word_coverage.png",
    )
    _plot_prompt_constructibility_coverage(
        combined_yn_token_coverage,
        "combined: yes/no prompt token-set coverage",
        f"Top {prompt_tokenizer_name} prompt token IDs in set",
        figures_dir / "combined_yes_no_prompt_token_coverage.png",
    )

    summary["combined"] = {
        "questions": {
            "lengths": _length_hist(combined_question_lengths),
            "vocab_size": len(combined_questions),
            "total_tokens": combined_questions.total(),
            "top_words": combined_questions.most_common(top_k),
            "normalized_word_frequencies": _normalized_word_frequencies(combined_questions, top_k),
            "coverage": combined_q_coverage,
        },
        "answers": {
            "lengths": _length_hist(combined_answer_lengths),
            "vocab_size": len(combined_answers),
            "total_tokens": combined_answers.total(),
            "top_words": combined_answers.most_common(top_k),
            "normalized_word_frequencies": _normalized_word_frequencies(combined_answers, top_k),
            "coverage": combined_a_coverage,
        },
        "yes_no_prompts": {
            "answers": dict(combined_yes_no_answer_counts),
            "word_lengths": _length_hist(combined_yes_no_prompt_word_lengths),
            "word_length_coverage": _length_coverage(combined_yes_no_prompt_word_lengths),
            "word_coverage": combined_yn_word_coverage,
            "token_lengths": _length_hist(combined_yes_no_prompt_token_lengths),
            "token_length_coverage": _length_coverage(combined_yes_no_prompt_token_lengths),
            "token_coverage": combined_yn_token_coverage,
        },
    }

    output = output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
