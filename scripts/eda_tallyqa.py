from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vlm_micro.cauldron import (
    _coverage,
    _length_coverage,
    _length_hist,
    _normalized_word_frequencies,
    _plot_coverage,
    _plot_length_coverage,
    _plot_length_hist,
    _plot_normalized_word_frequencies,
    _plot_top_words,
    normalize_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local EDA for TallyQA questions.")
    parser.add_argument("--root", type=Path, default=Path("data/TallyQA/tallyqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/tallyqa_eda"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--coverage-points", type=int, nargs="*", default=[10, 25, 50, 100, 250, 500, 1000])
    parser.add_argument("--suffix-min-support", type=int, default=250)
    parser.add_argument("--suffix-min-children", type=int, default=25)
    parser.add_argument("--suffix-max-depth", type=int, default=8)
    parser.add_argument("--suffix-max-top-child-fraction", type=float, default=0.5)
    parser.add_argument(
        "--pruned-suffixes",
        type=Path,
        default=Path("artifacts/reports/tallyqa_eda/frontier_suffixes_pruned.txt"),
        help="Optional tab-separated pruned suffix list to use for template item filtering.",
    )
    return parser.parse_args()


def load_split(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def image_root(image: str) -> str:
    return image.split("/", 1)[0] if "/" in image else image


def question_prefix(tokens: list[str], n: int) -> str:
    return " ".join(tokens[:n])


def question_suffix(tokens: list[str], n: int) -> str:
    return " ".join(tokens[-n:])


def how_many_template(tokens: list[str], suffix_len: int) -> str | None:
    if len(tokens) <= 2 + suffix_len:
        return None
    if tokens[:2] != ["how", "many"]:
        return None
    suffix = " ".join(tokens[-suffix_len:])
    return f"how many ... {suffix}"


@dataclass
class TrieNode:
    token: str
    count: int = 0
    children: dict[str, "TrieNode"] = field(default_factory=dict)


def add_suffix_path(root: TrieNode, tokens: list[str]) -> None:
    root.count += 1
    node = root
    for token in reversed(tokens):
        node = node.children.setdefault(token, TrieNode(token))
        node.count += 1


def iter_trie_nodes(
    node: TrieNode,
    reversed_path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], TrieNode]]:
    rows = [(reversed_path, node)]
    for token, child in node.children.items():
        rows.extend(iter_trie_nodes(child, (*reversed_path, token)))
    return rows


def suffix_from_reversed_path(reversed_path: tuple[str, ...]) -> str:
    return " ".join(reversed(reversed_path))


def load_pruned_suffixes(path: Path) -> list[tuple[str, ...]]:
    suffixes: list[tuple[str, ...]] = []
    if not path.exists():
        return suffixes
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        suffix_text = parts[1] if len(parts) >= 2 else parts[0]
        tokens = tuple(normalize_tokens(suffix_text))
        if tokens:
            suffixes.append(tokens)
    return suffixes


def summarize_selected_suffixes(
    stripped_questions: list[list[str]],
    suffixes: list[tuple[str, ...]],
) -> list[dict[str, Any]]:
    total = len(stripped_questions)
    rows = []
    for suffix in suffixes:
        support = sum(
            1
            for tokens in stripped_questions
            if len(tokens) >= len(suffix) and tuple(tokens[-len(suffix) :]) == suffix
        )
        rows.append(
            {
                "suffix": " ".join(suffix),
                "tokens": list(suffix),
                "support": support,
                "frequency": support / total if total else 0.0,
                "depth": len(suffix),
            }
        )
    rows.sort(key=lambda row: int(row["support"]), reverse=True)
    return rows


def discover_covering_suffixes(
    stripped_questions: list[list[str]],
    min_support: int,
    min_children: int,
    max_depth: int,
    max_top_child_fraction: float,
) -> dict[str, Any]:
    root = TrieNode("<root>")
    for tokens in stripped_questions:
        if tokens:
            add_suffix_path(root, tokens)

    candidates: list[dict[str, Any]] = []
    for reversed_path, node in iter_trie_nodes(root):
        if not reversed_path:
            continue
        depth = len(reversed_path)
        if depth > max_depth:
            continue
        child_count = len(node.children)
        top_child_fraction = max((child.count for child in node.children.values()), default=0) / node.count
        if (
            node.count < min_support
            or child_count < min_children
            or top_child_fraction > max_top_child_fraction
        ):
            continue
        suffix = suffix_from_reversed_path(reversed_path)
        candidates.append(
            {
                "suffix": suffix,
                "tokens": list(reversed(reversed_path)),
                "support": node.count,
                "frequency": node.count / root.count if root.count else 0.0,
                "depth": depth,
                "preceding_word_types": child_count,
                "top_preceding_word_fraction": top_child_fraction,
                "top_preceding_words": [
                    {"word": child.token, "count": child.count}
                    for child in sorted(node.children.values(), key=lambda item: item.count, reverse=True)[:25]
                ],
            }
        )

    candidate_suffixes = {tuple(row["tokens"]) for row in candidates}
    candidate_by_tokens = {tuple(row["tokens"]): row for row in candidates}
    maximal: list[dict[str, Any]] = []
    for row in candidates:
        tokens = tuple(row["tokens"])
        has_deeper_candidate = False
        for other in candidate_suffixes:
            if len(other) <= len(tokens):
                continue
            if other[-len(tokens) :] == tokens:
                has_deeper_candidate = True
                break
        if not has_deeper_candidate:
            maximal.append(row)

    frontier: list[dict[str, Any]] = []
    for row in candidates:
        tokens = tuple(row["tokens"])
        has_shorter_candidate = False
        for length in range(1, len(tokens)):
            if tokens[-length:] in candidate_by_tokens:
                has_shorter_candidate = True
                break
        if not has_shorter_candidate:
            frontier.append(row)

    maximal.sort(key=lambda row: (int(row["support"]), int(row["preceding_word_types"]), int(row["depth"])), reverse=True)
    frontier.sort(key=lambda row: (int(row["support"]), int(row["preceding_word_types"]), int(row["depth"])), reverse=True)
    candidates.sort(key=lambda row: (int(row["support"]), int(row["preceding_word_types"]), int(row["depth"])), reverse=True)
    return {
        "min_support": min_support,
        "min_children": min_children,
        "max_depth": max_depth,
        "max_top_child_fraction": max_top_child_fraction,
        "trie_questions": len(stripped_questions),
        "trie_nodes": len(iter_trie_nodes(root)) - 1,
        "candidate_suffixes": candidates[:500],
        "frontier_covering_suffixes": frontier[:500],
        "maximal_covering_suffixes": maximal[:500],
        "_trie_root": root,
        "_selected_suffixes": {tuple(row["tokens"]) for row in frontier},
    }


def plot_suffix_trie(
    root: TrieNode,
    selected_suffixes: set[tuple[str, ...]],
    output: Path,
    max_depth: int = 6,
    top_children: int = 8,
    min_support: int = 250,
) -> None:
    nodes: list[dict[str, Any]] = []
    edges: list[tuple[int, int]] = []

    def add_node(node: TrieNode, reversed_path: tuple[str, ...], depth: int, parent: int | None) -> None:
        node_id = len(nodes)
        suffix_tokens = tuple(reversed(reversed_path))
        parent_is_beyond = bool(nodes[parent]["beyond_filtering"]) if parent is not None else False
        beyond_filtering = parent_is_beyond or (
            parent is not None and bool(nodes[parent]["selected"])
        )
        label = "root" if not reversed_path else node.token
        if suffix_tokens in selected_suffixes:
            label = f"* {label}"
        nodes.append(
            {
                "label": label,
                "count": node.count,
                "children": len(node.children),
                "depth": depth,
                "selected": suffix_tokens in selected_suffixes,
                "beyond_filtering": beyond_filtering,
            }
        )
        if parent is not None:
            edges.append((parent, node_id))
        if depth >= max_depth:
            return
        children = [
            child
            for child in sorted(node.children.values(), key=lambda item: item.count, reverse=True)
            if child.count >= min_support
        ][:top_children]
        for child in children:
            add_node(child, (*reversed_path, child.token), depth + 1, node_id)

    add_node(root, (), 0, None)
    levels: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        levels[int(node["depth"])].append(index)

    positions: dict[int, tuple[float, float]] = {}
    for depth, level_nodes in levels.items():
        for offset, node_id in enumerate(level_nodes):
            positions[node_id] = (depth, -offset)

    height = max(6, max(len(level) for level in levels.values()) * 0.34)
    width = max(10, (max(levels) + 1) * 2.0)
    plt.figure(figsize=(width, height))
    ax = plt.gca()
    for parent, child in edges:
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        color = "#c77c2b" if nodes[child]["beyond_filtering"] else "#9aa0a6"
        linewidth = 1.2 if nodes[child]["beyond_filtering"] else 0.8
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth)
    for node_id, node in enumerate(nodes):
        x, y = positions[node_id]
        color = "#d95f02" if node["selected"] else "#4f8f66"
        edgecolor = "#c77c2b" if node["beyond_filtering"] else "#2f3a3f"
        linewidth = 2.0 if node["beyond_filtering"] else 0.7
        ax.scatter(
            [x],
            [y],
            s=95,
            color=color,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=3,
        )
        ax.text(
            x + 0.05,
            y,
            f"{node['label']}\n{node['count']} / {node['children']}",
            va="center",
            fontsize=7,
        )
    ax.set_title("Reversed suffix trie after stripping 'how many' (label: token, count / children)")
    selected_handle = ax.scatter([], [], s=95, color="#d95f02", edgecolors="#2f3a3f", linewidths=0.7)
    active_handle = ax.scatter([], [], s=95, color="#4f8f66", edgecolors="#2f3a3f", linewidths=0.7)
    beyond_handle = ax.scatter([], [], s=95, color="#4f8f66", edgecolors="#c77c2b", linewidths=2.0)
    active_edge = plt.Line2D([0], [0], color="#9aa0a6", linewidth=0.8)
    beyond_edge = plt.Line2D([0], [0], color="#c77c2b", linewidth=1.2)
    ax.legend(
        [selected_handle, active_handle, beyond_handle, active_edge, beyond_edge],
        [
            "selected suffix frontier",
            "candidate trie context",
            "beyond filtering frontier",
            "context edge",
            "beyond-filter edge",
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_suffix_frequency_rows(rows: list[dict[str, Any]], title: str, output: Path, top_k: int) -> None:
    shown = rows[:top_k]
    labels = [str(row["suffix"]) for row in shown]
    values = [100 * float(row["frequency"]) for row in shown]
    plt.figure(figsize=(max(10, math.ceil(len(shown) * 0.3)), 5))
    plt.bar(labels, values, color="#8a5f2d")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Questions after stripping 'how many' (%)")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def match_frontier_suffix(tokens: list[str], suffixes: list[tuple[str, ...]]) -> tuple[str, tuple[str, ...]] | None:
    for suffix in suffixes:
        if len(tokens) <= len(suffix):
            continue
        if tuple(tokens[-len(suffix) :]) == suffix:
            item_tokens = tokens[: -len(suffix)]
            if 1 <= len(item_tokens) <= 2:
                return " ".join(item_tokens), suffix
    return None


def analyze_template_items(
    stripped_questions: list[list[str]],
    suffix_rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    item_output_k = max(top_k, 200)
    suffixes = [
        tuple(str(token) for token in row["tokens"])
        for row in sorted(suffix_rows, key=lambda item: int(item["depth"]), reverse=True)
    ]
    item_counter: Counter[str] = Counter()
    suffix_counter: Counter[str] = Counter()
    item_length_counter: Counter[int] = Counter()
    matched = 0
    for tokens in stripped_questions:
        match = match_frontier_suffix(tokens, suffixes)
        if match is None:
            continue
        item, suffix = match
        matched += 1
        item_counter.update([item])
        suffix_counter.update([" ".join(suffix)])
        item_length_counter.update([len(item.split())])

    total = len(stripped_questions)
    return {
        "matched_questions": matched,
        "candidate_questions": total,
        "matched_fraction": matched / total if total else 0.0,
        "unique_items": len(item_counter),
        "item_lengths": dict(sorted(item_length_counter.items())),
        "top_items": counter_rows(item_counter, item_output_k),
        "top_item_frequencies": [
            {"value": item, "count": count, "frequency": count / matched if matched else 0.0}
            for item, count in item_counter.most_common(item_output_k)
        ],
        "top_matched_suffixes": counter_rows(suffix_counter, top_k),
        "_item_counter": item_counter,
    }


def add_bar_plot(rows: list[tuple[Any, int]], title: str, ylabel: str, output: Path) -> None:
    labels = [str(key) for key, _ in rows]
    values = [count for _, count in rows]
    plt.figure(figsize=(max(10, math.ceil(len(rows) * 0.24)), 5))
    plt.bar(labels, values, color="#5f7d8c")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def add_prevalence_plot(rows: list[dict[str, Any]], title: str, output: Path, denominator: int) -> None:
    labels = [str(row["value"]) for row in rows]
    values = [100 * float(row["count"]) / denominator for row in rows] if denominator else []
    plt.figure(figsize=(max(10, math.ceil(len(rows) * 0.28)), 5))
    plt.bar(labels, values, color="#6f6a9e")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Questions (%)")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def counter_rows(counter: Counter[Any], top_k: int) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(top_k)]


def analyze_rows(
    rows: list[dict[str, Any]],
    split: str,
    top_k: int,
    coverage_points: list[int],
    suffix_min_support: int,
    suffix_min_children: int,
    suffix_max_depth: int,
    suffix_max_top_child_fraction: float,
    filter_suffix_tokens: list[tuple[str, ...]] | None,
    filter_suffix_source: str,
) -> dict[str, Any]:
    question_counter: Counter[str] = Counter()
    answer_counter: Counter[int] = Counter()
    data_source_counter: Counter[str] = Counter()
    image_root_counter: Counter[str] = Counter()
    prefix_counters: dict[int, Counter[str]] = defaultdict(Counter)
    suffix_counters: dict[int, Counter[str]] = defaultdict(Counter)
    how_many_suffix_counters: dict[int, Counter[str]] = defaultdict(Counter)
    how_many_template_counters: dict[int, Counter[str]] = defaultdict(Counter)
    how_many_next_word_counter: Counter[str] = Counter()
    normalized_question_counter: Counter[str] = Counter()
    question_lengths: list[int] = []
    question_sets: list[set[str]] = []
    per_image_counter: Counter[str] = Counter()
    missing_by_field: Counter[str] = Counter()
    non_integer_answers = 0
    is_simple_counter: Counter[str] = Counter()
    stripped_how_many_questions: list[list[str]] = []

    required = {"image", "answer", "data_source", "question", "image_id", "question_id"}
    for row in rows:
        missing_by_field.update(sorted(field for field in required if field not in row))
        question = row.get("question")
        answer = row.get("answer")
        image = row.get("image")
        data_source = row.get("data_source")
        tokens = normalize_tokens(question) if isinstance(question, str) else []
        if tokens:
            question_counter.update(tokens)
            question_lengths.append(len(tokens))
            question_sets.append(set(tokens))
            normalized_question_counter.update([" ".join(tokens)])
            for n in range(1, min(6, len(tokens)) + 1):
                prefix_counters[n].update([question_prefix(tokens, n)])
                suffix_counters[n].update([question_suffix(tokens, n)])
            if tokens[:2] == ["how", "many"]:
                stripped_how_many_questions.append(tokens[2:])
                if len(tokens) > 2:
                    how_many_next_word_counter.update([tokens[2]])
                for n in range(1, min(6, len(tokens) - 2) + 1):
                    how_many_suffix_counters[n].update([question_suffix(tokens, n)])
                    template = how_many_template(tokens, n)
                    if template is not None:
                        how_many_template_counters[n].update([template])
        if isinstance(answer, int):
            answer_counter.update([answer])
        else:
            non_integer_answers += 1
        if isinstance(image, str):
            image_root_counter.update([image_root(image)])
            per_image_counter.update([image])
        if isinstance(data_source, str):
            data_source_counter.update([data_source])
        if "issimple" in row:
            is_simple_counter.update([str(bool(row["issimple"]))])

    total = len(rows)
    how_many_questions = sum(count for text, count in prefix_counters[2].items() if text == "how many")
    unique_images = len(per_image_counter)

    suffix_analysis = discover_covering_suffixes(
        stripped_how_many_questions,
        min_support=suffix_min_support,
        min_children=suffix_min_children,
        max_depth=suffix_max_depth,
        max_top_child_fraction=suffix_max_top_child_fraction,
    )
    if filter_suffix_tokens:
        selected_filter_suffixes = summarize_selected_suffixes(
            stripped_how_many_questions,
            filter_suffix_tokens,
        )
        selected_filter_suffix_set = set(filter_suffix_tokens)
    else:
        selected_filter_suffixes = suffix_analysis["frontier_covering_suffixes"]
        selected_filter_suffix_set = suffix_analysis["_selected_suffixes"]
    template_item_analysis = analyze_template_items(
        stripped_how_many_questions,
        selected_filter_suffixes,
        top_k,
    )

    return {
        "split": split,
        "rows": total,
        "unique_questions": len(normalized_question_counter),
        "unique_images": unique_images,
        "questions_per_image": _length_hist(list(per_image_counter.values())),
        "duplicate_normalized_questions": [
            {"question": question, "count": count}
            for question, count in normalized_question_counter.most_common(top_k)
            if count > 1
        ],
        "missing_fields": dict(missing_by_field),
        "non_integer_answers": non_integer_answers,
        "data_sources": counter_rows(data_source_counter, top_k),
        "image_roots": counter_rows(image_root_counter, top_k),
        "issimple": dict(is_simple_counter),
        "answers": {
            "counts": counter_rows(answer_counter, top_k),
            "min": min(answer_counter) if answer_counter else None,
            "max": max(answer_counter) if answer_counter else None,
            "mean": sum(answer * count for answer, count in answer_counter.items()) / answer_counter.total()
            if answer_counter
            else None,
        },
        "questions": {
            "how_many_fraction": how_many_questions / total if total else 0.0,
            "lengths": _length_hist(question_lengths),
            "length_coverage": _length_coverage(question_lengths),
            "vocab_size": len(question_counter),
            "total_tokens": question_counter.total(),
            "top_words": question_counter.most_common(top_k),
            "normalized_word_frequencies": _normalized_word_frequencies(question_counter, top_k),
            "coverage": _coverage(question_counter, question_sets, coverage_points),
        },
        "normalized_prefixes": {
            str(n): counter_rows(counter, top_k) for n, counter in sorted(prefix_counters.items())
        },
        "normalized_suffixes": {
            str(n): counter_rows(counter, top_k) for n, counter in sorted(suffix_counters.items())
        },
        "how_many_suffixes": {
            str(n): counter_rows(counter, top_k) for n, counter in sorted(how_many_suffix_counters.items())
        },
        "how_many_templates": {
            str(n): counter_rows(counter, top_k) for n, counter in sorted(how_many_template_counters.items())
        },
        "how_many_next_words": counter_rows(how_many_next_word_counter, 1000),
        "suffix_trie": {
            **{key: value for key, value in suffix_analysis.items() if not key.startswith("_")},
            "filter_suffix_source": filter_suffix_source,
            "selected_filter_suffixes": selected_filter_suffixes,
        },
        "template_items": {
            key: value for key, value in template_item_analysis.items() if not key.startswith("_")
        },
        "_plot_data": {
            "question_counter": question_counter,
            "answer_counter": answer_counter,
            "data_source_counter": data_source_counter,
            "how_many_next_word_counter": how_many_next_word_counter,
            "suffix_trie_root": suffix_analysis["_trie_root"],
            "selected_suffixes": selected_filter_suffix_set,
            "template_item_counter": template_item_analysis["_item_counter"],
            "question_lengths": question_lengths,
            "question_coverage": _coverage(question_counter, question_sets, coverage_points),
        },
    }


def write_split_plots(summary: dict[str, Any], output_dir: Path, top_k: int) -> None:
    split = str(summary["split"])
    plot_data = summary["_plot_data"]
    rows = int(summary["rows"])
    split_dir = output_dir / "figures" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    _plot_length_hist(plot_data["question_lengths"], f"{split}: question length", split_dir / "question_lengths.png")
    _plot_length_coverage(
        plot_data["question_lengths"],
        f"{split}: question length coverage",
        "Normalized words in question",
        split_dir / "question_length_coverage.png",
    )
    _plot_top_words(plot_data["question_counter"], f"{split}: question words", split_dir / "question_top_words.png", top_k)
    _plot_normalized_word_frequencies(
        plot_data["question_counter"],
        f"{split}: normalized question word frequency",
        split_dir / "question_word_frequency_normalized.png",
        top_k,
    )
    _plot_coverage(plot_data["question_coverage"], f"{split}: question coverage", split_dir / "question_coverage.png")
    add_bar_plot(
        plot_data["answer_counter"].most_common(top_k),
        f"{split}: answer distribution",
        "Questions",
        split_dir / "answer_distribution.png",
    )
    add_bar_plot(
        plot_data["data_source_counter"].most_common(top_k),
        f"{split}: data source distribution",
        "Questions",
        split_dir / "data_sources.png",
    )
    add_bar_plot(
        plot_data["how_many_next_word_counter"].most_common(100),
        f"{split}: words immediately after 'how many' ranks 1-100",
        "Questions",
        split_dir / "how_many_next_words_rank_001_100.png",
    )
    add_bar_plot(
        plot_data["how_many_next_word_counter"].most_common(200)[100:200],
        f"{split}: words immediately after 'how many' ranks 101-200",
        "Questions",
        split_dir / "how_many_next_words_rank_101_200.png",
    )
    plot_suffix_trie(
        plot_data["suffix_trie_root"],
        plot_data["selected_suffixes"],
        split_dir / "suffix_trie_top_branches.png",
        min_support=max(50, int(summary["suffix_trie"]["min_support"])),
    )
    plot_suffix_frequency_rows(
        summary["suffix_trie"]["selected_filter_suffixes"],
        f"{split}: selected filter suffix normalized frequencies",
        split_dir / "frontier_suffix_normalized_frequencies.png",
        top_k,
    )
    add_bar_plot(
        plot_data["template_item_counter"].most_common(top_k),
        f"{split}: matched item distribution",
        "Matched questions",
        split_dir / "template_item_distribution.png",
    )
    add_bar_plot(
        plot_data["template_item_counter"].most_common(100),
        f"{split}: matched items ranks 1-100",
        "Matched questions",
        split_dir / "template_items_rank_001_100.png",
    )
    add_bar_plot(
        plot_data["template_item_counter"].most_common(200)[100:200],
        f"{split}: matched items ranks 101-200",
        "Matched questions",
        split_dir / "template_items_rank_101_200.png",
    )
    _plot_normalized_word_frequencies(
        plot_data["template_item_counter"],
        f"{split}: matched item normalized frequency",
        split_dir / "template_item_frequency_normalized.png",
        top_k,
    )
    for key in ("2", "3", "4"):
        add_prevalence_plot(
            summary["normalized_prefixes"][key],
            f"{split}: normalized {key}-word prefix prevalence",
            split_dir / f"prefix_{key}_word_prevalence.png",
            rows,
        )
        add_prevalence_plot(
            summary["normalized_suffixes"][key],
            f"{split}: normalized {key}-word suffix prevalence",
            split_dir / f"suffix_{key}_word_prevalence.png",
            rows,
        )
        add_prevalence_plot(
            summary["how_many_templates"][key],
            f"{split}: how many ... {key}-word suffix prevalence",
            split_dir / f"how_many_template_{key}_word_suffix_prevalence.png",
            rows,
        )


def write_item_top200(summary: dict[str, Any], output_dir: Path) -> Path:
    split = str(summary["split"])
    output = output_dir / f"template_items_top200_{split}.txt"
    rows = summary["template_items"]["top_item_frequencies"][:200]
    output.write_text(
        "\n".join(
            f"{index:03d}\t{row['value']}\t{row['count']}\t{row['frequency']:.8f}"
            for index, row in enumerate(rows, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def strip_plot_data(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "_plot_data"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pruned_suffixes = load_pruned_suffixes(args.pruned_suffixes)
    filter_suffix_source = str(args.pruned_suffixes) if pruned_suffixes else "auto_frontier_covering_suffixes"

    split_rows = {
        "train": load_split(args.root / "train.json"),
        "test": load_split(args.root / "test.json"),
    }
    combined_rows = [row for rows in split_rows.values() for row in rows]
    summaries = {
        split: analyze_rows(
            rows,
            split,
            args.top_k,
            args.coverage_points,
            args.suffix_min_support,
            args.suffix_min_children,
            args.suffix_max_depth,
            args.suffix_max_top_child_fraction,
            pruned_suffixes or None,
            filter_suffix_source,
        )
        for split, rows in split_rows.items()
    }
    summaries["combined"] = analyze_rows(
        combined_rows,
        "combined",
        args.top_k,
        args.coverage_points,
        args.suffix_min_support,
        args.suffix_min_children,
        args.suffix_max_depth,
        args.suffix_max_top_child_fraction,
        pruned_suffixes or None,
        filter_suffix_source,
    )

    for summary in summaries.values():
        write_split_plots(summary, args.output_dir, args.top_k)
    item_outputs = [write_item_top200(summary, args.output_dir) for summary in summaries.values()]

    train_question_ids = {row.get("question_id") for row in split_rows["train"]}
    test_question_ids = {row.get("question_id") for row in split_rows["test"]}
    train_image_ids = {row.get("image_id") for row in split_rows["train"]}
    test_image_ids = {row.get("image_id") for row in split_rows["test"]}
    train_questions = {" ".join(normalize_tokens(row.get("question", ""))) for row in split_rows["train"]}
    test_questions = {" ".join(normalize_tokens(row.get("question", ""))) for row in split_rows["test"]}

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "uv run python scripts/eda_tallyqa.py",
        "dataset_root": str(args.root),
        "splits": {split: strip_plot_data(summary) for split, summary in summaries.items() if split != "combined"},
        "combined": strip_plot_data(summaries["combined"]),
        "cross_split_overlap": {
            "question_ids": len(train_question_ids & test_question_ids),
            "image_ids": len(train_image_ids & test_image_ids),
            "normalized_questions": len(train_questions & test_questions),
        },
        "normalization": "lowercase plus regex [a-z0-9]+(?:'[a-z0-9]+)?",
        "prefix_suffix_method": (
            "normalized_prefixes and normalized_suffixes count first/last n normalized tokens; "
            "how_many_templates count forms like 'how many ... are there' by preserving the first "
            "two tokens and the last n tokens."
        ),
        "suffix_trie_method": (
            "Questions beginning with normalized 'how many' are stripped to the remaining tokens. "
            "A reversed trie is built from those stripped questions. Candidate covering suffixes "
            "are trie nodes with support >= --suffix-min-support and distinct preceding tokens "
            ">= --suffix-min-children, excluding nodes where one preceding token dominates more "
            "than --suffix-max-top-child-fraction. Frontier covering suffixes choose the first "
            "accepted high-branching node along each reversed-trie path."
        ),
        "template_item_filter_method": (
            "For each normalized question beginning with 'how many', strip that prefix, match the "
            "longest selected filter suffix, and keep only prompts where the remaining leading "
            "item span is 1 or 2 normalized words. When --pruned-suffixes exists, that pruned "
            "suffix set is used instead of the automatically discovered frontier."
        ),
        "reasoning": (
            "This EDA checks whether TallyQA can support a compact tallying target with minimal "
            "language processing by measuring question regularity, answer range, source mix, and "
            "common normalized prefixes/suffixes."
        ),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    suffix_list = args.output_dir / "frontier_suffixes.txt"
    combined_suffixes = report["combined"]["suffix_trie"]["frontier_covering_suffixes"]
    suffix_list.write_text(
        "\n".join(
            f"{index:03d}\t{row['suffix']}\t{row['support']}\t{row['frequency']:.8f}"
            for index, row in enumerate(combined_suffixes, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    selected_suffix_list = args.output_dir / "filter_suffixes_used.txt"
    selected_suffixes = report["combined"]["suffix_trie"]["selected_filter_suffixes"]
    selected_suffix_list.write_text(
        "\n".join(
            f"{index:03d}\t{row['suffix']}\t{row['support']}\t{row['frequency']:.8f}"
            for index, row in enumerate(selected_suffixes, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(suffix_list)
    print(selected_suffix_list)
    for item_output in item_outputs:
        print(item_output)


if __name__ == "__main__":
    main()
