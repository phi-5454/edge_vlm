#!/usr/bin/env python3
"""Cache Coral Micro on-device TallyQA predictions over a serial benchmark app."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter, sleep
from typing import Any

import numpy as np
from tqdm import tqdm

from scripts.cache_smolvlm_tallyqa_teacher import (
    Uint8ImageStore,
    load_examples,
    load_metadata,
)
from scripts.cache_tflite_tallyqa_teacher import (
    aggregate_stats,
    candidate_scores,
    normalized_answer,
    numeric_metrics,
    softmax,
    update_stats,
)


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/coral_micro_tallyqa_benchmark_smoke.jsonl")
DEFAULT_PROMPT_LOOKUP_MANIFEST = Path(
    "artifacts/exports/coral/prompt_embedding_lookup/prompt_embedding_lookup_manifest.json"
)
PREFIX_READY = "VLM_MICRO_READY "
PREFIX_RESULT = "VLM_MICRO_RESULT "
PREFIX_ERROR = "VLM_MICRO_ERROR "
PREFIX_INPUT = "VLM_MICRO_INPUT "
PREFIX_RX_READY = "VLM_MICRO_RX_READY "
PREFIX_EVENT = "VLM_MICRO_EVENT "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial device, for example /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--serial-timeout-s",
        type=float,
        default=30.0,
        help="Default serial read/write timeout. Phase-specific timeouts below override it.",
    )
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=None,
        help="Timeout while waiting for board_ready. Defaults to --serial-timeout-s.",
    )
    parser.add_argument(
        "--rx-ready-timeout-s",
        type=float,
        default=5.0,
        help="Timeout after sending a JSON header while waiting for board RX_READY.",
    )
    parser.add_argument(
        "--result-timeout-s",
        type=float,
        default=30.0,
        help="Timeout after sending image bytes while waiting for board RESULT.",
    )
    parser.add_argument("--payload-chunk-size", type=int, default=512)
    parser.add_argument("--payload-chunk-delay-s", type=float, default=0.0005)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="coral_micro_tallyqa_benchmark")
    parser.add_argument(
        "--prompt-lookup-manifest",
        type=Path,
        default=DEFAULT_PROMPT_LOOKUP_MANIFEST,
        help=(
            "Prompt lookup manifest matching the firmware-staged lookup header. "
            "Required when the board model has a prompt-embedding input."
        ),
    )
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=5)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--output-tensor-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--debug-protocol",
        action="store_true",
        help="Print host-side serial protocol milestones for each example.",
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=Path("artifacts/profiles/coral/tallyqa_benchmark_serial_raw.log"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    if args.max_examples is not None and args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")
    if args.collapse_at < args.answer_min or args.collapse_at > args.answer_max:
        raise ValueError("--collapse-at must be inside [answer-min, answer-max]")
    if args.payload_chunk_size <= 0:
        raise ValueError("--payload-chunk-size must be positive")
    if args.payload_chunk_delay_s < 0:
        raise ValueError("--payload-chunk-delay-s must be non-negative")
    for name in ("serial_timeout_s", "ready_timeout_s", "rx_ready_timeout_s", "result_timeout_s"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def selected_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    validate_args(args)
    stop = len(examples) if args.end_index is None else min(len(examples), args.end_index)
    selected: list[int] = []
    for index in range(args.start_index, stop):
        if index % args.shard_count != args.shard_index:
            continue
        selected.append(index)
        if args.max_examples is not None and len(selected) >= args.max_examples:
            break
    return selected


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["dataset_index"]))
            except (KeyError, json.JSONDecodeError, TypeError, ValueError):
                continue
    return completed


def prefixed_payload(line: str) -> tuple[str, dict[str, Any]] | None:
    for event, prefix in (
        ("ready", PREFIX_READY),
        ("rx_ready", PREFIX_RX_READY),
        ("result", PREFIX_RESULT),
        ("error", PREFIX_ERROR),
        ("event", PREFIX_EVENT),
    ):
        if line.startswith(prefix):
            payload = line[len(prefix) :]
            try:
                return event, json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed Coral Micro {event} JSON after prefix {prefix!r}: {payload!r}"
                ) from exc
    return None


def read_prefixed(
    ser: Any,
    raw_handle: Any,
    recent_lines: deque[str] | None = None,
    phase: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str, dict[str, Any]]:
    previous_timeout = ser.timeout
    if timeout_s is not None:
        ser.timeout = timeout_s
    try:
        while True:
            raw = ser.readline()
            if not raw:
                suffix = f" during {phase}" if phase else ""
                raise TimeoutError(f"Timed out waiting for Coral Micro serial output{suffix}.")
            line = raw.decode("utf-8", errors="replace").strip()
            if recent_lines is not None:
                recent_lines.append(line)
            raw_handle.write(line + "\n")
            raw_handle.flush()
            parsed = prefixed_payload(line)
            if parsed is not None:
                return parsed
            print(line)
    finally:
        if timeout_s is not None:
            ser.timeout = previous_timeout


def wait_ready(
    ser: Any,
    raw_handle: Any,
    recent_lines: deque[str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    while True:
        event, payload = read_prefixed(
            ser,
            raw_handle,
            recent_lines,
            "waiting for board_ready",
            timeout_s,
        )
        if event == "ready":
            return payload
        if event == "error":
            print(json.dumps({"board_error_before_ready": payload}, sort_keys=True), file=sys.stderr)


def serial_port_summary(serial_module: Any) -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    ports: list[str] = []
    for port in list_ports.comports():
        ports.append(
            " ".join(
                str(value)
                for value in (
                    port.device,
                    port.description,
                    f"VID:PID={port.vid:04X}:{port.pid:04X}"
                    if port.vid is not None and port.pid is not None
                    else None,
                    f"SER={port.serial_number}" if port.serial_number else None,
                    f"LOCATION={port.location}" if port.location else None,
                )
                if value
            )
        )
    return ports


def raw_log_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def serial_timeout_message(
    args: argparse.Namespace,
    serial_module: Any,
    exc: TimeoutError,
    recent_lines: deque[str] | None = None,
    dataset_index: int | None = None,
) -> str:
    ports = serial_port_summary(serial_module)
    raw_size = raw_log_size(args.raw_log)
    context = []
    if dataset_index is not None:
        context.append(f"Current dataset index: {dataset_index}")
    if recent_lines:
        context.append("Recent serial lines:")
        context.extend(f"  {line}" for line in recent_lines)
    return "\n".join(
        [
            f"{exc}",
            f"No Coral Micro protocol output was received from {args.port} "
            f"within {args.serial_timeout_s:.1f}s.",
            f"Raw serial log: {args.raw_log} (size={raw_size} bytes)",
            "Available serial ports:",
            *(f"  - {port}" for port in ports),
            *context,
            "Checks: confirm the benchmark app flashed successfully, press board reset, "
            "or rerun with a longer --serial-timeout-s and the explicit --port.",
        ]
    )


def image_to_hwc_uint8(image: Any, ready: dict[str, Any]) -> np.ndarray:
    input_info = ready["input"]
    shape = [int(value) for value in input_info["shape"]]
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
        raise ValueError(f"Expected board input shape [1,H,W,3], got {shape}")
    resized = image.convert("RGB").resize((shape[2], shape[1]))
    return np.asarray(resized, dtype=np.uint8)


def quantize_for_board_input(image: Any, ready: dict[str, Any]) -> bytes:
    input_info = ready["input"]
    array = image_to_hwc_uint8(image, ready)
    input_type = str(input_info["type"])
    if input_type == "uint8":
        return array.tobytes()
    if input_type == "int8":
        scale = float(input_info["scale"])
        zero_point = int(input_info["zero_point"])
        if not scale:
            raise ValueError("Board reported int8 input with zero quantization scale.")
        preprocessed = array.astype(np.float32) / 127.5 - 1.0
        quantized = np.rint(preprocessed / scale + zero_point)
        return np.clip(quantized, -128, 127).astype(np.int8).tobytes()
    raise ValueError(f"Unsupported board input type: {input_type}")


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def load_prompt_lookup(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for index, entry in enumerate(manifest.get("entries", [])):
        prompt = normalize_prompt(str(entry.get("prompt", "")))
        if prompt and prompt not in mapping:
            mapping[prompt] = index
    if not mapping:
        raise ValueError(f"No prompt entries found in {path}")
    return mapping, manifest


def prompt_id_for_row(row: dict[str, Any], prompt_lookup: dict[str, int]) -> int:
    candidates = [
        row.get("student_prompt"),
        row.get("item"),
        row.get("teacher_prompt_clean"),
        row.get("teacher_prompt"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = normalize_prompt(str(candidate))
        if normalized in prompt_lookup:
            return int(prompt_lookup[normalized])
    prompt = str(row.get("student_prompt", row.get("item", "")))
    raise KeyError(
        f"Prompt {prompt!r} is not present in the prompt lookup manifest. "
        "Regenerate artifacts/exports/coral/prompt_embedding_lookup or pass the "
        "matching --prompt-lookup-manifest."
    )


def send_header(
    ser: Any,
    dataset_index: int,
    image_index: int,
    payload_size: int,
    prompt_id: int | None = None,
) -> None:
    header = {
        "dataset_index": int(dataset_index),
        "image_index": int(image_index),
        "bytes": int(payload_size),
    }
    if prompt_id is not None:
        header["prompt_id"] = int(prompt_id)
    ser.write((PREFIX_INPUT + json.dumps(header, separators=(",", ":")) + "\n").encode("utf-8"))
    ser.flush()


def send_payload(ser: Any, payload: bytes, chunk_size: int, chunk_delay_s: float) -> None:
    for offset in range(0, len(payload), chunk_size):
        ser.write(payload[offset : offset + chunk_size])
        if chunk_delay_s:
            sleep(chunk_delay_s)
    ser.flush()


def debug_protocol(args: argparse.Namespace, message: str, **fields: Any) -> None:
    if not args.debug_protocol:
        return
    print(json.dumps({"event": f"host_{message}", **fields}, sort_keys=True), flush=True)


def result_logits(result: dict[str, Any], output_tensor_index: int) -> np.ndarray:
    outputs = result.get("outputs", [])
    if output_tensor_index < 0 or output_tensor_index >= len(outputs):
        raise ValueError(
            f"Requested output tensor {output_tensor_index}, but result has {len(outputs)} outputs."
        )
    output = outputs[output_tensor_index]
    values = np.asarray(output["values"], dtype=np.float32)
    output_type = str(output["type"])
    if output_type in {"int8", "uint8", "int32"}:
        scale = float(output.get("scale", 0.0))
        zero_point = int(output.get("zero_point", 0))
        if scale:
            values = (values - zero_point) * scale
    if values.size < 2:
        raise ValueError(f"Classifier output tensor is too small: {values.size}")
    return values


def write_record(
    args: argparse.Namespace,
    row: dict[str, Any],
    dataset_index: int,
    image_identity: dict[str, Any],
    board_ready: dict[str, Any],
    board_result: dict[str, Any],
    host_roundtrip_us: float,
    prompt_lookup_entry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray, int]:
    logits = result_logits(board_result, args.output_tensor_index)
    class_count = min(logits.size, args.answer_max - args.answer_min + 1)
    logits = logits[:class_count]
    probabilities = softmax(logits)
    prediction = int(np.argmax(probabilities) + args.answer_min)
    answer = int(row["answer"])
    metrics = numeric_metrics(
        probabilities,
        prediction,
        answer,
        args.answer_min,
        args.collapse_at,
    )
    prompt = str(row["student_prompt"])
    candidates = candidate_scores(
        probabilities,
        logits,
        args.answer_min,
        args.answer_min + class_count - 1,
    )
    record = {
        "cache_schema_version": 1,
        "dataset_index": int(dataset_index),
        "example_id": row["example_id"],
        "source_subset": row["source_subset"],
        "source": row["source"],
        "source_row_index": int(row["source_row_index"]),
        "qa_index": int(row["qa_index"]),
        "answer": answer,
        "answer_text": row["answer_text"],
        "teacher_prompt": row.get("teacher_prompt_clean", prompt),
        "teacher_prompt_clean": row.get("teacher_prompt_clean", prompt),
        "student_prompt": prompt,
        "item": row["item"],
        "item_class_id": int(row["item_class_id"]),
        "matched_suffix": row["matched_suffix"],
        "image_id": row["image_id"],
        "image_index": int(row["image_index"]),
        "input_identity": {
            "filtered_dataset": str(args.dataset),
            "image_source": image_identity["image_source"],
            "model_name": args.model_name,
            "board_model_path": board_ready.get("model_path"),
            "student_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "image_identity": image_identity,
            "prompt_lookup_entry": prompt_lookup_entry,
            "prompt_lookup_manifest": str(args.prompt_lookup_manifest),
        },
        "image_preprocessing": {
            "cached_image_size": [224, 224],
            "cached_image_mode": "RGB",
            "image_source": image_identity["image_source"],
            "board_input": board_ready.get("input"),
        },
        "teacher_logits": {
            "numeric_answer_candidates": candidates,
            "raw_logits": [float(value) for value in logits.tolist()],
            "board_outputs": board_result.get("outputs", []),
        },
        "teacher_metrics": {
            "numeric_answer": metrics,
            "metric_definitions": {
                "accuracy": "argmax over board-emitted count classes equals answer collapsed at --collapse-at",
                "nll": "negative log-likelihood of collapsed answer under softmax over board logits",
                "target_probability": "probability assigned to collapsed answer class",
            },
        },
        "board_timing": {
            "receive_us": board_result.get("receive_us"),
            "copy_us": board_result.get("copy_us"),
            "invoke_us": board_result.get("invoke_us"),
            "host_roundtrip_us": host_roundtrip_us,
        },
        "board_result": board_result,
    }
    return record, probabilities, prediction


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force or --resume.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_log.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset)
    examples = load_examples(args.dataset)
    prompt_lookup: dict[str, int] | None = None
    prompt_lookup_manifest: dict[str, Any] | None = None
    if args.prompt_lookup_manifest.exists():
        prompt_lookup, prompt_lookup_manifest = load_prompt_lookup(args.prompt_lookup_manifest)
    selected = selected_indices(examples, args)
    completed = completed_indices(args.output) if args.resume else set()
    indices = [index for index in selected if index not in completed]
    selection_hash = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    if args.dry_run:
        prompt_counts = Counter(str(examples[index]["student_prompt"]) for index in selected)
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "output": str(args.output),
                    "selected_records": len(selected),
                    "remaining_records": len(indices),
                    "selected_prompt_classes": len(prompt_counts),
                    "top_prompt_classes": prompt_counts.most_common(20),
                    "selection_sha256": selection_hash,
                    "port": args.port,
                    "baud": args.baud,
                    "prompt_lookup_manifest": str(args.prompt_lookup_manifest),
                    "prompt_lookup_entries": len(prompt_lookup or {}),
                },
                indent=2,
            )
        )
        return

    try:
        import serial
    except ImportError as exc:
        raise SystemExit(
            "pyserial is required. Install with `uv sync --extra coral` or `uv pip install pyserial`."
        ) from exc

    stats: Counter = Counter()
    class_counts: Counter = Counter()
    confusion: dict[int, Counter] = defaultdict(Counter)
    latency: dict[str, list[float]] = defaultdict(list)
    image_store = Uint8ImageStore(args.dataset, metadata)
    output_mode = "a" if args.resume else "w"
    if args.force and args.raw_log.exists():
        try:
            args.raw_log.unlink()
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot remove stale raw serial log {args.raw_log}. "
                "If it was created by a sudo run, fix ownership with "
                "`sudo chown -R \"$USER\":\"$USER\" artifacts/profiles/coral`."
            ) from exc

    started_at = datetime.now(timezone.utc)
    with (
        serial.Serial(args.port, args.baud, timeout=args.serial_timeout_s, write_timeout=args.serial_timeout_s) as ser,
        args.raw_log.open("a", encoding="utf-8") as raw_log,
        args.output.open(output_mode, encoding="utf-8") as handle,
    ):
        recent_lines: deque[str] = deque(maxlen=20)
        try:
            board_ready = wait_ready(
                ser,
                raw_log,
                recent_lines,
                args.ready_timeout_s or args.serial_timeout_s,
            )
        except TimeoutError as exc:
            raise TimeoutError(serial_timeout_message(args, serial, exc, recent_lines)) from exc
        print(json.dumps({"event": "board_ready", **board_ready}, sort_keys=True))
        prompt_input_index = board_ready.get("prompt_input_index")
        has_prompt_input = prompt_input_index is not None and int(prompt_input_index) >= 0
        if has_prompt_input and prompt_lookup is None:
            raise FileNotFoundError(
                f"Board reports prompt_input_index={prompt_input_index}, but "
                f"{args.prompt_lookup_manifest} does not exist."
            )
        manifest_entries = (
            list(prompt_lookup_manifest.get("entries", []))
            if isinstance(prompt_lookup_manifest, dict)
            else []
        )
        progress = tqdm(
            total=len(selected),
            initial=len(completed),
            desc="Caching Coral Micro TallyQA predictions",
            unit="example",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
        for dataset_index in indices:
            row = examples[dataset_index]
            image, image_identity = image_store.get(int(row["image_index"]))
            payload = quantize_for_board_input(image, board_ready)
            prompt_id = (
                prompt_id_for_row(row, prompt_lookup)
                if has_prompt_input and prompt_lookup is not None
                else None
            )
            prompt_lookup_entry = (
                manifest_entries[prompt_id]
                if prompt_id is not None and 0 <= prompt_id < len(manifest_entries)
                else None
            )
            roundtrip_start = perf_counter()
            debug_protocol(
                args,
                "send_header",
                dataset_index=int(dataset_index),
                image_index=int(row["image_index"]),
                prompt_id=prompt_id,
                payload_bytes=len(payload),
            )
            send_header(
                ser,
                dataset_index,
                int(row["image_index"]),
                len(payload),
                prompt_id=prompt_id,
            )
            while True:
                try:
                    event, result = read_prefixed(
                        ser,
                        raw_log,
                        recent_lines,
                        f"waiting for RX_READY for dataset index {dataset_index}",
                        args.rx_ready_timeout_s,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        serial_timeout_message(args, serial, exc, recent_lines, dataset_index)
                    ) from exc
                if event == "error":
                    if int(result.get("dataset_index", -1)) == int(dataset_index):
                        raise RuntimeError(f"Board error for dataset index {dataset_index}: {result}")
                    print(json.dumps({"board_error": result}, sort_keys=True), file=sys.stderr)
                    continue
                if event == "event":
                    print(json.dumps({"board_event": result}, sort_keys=True))
                    continue
                if event == "rx_ready" and int(result["dataset_index"]) == int(dataset_index):
                    break
            debug_protocol(
                args,
                "send_payload_start",
                dataset_index=int(dataset_index),
                payload_bytes=len(payload),
                chunk_size=int(args.payload_chunk_size),
                chunk_delay_s=float(args.payload_chunk_delay_s),
            )
            send_payload(ser, payload, args.payload_chunk_size, args.payload_chunk_delay_s)
            debug_protocol(args, "send_payload_done", dataset_index=int(dataset_index))
            while True:
                debug_protocol(args, "wait_result", dataset_index=int(dataset_index))
                try:
                    event, result = read_prefixed(
                        ser,
                        raw_log,
                        recent_lines,
                        f"waiting for RESULT for dataset index {dataset_index}",
                        args.result_timeout_s,
                    )
                except TimeoutError as exc:
                    raise TimeoutError(
                        serial_timeout_message(args, serial, exc, recent_lines, dataset_index)
                    ) from exc
                if event == "error":
                    if int(result.get("dataset_index", -1)) == int(dataset_index):
                        raise RuntimeError(f"Board error for dataset index {dataset_index}: {result}")
                    print(json.dumps({"board_error": result}, sort_keys=True), file=sys.stderr)
                    continue
                if event == "event":
                    print(json.dumps({"board_event": result}, sort_keys=True))
                    continue
                if event == "result" and int(result["dataset_index"]) == int(dataset_index):
                    break
            host_roundtrip_us = (perf_counter() - roundtrip_start) * 1_000_000.0
            record, probabilities, prediction = write_record(
                args,
                row,
                dataset_index,
                image_identity,
                board_ready,
                result,
                host_roundtrip_us,
                prompt_lookup_entry,
            )
            answer = int(row["answer"])
            target = normalized_answer(answer, args.collapse_at)
            prompt = str(row["student_prompt"])
            update_stats(
                stats,
                prompt,
                answer,
                prediction,
                probabilities,
                args.answer_min,
                args.collapse_at,
            )
            class_counts[target] += 1
            confusion[target][prediction] += 1
            for key in ("receive_us", "copy_us", "invoke_us"):
                if key in result:
                    latency[key].append(float(result[key]))
            latency["host_roundtrip_us"].append(host_roundtrip_us)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.flush_every > 0 and stats[("overall", "total")] % args.flush_every == 0:
                handle.flush()
            progress.set_postfix(
                {
                    "pred": prediction,
                    "invoke_ms": f"{float(result.get('invoke_us', 0.0)) / 1000.0:.2f}",
                    "rt_ms": f"{host_roundtrip_us / 1000.0:.1f}",
                }
            )
            progress.update(1)
        progress.close()

    prompt_keys = sorted(
        key.removeprefix("prompt::")
        for key, metric in stats
        if metric == "total" and key.startswith("prompt::")
    )
    latency_summary = {
        key: {
            "mean_us": float(np.mean(values)) if values else None,
            "median_us": float(np.median(values)) if values else None,
            "p95_us": float(np.percentile(values, 95)) if values else None,
            "min_us": float(np.min(values)) if values else None,
            "max_us": float(np.max(values)) if values else None,
        }
        for key, values in latency.items()
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "raw_log": str(args.raw_log),
        "model_name": args.model_name,
        "port": args.port,
        "baud": args.baud,
        "selected_records": len(selected),
        "records_already_done": len(completed),
        "written_records_this_invocation": int(stats[("overall", "total")]),
        "selection": {
            "first_index": selected[0] if selected else None,
            "last_index": selected[-1] if selected else None,
            "selected_indices_sha256": selection_hash,
        },
        "teacher_metrics": {
            "numeric_answer": aggregate_stats(stats, "overall"),
            "by_student_prompt": {
                prompt: aggregate_stats(stats, f"prompt::{prompt}") for prompt in prompt_keys
            },
            "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
            "confusion": {
                str(true_label): {
                    str(pred_label): int(count)
                    for pred_label, count in sorted(pred_counts.items())
                }
                for true_label, pred_counts in sorted(confusion.items())
            },
        },
        "latency": latency_summary,
        "board_ready": board_ready,
        "prompt_lookup": {
            "manifest": str(args.prompt_lookup_manifest),
            "entries": len(prompt_lookup or {}),
            "quantization": (
                prompt_lookup_manifest.get("quantization")
                if isinstance(prompt_lookup_manifest, dict)
                else None
            ),
        },
        "board_memory": {
            "tensor_arena_bytes": board_ready.get("tensor_arena_bytes"),
            "arena_used_bytes": board_ready.get("arena_used_bytes"),
            "arena_recorded_used_bytes": board_ready.get("arena_recorded_used_bytes"),
            "arena_recorded_requested_bytes": board_ready.get("arena_recorded_requested_bytes"),
            "arena_recorded_alloc_count": board_ready.get("arena_recorded_alloc_count"),
            "recorded_allocations": board_ready.get("recorded_allocations"),
            "inputs": board_ready.get("inputs"),
            "outputs": board_ready.get("outputs"),
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote cache: {args.output}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote raw serial log: {args.raw_log}")


if __name__ == "__main__":
    main()
