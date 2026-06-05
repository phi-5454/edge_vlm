from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import torch
from tqdm.auto import tqdm
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.transforms import functional as TF

from scripts.cache_smolvlm_tallyqa_teacher import (
    Uint8ImageStore,
    load_examples,
    load_metadata,
    planned_indices,
)


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_OUTPUT = Path(
    "artifacts/teacher_cache/torchvision_person_tallyqa_target_mobilenet224_letterbox.jsonl"
)
PERSON_COCO_LABEL = 1
VERY_LOW_LOG_LIKELIHOOD = -1.0e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a torchvision COCO person-detector counting baseline for TallyQA."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        choices=["fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"],
        default="fasterrcnn_resnet50_fpn_v2",
    )
    parser.add_argument("--prompt", default="people")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def device_from_arg(value: str, require_cuda: bool) -> torch.device:
    if value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set, but CUDA is not available/selected.")
    return device


def load_detector(name: str, device: torch.device) -> torch.nn.Module:
    if name == "fasterrcnn_resnet50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    elif name == "retinanet_resnet50_fpn_v2":
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    elif name == "fcos_resnet50_fpn":
        model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.to(device)
    model.eval()
    return model


def selected_people_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    selected = planned_indices(examples, args)
    return [index for index in selected if str(examples[index]["student_prompt"]) == args.prompt]


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
            except json.JSONDecodeError:
                continue
    return completed


def detector_count(output: dict[str, torch.Tensor], threshold: float) -> tuple[int, list[dict[str, Any]]]:
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()
    boxes = output["boxes"].detach().cpu()
    keep = (labels == PERSON_COCO_LABEL) & (scores >= threshold)
    detections = [
        {
            "box_xyxy": [float(value) for value in box.tolist()],
            "score": float(score),
            "label": int(label),
            "label_name": "person",
        }
        for box, score, label in zip(boxes[keep], scores[keep], labels[keep], strict=True)
    ]
    return len(detections), detections


def candidate_scores(prediction: int, answer_min: int, answer_max: int) -> list[dict[str, Any]]:
    clamped = max(answer_min, min(answer_max, int(prediction)))
    return [
        {
            "answer": answer,
            "candidate_probability": 1.0 if answer == clamped else 0.0,
            "candidate_log_likelihood": 0.0
            if answer == clamped
            else VERY_LOW_LOG_LIKELIHOOD,
        }
        for answer in range(answer_min, answer_max + 1)
    ]


def update_stats(stats: Counter, answer: int, prediction: int, collapse_at: int) -> None:
    stats["total"] += 1
    stats["correct"] += int(prediction == answer)
    stats["within_1"] += int(abs(prediction - answer) <= 1)
    stats["collapsed_correct"] += int(min(prediction, collapse_at) == min(answer, collapse_at))


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force or --resume.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset)
    examples = load_examples(args.dataset)
    selected_indices = selected_people_indices(examples, args)
    completed = completed_indices(args.output) if args.resume else set()
    indices = [index for index in selected_indices if index not in completed]
    device = device_from_arg(args.device, args.require_cuda)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "selected_prompt": args.prompt,
                    "selected_records": len(selected_indices),
                    "remaining_records": len(indices),
                    "model": args.model,
                    "score_threshold": args.score_threshold,
                    "device": str(device),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return

    model = load_detector(args.model, device)
    image_store = Uint8ImageStore(args.dataset, metadata)
    output_mode = "a" if args.resume else "w"
    stats: Counter = Counter()

    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(total=len(selected_indices), initial=len(completed), desc="Caching person detector counts", unit="example")
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            rows = [examples[index] for index in batch_indices]
            images_and_identities = [
                image_store.get(int(row["image_index"])) for row in rows
            ]
            tensors = [
                TF.convert_image_dtype(TF.pil_to_tensor(image), torch.float32).to(device)
                for image, _identity in images_and_identities
            ]
            with torch.inference_mode():
                outputs = model(tensors)
            for dataset_index, row, (_image, image_identity), output in zip(
                batch_indices,
                rows,
                images_and_identities,
                outputs,
                strict=True,
            ):
                prediction, detections = detector_count(output, args.score_threshold)
                answer = int(row["answer"])
                update_stats(stats, answer, prediction, args.collapse_at)
                record = {
                    "schema_version": 1,
                    "dataset_index": int(dataset_index),
                    "example_id": row.get("example_id"),
                    "image_id": row.get("image_id"),
                    "image_identity": image_identity,
                    "student_prompt": str(row["student_prompt"]),
                    "teacher_prompt": "person",
                    "answer": answer,
                    "teacher_model": {
                        "name": args.model,
                        "family": "torchvision-detection-coco",
                        "score_threshold": args.score_threshold,
                        "person_coco_label": PERSON_COCO_LABEL,
                    },
                    "teacher_metrics": {
                        "numeric_answer": {
                            "prediction": int(prediction),
                            "prediction_text": str(prediction),
                            "correct": prediction == answer,
                            "within_1": abs(prediction - answer) <= 1,
                            "collapsed_correct": min(prediction, args.collapse_at)
                            == min(answer, args.collapse_at),
                        }
                    },
                    "teacher_logits": {
                        "numeric_answer_candidates": candidate_scores(
                            prediction,
                            args.answer_min,
                            args.answer_max,
                        )
                    },
                    "detections": detections,
                }
                handle.write(json.dumps(record) + "\n")
                if args.flush_every > 0 and stats["total"] % args.flush_every == 0:
                    handle.flush()
            progress.update(len(batch_indices))
        progress.close()

    total = int(stats["total"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "model": args.model,
        "prompt": args.prompt,
        "score_threshold": args.score_threshold,
        "selected_records": len(selected_indices),
        "written_records_this_invocation": total,
        "accuracy": stats["correct"] / total if total else None,
        "within_1_accuracy": stats["within_1"] / total if total else None,
        "collapsed_accuracy": stats["collapsed_correct"] / total if total else None,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
