#!/usr/bin/env python3
"""Pretrain the folded MAX78000 frontend against a frozen MobileNetV3 feature cut.

Run this in an environment where ADI's ``ai8x`` module is importable, for example
from ``../MAX78000/ai8x-training`` after staging the repo-owned model file.
The loss is MSE between:

- frozen torchvision MobileNetV3-large features at cutoff 13: ``N x 112 x 14 x 14``
- folded MAX78000 frontend ``forward_features()`` output: ``N x 112 x 14 x 14``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_MODEL_FILE = Path("max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py")
DEFAULT_OUTPUT = Path("artifacts/models/max78000/tallyqa_folded_frontend_mbv3_large_cut13.pt")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class Batch:
    teacher_images: torch.Tensor
    folded_images: torch.Tensor
    dataset_indices: torch.Tensor
    image_indices: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--model-factory", default="ai85tallyqambv3smallpeople")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--teacher-backbone", choices=["mobilenet_v3_large", "mobilenet_v3_small"], default="mobilenet_v3_large")
    parser.add_argument("--teacher-cutoff", type=int, default=13)
    parser.add_argument("--prompt-class", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unique-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def load_rows(dataset: Path) -> list[dict[str, Any]]:
    columns = ["example_id", "student_prompt", "item", "image_index"]
    return pq.read_table(dataset / "examples.parquet", columns=columns).to_pylist()


def load_metadata(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))


def select_indices(
    rows: list[dict[str, Any]],
    prompt_class: str | None,
    unique_images: bool,
    max_examples: int | None,
) -> list[int]:
    selected: list[int] = []
    seen_images: set[int] = set()
    prompt_lower = prompt_class.lower() if prompt_class else None
    for index, row in enumerate(rows):
        if prompt_lower is not None:
            item = str(row.get("item") or row.get("student_prompt") or "").strip().lower()
            if item != prompt_lower:
                continue
        image_index = int(row["image_index"])
        if unique_images and image_index in seen_images:
            continue
        seen_images.add(image_index)
        selected.append(index)
        if max_examples is not None and len(selected) >= max_examples:
            break
    if not selected:
        raise ValueError("No examples selected for MAX78000 frontend pretraining.")
    return selected


class TallyQAFeatureDistillationDataset(Dataset):
    def __init__(
        self,
        dataset: Path,
        prompt_class: str | None,
        unique_images: bool,
        max_examples: int | None,
    ):
        self.dataset = dataset
        self.rows = load_rows(dataset)
        self.indices = select_indices(self.rows, prompt_class, unique_images, max_examples)
        metadata = load_metadata(dataset)
        image_meta = metadata["image"]
        self.image_shape = tuple(int(dim) for dim in image_meta["shape"])
        self.image_path = dataset / image_meta.get("tensor_file", "images.uint8.bin")
        self.images = np.memmap(self.image_path, dtype=np.uint8, mode="r", shape=self.image_shape)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, offset: int) -> dict[str, torch.Tensor]:
        dataset_index = self.indices[offset]
        row = self.rows[dataset_index]
        image_index = int(row["image_index"])
        chw = np.asarray(self.images[image_index], dtype=np.uint8)
        image = Image.fromarray(np.transpose(chw, (1, 2, 0)))

        teacher = TF.to_tensor(image)
        teacher = (teacher - IMAGENET_MEAN) / IMAGENET_STD

        folded_source = TF.resize(image, [112, 112], interpolation=InterpolationMode.BILINEAR)
        folded = fold_2x2(TF.to_tensor(folded_source))
        folded = (folded - 0.5) * 256.0

        return {
            "teacher_images": teacher,
            "folded_images": folded,
            "dataset_indices": torch.tensor(dataset_index, dtype=torch.int64),
            "image_indices": torch.tensor(image_index, dtype=torch.int64),
        }


def fold_2x2(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW tensor, got {tuple(image.shape)}.")
    channels, height, width = image.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Fold requires even spatial dimensions, got {height}x{width}.")
    folded = image.reshape(channels, height // 2, 2, width // 2, 2)
    folded = folded.permute(0, 2, 4, 1, 3).contiguous()
    return folded.reshape(channels * 4, height // 2, width // 2)


def collate(batch: list[dict[str, torch.Tensor]]) -> Batch:
    return Batch(
        teacher_images=torch.stack([item["teacher_images"] for item in batch]),
        folded_images=torch.stack([item["folded_images"] for item in batch]),
        dataset_indices=torch.stack([item["dataset_indices"] for item in batch]),
        image_indices=torch.stack([item["image_indices"] for item in batch]),
    )


def build_teacher(backbone: str, cutoff: int) -> nn.Module:
    if backbone == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    elif backbone == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported teacher backbone: {backbone}")
    features = nn.Sequential(*list(model.features.children())[:cutoff])
    features.eval()
    for parameter in features.parameters():
        parameter.requires_grad = False
    return features


def import_student_factory(model_file: Path, factory_name: str):
    model_file = model_file.resolve()
    sys.path.insert(0, str(model_file.parent))
    spec = importlib.util.spec_from_file_location("edge_vlm_max78000_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model file {model_file}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return getattr(module, factory_name)
    except AttributeError as exc:
        raise AttributeError(f"{model_file} does not define {factory_name!r}.") from exc


def save_checkpoint(
    path: Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "epoch": int(epoch),
            "metrics": metrics,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    dataset = TallyQAFeatureDistillationDataset(
        args.dataset,
        prompt_class=args.prompt_class,
        unique_images=bool(args.unique_images),
        max_examples=args.max_examples,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    teacher = build_teacher(args.teacher_backbone, int(args.teacher_cutoff)).to(device)
    student_factory = import_student_factory(args.model_file, args.model_factory)
    student = student_factory(num_classes=5, num_channels=12, dimensions=(56, 56)).to(device)
    if not hasattr(student, "forward_features"):
        raise AttributeError("MAX78000 student model must define forward_features().")
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    best_loss = math.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        running_loss = 0.0
        examples = 0
        progress = tqdm(loader, desc=f"frontend pretrain {epoch}/{args.epochs}", unit="batch")
        for batch in progress:
            teacher_images = batch.teacher_images.to(device, non_blocking=True)
            folded_images = batch.folded_images.to(device, non_blocking=True)
            with torch.no_grad():
                target = teacher(teacher_images)
            prediction = student.forward_features(folded_images)
            if prediction.shape != target.shape:
                raise RuntimeError(
                    "Feature shape mismatch: "
                    f"student {tuple(prediction.shape)} vs teacher {tuple(target.shape)}"
                )
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = folded_images.shape[0]
            running_loss += float(loss.detach().cpu()) * batch_size
            examples += batch_size
            progress.set_postfix(mse=f"{running_loss / max(1, examples):.6f}")
        epoch_metrics = {"epoch": float(epoch), "mse": running_loss / max(1, examples)}
        history.append(epoch_metrics)
        if args.save_every_epoch:
            save_checkpoint(
                args.output.with_name(f"{args.output.stem}.epoch{epoch:03d}{args.output.suffix}"),
                student,
                optimizer,
                epoch,
                epoch_metrics,
                args,
            )
        if epoch_metrics["mse"] < best_loss:
            best_loss = epoch_metrics["mse"]
            save_checkpoint(args.output, student, optimizer, epoch, epoch_metrics, args)

    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "selected_examples": len(dataset),
        "teacher": {
            "backbone": args.teacher_backbone,
            "cutoff": int(args.teacher_cutoff),
            "output_shape": [112, 14, 14],
        },
        "student": {
            "model_file": str(args.model_file),
            "factory": args.model_factory,
            "input_shape": [12, 56, 56],
            "output_shape": [112, 14, 14],
        },
        "history": history,
        "best_mse": best_loss,
        "checkpoint": str(args.output),
        "config": asdict(args) if hasattr(args, "__dataclass_fields__") else vars(args),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Best MSE: {best_loss:.6f}")
    print(f"Wrote checkpoint: {args.output}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
