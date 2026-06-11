###################################################################################################
#
# TallyQA count dataset adapter for ADI ai8x-training.
#
###################################################################################################
"""TallyQA count dataset for MAX78000 experiments.

The preferred input is the materialized manifest created by
``scripts/materialize_max78000_tallyqa_dataset.py`` in edge_vlm.  The adapter can
also read the full TallyQA target dataset directly if ``examples.jsonl`` and
``metadata.json`` are present, in which case it falls back to the original
people-only positive-count view.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import ai8x


COUNT_LABELS = ("0", "1", "2", "3", "4", "5+")
DEFAULT_SOURCE_SUBDIR = "tallyqa_cauldron_target_mobilenet224_letterbox"
DEFAULT_MANIFEST_SUBDIR = "max78000_tallyqa_count_fold2_56"
RESIZE_SIZE = 112
FOLDED_SIZE = 56
FOLDED_CHANNELS = 12


def _split_for_image(image_id: str, seed: int) -> str:
    digest = hashlib.blake2b(f"{seed}:{image_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def _resolve_dataset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    candidates = (
        root,
        root / DEFAULT_MANIFEST_SUBDIR,
        root / DEFAULT_SOURCE_SUBDIR,
    )
    for candidate in candidates:
        if (candidate / "metadata.json").exists() and (
            (candidate / "manifest.jsonl").exists() or (candidate / "examples.jsonl").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find MAX78000 TallyQA count dataset. Expected metadata.json plus "
        "manifest.jsonl or examples.jsonl in one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _count_to_label(answer: int) -> int | None:
    count = int(answer)
    if count <= 0:
        return None
    return min(count, 5) - 1


def _load_records(root: Path, split: str, seed: int) -> list[dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        return [row for row in _load_jsonl(manifest) if row["split"] == split]

    records: list[dict[str, Any]] = []
    for row in _load_jsonl(root / "examples.jsonl"):
        item = str(row.get("item") or row.get("student_prompt") or "").strip().lower()
        if item != "people":
            continue
        label = _count_to_label(int(row["answer"]))
        if label is None:
            continue
        image_id = str(row["image_id"])
        if _split_for_image(image_id, seed) != split:
            continue
        records.append(
            {
                "example_id": int(row["example_id"]),
                "image_id": image_id,
                "image_index": int(row["image_index"]),
                "answer": int(row["answer"]),
                "label": label,
                "split": split,
            }
        )
    return records


def _resolve_tensor_file(root: Path, metadata: dict[str, Any]) -> Path:
    tensor_file = Path(metadata["image"]["tensor_file"])
    if tensor_file.is_absolute():
        return tensor_file
    return root / tensor_file


class TallyQACount(Dataset):
    """TallyQA count classification dataset from a materialized manifest."""

    def __init__(
        self,
        root_dir: str | Path,
        d_type: str,
        transform=None,
        seed: int = 0,
    ):
        if d_type not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {d_type!r}; expected train, val, or test.")
        self.root_dir = _resolve_dataset_root(root_dir)
        self.d_type = d_type
        self.transform = transform
        self.records = _load_records(self.root_dir, d_type, seed)
        if not self.records:
            raise RuntimeError(f"No TallyQA count records found for split {d_type!r}.")

        self.metadata = json.loads((self.root_dir / "metadata.json").read_text(encoding="utf-8"))
        image_meta = self.metadata["image"]
        shape = tuple(int(v) for v in image_meta["shape"])
        if len(shape) != 4 or shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected image shape (N, 3, 224, 224), got {shape}.")
        if image_meta.get("layout") != "CHW" or image_meta.get("dtype") != "uint8":
            raise ValueError("Expected CHW uint8 image tensor metadata.")
        tensor_file = _resolve_tensor_file(self.root_dir, self.metadata)
        if not tensor_file.exists():
            raise FileNotFoundError(tensor_file)
        self.images = np.memmap(tensor_file, dtype=np.uint8, mode="r", shape=shape)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_chw = np.asarray(self.images[int(record["image_index"])])
        image_hwc = np.transpose(image_chw, (1, 2, 0)).copy()
        label = torch.tensor(int(record["label"]), dtype=torch.int64)
        if self.transform:
            image = self.transform(Image.fromarray(image_hwc))
        else:
            image = torch.from_numpy(image_chw.copy()).float().div(255.0)
        return image, label


class Fold2x2:
    """Fold a 3xHxW tensor into 12x(H/2)x(W/2)."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}.")
        channels, height, width = image.shape
        if channels != 3:
            raise ValueError(f"Expected RGB tensor with 3 channels, got {channels}.")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Fold2x2 requires even spatial dimensions, got {height}x{width}.")
        folded = image.reshape(channels, height // 2, 2, width // 2, 2)
        folded = folded.permute(0, 2, 4, 1, 3).contiguous()
        return folded.reshape(channels * 4, height // 2, width // 2)


def get_tallyqa_count_dataset(data, load_train, load_test):
    """Load TallyQA count train/test datasets."""
    data_dir, args = data
    seed = int(getattr(args, "seed", 0) or 0)

    transform = transforms.Compose(
        [
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            Fold2x2(),
            ai8x.normalize(args=args),
        ]
    )

    train_dataset = (
        TallyQACount(root_dir=data_dir, d_type="train", transform=transform, seed=seed)
        if load_train
        else None
    )
    test_dataset = (
        TallyQACount(root_dir=data_dir, d_type="test", transform=transform, seed=seed)
        if load_test
        else None
    )

    if test_dataset is not None and getattr(args, "truncate_testset", False):
        test_dataset.records = test_dataset.records[:1]

    return train_dataset, test_dataset


datasets = [
    {
        "name": "tallyqa_count_fold2_56",
        "input": (FOLDED_CHANNELS, FOLDED_SIZE, FOLDED_SIZE),
        "output": COUNT_LABELS,
        "loader": get_tallyqa_count_dataset,
    },
]
