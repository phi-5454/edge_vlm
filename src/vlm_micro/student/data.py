from __future__ import annotations

import hashlib
import io
import json
import warnings
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lightning as L
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class CompactVocabulary:
    """Maps sparse teacher token IDs to a dense student embedding table."""

    teacher_token_ids: tuple[int, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> CompactVocabulary:
        token_ids = sorted({int(token_id) for row in rows for token_id in row["student_token_ids"]})
        if not token_ids:
            raise ValueError("Cannot build a compact vocabulary from an empty dataset.")
        return cls(tuple(token_ids))

    @property
    def size(self) -> int:
        return len(self.teacher_token_ids) + 1

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def teacher_to_compact(self) -> dict[int, int]:
        return {teacher_id: index + 1 for index, teacher_id in enumerate(self.teacher_token_ids)}

    def remap(self, teacher_ids: Iterable[int]) -> list[int]:
        mapping = self.teacher_to_compact
        try:
            return [mapping[int(token_id)] for token_id in teacher_ids]
        except KeyError as error:
            raise ValueError(f"Teacher token ID {error.args[0]} is outside the compact vocabulary.") from error


def load_prompt_rows(dataset_root: Path) -> list[dict[str, Any]]:
    columns = [
        "source_subset",
        "original_index",
        "qa_index",
        "answer",
        "student_prompt",
        "student_token_ids",
        "student_image_id",
    ]
    return pq.read_table(dataset_root / "combined.parquet", columns=columns).to_pylist()


def load_teacher_targets(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    targets: dict[int, float] = {}

    def load_line(line: str, line_number: int, allow_truncated: bool) -> None:
        if not line.strip():
            return
        try:
            payload = json.loads(line)
            targets[int(payload["dataset_index"])] = float(
                payload["teacher_logits"]["standalone"]["yes_minus_no_logit"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if allow_truncated:
                warnings.warn(
                    f"Ignoring invalid trailing teacher cache record at {path}:{line_number}: {error}",
                    stacklevel=2,
                )
                return
            raise ValueError(f"Invalid teacher cache record at {path}:{line_number}") from error

    with path.open("r", encoding="utf-8") as handle:
        previous: tuple[int, str] | None = None
        for line_number, line in enumerate(handle, start=1):
            if previous is not None:
                load_line(*previous, allow_truncated=False)
            previous = (line, line_number)
        if previous is not None:
            load_line(*previous, allow_truncated=True)
    return targets


def split_for_image(image_id: str, seed: int) -> str:
    digest = hashlib.blake2b(f"{seed}:{image_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def collapse_count(answer: int, collapse_at: int = 5) -> int:
    return min(int(answer), collapse_at)


def load_tallyqa_rows(dataset_root: Path) -> list[dict[str, Any]]:
    columns = [
        "example_id",
        "source_subset",
        "source_row_index",
        "qa_index",
        "answer",
        "student_prompt",
        "item",
        "item_class_id",
        "image_id",
        "image_index",
    ]
    return pq.read_table(dataset_root / "examples.parquet", columns=columns).to_pylist()


def load_tallyqa_prompt_artifact(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"prompt_token_ids", "prompt_attention_mask", "embedding_rows"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing required prompt embedding fields: {sorted(missing)}")
    return payload


def load_prompt_accuracy_filter(path: Path | None, min_accuracy: float | None) -> set[str] | None:
    if path is None:
        return None
    if min_accuracy is None:
        raise ValueError("min_prompt_accuracy is required when prompt_class_filter_csv is set.")
    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_prompt_accuracy must be between 0 and 1.")
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["accuracy"]) >= min_accuracy:
                prompts.add(str(row["student_prompt"]))
    if not prompts:
        raise ValueError(f"No prompt classes in {path} have accuracy >= {min_accuracy}.")
    return prompts


def parse_prompt_class_names(value: str | None) -> set[str] | None:
    if value is None:
        return None
    prompts = {prompt.strip() for prompt in value.split(",") if prompt.strip()}
    if not prompts:
        raise ValueError("prompt_class_names must contain at least one prompt name when set.")
    return prompts


def load_prompt_class_names_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    prompts = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not prompts:
        raise ValueError(f"{path} does not contain any prompt class names.")
    return prompts


def load_curriculum_schedule(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of curriculum stages.")
    stages: list[dict[str, Any]] = []
    for stage in payload:
        if not isinstance(stage, dict):
            raise ValueError(f"{path} curriculum stages must be JSON objects.")
        start_epoch = int(stage["start_epoch"])
        if start_epoch <= 0:
            raise ValueError("curriculum start_epoch values are 1-based and must be positive.")
        train_sampling = str(stage.get("train_sampling", "natural"))
        if train_sampling not in {"natural", "prompt_class_tempered"}:
            raise ValueError("curriculum train_sampling must be 'natural' or 'prompt_class_tempered'.")
        stages.append({**stage, "start_epoch": start_epoch, "train_sampling": train_sampling})
    stages.sort(key=lambda row: int(row["start_epoch"]))
    return stages


def load_tallyqa_teacher_targets(
    path: Path | None,
    num_classes: int = 6,
    collapse_at: int = 5,
    probability_temperature: float = 1.0,
) -> dict[int, torch.Tensor]:
    if path is None:
        return {}
    if probability_temperature <= 0:
        raise ValueError("teacher probability temperature must be positive.")
    targets: dict[int, torch.Tensor] = {}

    def load_line(line: str, line_number: int, allow_truncated: bool) -> None:
        if not line.strip():
            return
        try:
            payload = json.loads(line)
            probabilities = torch.zeros(num_classes, dtype=torch.float32)
            for candidate in payload["teacher_logits"]["numeric_answer_candidates"]:
                class_id = collapse_count(int(candidate["answer"]), collapse_at)
                probabilities[class_id] += float(candidate["candidate_probability"])
            total = float(probabilities.sum().item())
            if total <= 0:
                raise ValueError("teacher candidate probabilities sum to zero")
            if probability_temperature != 1.0:
                probabilities = torch.where(
                    probabilities > 0,
                    probabilities.pow(1.0 / probability_temperature),
                    probabilities,
                )
                total = float(probabilities.sum().item())
            targets[int(payload["dataset_index"])] = probabilities / total
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if allow_truncated:
                warnings.warn(
                    f"Ignoring invalid trailing teacher cache record at {path}:{line_number}: {error}",
                    stacklevel=2,
                )
                return
            raise ValueError(f"Invalid teacher cache record at {path}:{line_number}") from error

    with path.open("r", encoding="utf-8") as handle:
        previous: tuple[str, int] | None = None
        for line_number, line in enumerate(handle, start=1):
            if previous is not None:
                load_line(*previous, allow_truncated=False)
            previous = (line, line_number)
        if previous is not None:
            load_line(*previous, allow_truncated=True)
    return targets


class ParquetImageStore:
    """Random-access image parquet reader with worker-local LRU caches."""

    def __init__(self, path: Path, row_group_cache_size: int, tensor_cache_size: int, image_size: int):
        self.path = Path(path)
        self.row_group_cache_size = row_group_cache_size
        self.tensor_cache_size = tensor_cache_size
        self.image_size = image_size
        self._file: pq.ParquetFile | None = None
        self._locations: dict[str, tuple[int, int]] | None = None
        self._row_groups: OrderedDict[int, list[bytes]] = OrderedDict()
        self._tensors: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_file"] = None
        state["_row_groups"] = OrderedDict()
        state["_tensors"] = OrderedDict()
        return state

    @property
    def file(self) -> pq.ParquetFile:
        if self._file is None:
            self._file = pq.ParquetFile(self.path)
        return self._file

    def _ensure_locations(self) -> dict[str, tuple[int, int]]:
        if self._locations is None:
            locations: dict[str, tuple[int, int]] = {}
            for group_index in range(self.file.num_row_groups):
                image_ids = self.file.read_row_group(group_index, columns=["student_image_id"]).column(0)
                for row_index, image_id in enumerate(image_ids.to_pylist()):
                    locations[str(image_id)] = (group_index, row_index)
            self._locations = locations
        return self._locations

    def _image_bytes(self, image_id: str) -> bytes:
        group_index, row_index = self._ensure_locations()[image_id]
        if group_index not in self._row_groups:
            table = self.file.read_row_group(group_index, columns=["image_bytes"])
            self._row_groups[group_index] = table.column(0).to_pylist()
            while len(self._row_groups) > self.row_group_cache_size:
                self._row_groups.popitem(last=False)
        self._row_groups.move_to_end(group_index)
        return self._row_groups[group_index][row_index]

    def tensor(self, image_id: str) -> torch.Tensor:
        if image_id not in self._tensors:
            with Image.open(io.BytesIO(self._image_bytes(image_id))) as image:
                image = image.convert("RGB")
                tensor = TF.pil_to_tensor(image)
                tensor = TF.resize(tensor, [self.image_size, self.image_size], antialias=True)
                tensor = TF.convert_image_dtype(tensor, torch.float32)
                tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self._tensors[image_id] = tensor
            while len(self._tensors) > self.tensor_cache_size:
                self._tensors.popitem(last=False)
        self._tensors.move_to_end(image_id)
        return self._tensors[image_id]


class Uint8MemmapImageStore:
    """Random-access CHW uint8 image memmap with ImageNet normalization."""

    def __init__(self, dataset_root: Path, tensor_cache_size: int = 256):
        self.dataset_root = Path(dataset_root)
        metadata = json.loads((self.dataset_root / "metadata.json").read_text(encoding="utf-8"))
        image_meta = metadata["image"]
        self.shape = tuple(int(dim) for dim in image_meta["shape"])
        self.tensor_path = self.dataset_root / image_meta.get("tensor_file", "images.uint8.bin")
        self.tensor_cache_size = tensor_cache_size
        self._images: np.memmap | None = None
        self._tensors: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_images"] = None
        state["_tensors"] = OrderedDict()
        return state

    @property
    def images(self) -> np.memmap:
        if self._images is None:
            self._images = np.memmap(
                self.tensor_path,
                dtype=np.uint8,
                mode="r",
                shape=self.shape,
            )
        return self._images

    def tensor(self, image_index: int) -> torch.Tensor:
        image_index = int(image_index)
        if image_index not in self._tensors:
            tensor = torch.from_numpy(np.asarray(self.images[image_index]).copy()).float() / 255.0
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self._tensors[image_index] = tensor
            while len(self._tensors) > self.tensor_cache_size:
                self._tensors.popitem(last=False)
        self._tensors.move_to_end(image_index)
        return self._tensors[image_index]


class StudentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        indices: list[int],
        vocabulary: CompactVocabulary,
        teacher_targets: dict[int, float],
        image_store: ParquetImageStore,
    ):
        self.rows = rows
        self.indices = indices
        self.vocabulary = vocabulary
        self.teacher_targets = teacher_targets
        self.image_store = image_store

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        dataset_index = self.indices[item]
        row = self.rows[dataset_index]
        return {
            "dataset_index": dataset_index,
            "token_ids": self.vocabulary.remap(row["student_token_ids"]),
            "image": self.image_store.tensor(str(row["student_image_id"])),
            "label": 1.0 if row["answer"] == "yes" else 0.0,
            "teacher_logit": self.teacher_targets.get(dataset_index, float("nan")),
        }


class TallyQAStudentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        indices: list[int],
        prompt_token_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        teacher_targets: dict[int, torch.Tensor],
        image_store: Uint8MemmapImageStore,
        collapse_at: int = 5,
        num_classes: int = 6,
    ):
        self.rows = rows
        self.indices = indices
        self.prompt_token_ids = prompt_token_ids
        self.prompt_attention_mask = prompt_attention_mask
        self.teacher_targets = teacher_targets
        self.image_store = image_store
        self.collapse_at = collapse_at
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        dataset_index = self.indices[item]
        row = self.rows[dataset_index]
        item_class_id = int(row["item_class_id"])
        teacher_probs = self.teacher_targets.get(
            dataset_index,
            torch.full((self.num_classes,), float("nan"), dtype=torch.float32),
        )
        return {
            "dataset_index": dataset_index,
            "token_ids": self.prompt_token_ids[item_class_id],
            "attention_mask": self.prompt_attention_mask[item_class_id],
            "image": self.image_store.tensor(int(row["image_index"])),
            "label": collapse_count(int(row["answer"]), self.collapse_at),
            "teacher_probs": teacher_probs,
            "item_class_id": item_class_id,
            "image_id": str(row["image_id"]),
            "student_prompt": str(row["student_prompt"]),
        }


class ImageGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle image groups while keeping repeated-image prompts close together."""

    def __init__(self, dataset: StudentDataset, batch_size: int, seed: int, shuffle_block_size: int):
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle_block_size = shuffle_block_size
        self.epoch = 0
        groups: dict[str, list[int]] = {}
        for local_index, dataset_index in enumerate(dataset.indices):
            image_id = str(dataset.rows[dataset_index]["student_image_id"])
            groups.setdefault(image_id, []).append(local_index)
        self.groups = list(groups.values())

    def __len__(self) -> int:
        return (sum(len(group) for group in self.groups) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterable[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        pending: list[int] = []
        blocks = [
            list(range(start, min(start + self.shuffle_block_size, len(self.groups))))
            for start in range(0, len(self.groups), self.shuffle_block_size)
        ]
        for block_index in torch.randperm(len(blocks), generator=generator).tolist():
            block = blocks[block_index]
            for offset in torch.randperm(len(block), generator=generator).tolist():
                pending.extend(self.groups[block[offset]])
                while len(pending) >= self.batch_size:
                    yield pending[: self.batch_size]
                    pending = pending[self.batch_size :]
        if pending:
            yield pending


class TallyQAImageGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle image groups while keeping repeated-image tally prompts close together."""

    def __init__(
        self,
        dataset: TallyQAStudentDataset,
        batch_size: int,
        seed: int,
        shuffle_block_size: int,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle_block_size = shuffle_block_size
        self.epoch = 0
        groups: dict[str, list[int]] = {}
        for local_index, dataset_index in enumerate(dataset.indices):
            image_id = str(dataset.rows[dataset_index]["image_id"])
            groups.setdefault(image_id, []).append(local_index)
        self.groups = list(groups.values())

    def __len__(self) -> int:
        return (sum(len(group) for group in self.groups) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterable[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        pending: list[int] = []
        blocks = [
            list(range(start, min(start + self.shuffle_block_size, len(self.groups))))
            for start in range(0, len(self.groups), self.shuffle_block_size)
        ]
        for block_index in torch.randperm(len(blocks), generator=generator).tolist():
            block = blocks[block_index]
            for offset in torch.randperm(len(block), generator=generator).tolist():
                pending.extend(self.groups[block[offset]])
                while len(pending) >= self.batch_size:
                    yield pending[: self.batch_size]
                    pending = pending[self.batch_size :]
        if pending:
            yield pending


def collate_student_batch(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_length = max(len(row["token_ids"]) for row in rows)
    token_ids = torch.zeros((len(rows), max_length), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["token_ids"])
        token_ids[index, :length] = torch.tensor(row["token_ids"], dtype=torch.long)
        attention_mask[index, :length] = True
    return {
        "dataset_index": torch.tensor([row["dataset_index"] for row in rows], dtype=torch.long),
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "images": torch.stack([row["image"] for row in rows]),
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
        "teacher_logits": torch.tensor([row["teacher_logit"] for row in rows], dtype=torch.float32),
    }


def collate_tallyqa_student_batch(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "dataset_index": torch.tensor([row["dataset_index"] for row in rows], dtype=torch.long),
        "token_ids": torch.stack([row["token_ids"] for row in rows]).long(),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]).bool(),
        "images": torch.stack([row["image"] for row in rows]),
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "teacher_probs": torch.stack([row["teacher_probs"] for row in rows]).float(),
        "item_class_ids": torch.tensor([row["item_class_id"] for row in rows], dtype=torch.long),
        "student_prompts": [str(row["student_prompt"]) for row in rows],
    }


class StudentDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root: Path,
        teacher_cache: Path | None,
        batch_size: int,
        num_workers: int,
        image_size: int,
        seed: int,
        row_group_cache_size: int = 4,
        tensor_cache_size: int = 256,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        group_train_by_image: bool = True,
        shuffle_train: bool = True,
        train_sampling: str = "natural",
        prompt_class_sampling_temperature: float = 0.5,
        train_epoch_size: int | None = None,
        shuffle_block_size: int = 256,
        missing_teacher_policy: str = "filter",
    ):
        super().__init__()
        if missing_teacher_policy not in {"filter", "keep"}:
            raise ValueError("missing_teacher_policy must be 'filter' or 'keep'.")
        self.save_hyperparameters()
        # Keep Lightning checkpoints compatible with PyTorch's safe
        # weights-only loader. pathlib.PosixPath is not allowlisted by default.
        self.hparams.dataset_root = str(dataset_root)
        self.hparams.teacher_cache = str(teacher_cache) if teacher_cache is not None else None
        self.rows = load_prompt_rows(dataset_root)
        self.vocabulary = CompactVocabulary.from_rows(self.rows)
        self.teacher_targets = load_teacher_targets(teacher_cache)
        self.full_indices = {"train": [], "val": [], "test": []}
        self.indices = {"train": [], "val": [], "test": []}
        for index, row in enumerate(self.rows):
            split = split_for_image(str(row["student_image_id"]), seed)
            self.full_indices[split].append(index)
            if missing_teacher_policy == "keep" or index in self.teacher_targets:
                self.indices[split].append(index)
        if missing_teacher_policy == "filter" and not sum(map(len, self.indices.values())):
            raise ValueError("The teacher cache does not contain any usable prompt targets.")

    def _dataset(self, split: str) -> StudentDataset:
        store = ParquetImageStore(
            Path(self.hparams.dataset_root) / "images.parquet",
            self.hparams.row_group_cache_size,
            self.hparams.tensor_cache_size,
            self.hparams.image_size,
        )
        return StudentDataset(self.rows, self.indices[split], self.vocabulary, self.teacher_targets, store)

    def _loader(self, split: str, shuffle: bool) -> DataLoader[dict[str, torch.Tensor]]:
        workers = int(self.hparams.num_workers)
        dataset = self._dataset(split)
        common = {
            "dataset": dataset,
            "num_workers": workers,
            "collate_fn": collate_student_batch,
            "pin_memory": self.hparams.pin_memory,
            "persistent_workers": self.hparams.persistent_workers and workers > 0,
            "prefetch_factor": self.hparams.prefetch_factor if workers > 0 else None,
        }
        if split == "train" and self.hparams.group_train_by_image and self.hparams.shuffle_train:
            return DataLoader(
                **common,
                batch_sampler=ImageGroupedBatchSampler(
                    dataset,
                    self.hparams.batch_size,
                    self.hparams.seed,
                    self.hparams.shuffle_block_size,
                ),
            )
        return DataLoader(
            **common,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader("train", shuffle=bool(self.hparams.shuffle_train))

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader("test", shuffle=False)

    def split_sizes(self) -> dict[str, int]:
        return {split: len(indices) for split, indices in self.indices.items()}

    def full_split_sizes(self) -> dict[str, int]:
        return {split: len(indices) for split, indices in self.full_indices.items()}

    def cache_coverage(self) -> dict[str, float | int | str]:
        covered = sum(index in self.teacher_targets for index in range(len(self.rows)))
        return {
            "policy": str(self.hparams.missing_teacher_policy),
            "covered_prompts": covered,
            "total_prompts": len(self.rows),
            "covered_fraction": covered / len(self.rows) if self.rows else 0.0,
            "active_prompts": sum(map(len, self.indices.values())),
        }


class TallyQAStudentDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root: Path,
        prompt_embeddings: Path,
        teacher_cache: Path | None,
        batch_size: int,
        num_workers: int,
        seed: int,
        tensor_cache_size: int = 256,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        group_train_by_image: bool = True,
        shuffle_train: bool = True,
        train_sampling: str = "natural",
        prompt_class_sampling_temperature: float = 0.5,
        train_epoch_size: int | None = None,
        shuffle_block_size: int = 256,
        missing_teacher_policy: str = "filter",
        collapse_at: int = 5,
        num_classes: int = 6,
        prompt_class_filter_csv: Path | None = None,
        min_prompt_accuracy: float | None = None,
        prompt_class_names: str | None = None,
        prompt_class_names_file: Path | None = None,
        curriculum_schedule: Path | None = None,
        train_example_limit: int | None = None,
        teacher_probability_temperature: float = 1.0,
    ):
        super().__init__()
        if missing_teacher_policy not in {"filter", "keep"}:
            raise ValueError("missing_teacher_policy must be 'filter' or 'keep'.")
        if collapse_at != num_classes - 1:
            raise ValueError("This TallyQA path expects collapse_at == num_classes - 1.")
        if train_example_limit is not None and train_example_limit <= 0:
            raise ValueError("train_example_limit must be positive when provided.")
        if train_sampling not in {"natural", "prompt_class_tempered"}:
            raise ValueError("train_sampling must be 'natural' or 'prompt_class_tempered'.")
        if prompt_class_sampling_temperature < 0:
            raise ValueError("prompt_class_sampling_temperature must be non-negative.")
        if train_epoch_size is not None and train_epoch_size <= 0:
            raise ValueError("train_epoch_size must be positive when provided.")
        if teacher_probability_temperature <= 0:
            raise ValueError("teacher_probability_temperature must be positive.")
        self.save_hyperparameters()
        self.hparams.dataset_root = str(dataset_root)
        self.hparams.prompt_embeddings = str(prompt_embeddings)
        self.hparams.teacher_cache = str(teacher_cache) if teacher_cache is not None else None
        self.hparams.prompt_class_filter_csv = (
            str(prompt_class_filter_csv) if prompt_class_filter_csv is not None else None
        )
        self.hparams.prompt_class_names_file = (
            str(prompt_class_names_file) if prompt_class_names_file is not None else None
        )
        self.hparams.curriculum_schedule = (
            str(curriculum_schedule) if curriculum_schedule is not None else None
        )

        prompt_payload = load_tallyqa_prompt_artifact(prompt_embeddings)
        self.prompt_token_ids = prompt_payload["prompt_token_ids"].long()
        self.prompt_attention_mask = prompt_payload["prompt_attention_mask"].bool()
        self.embedding_rows = prompt_payload["embedding_rows"].float()
        self.prompt_classes = prompt_payload.get("prompt_classes", [])

        self.rows = load_tallyqa_rows(dataset_root)
        self.teacher_targets = load_tallyqa_teacher_targets(
            teacher_cache,
            num_classes=num_classes,
            collapse_at=collapse_at,
            probability_temperature=teacher_probability_temperature,
        )
        prompt_filters = [
            prompt_filter
            for prompt_filter in (
                load_prompt_accuracy_filter(prompt_class_filter_csv, min_prompt_accuracy),
                parse_prompt_class_names(prompt_class_names),
                load_prompt_class_names_file(prompt_class_names_file),
            )
            if prompt_filter is not None
        ]
        prompt_filter = set.intersection(*prompt_filters) if prompt_filters else None
        if prompt_filter is not None and not prompt_filter:
            raise ValueError("Prompt class filters produced an empty prompt set.")
        self.full_indices = {"train": [], "val": [], "test": []}
        self.indices = {"train": [], "val": [], "test": []}
        for index, row in enumerate(self.rows):
            split = split_for_image(str(row["image_id"]), seed)
            self.full_indices[split].append(index)
            prompt_allowed = prompt_filter is None or str(row["student_prompt"]) in prompt_filter
            teacher_allowed = missing_teacher_policy == "keep" or index in self.teacher_targets
            if prompt_allowed and teacher_allowed:
                self.indices[split].append(index)
        if train_example_limit is not None:
            self.indices["train"] = self.indices["train"][:train_example_limit]
        if missing_teacher_policy == "filter" and not sum(map(len, self.indices.values())):
            raise ValueError("The teacher cache does not contain any usable TallyQA targets.")
        self._base_train_indices = list(self.indices["train"])
        self._curriculum_schedule = load_curriculum_schedule(curriculum_schedule)
        self._curriculum_epoch = 0
        self._curriculum_stage: dict[str, Any] | None = None
        self.set_train_epoch(0)

    def _stage_prompt_filter(self, stage: dict[str, Any] | None) -> set[str] | None:
        if stage is None:
            return None
        filters = [
            prompt_filter
            for prompt_filter in (
                parse_prompt_class_names(stage.get("prompt_class_names")),
                load_prompt_class_names_file(
                    Path(stage["prompt_class_names_file"])
                    if stage.get("prompt_class_names_file") is not None
                    else None
                ),
            )
            if prompt_filter is not None
        ]
        return set.intersection(*filters) if filters else None

    def set_train_epoch(self, zero_based_epoch: int) -> None:
        if not self._curriculum_schedule:
            return
        one_based_epoch = int(zero_based_epoch) + 1
        active = self._curriculum_schedule[0]
        for stage in self._curriculum_schedule:
            if int(stage["start_epoch"]) <= one_based_epoch:
                active = stage
            else:
                break
        self._curriculum_epoch = one_based_epoch
        self._curriculum_stage = active

    def _train_indices(self) -> list[int]:
        prompt_filter = self._stage_prompt_filter(self._curriculum_stage)
        if prompt_filter is None:
            return list(self._base_train_indices)
        indices = [
            index
            for index in self._base_train_indices
            if str(self.rows[index]["student_prompt"]) in prompt_filter
        ]
        if not indices:
            raise ValueError(f"Curriculum stage produced no train examples: {self._curriculum_stage}")
        return indices

    def _train_sampling(self) -> str:
        if self._curriculum_stage is not None:
            return str(self._curriculum_stage.get("train_sampling", self.hparams.train_sampling))
        return str(self.hparams.train_sampling)

    def _prompt_sampling_temperature(self) -> float:
        if self._curriculum_stage is not None:
            return float(
                self._curriculum_stage.get(
                    "prompt_class_sampling_temperature",
                    self.hparams.prompt_class_sampling_temperature,
                )
            )
        return float(self.hparams.prompt_class_sampling_temperature)

    def _train_epoch_size(self, dataset: TallyQAStudentDataset) -> int:
        if self._curriculum_stage is not None and self._curriculum_stage.get("train_epoch_size") is not None:
            return int(self._curriculum_stage["train_epoch_size"])
        if self.hparams.train_epoch_size is not None:
            return int(self.hparams.train_epoch_size)
        return len(dataset)

    def _prompt_class_sampling_weights(self, dataset: TallyQAStudentDataset) -> torch.Tensor:
        temperature = self._prompt_sampling_temperature()
        counts: dict[int, int] = {}
        for dataset_index in dataset.indices:
            class_id = int(dataset.rows[dataset_index]["item_class_id"])
            counts[class_id] = counts.get(class_id, 0) + 1
        return torch.tensor(
            [
                counts[int(dataset.rows[dataset_index]["item_class_id"])] ** (-temperature)
                for dataset_index in dataset.indices
            ],
            dtype=torch.double,
        )

    def _dataset(self, split: str) -> TallyQAStudentDataset:
        store = Uint8MemmapImageStore(
            Path(self.hparams.dataset_root),
            tensor_cache_size=self.hparams.tensor_cache_size,
        )
        return TallyQAStudentDataset(
            rows=self.rows,
            indices=self._train_indices() if split == "train" else self.indices[split],
            prompt_token_ids=self.prompt_token_ids,
            prompt_attention_mask=self.prompt_attention_mask,
            teacher_targets=self.teacher_targets,
            image_store=store,
            collapse_at=self.hparams.collapse_at,
            num_classes=self.hparams.num_classes,
        )

    def _loader(self, split: str, shuffle: bool) -> DataLoader[dict[str, torch.Tensor]]:
        workers = int(self.hparams.num_workers)
        dataset = self._dataset(split)
        common = {
            "dataset": dataset,
            "num_workers": workers,
            "collate_fn": collate_tallyqa_student_batch,
            "pin_memory": self.hparams.pin_memory,
            "persistent_workers": self.hparams.persistent_workers and workers > 0,
            "prefetch_factor": self.hparams.prefetch_factor if workers > 0 else None,
        }
        if split == "train" and self._train_sampling() == "prompt_class_tempered":
            if not self.hparams.shuffle_train:
                raise ValueError("prompt_class_tempered train sampling requires shuffle_train=true.")
            epoch_size = self._train_epoch_size(dataset)
            return DataLoader(
                **common,
                batch_size=self.hparams.batch_size,
                sampler=WeightedRandomSampler(
                    weights=self._prompt_class_sampling_weights(dataset),
                    num_samples=epoch_size,
                    replacement=True,
                    generator=torch.Generator().manual_seed(
                        int(self.hparams.seed) + int(self._curriculum_epoch)
                    ),
                ),
            )
        if split == "train" and self.hparams.group_train_by_image and self.hparams.shuffle_train:
            return DataLoader(
                **common,
                batch_sampler=TallyQAImageGroupedBatchSampler(
                    dataset,
                    self.hparams.batch_size,
                    self.hparams.seed,
                    self.hparams.shuffle_block_size,
                ),
            )
        return DataLoader(
            **common,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader("train", shuffle=bool(self.hparams.shuffle_train))

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return self._loader("test", shuffle=False)

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": len(self._train_indices()),
            "val": len(self.indices["val"]),
            "test": len(self.indices["test"]),
        }

    def full_split_sizes(self) -> dict[str, int]:
        return {split: len(indices) for split, indices in self.full_indices.items()}

    def label_counts(self, split: str = "train") -> dict[int, int]:
        counts = {class_id: 0 for class_id in range(int(self.hparams.num_classes))}
        for index in self.indices[split]:
            label = collapse_count(int(self.rows[index]["answer"]), int(self.hparams.collapse_at))
            counts[label] += 1
        return counts

    def cache_coverage(self) -> dict[str, float | int | str]:
        covered = sum(index in self.teacher_targets for index in range(len(self.rows)))
        return {
            "policy": str(self.hparams.missing_teacher_policy),
            "covered_prompts": covered,
            "total_prompts": len(self.rows),
            "covered_fraction": covered / len(self.rows) if self.rows else 0.0,
            "active_prompts": sum(map(len, self.indices.values())),
            "prompt_class_filter_csv": self.hparams.prompt_class_filter_csv,
            "min_prompt_accuracy": self.hparams.min_prompt_accuracy,
            "prompt_class_names": self.hparams.prompt_class_names,
            "prompt_class_names_file": self.hparams.prompt_class_names_file,
            "train_sampling": self.hparams.train_sampling,
            "prompt_class_sampling_temperature": self.hparams.prompt_class_sampling_temperature,
            "teacher_probability_temperature": self.hparams.teacher_probability_temperature,
            "train_epoch_size": self.hparams.train_epoch_size,
            "curriculum_schedule": self.hparams.curriculum_schedule,
            "curriculum_epoch": self._curriculum_epoch,
            "curriculum_stage": self._curriculum_stage,
        }
