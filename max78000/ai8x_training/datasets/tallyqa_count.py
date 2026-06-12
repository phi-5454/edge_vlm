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
import os
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
PROMPT_EMBEDDING_CHANNELS = 576
PROMPT_PLANE_CHANNELS = 16
DEFAULT_PROMPT_EMBEDDINGS = "artifacts/models/tallyqa_smolvlm_prompt_embeddings_letterbox.pt"
DEFAULT_PROMPT_PLANE_LOOKUP = "max78000/prompt_embeddings/tallyqa_prompt_planes16_random.json"


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


def _class_weights_from_env(num_classes: int) -> dict[str, list[float]]:
    raw = os.environ.get("EDGE_VLM_TALLYQA_CLASS_WEIGHTS")
    if not raw:
        return {}
    weights = [float(value) for value in json.loads(raw)]
    if len(weights) != num_classes:
        raise ValueError(
            f"EDGE_VLM_TALLYQA_CLASS_WEIGHTS has {len(weights)} values; "
            f"expected {num_classes}."
        )
    return {"weight": weights}


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
                "student_prompt": "people",
            }
        )
    return records


def _resolve_tensor_file(root: Path, metadata: dict[str, Any]) -> Path:
    tensor_file = Path(metadata["image"]["tensor_file"])
    if tensor_file.is_absolute():
        return tensor_file
    return root / tensor_file


def _resolve_prompt_embeddings_file(root: Path, metadata: dict[str, Any]) -> Path:
    configured = metadata.get("prompt_embeddings")
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured)
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            candidates.append(root / configured_path)
    env_path = Path(value) if (value := os.environ.get("EDGE_VLM_PROMPT_EMBEDDINGS")) else None
    if env_path is not None:
        candidates.append(env_path)
    for parent in (root, *root.parents):
        candidates.append(parent / DEFAULT_PROMPT_EMBEDDINGS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find prompt embeddings artifact. Set EDGE_VLM_PROMPT_EMBEDDINGS "
        f"or place {DEFAULT_PROMPT_EMBEDDINGS} under the edge_vlm repo. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _resolve_prompt_plane_lookup_file(root: Path, metadata: dict[str, Any]) -> Path:
    configured = metadata.get("prompt_plane_lookup")
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured)
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            candidates.append(root / configured_path)
    env_path = Path(value) if (value := os.environ.get("EDGE_VLM_PROMPT_PLANE_LOOKUP")) else None
    if env_path is not None:
        candidates.append(env_path)
    for parent in (root, *root.parents):
        candidates.append(parent / DEFAULT_PROMPT_PLANE_LOOKUP)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find prompt-plane lookup. Set EDGE_VLM_PROMPT_PLANE_LOOKUP "
        f"or place {DEFAULT_PROMPT_PLANE_LOOKUP} under the edge_vlm repo. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_prompt_embeddings_by_class(
    root: Path,
    metadata: dict[str, Any],
) -> dict[str, torch.Tensor]:
    path = _resolve_prompt_embeddings_file(root, metadata)
    payload = torch.load(path, map_location="cpu")
    required = ("prompt_token_ids", "prompt_attention_mask", "embedding_rows", "prompt_classes")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Prompt embedding artifact {path} is missing keys: {missing}")

    token_ids = payload["prompt_token_ids"].long()
    attention_mask = payload["prompt_attention_mask"].bool()
    embedding_rows = payload["embedding_rows"].float()
    prompt_classes = [
        str(item.get("item", item) if isinstance(item, dict) else item).strip().lower()
        for item in payload["prompt_classes"]
    ]
    if embedding_rows.shape[1] != PROMPT_EMBEDDING_CHANNELS:
        raise ValueError(
            f"Expected {PROMPT_EMBEDDING_CHANNELS}-d prompt embeddings, "
            f"got {embedding_rows.shape[1]} from {path}."
        )

    pad_row = torch.zeros((1, embedding_rows.shape[1]), dtype=embedding_rows.dtype)
    embedding_table = torch.cat((pad_row, embedding_rows), dim=0)
    token_vectors = embedding_table[token_ids]
    mask = attention_mask.unsqueeze(-1).to(token_vectors.dtype)
    prompt_vectors = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return {
        prompt_class: prompt_vectors[index].contiguous()
        for index, prompt_class in enumerate(prompt_classes)
    }


def _load_prompt_planes_by_class(
    root: Path,
    metadata: dict[str, Any],
) -> dict[str, torch.Tensor]:
    path = _resolve_prompt_plane_lookup_file(root, metadata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("plane_channels", 0)) != PROMPT_PLANE_CHANNELS:
        raise ValueError(
            f"Expected {PROMPT_PLANE_CHANNELS} prompt-plane channels in {path}, "
            f"got {payload.get('plane_channels')}."
        )
    vectors = payload.get("prompt_vectors")
    if not isinstance(vectors, dict):
        raise ValueError(f"{path} does not contain a prompt_vectors object.")
    return {
        str(prompt).strip().lower(): torch.tensor(values, dtype=torch.float32)
        for prompt, values in vectors.items()
    }


def _prompt_planes(prompt_vector: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Project a 576-d prompt vector to 16 broadcast planes.

    This is a synthesis-friendly replacement for the dynamic FiLM path: the
    prompt remains explicit in the input tensor, but the accelerator sees a
    single CHW input.
    """
    prompt_vector = prompt_vector.to(dtype=torch.float32)
    prompt_vector = prompt_vector / prompt_vector.norm(p=2).clamp_min(1e-6)
    prompt_vector = prompt_vector[:PROMPT_PLANE_CHANNELS].clamp(-1.0, 1.0)
    return prompt_vector.view(PROMPT_PLANE_CHANNELS, 1, 1).expand(-1, height, width)


class TallyQACount(Dataset):
    """TallyQA count classification dataset from a materialized manifest."""

    def __init__(
        self,
        root_dir: str | Path,
        d_type: str,
        transform=None,
        seed: int = 0,
        prompt_embedding_channels: int = 0,
        prompt_plane_channels: int = 0,
    ):
        if d_type not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {d_type!r}; expected train, val, or test.")
        self.root_dir = _resolve_dataset_root(root_dir)
        self.d_type = d_type
        self.transform = transform
        self.prompt_embedding_channels = int(prompt_embedding_channels)
        self.prompt_plane_channels = int(prompt_plane_channels)
        if self.prompt_embedding_channels and self.prompt_plane_channels:
            raise ValueError("Use either prompt_embedding_channels or prompt_plane_channels, not both.")
        self.records = _load_records(self.root_dir, d_type, seed)
        if not self.records:
            raise RuntimeError(f"No TallyQA count records found for split {d_type!r}.")

        self.metadata = json.loads((self.root_dir / "metadata.json").read_text(encoding="utf-8"))
        self.prompt_classes = [
            str(item).strip().lower() for item in self.metadata.get("prompt_classes") or ["people"]
        ]
        if self.prompt_embedding_channels or self.prompt_plane_channels:
            if self.prompt_embedding_channels and self.prompt_embedding_channels != PROMPT_EMBEDDING_CHANNELS:
                raise ValueError(
                    f"Prompt embedding channels must be {PROMPT_EMBEDDING_CHANNELS}, "
                    f"got {self.prompt_embedding_channels}."
                )
            if self.prompt_plane_channels and self.prompt_plane_channels != PROMPT_PLANE_CHANNELS:
                raise ValueError(
                    f"Prompt plane channels must be {PROMPT_PLANE_CHANNELS}, "
                    f"got {self.prompt_plane_channels}."
                )
            if self.prompt_plane_channels:
                self.prompt_embeddings_by_class = _load_prompt_planes_by_class(
                    self.root_dir,
                    self.metadata,
                )
            else:
                self.prompt_embeddings_by_class = _load_prompt_embeddings_by_class(
                    self.root_dir,
                    self.metadata,
                )
            missing_prompts = [
                prompt for prompt in self.prompt_classes if prompt not in self.prompt_embeddings_by_class
            ]
            if missing_prompts:
                raise KeyError(
                    "Prompt embedding artifact is missing dataset prompt classes: "
                    + ", ".join(missing_prompts[:10])
                )
        else:
            self.prompt_embeddings_by_class = {}
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
        label_value = int(record["label"])
        label = torch.tensor(label_value, dtype=torch.int64)
        if self.transform:
            image = self.transform(Image.fromarray(image_hwc))
        else:
            image = torch.from_numpy(image_chw.copy()).float().div(255.0)
        prompt_vector = None
        if self.prompt_embedding_channels or self.prompt_plane_channels:
            prompt = str(record.get("student_prompt") or "people").strip().lower()
            if prompt not in self.prompt_embeddings_by_class:
                raise KeyError(
                    f"Prompt {prompt!r} is not in prompt embeddings for {self.root_dir}."
                )
            prompt_vector = self.prompt_embeddings_by_class[prompt].to(dtype=image.dtype)
            if self.prompt_embedding_channels:
                image = (image, prompt_vector)
            else:
                if prompt_vector.numel() == PROMPT_PLANE_CHANNELS:
                    prompt_planes = prompt_vector.view(PROMPT_PLANE_CHANNELS, 1, 1).expand(
                        -1,
                        image.shape[-2],
                        image.shape[-1],
                    )
                else:
                    prompt_planes = _prompt_planes(prompt_vector, image.shape[-2], image.shape[-1])
                image = torch.cat((image, prompt_planes.to(dtype=image.dtype)), dim=0)
        if "teacher_probs" in record:
            teacher_probs = torch.tensor(record["teacher_probs"], dtype=torch.float32)
            if teacher_probs.numel() != len(COUNT_LABELS):
                raise ValueError(
                    f"Expected {len(COUNT_LABELS)} teacher probabilities, got "
                    f"{teacher_probs.numel()} for example {record.get('example_id')}."
                )
            target = torch.cat(
                (
                    torch.tensor([float(label_value)], dtype=torch.float32),
                    teacher_probs,
                )
            )
            return image, target
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


def _get_tallyqa_count_dataset(
    data,
    load_train,
    load_test,
    prompt_embedding_channels: int,
    prompt_plane_channels: int = 0,
):
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
        TallyQACount(
            root_dir=data_dir,
            d_type="train",
            transform=transform,
            seed=seed,
            prompt_embedding_channels=prompt_embedding_channels,
            prompt_plane_channels=prompt_plane_channels,
        )
        if load_train
        else None
    )
    test_dataset = (
        TallyQACount(
            root_dir=data_dir,
            d_type="test",
            transform=transform,
            seed=seed,
            prompt_embedding_channels=prompt_embedding_channels,
            prompt_plane_channels=prompt_plane_channels,
        )
        if load_test
        else None
    )

    if test_dataset is not None and getattr(args, "truncate_testset", False):
        test_dataset.records = test_dataset.records[:1]

    return train_dataset, test_dataset


def get_tallyqa_count_dataset(data, load_train, load_test):
    """Load TallyQA count datasets without prompt conditioning channels."""
    return _get_tallyqa_count_dataset(data, load_train, load_test, prompt_embedding_channels=0)


def get_tallyqa_count_prompt_embed576_dataset(data, load_train, load_test):
    """Load TallyQA count datasets with 576-d precomputed prompt vectors."""
    return _get_tallyqa_count_dataset(
        data,
        load_train,
        load_test,
        prompt_embedding_channels=PROMPT_EMBEDDING_CHANNELS,
    )


def get_tallyqa_count_prompt_planes16_dataset(data, load_train, load_test):
    """Load TallyQA count datasets with 16 broadcast prompt planes."""
    return _get_tallyqa_count_dataset(
        data,
        load_train,
        load_test,
        prompt_embedding_channels=0,
        prompt_plane_channels=PROMPT_PLANE_CHANNELS,
    )


datasets = [
    {
        "name": "tallyqa_count_fold2_56",
        "input": (FOLDED_CHANNELS, FOLDED_SIZE, FOLDED_SIZE),
        "output": COUNT_LABELS,
        "loader": get_tallyqa_count_dataset,
        **_class_weights_from_env(len(COUNT_LABELS)),
    },
    {
        "name": "tallyqa_count_fold2_56_prompt_embed576",
        "input": (FOLDED_CHANNELS, FOLDED_SIZE, FOLDED_SIZE),
        "output": COUNT_LABELS,
        "loader": get_tallyqa_count_prompt_embed576_dataset,
        **_class_weights_from_env(len(COUNT_LABELS)),
    },
    {
        "name": "tallyqa_count_fold2_56_prompt_planes16",
        "input": (FOLDED_CHANNELS + PROMPT_PLANE_CHANNELS, FOLDED_SIZE, FOLDED_SIZE),
        "output": COUNT_LABELS,
        "loader": get_tallyqa_count_prompt_planes16_dataset,
        **_class_weights_from_env(len(COUNT_LABELS)),
    },
]
