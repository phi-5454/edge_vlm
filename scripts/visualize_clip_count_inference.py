from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from scripts.cache_smolvlm_tallyqa_teacher import Uint8ImageStore, load_examples, load_metadata


DEFAULT_CLIP_COUNT_REPO = Path("../CLIP-Count")
DEFAULT_CHECKPOINT = Path("external_models/clipcount_pretrained.ckpt")
DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_SOURCE_DATASET = Path("data/the_cauldron/tallyqa")
DEFAULT_OUTPUT = Path("artifacts/reports/clip_count_examples/example_inference.png")
SCALE_FACTOR = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one CLIP-Count example and save image, density heatmap, and count."
    )
    parser.add_argument("--clip-count-repo", type=Path, default=DEFAULT_CLIP_COUNT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument(
        "--image-source",
        choices=["original", "target"],
        default="original",
        help="Use original Cauldron TallyQA images or target MobileNet-ready images.",
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--dataset-index", type=int, default=None)
    parser.add_argument(
        "--dataset-indices",
        default=None,
        help="Comma-separated dataset indices. If set, produces one figure row per input.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of default examples to select when neither index option is set.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--height", type=int, default=384)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


@contextmanager
def redirect_hardcoded_cuda_to_cpu(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    original_module_to = torch.nn.Module.to
    original_tensor_to = torch.Tensor.to

    def normalize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args = tuple("cpu" if value == "cuda" else value for value in args)
        kwargs = dict(kwargs)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return args, kwargs

    def module_to_cpu(self: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        args, kwargs = normalize_args(args, kwargs)
        return original_module_to(self, *args, **kwargs)

    def tensor_to_cpu(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        args, kwargs = normalize_args(args, kwargs)
        return original_tensor_to(self, *args, **kwargs)

    torch.nn.Module.to = module_to_cpu  # type: ignore[method-assign]
    torch.Tensor.to = tensor_to_cpu  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.Module.to = original_module_to  # type: ignore[method-assign]
        torch.Tensor.to = original_tensor_to  # type: ignore[method-assign]


def import_clip_count(repo: Path) -> type[torch.nn.Module]:
    repo = repo.resolve()
    if not (repo / "models" / "clip_count.py").exists():
        raise FileNotFoundError(f"Could not find CLIP-Count model code under {repo}")
    sys.path.insert(0, str(repo))
    from models.clip_count import CLIPCount  # type: ignore[import-not-found]

    return CLIPCount


def build_model(clip_count_repo: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    CLIPCount = import_clip_count(clip_count_repo)
    with redirect_hardcoded_cuda_to_cpu(device.type == "cpu"):
        model = CLIPCount(
            fim_depth=int(hparams.get("decoder_depth", 4)),
            fim_num_heads=int(hparams.get("decoder_head", 8)),
            use_coop=bool(hparams.get("use_coop", True)),
            use_vpt=bool(hparams.get("use_vpt", True)),
            coop_width=int(hparams.get("coop_width", 2)),
            vpt_width=int(hparams.get("vpt_width", 20)),
            vpt_depth=int(hparams.get("vpt_depth", 10)),
            backbone=str(hparams.get("backbone", "b16")),
            use_fim=bool(hparams.get("use_fim", False)),
            use_mixed_fim=bool(hparams.get("use_mixed_fim", True)),
            unfreeze_vit=bool(hparams.get("unfreeze_vit", False)),
        )

    state_dict = {
        key.removeprefix("model."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        non_clip_missing = [
            key
            for key in missing
            if not key.startswith(("clip.", "img_encoder.vit.", "text_encoder.clip_model."))
        ]
        print(
            json.dumps(
                {
                    "load_state_dict": {
                        "missing_key_count": len(missing),
                        "unexpected_key_count": len(unexpected),
                        "non_clip_missing_keys": non_clip_missing,
                        "unexpected_keys": unexpected,
                    }
                },
                indent=2,
            )
        )
    return model.to(device).eval()


def parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise ValueError("--dataset-indices was set but no indices were parsed.")
    return indices


def load_input_image(args: argparse.Namespace) -> tuple[Image.Image, str, int | None, int | None]:
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        if args.prompt is None:
            raise ValueError("--prompt is required when --image is provided.")
        return image, args.prompt, None, None

    examples = load_examples(args.dataset)
    metadata = load_metadata(args.dataset)
    if args.dataset_index is None:
        dataset_index = next(
            index for index, row in enumerate(examples) if str(row["student_prompt"]) == "people"
        )
    else:
        dataset_index = args.dataset_index
    row = examples[dataset_index]
    prompt = args.prompt or str(row["student_prompt"])
    if args.image_source == "original":
        from datasets import load_from_disk

        source_dataset = load_from_disk(str(args.source_dataset))
        image = source_dataset[int(row["source_row_index"])]["images"][0]
    else:
        image_store = Uint8ImageStore(args.dataset, metadata)
        image, _identity = image_store.get(int(row["image_index"]))
    return image.convert("RGB"), prompt, dataset_index, int(row["answer"])


def load_dataset_examples(args: argparse.Namespace) -> list[tuple[Image.Image, str, int, int]]:
    if args.image is not None:
        image, prompt, dataset_index, true_count = load_input_image(args)
        return [(image, prompt, dataset_index if dataset_index is not None else -1, true_count or -1)]

    examples = load_examples(args.dataset)
    metadata = load_metadata(args.dataset)
    if args.image_source == "original":
        from datasets import load_from_disk

        source_dataset = load_from_disk(str(args.source_dataset))
        image_store = None
    else:
        source_dataset = None
        image_store = Uint8ImageStore(args.dataset, metadata)
    indices = parse_indices(args.dataset_indices)
    if indices is None:
        if args.dataset_index is not None:
            indices = [args.dataset_index]
        else:
            seen_prompts: set[str] = set()
            indices = []
            preferred_prompts = ["people", "cars", "chairs", "dogs"]
            for prompt in preferred_prompts:
                for index, row in enumerate(examples):
                    if str(row["student_prompt"]) == prompt:
                        indices.append(index)
                        seen_prompts.add(prompt)
                        break
            if len(indices) < args.count:
                for index, row in enumerate(examples):
                    prompt = str(row["student_prompt"])
                    if prompt in seen_prompts:
                        continue
                    indices.append(index)
                    seen_prompts.add(prompt)
                    if len(indices) >= args.count:
                        break
            indices = indices[: args.count]

    rows: list[tuple[Image.Image, str, int, int]] = []
    for index in indices:
        row = examples[index]
        prompt = args.prompt or str(row["student_prompt"])
        if args.image_source == "original":
            image = source_dataset[int(row["source_row_index"])]["images"][0]
        else:
            image, _identity = image_store.get(int(row["image_index"]))
        rows.append((image.convert("RGB"), prompt, index, int(row["answer"])))
    return rows


def prepare_image(image: Image.Image, height: int, device: torch.device) -> torch.Tensor:
    tensor = TF.pil_to_tensor(image).unsqueeze(0)
    original_width, original_height = image.size
    resized_width = max(1, round(original_width * height / original_height))
    tensor = TF.resize(tensor, [height, resized_width], antialias=True).float() / 255.0
    return tensor.clamp(0, 1).to(device)


def sliding_window(image: torch.Tensor, stride: int) -> torch.Tensor:
    if image.shape[0] == 1:
        image = image.squeeze(0)
    if image.shape[1] != 384:
        raise ValueError(f"CLIP-Count sliding window expects height 384, got {image.shape[1]}")
    chw = image.detach().cpu().numpy()
    hwc = np.transpose(chw, (1, 2, 0))
    target_width = max(384, hwc.shape[1])
    remainder = target_width % stride
    if remainder:
        target_width += stride - remainder
    pad = target_width - hwc.shape[1]
    hwc = np.pad(hwc, ((0, 0), (0, pad), (0, 0)), "constant")
    patches = [hwc[:, start : start + 384, :] for start in range(0, hwc.shape[1] - 384 + 1, stride)]
    return torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float()


def window_composite(patches: torch.Tensor, stride: int) -> torch.Tensor:
    image = patches[0]
    blend_width = 384 - stride
    for patch in patches[1:]:
        blend_factor = torch.sigmoid(torch.linspace(-3, 3, blend_width, device=patches.device))
        blend_factor = blend_factor.view(1, 1, -1)
        blend = (1 - blend_factor) * image[:, :, -blend_width:] + blend_factor * patch[:, :, :blend_width]
        image[:, :, -blend_width:] = blend
        image = torch.cat([image, patch[:, :, blend_width:]], dim=-1)
    return image.unsqueeze(0)


@torch.inference_mode()
def infer(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    prompt: str,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    raw_width = image_tensor.shape[-1]
    patches = sliding_window(image_tensor, stride=stride).to(device)
    prompts = np.repeat([prompt], patches.shape[0], axis=0)
    density = model(patches, prompts)
    density = density.unsqueeze(1)
    density = window_composite(density, stride=stride).squeeze(1)
    return density[:, :, :raw_width]


def save_plot(
    image_tensor: torch.Tensor,
    density: torch.Tensor,
    prompt: str,
    output: Path,
    dataset_index: int | None,
    true_count: int | None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    image_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    density_np = density[0].detach().cpu().float().numpy()
    pred_count = float(density_np.sum() / SCALE_FACTOR)
    density_norm = density_np / max(float(density_np.max()), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(density_np, cmap="magma")
    axes[1].set_title("Density")
    axes[2].imshow(image_np)
    axes[2].imshow(density_norm, cmap="magma", alpha=0.65)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    title = f"prompt={prompt!r} pred={pred_count:.2f}"
    if true_count is not None:
        title += f" true={true_count}"
    if dataset_index is not None:
        title += f" dataset_index={dataset_index}"
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    density_path = output.with_suffix(".density.npy")
    np.save(density_path, density_np)
    summary = {
        "output": str(output),
        "density": str(density_path),
        "prompt": prompt,
        "predicted_count": pred_count,
        "true_count": true_count,
        "dataset_index": dataset_index,
        "density_shape": list(density_np.shape),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def save_multi_plot(
    rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13, max(3.2, 3.2 * len(rows))),
        constrained_layout=True,
        squeeze=False,
    )
    summaries: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        image_np = row["image_tensor"][0].permute(1, 2, 0).detach().cpu().numpy()
        density_np = row["density"][0].detach().cpu().float().numpy()
        density_norm = density_np / max(float(density_np.max()), 1e-8)
        pred_count = float(density_np.sum() / SCALE_FACTOR)

        axes[row_index, 0].imshow(image_np)
        axes[row_index, 0].set_title("Input")
        axes[row_index, 1].imshow(density_np, cmap="magma")
        axes[row_index, 1].set_title("Density")
        axes[row_index, 2].imshow(image_np)
        axes[row_index, 2].imshow(density_norm, cmap="magma", alpha=0.65)
        axes[row_index, 2].set_title("Overlay")
        for ax in axes[row_index]:
            ax.axis("off")
        label = (
            f"idx={row['dataset_index']}  prompt={row['prompt']!r}\n"
            f"pred={pred_count:.2f}  true={row['true_count']}"
        )
        axes[row_index, 0].text(
            0.02,
            0.98,
            label,
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 4, "edgecolor": "none"},
        )

        density_path = output.with_name(
            f"{output.stem}_{row['dataset_index']:06d}_{row['prompt'].replace(' ', '_')}.density.npy"
        )
        np.save(density_path, density_np)
        summaries.append(
            {
                "dataset_index": row["dataset_index"],
                "prompt": row["prompt"],
                "true_count": row["true_count"],
                "predicted_count": pred_count,
                "density": str(density_path),
                "density_shape": list(density_np.shape),
            }
        )

    fig.suptitle("CLIP-Count example density maps")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    summary = {"output": str(output), "examples": summaries}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args.clip_count_repo, args.checkpoint, device=device)
    examples = load_dataset_examples(args)
    outputs: list[dict[str, Any]] = []
    for image, prompt, dataset_index, true_count in examples:
        image_tensor = prepare_image(image, height=args.height, device=device)
        density = infer(model, image_tensor, prompt=prompt, stride=args.stride, device=device)
        outputs.append(
            {
                "image_tensor": image_tensor,
                "density": density,
                "prompt": prompt,
                "dataset_index": dataset_index,
                "true_count": true_count,
            }
        )
    if len(outputs) == 1:
        only = outputs[0]
        summary = save_plot(
            image_tensor=only["image_tensor"],
            density=only["density"],
            prompt=only["prompt"],
            output=args.output,
            dataset_index=only["dataset_index"],
            true_count=only["true_count"],
        )
    else:
        summary = save_multi_plot(outputs, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
