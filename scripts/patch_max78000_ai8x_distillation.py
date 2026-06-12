#!/usr/bin/env python3
"""Patch ADI ai8x-training train.py for edge_vlm cached teacher distillation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HELPER_MARKER = "# edge_vlm cached-teacher distillation helpers"


HELPERS = f'''
{HELPER_MARKER}
EDGE_VLM_LAST_DISTILLATION_STATS = {{}}


def edge_vlm_split_distillation_target(target):
    """Return hard labels and optional cached teacher probabilities."""
    if (
        torch.is_tensor(target)
        and target.ndim == 2
        and target.size(1) > 1
        and torch.is_floating_point(target)
    ):
        labels = target[:, 0].long()
        teacher_probs = target[:, 1:].float()
        return labels, teacher_probs
    return target, None


def edge_vlm_distillation_loss(output, target, criterion):
    """Cross-entropy plus optional cached-teacher KL distillation."""
    global EDGE_VLM_LAST_DISTILLATION_STATS
    labels, teacher_probs = edge_vlm_split_distillation_target(target)
    ce_loss = criterion(output, labels)
    EDGE_VLM_LAST_DISTILLATION_STATS = {{
        "ce_loss": ce_loss.detach(),
        "kl_loss": None,
        "distillation_loss": ce_loss.detach(),
    }}
    if teacher_probs is None:
        return ce_loss
    beta = float(os.environ.get("EDGE_VLM_DISTILLATION_BETA", "0.25"))
    if beta <= 0:
        return ce_loss
    alpha = float(os.environ.get("EDGE_VLM_DISTILLATION_ALPHA", "1.0"))
    temperature = float(os.environ.get("EDGE_VLM_DISTILLATION_TEMPERATURE", "2.0"))
    teacher_probs = teacher_probs.clamp_min(1e-8)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student_log_probs = nn.functional.log_softmax(output / temperature, dim=1)
    kl_loss = nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)
    total_loss = alpha * ce_loss + beta * kl_loss
    EDGE_VLM_LAST_DISTILLATION_STATS = {{
        "ce_loss": ce_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "distillation_loss": total_loss.detach(),
    }}
    return total_loss


def edge_vlm_add_distillation_meters(losses):
    """Expose cached-teacher loss components through ai8x's normal meters."""
    for meter_name, stats_key in (
        ("CE Loss", "ce_loss"),
        ("KL Loss", "kl_loss"),
        ("Distillation Loss", "distillation_loss"),
    ):
        value = EDGE_VLM_LAST_DISTILLATION_STATS.get(stats_key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = float(value.detach().item())
        if meter_name not in losses:
            losses[meter_name] = tnt.AverageValueMeter()
        losses[meter_name].add(float(value))


def edge_vlm_hard_labels(target):
    """Return hard labels from scalar or cached-teacher target tensors."""
    labels, _teacher_probs = edge_vlm_split_distillation_target(target)
    return labels

'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai8x-training", type=Path, default=Path("../MAX78000/ai8x-training"))
    return parser.parse_args()


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new, 1)


def main() -> None:
    args = parse_args()
    train_py = args.ai8x_training / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(train_py)
    text = train_py.read_text(encoding="utf-8")
    had_meter_patch = "edge_vlm_add_distillation_meters(losses)" in text

    if HELPER_MARKER not in text:
        text = text.replace('matplotlib.use("pgf")\n', 'matplotlib.use("pgf")\n\n' + HELPERS, 1)
    elif "edge_vlm_add_distillation_meters" not in text:
        text = text.replace("def main():\n", HELPERS + "\ndef main():\n", 1)

    text = text.replace("loss = criterion(output, target)", "loss = edge_vlm_distillation_loss(output, target, criterion)")
    if not had_meter_patch:
        text = re.sub(
            r"(?m)^(?P<indent>\s*)losses\[OBJECTIVE_LOSS_KEY\]\.add\(loss\.item\(\)\)$",
            "\\g<indent>losses[OBJECTIVE_LOSS_KEY].add(loss.item())\n"
            "\\g<indent>edge_vlm_add_distillation_meters(losses)",
            text,
        )
    text = text.replace("classerr.add(output.data, target)", "classerr.add(output.data, edge_vlm_hard_labels(target))")
    text = text.replace(
        "target.flatten())",
        "edge_vlm_hard_labels(target).flatten())",
    )
    text = text.replace(
        "confusion.add(output.data, target)",
        "confusion.add(output.data, edge_vlm_hard_labels(target))",
    )
    text = replace_once(
        text,
        "sample.generate(args.generate_sample, inputs, target, output,\n"
        "                                args.dataset, False, args.slice_sample)",
        "sample.generate(args.generate_sample, inputs, edge_vlm_hard_labels(target), output,\n"
        "                                args.dataset, False, args.slice_sample)",
    )

    train_py.write_text(text, encoding="utf-8")
    print(f"Patched cached-teacher distillation support into {train_py}")


if __name__ == "__main__":
    main()
