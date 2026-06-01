from __future__ import annotations

import gc
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from transformers import AutoModelForImageTextToText


def load_teacher_embedding_rows(
    model_name: str,
    teacher_token_ids: tuple[int, ...],
    local_files_only: bool,
    trust_remote_code: bool,
) -> torch.Tensor:
    """Load only the selected rows into the returned tensor, then release the teacher."""

    teacher = AutoModelForImageTextToText.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    embedding = teacher.get_input_embeddings()
    if embedding is None:
        raise ValueError(f"{model_name} does not expose input embeddings.")
    rows = embedding.weight.detach().cpu()[list(teacher_token_ids)].float().clone()
    del teacher
    gc.collect()
    return rows


class MobileViTFusionBlock(nn.Module):
    """Local convolution plus global attention over compact modality tokens."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.local_norm = nn.LayerNorm(dim)
        self.local = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.global_block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        local = self.local_norm(tokens).transpose(1, 2)
        tokens = tokens + self.local(local).transpose(1, 2)
        return self.global_block(tokens)


class StudentBaseline(nn.Module):
    def __init__(
        self,
        embedding_rows: torch.Tensor,
        freeze_embeddings: bool = True,
        image_pretrained: bool = True,
        query_dim: int = 128,
        image_dim: int = 128,
        fusion_dim: int = 128,
        fusion_depth: int = 2,
        fusion_heads: int = 4,
        fusion_mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if query_dim != fusion_dim or image_dim != fusion_dim:
            raise ValueError("query_dim and image_dim must equal fusion_dim for two-token fusion.")
        pad_row = torch.zeros((1, embedding_rows.shape[1]), dtype=embedding_rows.dtype)
        self.token_embedding = nn.Embedding.from_pretrained(
            torch.cat([pad_row, embedding_rows], dim=0),
            freeze=freeze_embeddings,
            padding_idx=0,
        )
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_rows.shape[1], query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim),
        )
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if image_pretrained else None
        )
        self.image_features = backbone.features
        self.image_pool = backbone.avgpool
        self.image_projection = nn.Sequential(
            nn.Linear(backbone.classifier[0].in_features, image_dim),
            nn.GELU(),
            nn.LayerNorm(image_dim),
        )
        self.fusion = nn.Sequential(
            *[
                MobileViTFusionBlock(fusion_dim, fusion_heads, fusion_mlp_ratio, dropout)
                for _ in range(fusion_depth)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.token_embedding(token_ids)
        mask = attention_mask.unsqueeze(-1).to(embedded.dtype)
        query = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        query = self.query_projection(query)

        image = self.image_pool(self.image_features(images))
        image = self.image_projection(torch.flatten(image, 1))

        tokens = torch.stack([query, image], dim=1)
        fused = self.fusion(tokens).mean(dim=1)
        return self.classifier(fused).squeeze(-1)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
    }


def architecture_report(model: nn.Module) -> dict[str, Any]:
    return {
        "class": type(model).__name__,
        "parameter_counts": parameter_counts(model),
        "architecture": str(model),
    }
