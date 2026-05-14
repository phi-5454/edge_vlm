from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(slots=True)
class TinyVlmConfig:
    vocab_size: int = 49_152
    max_text_tokens: int = 64
    image_size: int = 224
    text_width: int = 128
    text_layers: int = 2
    text_heads: int = 4
    projection_dim: int = 128
    fusion_hidden_dim: int = 256
    teacher_dim: int = 576
    num_answer_classes: int = 0
    pretrained_vision: bool = True
    freeze_vision: bool = False


class TinyTextEncoder(nn.Module):
    def __init__(self, config: TinyVlmConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.position_embedding = nn.Embedding(config.max_text_tokens, config.text_width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.text_width,
            nhead=config.text_heads,
            dim_feedforward=config.text_width * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.text_layers)
        self.norm = nn.LayerNorm(config.text_width)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device).expand(batch_size, -1)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = attention_mask == 0 if attention_mask is not None else None
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        if attention_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.norm(pooled)


class TinyVLM(nn.Module):
    """Small dual-encoder VLM intended for distillation and fast inference."""

    def __init__(self, config: TinyVlmConfig) -> None:
        super().__init__()
        self.config = config
        weights = MobileNet_V3_Small_Weights.DEFAULT if config.pretrained_vision else None
        mobilenet = mobilenet_v3_small(weights=weights)
        self.vision = mobilenet.features
        self.vision_pool = nn.AdaptiveAvgPool2d(1)
        vision_dim = mobilenet.classifier[0].in_features

        if config.freeze_vision:
            for parameter in self.vision.parameters():
                parameter.requires_grad = False

        self.text = TinyTextEncoder(config)
        self.image_projection = nn.Sequential(
            nn.Linear(vision_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(config.text_width, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.projection_dim * 2, config.fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_hidden_dim),
        )
        self.teacher_projection = nn.Linear(config.fusion_hidden_dim, config.teacher_dim)
        self.answer_head = (
            nn.Linear(config.fusion_hidden_dim, config.num_answer_classes)
            if config.num_answer_classes > 0
            else None
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        fused = self.fuse(image_features, text_features)
        outputs = {
            "image_embeds": image_features,
            "text_embeds": text_features,
            "fused_embeds": fused,
            "teacher_embeds": self.teacher_projection(fused),
        }
        if self.answer_head is not None:
            outputs["answer_logits"] = self.answer_head(fused)
        return outputs

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision(pixel_values)
        pooled = self.vision_pool(features).flatten(1)
        return self.image_projection(pooled)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_features = self.text(input_ids, attention_mask)
        return self.text_projection(text_features)

    def fuse(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([image_features, text_features], dim=-1))
