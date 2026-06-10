from __future__ import annotations

import gc
from typing import Any

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
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


class FeatureFiLM(nn.Module):
    """Prompt-conditioned affine modulation for convolutional feature maps."""

    def __init__(self, query_dim: int, channels: int):
        super().__init__()
        self.channels = channels
        self.to_scale_shift = nn.Linear(query_dim, channels * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(query).chunk(2, dim=1)
        scale = scale.view(scale.shape[0], self.channels, 1, 1)
        shift = shift.view(shift.shape[0], self.channels, 1, 1)
        return features * (1 + scale) + shift


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
        image_backbone: str = "mobilenet_v3_large",
        image_feature_cutoff: int | str | None = "auto",
        image_token_mode: str = "spatial",
        fusion_mode: str = "transformer",
        freeze_image_features: bool = False,
        image_film_at: int | str | None = None,
        image_film_position: str = "post_block",
        use_prompt_identity: bool = True,
        use_image_positional_embeddings: bool = True,
        image_position_tokens: int = 196,
        zero_image_tokens: bool = False,
        zero_query_token: bool = False,
        num_outputs: int = 1,
    ):
        super().__init__()
        if num_outputs <= 0:
            raise ValueError("num_outputs must be positive.")
        if query_dim != fusion_dim or image_dim != fusion_dim:
            raise ValueError("query_dim and image_dim must equal fusion_dim for token fusion.")
        if image_token_mode not in {"spatial", "pooled"}:
            raise ValueError("image_token_mode must be 'spatial' or 'pooled'.")
        if fusion_mode not in {"transformer", "concat_mlp", "film_mlp", "prompt_patch_mlp"}:
            raise ValueError(
                "fusion_mode must be one of "
                "{'transformer', 'concat_mlp', 'film_mlp', 'prompt_patch_mlp'}."
            )
        if image_film_position not in {"post_block", "pre_depthwise"}:
            raise ValueError("image_film_position must be one of {'post_block', 'pre_depthwise'}.")
        self.num_outputs = num_outputs
        self.query_dim = query_dim
        self.fusion_dim = fusion_dim
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
        if image_backbone == "mobilenet_v3_large":
            backbone = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT if image_pretrained else None
            )
        elif image_backbone == "mobilenet_v3_small":
            backbone = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT if image_pretrained else None
            )
        else:
            raise ValueError(
                "image_backbone must be one of {'mobilenet_v3_large', 'mobilenet_v3_small'}."
            )
        resolved_cutoff = self._resolve_image_feature_cutoff(image_backbone, image_feature_cutoff)
        self.image_backbone_name = image_backbone
        self.image_feature_cutoff = resolved_cutoff
        self.image_token_mode = image_token_mode
        self.fusion_mode = fusion_mode
        self.image_film_at = image_film_at
        self.image_film_indices = self._resolve_image_film_at(image_film_at)
        self.image_film_position = image_film_position
        self.use_prompt_identity = use_prompt_identity
        self.use_image_positional_embeddings = use_image_positional_embeddings
        self.image_position_tokens = image_position_tokens
        self.zero_image_tokens = zero_image_tokens
        self.zero_query_token = zero_query_token
        self.image_features = (
            backbone.features
            if resolved_cutoff is None
            else nn.Sequential(*list(backbone.features.children())[:resolved_cutoff])
        )
        if freeze_image_features:
            for parameter in self.image_features.parameters():
                parameter.requires_grad = False
        self.image_pool = backbone.avgpool
        image_feature_channels = self._image_feature_channels(image_backbone, resolved_cutoff)
        self.image_feature_channels = image_feature_channels
        if self.image_film_indices:
            film_layers: dict[str, FeatureFiLM] = {}
            for feature_index in self.image_film_indices:
                if resolved_cutoff is not None and feature_index >= resolved_cutoff:
                    raise ValueError(
                        f"image_film_at={feature_index} must be before "
                        f"image_feature_cutoff={resolved_cutoff}."
                    )
                film_channels = self._image_feature_channels_after_index(
                    image_backbone,
                    feature_index,
                ) if image_film_position == "post_block" else self._image_feature_pre_depthwise_channels(
                    image_backbone,
                    feature_index,
                )
                film_layers[str(feature_index)] = FeatureFiLM(query_dim, film_channels)
            self.image_film_layers = nn.ModuleDict(film_layers)
            self.image_film = (
                next(iter(self.image_film_layers.values()))
                if len(self.image_film_layers) == 1
                else self.image_film_layers
            )
        else:
            self.image_film_layers = None
            self.image_film = None
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_channels, image_dim),
            nn.GELU(),
            nn.LayerNorm(image_dim),
        )
        self.prompt_identity = (
            nn.Parameter(torch.zeros(1, 1, fusion_dim)) if use_prompt_identity else None
        )
        if image_position_tokens <= 0:
            raise ValueError("image_position_tokens must be positive.")
        self.image_position_embeddings = (
            nn.Parameter(torch.zeros(1, image_position_tokens, fusion_dim))
            if use_image_positional_embeddings
            else None
        )
        if fusion_mode == "transformer":
            self.fusion = nn.Sequential(
                *[
                    MobileViTFusionBlock(fusion_dim, fusion_heads, fusion_mlp_ratio, dropout)
                    for _ in range(fusion_depth)
                ]
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, num_outputs),
            )
        elif fusion_mode == "concat_mlp":
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim * fusion_mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * fusion_mlp_ratio, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, num_outputs),
            )
        elif fusion_mode == "prompt_patch_mlp":
            self.fusion = nn.Identity()
            patch_hidden_dim = fusion_dim * fusion_mlp_ratio
            self.patch_mlp = nn.Sequential(
                nn.Conv2d(image_feature_channels + query_dim, patch_hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(patch_hidden_dim, fusion_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Linear(fusion_dim * 7 * 7, num_outputs)
        else:
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, fusion_dim * fusion_mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * fusion_mlp_ratio, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, num_outputs),
            )

    def encode_query(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.token_embedding(token_ids)
        mask = attention_mask.unsqueeze(-1).to(embedded.dtype)
        query = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.query_projection(query)

    def encode_image_features(
        self,
        images: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.image_film_layers is None:
            return self.image_features(images)
        if query is None:
            raise ValueError("query is required when image FiLM conditioning is enabled.")
        features = images
        for index, block in enumerate(self.image_features):
            key = str(index)
            film = self.image_film_layers[key] if key in self.image_film_layers else None
            if film is not None and self.image_film_position == "pre_depthwise":
                features = self._forward_block_with_pre_depthwise_film(block, features, query, film)
            else:
                features = block(features)
            if film is not None and self.image_film_position == "post_block":
                features = film(features, query)
        return features

    def prompt_patch_mlp_features(
        self,
        image_features: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if image_features.shape[-2:] != (14, 14):
            raise ValueError(
                "prompt_patch_mlp expects 14x14 image features before 2x2 average pooling; "
                f"got spatial shape {tuple(image_features.shape[-2:])}."
            )
        query_map = query[:, :, None, None].expand(
            -1,
            -1,
            image_features.shape[-2],
            image_features.shape[-1],
        )
        conditioned = torch.cat([image_features, query_map], dim=1)
        return self.patch_mlp(conditioned).flatten(1)

    @staticmethod
    def _forward_block_with_pre_depthwise_film(
        block: nn.Module,
        features: torch.Tensor,
        query: torch.Tensor,
        film: FeatureFiLM,
    ) -> torch.Tensor:
        if not hasattr(block, "block") or not isinstance(block.block, nn.Sequential):
            return block(film(features, query))

        residual = features
        layers = list(block.block.children())
        if not layers:
            return block(features)

        first_conv = layers[0][0] if isinstance(layers[0], nn.Sequential) and layers[0] else None
        if isinstance(first_conv, nn.Conv2d) and first_conv.groups == first_conv.in_channels:
            x = film(features, query)
            start_index = 0
        else:
            x = layers[0](features)
            x = film(x, query)
            start_index = 1
        for layer in layers[start_index:]:
            x = layer(x)
        if getattr(block, "use_res_connect", False):
            x = residual + x
        return x

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.zero_query_token:
            query = torch.zeros(
                (images.shape[0], self.query_dim),
                device=images.device,
                dtype=images.dtype,
            )
        else:
            query = self.encode_query(token_ids, attention_mask)

        image_features = None
        if self.zero_image_tokens:
            image_token_count = 1 if self.image_token_mode == "pooled" else self.image_position_tokens
            image_tokens = torch.zeros(
                (images.shape[0], image_token_count, self.fusion_dim),
                device=query.device,
                dtype=query.dtype,
            )
        else:
            image_features = self.encode_image_features(images, query)
            if self.fusion_mode == "prompt_patch_mlp":
                image_tokens = None
            elif self.image_token_mode == "pooled":
                image = self.image_pool(image_features)
                image_tokens = self.image_projection(torch.flatten(image, 1)).unsqueeze(1)
            else:
                image_tokens = image_features.flatten(2).transpose(1, 2)
                image_tokens = self.image_projection(image_tokens)
                image_tokens = image_tokens + self._image_position_embeddings(
                    image_tokens.shape[1],
                    image_tokens.device,
                    image_tokens.dtype,
                )

        query_token = query.unsqueeze(1)
        if self.prompt_identity is not None:
            query_token = query_token + self.prompt_identity.to(
                device=query_token.device,
                dtype=query_token.dtype,
            )
        if self.zero_query_token:
            query_token = torch.zeros_like(query_token)

        if self.fusion_mode == "prompt_patch_mlp":
            if image_features is None:
                image_features = torch.zeros(
                    (images.shape[0], self.image_feature_channels, 14, 14),
                    device=query.device,
                    dtype=query.dtype,
                )
            fused = self.prompt_patch_mlp_features(image_features, query_token.squeeze(1))
        elif self.fusion_mode == "concat_mlp":
            image = image_tokens.mean(dim=1)
            query = query_token.squeeze(1)
            fused = torch.cat([query, image], dim=1)
        elif self.fusion_mode == "film_mlp":
            fused = image_tokens.mean(dim=1)
        else:
            tokens = torch.cat([query_token, image_tokens], dim=1)
            fused = self.fusion(tokens).mean(dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(-1) if self.num_outputs == 1 else logits

    def _image_position_embeddings(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.use_image_positional_embeddings:
            dim = int(self.image_projection[-1].normalized_shape[0])
            return torch.zeros((1, num_tokens, dim), device=device, dtype=dtype)
        if self.image_position_embeddings is None:
            raise RuntimeError("image_position_embeddings is unexpectedly missing.")
        if num_tokens > self.image_position_embeddings.shape[1]:
            raise ValueError(
                f"Image produced {num_tokens} tokens, but image_position_tokens="
                f"{self.image_position_embeddings.shape[1]}."
            )
        return self.image_position_embeddings[:, :num_tokens].to(device=device, dtype=dtype)

    @staticmethod
    def _resolve_image_feature_cutoff(
        image_backbone: str,
        image_feature_cutoff: int | str | None,
    ) -> int | None:
        if image_feature_cutoff is None:
            return None
        if isinstance(image_feature_cutoff, str):
            if image_feature_cutoff == "auto":
                return 13 if image_backbone == "mobilenet_v3_large" else 9
            if image_feature_cutoff == "none":
                return None
            image_feature_cutoff = int(image_feature_cutoff)
        if image_feature_cutoff <= 0:
            raise ValueError("image_feature_cutoff must be positive, 'auto', 'none', or null.")
        return image_feature_cutoff

    @staticmethod
    def _resolve_image_film_at(image_film_at: int | str | None) -> tuple[int, ...]:
        if image_film_at is None:
            return ()
        if isinstance(image_film_at, str):
            if image_film_at == "none":
                return ()
            values = tuple(int(value.strip()) for value in image_film_at.split(",") if value.strip())
        else:
            values = (int(image_film_at),)
        if any(value < 0 for value in values):
            raise ValueError("image_film_at values must be non-negative, 'none', or null.")
        if len(set(values)) != len(values):
            raise ValueError("image_film_at must not contain duplicate indices.")
        return values

    @staticmethod
    def _image_feature_channels(image_backbone: str, image_feature_cutoff: int | None) -> int:
        channels = {
            "mobilenet_v3_large": {
                None: 960,
                7: 40,
                11: 80,
                13: 112,
                16: 160,
                17: 960,
            },
            "mobilenet_v3_small": {
                None: 576,
                4: 24,
                7: 40,
                9: 48,
                12: 96,
                13: 576,
            },
        }
        if image_feature_cutoff not in channels[image_backbone]:
            supported = ", ".join(str(value) for value in channels[image_backbone])
            raise ValueError(
                f"Unsupported {image_backbone} image_feature_cutoff={image_feature_cutoff}. "
                f"Supported cutoffs: {supported}."
            )
        return channels[image_backbone][image_feature_cutoff]

    @staticmethod
    def _image_feature_channels_after_index(image_backbone: str, feature_index: int) -> int:
        channels = {
            "mobilenet_v3_large": {
                0: 16,
                1: 16,
                2: 24,
                3: 24,
                4: 40,
                5: 40,
                6: 40,
                7: 80,
                8: 80,
                9: 80,
                10: 80,
                11: 112,
                12: 112,
                13: 160,
                14: 160,
                15: 160,
                16: 960,
            },
            "mobilenet_v3_small": {
                0: 16,
                1: 16,
                2: 24,
                3: 24,
                4: 40,
                5: 40,
                6: 40,
                7: 48,
                8: 48,
                9: 96,
                10: 96,
                11: 96,
                12: 576,
            },
        }
        if feature_index not in channels[image_backbone]:
            supported = ", ".join(str(value) for value in channels[image_backbone])
            raise ValueError(
                f"Unsupported {image_backbone} image_film_at={feature_index}. "
                f"Supported feature indices: {supported}."
            )
        return channels[image_backbone][feature_index]

    @staticmethod
    def _image_feature_pre_depthwise_channels(image_backbone: str, feature_index: int) -> int:
        channels = {
            "mobilenet_v3_large": {
                1: 16,
                2: 64,
                3: 72,
                4: 72,
                5: 120,
                6: 120,
                7: 240,
                8: 200,
                9: 184,
                10: 184,
                11: 480,
                12: 672,
                13: 672,
                14: 960,
                15: 960,
            },
            "mobilenet_v3_small": {
                1: 16,
                2: 72,
                3: 88,
                4: 96,
                5: 240,
                6: 240,
                7: 120,
                8: 144,
                9: 288,
                10: 576,
                11: 576,
            },
        }
        if feature_index not in channels[image_backbone]:
            supported = ", ".join(str(value) for value in channels[image_backbone])
            raise ValueError(
                f"Unsupported {image_backbone} pre-depthwise image_film_at={feature_index}. "
                f"Supported feature indices: {supported}."
            )
        return channels[image_backbone][feature_index]


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
        "image_backbone": getattr(model, "image_backbone_name", None),
        "image_feature_cutoff": getattr(model, "image_feature_cutoff", None),
        "image_token_mode": getattr(model, "image_token_mode", None),
        "fusion_mode": getattr(model, "fusion_mode", None),
        "image_film_at": getattr(model, "image_film_at", None),
        "image_film_indices": getattr(model, "image_film_indices", None),
        "image_film_position": getattr(model, "image_film_position", None),
        "use_prompt_identity": getattr(model, "use_prompt_identity", None),
        "use_image_positional_embeddings": getattr(
            model,
            "use_image_positional_embeddings",
            None,
        ),
        "image_position_tokens": getattr(model, "image_position_tokens", None),
        "architecture": str(model),
    }
