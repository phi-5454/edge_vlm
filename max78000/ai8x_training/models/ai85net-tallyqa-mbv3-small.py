###################################################################################################
#
# Repo-owned MAX78000 model definition.
#
# Stage this file into ../MAX78000/ai8x-training/models/ before running ADI train.py.
#
###################################################################################################
"""Folded-input count classifiers for MAX78000.

The default factory remains a conservative bring-up model. This file also
contains MobileNetV3-minimal-inspired folded encoders for architecture probes:

- input: 12-channel 56x56 tensor produced by downsampling 224x224 RGB to
  112x112 and then 2x2 folding
- output classes: 1, 2, 3, 4, 5+ for people-only bring-up, or 0, 1, 2, 3, 4, 5+
  for prompt-subset runs
- no prompt input
- only ordinary 3x3 and 1x1 convolutions
- no depthwise or depth-separable convolutions
- no strided convolutions; downsampling is max pooling fused before conv
- ReLU only
- no squeeze-excitation / hard-sigmoid path
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

import ai8x


@dataclass(frozen=True)
class StageSpec:
    """Simple MAX78000-friendly convolution stage.

    out_channels:
        Number of output channels after the downsampling 3x3 conv.
    extra_convs:
        Additional same-resolution 3x3 convs after the downsampling conv.
    """

    out_channels: int
    extra_convs: int = 1


@dataclass(frozen=True)
class MobileNetV3MinimalBlockSpec:
    """MobileNetV3-minimal block with MAX78000-compatible substitutions.

    expand_channels and out_channels mirror the Keras MobileNetV3 minimal
    channel schedule. Original depthwise convolutions are replaced by ordinary
    3x3 convolutions. Original stride-2 blocks use fused max pooling before the
    3x3 convolution.
    """

    expand_channels: int
    out_channels: int
    pool: bool = False
    residual: bool = True


def initialize_conv_linear(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight, mode="fan_out", nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Linear):
            nn.init.normal_(child.weight, mean=0.0, std=0.01)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class DownsampleConvStage(nn.Module):
    """MaxPool + ordinary 3x3 conv stage, followed by optional 3x3 convs."""

    def __init__(
        self,
        in_channels: int,
        spec: StageSpec,
        bias: bool,
        **kwargs,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            ai8x.FusedMaxPoolConv2dBNReLU(
                in_channels,
                spec.out_channels,
                3,
                pool_size=2,
                pool_stride=2,
                padding=1,
                bias=bias,
                **kwargs,
            )
        ]
        for _ in range(spec.extra_convs):
            layers.append(
                ai8x.FusedConv2dBNReLU(
                    spec.out_channels,
                    spec.out_channels,
                    3,
                    padding=1,
                    bias=bias,
                    **kwargs,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.layers(x)


class SameResolutionConvStage(nn.Module):
    """Ordinary same-resolution 3x3 conv stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            ai8x.FusedConv2dBNReLU(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=bias,
                **kwargs,
            ),
            ai8x.FusedConv2dBNReLU(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=bias,
                **kwargs,
            ),
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.layers(x)


class MobileNetV3MinimalBlock(nn.Module):
    """Inverted-residual-style block using ordinary 3x3 convs instead of depthwise convs."""

    def __init__(
        self,
        in_channels: int,
        spec: MobileNetV3MinimalBlockSpec,
        bias: bool,
        **kwargs,
    ):
        super().__init__()
        self.use_residual = spec.residual and not spec.pool and in_channels == spec.out_channels

        if spec.expand_channels == in_channels:
            self.expand = None
            spatial_channels = in_channels
        else:
            self.expand = ai8x.FusedConv2dBNReLU(
                in_channels,
                spec.expand_channels,
                1,
                bias=bias,
                **kwargs,
            )
            spatial_channels = spec.expand_channels

        if spec.pool:
            self.spatial = ai8x.FusedMaxPoolConv2dBNReLU(
                spatial_channels,
                spatial_channels,
                3,
                pool_size=2,
                pool_stride=2,
                padding=1,
                bias=bias,
                **kwargs,
            )
        else:
            self.spatial = ai8x.FusedConv2dBNReLU(
                spatial_channels,
                spatial_channels,
                3,
                padding=1,
                bias=bias,
                **kwargs,
            )

        # MobileNetV3 inverted residuals use a linear projection.
        self.project = ai8x.FusedConv2dBN(
            spatial_channels,
            spec.out_channels,
            1,
            bias=bias,
            **kwargs,
        )
        self.add = ai8x.Add() if self.use_residual else None

    def forward(self, x):  # pylint: disable=arguments-differ
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.spatial(x)
        x = self.project(x)
        if self.add is not None:
            x = self.add(x, identity)
        return x


class TallyQAFoldedSimpleCount(nn.Module):
    """Count classifier for folded RGB inputs."""

    # Input is 12x56x56. Each per-channel plane has 3136 bytes, below 8192.
    stage_specs = (
        StageSpec(out_channels=40, extra_convs=1),  # 56 -> 28
        StageSpec(out_channels=80, extra_convs=1),  # 28 -> 14
    )

    def __init__(
        self,
        num_classes: int = 5,
        num_channels: int = 12,
        dimensions: tuple[int, int] = (56, 56),
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        if dimensions != (56, 56):
            raise ValueError("This MAX78000 folded model expects 56x56 inputs.")
        if num_channels != 12:
            raise ValueError("This MAX78000 folded model expects 12 input channels.")
        if num_classes not in {5, 6}:
            raise ValueError("Count head supports either 5 classes (1..5+) or 6 classes (0..5+).")

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dimensions = dimensions

        # First convolution before spatial downsampling. This consumes the folded
        # RGB patch channels and gives the network a chance to mix local color and
        # subpixel position information before pooling.
        self.stem = ai8x.FusedConv2dBNReLU(
            num_channels,
            24,
            3,
            padding=1,
            bias=bias,
            **kwargs,
        )

        stages: list[nn.Module] = []
        in_channels = 24
        for spec in self.stage_specs:
            stages.append(DownsampleConvStage(in_channels, spec, bias=bias, **kwargs))
            in_channels = spec.out_channels
        self.features = nn.Sequential(*stages)
        self.cut_projection = SameResolutionConvStage(80, 112, bias=bias, **kwargs)

        self.avgpool = ai8x.AvgPool2d(kernel_size=14, stride=14, **kwargs)
        self.classifier = ai8x.Linear(112, num_classes, bias=True, wide=True, **kwargs)

        self._initialize()

    def _initialize(self) -> None:
        initialize_conv_linear(self)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.cut_projection(x)
        # Expected cut tensor for inspection/debug: N x 112 x 14 x 14.
        return x

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class TallyQAFoldedMobileNetV3MinimalCount(nn.Module):
    """Folded-input MobileNetV3-minimal-style count classifier.

    The architecture keeps MobileNetV3 minimal's channel schedule and residual
    pattern, but substitutes ordinary 3x3 convolutions for depthwise
    convolutions. The input-folded 56x56 tensor replaces the usual RGB image
    stem input, so the stem is same-resolution and later stride-2 MobileNetV3
    transitions are implemented as max-pool-plus-conv blocks.
    """

    # Folded 56x56 input replaces the first two spatial reductions in the
    # original 224x224 MobileNetV3 input path. For MobileNetV3-small this lands
    # at the 56x56x16 stage. Keep the 24, 40, and 48-channel stages and cut
    # before the 96/576 tail, matching the large-minimal fusion-cut principle.
    small_stem_channels = 16
    small_head_channels = 144
    small_pool_kernel = 14
    small_specs = (
        MobileNetV3MinimalBlockSpec(72, 24, pool=True),
        MobileNetV3MinimalBlockSpec(88, 24),
        MobileNetV3MinimalBlockSpec(96, 40, pool=True),
        MobileNetV3MinimalBlockSpec(240, 40),
        MobileNetV3MinimalBlockSpec(240, 40),
        MobileNetV3MinimalBlockSpec(120, 48),
        MobileNetV3MinimalBlockSpec(144, 48),
    )

    # For MobileNetV3-large, the folded input corresponds to the 56x56x24
    # stage. Keep the 40, 80, and 112-channel stages and cut before the 160
    # tail, which is both later than the intended fusion cut and parameter-heavy
    # when depthwise convolutions are replaced by full 3x3 convolutions.
    large_stem_channels = 24
    large_head_channels = 672
    large_pool_kernel = 14
    large_specs = (
        MobileNetV3MinimalBlockSpec(72, 40, pool=True),
        MobileNetV3MinimalBlockSpec(120, 40),
        MobileNetV3MinimalBlockSpec(120, 40),
        MobileNetV3MinimalBlockSpec(240, 80, pool=True),
        MobileNetV3MinimalBlockSpec(200, 80),
        MobileNetV3MinimalBlockSpec(184, 80),
        MobileNetV3MinimalBlockSpec(184, 80),
        MobileNetV3MinimalBlockSpec(480, 112),
        MobileNetV3MinimalBlockSpec(672, 112),
    )

    def __init__(
        self,
        variant: str,
        num_classes: int = 5,
        num_channels: int = 12,
        dimensions: tuple[int, int] = (56, 56),
        bias: bool = True,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        if dimensions != (56, 56):
            raise ValueError("This MAX78000 folded MobileNetV3 model expects 56x56 inputs.")
        if num_channels != 12:
            raise ValueError("This MAX78000 folded MobileNetV3 model expects 12 input channels.")
        if num_classes not in {5, 6}:
            raise ValueError("Count head supports either 5 classes (1..5+) or 6 classes (0..5+).")
        if variant not in {"small", "large"}:
            raise ValueError("variant must be one of {'small', 'large'}.")

        self.variant = variant
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dimensions = dimensions

        if variant == "small":
            specs = self.small_specs
            stem_channels = self.small_stem_channels
            head_channels = self.small_head_channels
            pool_kernel = self.small_pool_kernel
        else:
            specs = self.large_specs
            stem_channels = self.large_stem_channels
            head_channels = self.large_head_channels
            pool_kernel = self.large_pool_kernel

        self.stem = ai8x.FusedConv2dBNReLU(
            num_channels,
            stem_channels,
            3,
            padding=1,
            bias=bias,
            **kwargs,
        )

        blocks: list[nn.Module] = []
        in_channels = stem_channels
        for base_spec in specs:
            spec = MobileNetV3MinimalBlockSpec(
                expand_channels=base_spec.expand_channels,
                out_channels=base_spec.out_channels,
                pool=base_spec.pool,
                residual=base_spec.residual and residual,
            )
            blocks.append(MobileNetV3MinimalBlock(in_channels, spec, bias=bias, **kwargs))
            in_channels = spec.out_channels
        self.features = nn.Sequential(*blocks)
        self.head_conv = ai8x.FusedConv2dBNReLU(
            in_channels,
            head_channels,
            1,
            bias=bias,
            **kwargs,
        )

        self.avgpool = ai8x.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel, **kwargs)
        self.classifier = ai8x.Linear(head_channels, num_classes, bias=True, wide=True, **kwargs)

        initialize_conv_linear(self)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.head_conv(x)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def ai85tallyqambv3smallcount(pretrained: bool = False, **kwargs):
    """Construct the folded-input TallyQA count model."""
    assert not pretrained
    return TallyQAFoldedSimpleCount(**kwargs)


def ai85tallyqambv3smallminimalcount(pretrained: bool = False, **kwargs):
    """Construct the folded-input MobileNetV3-small-minimal-style TallyQA model."""
    assert not pretrained
    return TallyQAFoldedMobileNetV3MinimalCount(variant="small", **kwargs)


def ai85tallyqambv3largeminimalcount(pretrained: bool = False, **kwargs):
    """Construct the folded-input MobileNetV3-large-minimal-style TallyQA model."""
    assert not pretrained
    return TallyQAFoldedMobileNetV3MinimalCount(variant="large", **kwargs)


def ai85tallyqambv3smallpeople(pretrained: bool = False, **kwargs):
    """Backward-compatible alias for older people-count notebooks."""
    return ai85tallyqambv3smallcount(pretrained=pretrained, **kwargs)


models = [
    {
        "name": "ai85tallyqambv3smallcount",
        "min_input": 1,
        "dim": 2,
    },
    {
        "name": "ai85tallyqambv3smallpeople",
        "min_input": 1,
        "dim": 2,
    },
    {
        "name": "ai85tallyqambv3smallminimalcount",
        "min_input": 1,
        "dim": 2,
    },
    {
        "name": "ai85tallyqambv3largeminimalcount",
        "min_input": 1,
        "dim": 2,
    },
]
