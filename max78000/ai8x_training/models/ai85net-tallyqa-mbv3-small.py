###################################################################################################
#
# Repo-owned MAX78000 model definition.
#
# Stage this file into ../MAX78000/ai8x-training/models/ before running ADI train.py.
#
###################################################################################################
"""MobileNetV3-small-style people-count classifier for MAX78000.

This is intentionally a hardware-shaped approximation, not a literal torchvision
MobileNetV3 port:

- input: RGB image, default 224x224
- output classes: 1, 2, 3, 4, 5+ people
- no prompt input
- only 3x3 and 1x1 kernels
- no strided convolution; spatial downsampling is max pooling fused before conv
- depthwise convolutions are not used because this ADI ai8x tree restricts them
  to MAX78002; bottlenecks use ordinary 3x3 convolutions
- 5x5 MobileNetV3 layers are approximated by two stacked 3x3 convolutions
- only ReLU activations
- no squeeze-excitation / hard-sigmoid path

The feature extractor is cut at a 14x14x114 tensor. The classifier then average-pools
that tensor to 1x1 before a linear head because ai8x.Linear is limited to <=1024 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from torch import nn

import ai8x


@dataclass(frozen=True)
class BlockSpec:
    """Readable MobileNet-style block description.

    expand_channels:
        Number of channels after the 1x1 expansion. Set equal to in_channels for
        an identity expansion.
    out_channels:
        Number of channels after the final 1x1 projection.
    pool:
        If true, downsample by 2 using max-pooling before the 3x3 conv.
    stacked_3x3:
        If true, add a second 3x3 conv to approximate a MobileNetV3 5x5
        receptive field without using unsupported 5x5 kernels.
    residual:
        Enable residual add when shape is unchanged. Set false for first bring-up
        if the synthesis YAML should avoid eltwise layers initially.
    """

    expand_channels: int
    out_channels: int
    pool: bool = False
    stacked_3x3: bool = False
    residual: bool = True


class BottleneckBlock(nn.Module):
    """MAX78000-friendly MobileNet-style bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        spec: BlockSpec,
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

        spatial_cls = (
            ai8x.FusedMaxPoolConv2dBNReLU
            if spec.pool
            else ai8x.FusedConv2dBNReLU
        )
        self.spatial = spatial_cls(
            spatial_channels,
            spatial_channels,
            3,
            padding=1,
            pool_size=2,
            pool_stride=2,
            bias=bias,
            **kwargs,
        ) if spec.pool else spatial_cls(
            spatial_channels,
            spatial_channels,
            3,
            padding=1,
            bias=bias,
            **kwargs,
        )

        self.spatial_extra = (
            ai8x.FusedConv2dBNReLU(
                spatial_channels,
                spatial_channels,
                3,
                padding=1,
                bias=bias,
                **kwargs,
            )
            if spec.stacked_3x3
            else None
        )

        # Linear bottleneck projection: no activation here.
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
        if self.spatial_extra is not None:
            x = self.spatial_extra(x)
        x = self.project(x)
        if self.add is not None:
            x = self.add(x, identity)
        return x


class TallyQAMobileNetV3SmallPeople(nn.Module):
    """People-only count classifier with a MobileNetV3-small-like stem."""

    # This is the readable architecture table. Input is 224x224 by default.
    #
    # stem: 224 -> 112
    # block 1: 112 -> 56
    # block 2: 56 -> 28
    # block 4: 28 -> 14
    # block 10 output: 14x14x114
    feature_specs = (
        BlockSpec(expand_channels=16, out_channels=16, pool=True, stacked_3x3=False),
        BlockSpec(expand_channels=24, out_channels=24, pool=True, stacked_3x3=False),
        BlockSpec(expand_channels=24, out_channels=24, pool=False, stacked_3x3=False),
        BlockSpec(expand_channels=32, out_channels=40, pool=True, stacked_3x3=True),
        BlockSpec(expand_channels=32, out_channels=40, pool=False, stacked_3x3=True),
        BlockSpec(expand_channels=40, out_channels=40, pool=False, stacked_3x3=True),
        BlockSpec(expand_channels=48, out_channels=48, pool=False, stacked_3x3=True),
        BlockSpec(expand_channels=48, out_channels=48, pool=False, stacked_3x3=True),
        BlockSpec(expand_channels=64, out_channels=96, pool=False, stacked_3x3=True),
        BlockSpec(expand_channels=64, out_channels=114, pool=False, stacked_3x3=True),
    )

    def __init__(
        self,
        num_classes: int = 5,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (224, 224),
        bias: bool = True,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        if dimensions != (224, 224):
            raise ValueError("This first MAX78000 bring-up model expects 224x224 inputs.")
        if num_classes != 5:
            raise ValueError("People-count head is fixed to classes 1, 2, 3, 4, 5+.")

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dimensions = dimensions

        # Initial 3x3 conv with downsampling handled by fused max pool.
        self.stem = ai8x.FusedMaxPoolConv2dBNReLU(
            num_channels,
            16,
            3,
            pool_size=2,
            pool_stride=2,
            padding=1,
            bias=bias,
            **kwargs,
        )

        blocks: list[nn.Module] = []
        in_channels = 16
        for base_spec in self.feature_specs:
            spec = BlockSpec(
                expand_channels=base_spec.expand_channels,
                out_channels=base_spec.out_channels,
                pool=base_spec.pool,
                stacked_3x3=base_spec.stacked_3x3,
                residual=base_spec.residual and residual,
            )
            blocks.append(BottleneckBlock(in_channels, spec, bias=bias, **kwargs))
            in_channels = spec.out_channels
        self.features = nn.Sequential(*blocks)

        self.avgpool = ai8x.AvgPool2d(kernel_size=14, stride=14, **kwargs)
        self.classifier = ai8x.Linear(114, num_classes, bias=True, wide=True, **kwargs)

        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.stem(x)
        x = self.features(x)
        # Expected cut tensor for inspection/debug: N x 114 x 14 x 14.
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def ai85tallyqambv3smallpeople(pretrained: bool = False, **kwargs):
    """Construct the TallyQA people-count MobileNetV3-small-style model."""
    assert not pretrained
    return TallyQAMobileNetV3SmallPeople(**kwargs)


models = [
    {
        "name": "ai85tallyqambv3smallpeople",
        "min_input": 1,
        "dim": 2,
    },
]
