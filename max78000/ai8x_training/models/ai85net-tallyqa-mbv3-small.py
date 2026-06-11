###################################################################################################
#
# Repo-owned MAX78000 model definition.
#
# Stage this file into ../MAX78000/ai8x-training/models/ before running ADI train.py.
#
###################################################################################################
"""Folded-input simple-conv people-count classifier for MAX78000.

This is a conservative MAX78000 bring-up model, not a literal MobileNetV3 port.
The intent is to mirror the "simple" MobileNet direction while staying within
the ADI ai8x-training operator subset:

- input: 12-channel 56x56 tensor produced by downsampling 224x224 RGB to
  112x112 and then 2x2 folding
- output classes: 1, 2, 3, 4, 5+ people
- no prompt input
- only ordinary 3x3 and 1x1 convolutions
- no depthwise or depth-separable convolutions
- no strided convolutions; downsampling is max pooling fused before conv
- ReLU only
- no squeeze-excitation / hard-sigmoid path

The requested spatial progression is:

    56x56x12 -> 56x56x24 -> 28x28x40 -> 14x14x80 -> 14x14x112

The classifier average-pools the 14x14x112 tensor to 1x1 before a linear head.
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


class TallyQAFoldedSimplePeople(nn.Module):
    """People-only count classifier for folded RGB inputs."""

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
        if num_classes != 5:
            raise ValueError("People-count head is fixed to classes 1, 2, 3, 4, 5+.")

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
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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


def ai85tallyqambv3smallpeople(pretrained: bool = False, **kwargs):
    """Construct the folded-input TallyQA people-count model."""
    assert not pretrained
    return TallyQAFoldedSimplePeople(**kwargs)


models = [
    {
        "name": "ai85tallyqambv3smallpeople",
        "min_input": 1,
        "dim": 2,
    },
]
