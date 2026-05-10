"""
Res2Net-29-style **ANN** backbone matching `MSFRes2Net` topology.

- Same stem, stage widths (64→128→256→512), `(2,2,2,2)` blocks, `base_width`, `scale`.
- Conv + BN + ReLU instead of spiking layers; input `[B, 3, H, W]`, output `[B, num_classes]`.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from .attention_ann import CSAAnn


class SeqConvBnReLU(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Bottle2neckANN(nn.Module):
    """Res2Net block with split/concat on **dim=1** (channel) for `[B,C,H,W]`."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 32,
        scale: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.width = max(int(math.floor(planes * (base_width / 64.0))), 1)
        self.downsample = downsample

        self.branch = SeqConvBnReLU(
            inplanes, self.width * scale, kernel_size=1, stride=stride, padding=0
        )

        self.convs = nn.ModuleList()
        self.scales = nn.ModuleList()
        for _ in range(scale - 1):
            self.convs.append(
                nn.Conv2d(self.width, self.width, 3, stride=1, padding=1, bias=False)
            )
            self.scales.append(nn.Sequential(nn.BatchNorm2d(self.width), nn.ReLU(inplace=True)))

        self.tail = SeqConvBnReLU(self.width * scale, planes, kernel_size=1, stride=1, padding=0)

        self.use_attention = use_attention
        self.attn = CSAAnn(planes) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.branch(x)
        splits = torch.split(out, self.width, dim=1)
        feats: List[torch.Tensor] = [splits[0]]
        y = splits[0]
        for i in range(1, self.scale):
            y = self.convs[i - 1](y + splits[i])
            y = self.scales[i - 1](y)
            feats.append(y)
        out = torch.cat(feats, dim=1)
        out = self.tail(out)
        out = self.attn(out)
        return out + identity


class Res2NetANN(nn.Module):
    """Res2Net-29-style classifier (ANN), aligned with `MSFRes2Net` hyperparameters."""

    def __init__(
        self,
        num_classes: int = 38,
        layers: tuple = (2, 2, 2, 2),
        base_width: int = 32,
        scale: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()
        self.inplanes = 64
        self.base_width = base_width
        self.scale = scale
        self.use_attention = use_attention

        self.conv1 = SeqConvBnReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )

        layers_list: List[nn.Module] = [
            Bottle2neckANN(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                base_width=self.base_width,
                scale=self.scale,
                use_attention=self.use_attention,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers_list.append(
                Bottle2neckANN(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    base_width=self.base_width,
                    scale=self.scale,
                    use_attention=self.use_attention,
                )
            )
        return nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> logits [B, num_classes]."""
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_ann_model(cfg) -> Res2NetANN:
    """Build ANN from `TrainConfig`-like object (uses `layers`, `base_width`, `scale`, `use_attention`, `num_classes`)."""
    return Res2NetANN(
        num_classes=cfg.num_classes,
        layers=tuple(cfg.layers),
        base_width=cfg.base_width,
        scale=cfg.scale,
        use_attention=cfg.use_attention,
    )
