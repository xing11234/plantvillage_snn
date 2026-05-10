"""
Channel + spatial attention for standard ANN feature maps `[B, C, H, W]`.

Mirrors `attention.CSA` layout for fair comparison with the SNN path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttentionANN(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        b, c, _h, _w = x.shape
        y = self.pool(x).view(b, c)
        w = self.gate(self.fc2(self.act(self.fc1(y)))).view(b, c, 1, 1)
        return x * w


class SpatialAttentionANN(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vmax, _ = torch.max(x, dim=1, keepdim=True)
        vmean = torch.mean(x, dim=1, keepdim=True)
        z = torch.cat([vmax, vmean], dim=1)
        z = self.gate(self.conv(z))
        return x * z


class CSAAnn(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttentionANN(channels, reduction=reduction)
        self.sa = SpatialAttentionANN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x
