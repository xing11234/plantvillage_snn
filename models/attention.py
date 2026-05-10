"""
Channel (CA) + Spatial (SA) attention for multi-step feature maps `[T, B, C, H, W]`.

All split / concat / pooling keep **time in dim=0**; channel operations use **dim=2**.
Gating uses standard `nn` layers on aggregated statistics (common in SNN literature).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze–Excitation style MLP over channels (temporal mean then spatial squeeze)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T,B,C,H,W]
        T, B, C, H, W = x.shape
        y = x.mean(dim=0)  # [B,C,H,W]
        y = self.pool(y).view(B, C)
        w = self.gate(self.fc2(self.act(self.fc1(y)))).view(B, C, 1, 1)
        w = w.unsqueeze(0).expand(T, -1, -1, -1, -1)
        return x * w


class SpatialAttention(nn.Module):
    """7x7 conv over channel-wise max/mean maps, applied per time step."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        out = []
        for t in range(T):
            xt = x[t]
            vmax, _ = torch.max(xt, dim=1, keepdim=True)
            vmean = torch.mean(xt, dim=1, keepdim=True)
            z = torch.cat([vmax, vmean], dim=1)
            z = self.gate(self.conv(z))
            out.append(xt * z)
        return torch.stack(out, dim=0)


class CSA(nn.Module):
    """Channel attention followed by spatial attention (serial, lightweight)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x
