"""
Training helpers: first-spike coding, cosine LR, and **SpikeCounter** for energy-style metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional as sj_functional
from spikingjelly.activation_based import neuron

from models.msf_neuron import MSFNode

from .lr_schedule import cosine_lr


def reset_snn_state(net: nn.Module) -> None:
    """
    Reset SpikingJelly ``MemoryModule`` states (e.g. LIF membrane) before each forward.

    If this is skipped with ``LIFNode`` + multi-step ``[T,B,...]``, membrane can leak
    across batches and autograd may error (e.g. "backward through the graph a second time").
    ``MSFNode`` re-inits state inside ``forward``; calling this is still safe.
    """
    sj_functional.reset_net(net)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize_to_01(x: torch.Tensor, mean: Tuple[float, ...] = IMAGENET_MEAN, std: Tuple[float, ...] = IMAGENET_STD) -> torch.Tensor:
    """
    Map ImageNet-normalized tensors back to approximately `[0, 1]` for intensity-based coding.

    Parameters
    ----------
    x : Tensor
        Shape `[B, 3, H, W]` (normalized).
    """
    out = x.clone()
    for i in range(3):
        out[:, i] = out[:, i] * std[i] + mean[i]
    return torch.clamp(out, 0.0, 1.0)


def first_spike_coding(image: torch.Tensor, T: int = 8) -> torch.Tensor:
    """
    First-spike time coding on **intensities in [0, 1]**.

    Brighter pixels fire earlier. Returns `[T, B, C, H, W]`.

    Parameters
    ----------
    image : Tensor
        `[B, C, H, W]` with values in `[0, 1]`.
    """
    if image.dim() != 4:
        raise ValueError("first_spike_coding expects [B,C,H,W]")
    if T < 2:
        raise ValueError("T must be >= 2")
    spike_time = (1.0 - image) * (T - 1)
    spike_time = torch.round(spike_time).long().clamp(0, T - 1)
    T_, B, C, H, W = T, image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    spikes = image.new_zeros(T_, B, C, H, W)
    for t in range(T_):
        spikes[t] = (spike_time == t).to(image.dtype)
    return spikes


class SpikeCounter:
    """
    Registers forward hooks on `MSFNode` and `LIFNode` modules to estimate mean spike rates.

    After each epoch you can read:
    - ``global_spike_rate``: total spikes / total scalar outputs (synaptic events / neuron / step)
    - ``per_layer_spike_rate``: dict of the same quantity per hooked module name
    """

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.hooks: List = []
        self._total_spikes: float = 0.0
        self._total_elements: float = 0.0
        self._layer_spikes: Dict[str, float] = {}
        self._layer_elements: Dict[str, float] = {}
        self.global_spike_rate: float = 0.0
        self.per_layer_spike_rate: Dict[str, float] = {}
        if self.enabled:
            self._register_hooks()

    def _register_hooks(self) -> None:
        def make_hook(name: str):
            def _hook(_module, _inp, out):
                if not isinstance(out, torch.Tensor):
                    return
                if out.dim() != 5:
                    return
                # Count binary spikes in [T,B,C,H,W]
                spikes = float(out.detach().float().sum().cpu().item())
                elems = float(out.numel())
                self._total_spikes += spikes
                self._total_elements += elems
                self._layer_spikes[name] = self._layer_spikes.get(name, 0.0) + spikes
                self._layer_elements[name] = self._layer_elements.get(name, 0.0) + elems

            return _hook

        for name, m in self.model.named_modules():
            if isinstance(m, (MSFNode, neuron.LIFNode, neuron.ParametricLIFNode)):
                self.hooks.append(m.register_forward_hook(make_hook(name)))

    def reset_buffer(self) -> None:
        self._total_spikes = 0.0
        self._total_elements = 0.0
        self._layer_spikes.clear()
        self._layer_elements.clear()

    def finalize_epoch(self) -> None:
        """Call at epoch end after all training/validation forwards."""
        self.global_spike_rate = (self._total_spikes / self._total_elements) if self._total_elements > 0 else 0.0
        self.per_layer_spike_rate = {}
        for k, s in self._layer_spikes.items():
            denom = self._layer_elements.get(k, 0.0)
            self.per_layer_spike_rate[k] = (s / denom) if denom > 0 else 0.0

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
