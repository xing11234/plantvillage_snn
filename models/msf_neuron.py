"""
Multi-Synaptic Firing (MSF) neuron with multiple thresholds and configurable surrogate gradients.

Designed for multi-step tensors `[T, B, C, H, W]`. When `use_msf=False` in the network factory,
standard `LIFNode` from SpikingJelly is used instead (see `res2net_msf.py`).

**Hyperparameters (see ``TrainConfig.msf_surrogate`` / ``msf_surrogate_alpha``):**
``surrogate_alpha`` (α) scales the *width* of smooth surrogate derivatives (arctan / sigmoid /
gaussian). Larger α tends to sharpen the peak near the firing surface; too large can narrow the
effective support (weaker or vanishing gradients far from threshold). Tune α **together with**
optimizer learning rate when switching from ``rect`` to smooth surrogates.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based.base import MultiStepModule


def _surrogate_grad_rect(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Rectangular surrogate: flat in (-alpha, alpha)."""
    a = alpha if alpha > 0 else 1.0
    return ((x > -a) & (x < a)).float() / (2 * a)


def _surrogate_grad_sigmoid(x: torch.Tensor, alpha: float) -> torch.Tensor:
    a = alpha if alpha > 0 else 1.0
    ex = torch.exp(-torch.abs(x) / a)
    return (1.0 / a) * ex / (1.0 + ex).pow(2)


def _surrogate_grad_arctan(x: torch.Tensor, alpha: float) -> torch.Tensor:
    a = alpha if alpha > 0 else 1.0
    return a / (1.0 + (a * x).pow(2)) / math.pi


def _surrogate_grad_gaussian(x: torch.Tensor, alpha: float) -> torch.Tensor:
    a = alpha if alpha > 0 else 1.0
    return (1.0 / (a * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (x / a).pow(2))


def pick_surrogate(name: str):
    name = name.lower()
    if name == "rect":
        return _surrogate_grad_rect
    if name == "sigmoid":
        return _surrogate_grad_sigmoid
    if name == "arctan":
        return _surrogate_grad_arctan
    if name == "gaussian":
        return _surrogate_grad_gaussian
    raise ValueError(f"Unknown surrogate '{name}'. Choose rect|sigmoid|arctan|gaussian.")


class _MSFSpike(torch.autograd.Function):
    """Straight-through estimator with MSF-style multi-threshold surrogate blending."""

    @staticmethod
    def forward(
        ctx,
        v: torch.Tensor,
        thetas: torch.Tensor,
        surrogate_name: str,
        alpha: float,
    ) -> torch.Tensor:
        """
        v: [B, C, H, W]
        thetas: [D] ascending thresholds
        Returns hard spike [B,C,H,W] if v crosses the *lowest* threshold (standard IF),
        while backward uses a sum of surrogates at all D thresholds (MSF gradient path).
        """
        ctx.save_for_backward(v, thetas)
        ctx.surrogate_name = surrogate_name
        ctx.alpha = alpha
        spike = (v >= thetas[0]).to(v.dtype)
        return spike

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, thetas = ctx.saved_tensors
        fn = pick_surrogate(ctx.surrogate_name)
        # Blend surrogate gradients across thresholds (multi-synaptic signal in backward).
        g = torch.zeros_like(v)
        for theta in thetas:
            g = g + fn(v - theta, ctx.alpha)
        g = g / max(thetas.numel(), 1)
        return grad_output * g, None, None, None


class MSFNode(nn.Module, MultiStepModule):
    """
    Leaky integrate-and-fire style dynamics with MSF surrogate over D thresholds.

    - Membrane: v_t = decay * v_{t-1} + x_t
    - Spike (forward): s_t = H(v_t - theta_0)
    - Reset: v_t <- v_t - s_t * theta_0  (hard subtract; mimics charge removal)
    - Surrogate: gradients averaged over (v - theta_d) for d in [0, D)

    Input / output: `[T, B, C, H, W]` (multi-step mode compatible with SpikingJelly layout).
    """

    def __init__(
        self,
        decay: float = 0.25,
        D: int = 4,
        v_threshold: float = 1.0,
        surrogate: str = "rect",
        surrogate_alpha: float = 1.0,
        step_mode: str = "m",
        record_v: bool = False,
    ):
        # surrogate_alpha is forwarded to _MSFSpike; set via TrainConfig.msf_surrogate_alpha in builds.
        nn.Module.__init__(self)
        if step_mode != "m":
            raise ValueError("MSFNode in this project is fixed to step_mode='m' ([T,B,C,H,W]).")
        self.step_mode = step_mode
        self.decay = float(decay)
        self.D = int(D)
        self.surrogate = surrogate
        self.surrogate_alpha = float(surrogate_alpha)
        self.record_v = bool(record_v)

        thetas = torch.linspace(float(v_threshold), float(v_threshold) * (1.0 + 0.25 * max(D - 1, 0)), D)
        self.register_buffer("thetas", thetas)

        # Last forward stats (for TensorBoard / debugging)
        self.last_mean_spike_rate: float = 0.0
        # Optional visualization: list of [B,C,H,W] per time step (detached)
        self.last_v_trace: List[torch.Tensor] = []

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() != 5:
            raise ValueError(f"Expected x_seq [T,B,C,H,W], got shape {tuple(x_seq.shape)}")
        T, B, C, H, W = x_seq.shape
        v = x_seq.new_zeros(B, C, H, W)
        spikes: List[torch.Tensor] = []
        self.last_v_trace = []
        spike_accum = x_seq.new_zeros(())

        thetas = self.thetas.to(dtype=x_seq.dtype, device=x_seq.device)
        for t in range(T):
            v = self.decay * v + x_seq[t]
            if self.record_v:
                self.last_v_trace.append(v.detach().clone())
            s = _MSFSpike.apply(v, thetas, self.surrogate, self.surrogate_alpha)
            v = v - s * thetas[0]
            spikes.append(s)
            spike_accum = spike_accum + s.sum()

        out = torch.stack(spikes, dim=0)
        denom = float(T * B * C * H * W)
        self.last_mean_spike_rate = float((spike_accum / denom).detach().cpu().item())
        return out

    def extra_repr(self) -> str:
        return (
            f"decay={self.decay}, D={self.D}, surrogate={self.surrogate}, "
            f"surrogate_alpha={self.surrogate_alpha}, step_mode={self.step_mode}"
        )
