"""
Spiking **ResNet-18** from SpikingJelly (`spiking_resnet18`), multi-step `[T,B,C,H,W]`.

Use when you want a **lighter** backbone than MSF-Res2Net (fewer params, often faster / less OOM).
Same training loop as `res2net_msf.MSFRes2Net`: forward returns `[T, B, num_classes]`.

Neurons (see ``TrainConfig``): **MSF** (`use_msf=True`), **LIF** (`use_msf=False`, ``lif_variant="lif"``),
or **PLIF** / ``ParametricLIFNode`` (`use_msf=False`, ``lif_variant="plif"``) with learnable decay.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from spikingjelly.activation_based import functional as sj_functional
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.model import spiking_resnet

from .msf_neuron import MSFNode

if TYPE_CHECKING:
    from config import TrainConfig


def build_spiking_resnet18(cfg: "TrainConfig") -> spiking_resnet.SpikingResNet:
    """
    Build `spiking_resnet18` with MSFNode, LIFNode, or ParametricLIFNode (PLIF), `num_classes` from cfg.

    Call **after** `cfg.num_classes` is set (e.g. from dataloader).
    """
    nc = int(cfg.num_classes)
    if cfg.use_msf:
        model = spiking_resnet.spiking_resnet18(
            pretrained=False,
            progress=False,
            spiking_neuron=MSFNode,
            num_classes=nc,
            decay=float(cfg.msf_decay),
            D=int(cfg.msf_D),
            v_threshold=float(cfg.msf_v_threshold),
            surrogate=str(cfg.msf_surrogate),
            surrogate_alpha=float(cfg.msf_surrogate_alpha),
            step_mode="m",
            record_v=False,
        )
    elif getattr(cfg, "lif_variant", "lif") == "plif":
        # Parametric LIF (Fang et al.): learnable membrane time constant; init_tau matches LIF tau for fair comparison.
        model = spiking_resnet.spiking_resnet18(
            pretrained=False,
            progress=False,
            spiking_neuron=neuron.ParametricLIFNode,
            num_classes=nc,
            init_tau=float(cfg.lif_tau),
            decay_input=True,
            v_reset=0.0,
            detach_reset=True,
            step_mode="m",
        )
    else:
        model = spiking_resnet.spiking_resnet18(
            pretrained=False,
            progress=False,
            spiking_neuron=neuron.LIFNode,
            num_classes=nc,
            tau=float(cfg.lif_tau),
            decay_input=True,
            v_reset=0.0,
            detach_reset=True,
            step_mode="m",
        )
    sj_functional.set_step_mode(model, "m")
    return model
