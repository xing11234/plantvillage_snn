"""
MSF / LIF + Res2Net-style backbone for multi-step tensors `[T, B, C, H, W]`.

**Res2Net split / concat** operate on **channel dimension `dim=2`** so the time axis
`dim=0` is never mixed with channels. Inside each `Bottle2neckMSF` block, tensors keep
the ordering `[T, B, C, H, W]` end-to-end.
"""
from __future__ import annotations

import math
from typing import List, Optional, Type

import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron

from .attention import CSA
from .msf_neuron import MSFNode


def make_spiking_node(
    use_msf: bool,
    *,
    decay: float = 0.25,
    D: int = 4,
    v_threshold: float = 1.0,
    surrogate: str = "rect",
    surrogate_alpha: float = 1.0,
    lif_tau: float = 2.0,
    record_v: bool = False,
) -> nn.Module:
    if use_msf:
        return MSFNode(
            decay=decay,
            D=D,
            v_threshold=v_threshold,
            surrogate=surrogate,
            surrogate_alpha=float(surrogate_alpha),
            step_mode="m",
            record_v=record_v,
        )
    return neuron.LIFNode(
        tau=lif_tau,
        decay_input=True,
        v_reset=0.0,
        detach_reset=True,
        step_mode="m",
    )


class SeqConvBnMSF(nn.Module):
    """Conv2d (multi-step) + BN + spiking node."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        node: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv = layer.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias, step_mode="m"
        )
        self.bn = layer.BatchNorm2d(out_ch, step_mode="m")
        self.node = node if node is not None else neuron.LIFNode(step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.node(x)
        return x


class Bottle2neckMSF(nn.Module):
    """
    Res2Net multi-scale block with split/concat on **dim=2** (channel).

    Feature flow: 1x1 expand -> split into `scale` groups of `width` channels ->
    hierarchical 3x3 convolutions on summed groups -> concat -> 1x1 project.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 32,
        scale: int = 4,
        use_msf: bool = True,
        use_attention: bool = True,
        msf_decay: float = 0.25,
        msf_D: int = 4,
        msf_v_threshold: float = 1.0,
        msf_surrogate: str = "rect",
        msf_surrogate_alpha: float = 1.0,
        lif_tau: float = 2.0,
        record_v: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.width = max(int(math.floor(planes * (base_width / 64.0))), 1)
        self.downsample = downsample

        def node():
            return make_spiking_node(
                use_msf,
                decay=msf_decay,
                D=msf_D,
                v_threshold=msf_v_threshold,
                surrogate=msf_surrogate,
                surrogate_alpha=msf_surrogate_alpha,
                lif_tau=lif_tau,
                record_v=record_v,
            )

        # Spatial downsample is handled on the 1x1 expand so all scale splits stay aligned.
        self.branch = SeqConvBnMSF(
            inplanes, self.width * scale, kernel_size=1, stride=stride, padding=0, node=node()
        )

        self.convs = nn.ModuleList()
        self.scales = nn.ModuleList()
        for _ in range(scale - 1):
            self.convs.append(
                layer.Conv2d(self.width, self.width, 3, stride=1, padding=1, bias=False, step_mode="m")
            )
            self.scales.append(nn.Sequential(layer.BatchNorm2d(self.width, step_mode="m"), node()))

        self.tail = SeqConvBnMSF(
            self.width * scale, planes, kernel_size=1, stride=1, padding=0, node=node()
        )

        self.use_attention = use_attention
        self.attn = CSA(planes) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T,B,C,H,W]
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.branch(x)
        # Res2Net hierarchical fusion: split / 3x3 / concat all on **dim=2 (C)**
        splits = torch.split(out, self.width, dim=2)
        feats: List[torch.Tensor] = [splits[0]]
        y = splits[0]
        for i in range(1, self.scale):
            y = self.convs[i - 1](y + splits[i])
            y = self.scales[i - 1](y)
            feats.append(y)
        out = torch.cat(feats, dim=2)
        out = self.tail(out)
        out = self.attn(out)
        return out + identity


class MSFRes2Net(nn.Module):
    """
    Res2Net-29-style topology: stem + 4 stages with `(2,2,2,2)` blocks, `base_width=32`, `scale=4`.

    Classification logits are obtained by **temporal average** of the classifier output.
    """

    def __init__(
        self,
        num_classes: int = 38,
        layers: tuple = (2, 2, 2, 2),
        base_width: int = 32,
        scale: int = 4,
        use_msf: bool = True,
        use_attention: bool = True,
        msf_decay: float = 0.25,
        msf_D: int = 4,
        msf_v_threshold: float = 1.0,
        msf_surrogate: str = "rect",
        msf_surrogate_alpha: float = 1.0,
        lif_tau: float = 2.0,
        record_v: bool = False,
    ):
        super().__init__()
        self.inplanes = 64
        self.base_width = base_width
        self.scale = scale
        self.use_msf = use_msf
        self.use_attention = use_attention

        def node():
            return make_spiking_node(
                use_msf,
                decay=msf_decay,
                D=msf_D,
                v_threshold=msf_v_threshold,
                surrogate=msf_surrogate,
                surrogate_alpha=msf_surrogate_alpha,
                lif_tau=lif_tau,
                record_v=record_v,
            )

        self.conv1 = SeqConvBnMSF(3, 64, kernel_size=7, stride=2, padding=3, node=node())
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode="m")

        self.layer1 = self._make_layer(
            64, layers[0], stride=1, **self._msf_kwargs(msf_decay, msf_D, msf_v_threshold, msf_surrogate, msf_surrogate_alpha, lif_tau, record_v)
        )
        self.layer2 = self._make_layer(
            128, layers[1], stride=2, **self._msf_kwargs(msf_decay, msf_D, msf_v_threshold, msf_surrogate, msf_surrogate_alpha, lif_tau, record_v)
        )
        self.layer3 = self._make_layer(
            256, layers[2], stride=2, **self._msf_kwargs(msf_decay, msf_D, msf_v_threshold, msf_surrogate, msf_surrogate_alpha, lif_tau, record_v)
        )
        self.layer4 = self._make_layer(
            512, layers[3], stride=2, **self._msf_kwargs(msf_decay, msf_D, msf_v_threshold, msf_surrogate, msf_surrogate_alpha, lif_tau, record_v)
        )

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.fc = layer.Linear(512, num_classes, step_mode="m")

        self._init_weights()

    def _msf_kwargs(self, decay, D, vth, surr, surr_alpha, tau, rec):
        return dict(
            use_msf=self.use_msf,
            use_attention=self.use_attention,
            msf_decay=decay,
            msf_D=D,
            msf_v_threshold=vth,
            msf_surrogate=surr,
            msf_surrogate_alpha=surr_alpha,
            lif_tau=tau,
            record_v=rec,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            w = getattr(m, "weight", None)
            if w is not None and w.dim() >= 2 and isinstance(m, (nn.Conv2d, nn.Linear, layer.Conv2d, layer.Linear)):
                nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, layer.BatchNorm2d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int, **kwargs) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            ds_layers = [
                layer.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False, step_mode="m"),
                layer.BatchNorm2d(planes, step_mode="m"),
            ]
            # shortcut carries spikes; keep dynamics light with same node type
            ds_layers.append(
                make_spiking_node(
                    kwargs.get("use_msf", True),
                    decay=kwargs.get("msf_decay", 0.25),
                    D=kwargs.get("msf_D", 4),
                    v_threshold=kwargs.get("msf_v_threshold", 1.0),
                    surrogate=kwargs.get("msf_surrogate", "rect"),
                    surrogate_alpha=float(kwargs.get("msf_surrogate_alpha", 1.0)),
                    lif_tau=kwargs.get("lif_tau", 2.0),
                    record_v=kwargs.get("record_v", False),
                )
            )
            downsample = nn.Sequential(*ds_layers)

        layers: List[nn.Module] = []
        layers.append(
            Bottle2neckMSF(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                base_width=self.base_width,
                scale=self.scale,
                **kwargs,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                Bottle2neckMSF(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    base_width=self.base_width,
                    scale=self.scale,
                    **kwargs,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq : Tensor
            Shape `[T, B, 3, H, W]` (already spike-coded or analog drive per step).

        Returns
        -------
        Tensor
            Shape `[T, B, num_classes]` logits per timestep (average externally for loss).
        """
        x = self.conv1(x_seq)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # Merge C,H,W so the last dim is in_features (512). flatten(3) only merges H,W and leaves [T,B,C,1].
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x


def build_model(cfg) -> MSFRes2Net:
    """Build model from a `TrainConfig`-like object."""
    return MSFRes2Net(
        num_classes=cfg.num_classes,
        layers=tuple(cfg.layers),
        base_width=cfg.base_width,
        scale=cfg.scale,
        use_msf=cfg.use_msf,
        use_attention=cfg.use_attention,
        msf_decay=cfg.msf_decay,
        msf_D=cfg.msf_D,
        msf_v_threshold=cfg.msf_v_threshold,
        msf_surrogate=cfg.msf_surrogate,
        msf_surrogate_alpha=float(cfg.msf_surrogate_alpha),
        lif_tau=cfg.lif_tau,
        record_v=getattr(cfg, "record_v_for_viz", False),
    )


def set_msf_record_v_flag(module: nn.Module, enabled: bool) -> None:
    """Toggle membrane recording on all MSFNode instances (for visualization)."""
    for m in module.modules():
        if isinstance(m, MSFNode):
            m.record_v = bool(enabled)
