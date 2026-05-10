"""
Model package.

ANN modules (`res2net_ann`) do **not** require spikingjelly. SNN symbols are loaded
lazily so `from models.res2net_ann import build_ann_model` works in a plain torch env.
"""
from __future__ import annotations

from .attention import CSA, ChannelAttention, SpatialAttention
from .attention_ann import CSAAnn
from .res2net_ann import Res2NetANN, build_ann_model
from .resnet18_tv_ann import build_tv_resnet18_ann, build_tv_resnet18_ann_from_config

__all__ = [
    "CSA",
    "CSAAnn",
    "ChannelAttention",
    "SpatialAttention",
    "Res2NetANN",
    "build_ann_model",
    "build_tv_resnet18_ann",
    "build_tv_resnet18_ann_from_config",
    "MSFNode",
    "MSFRes2Net",
    "build_model",
    "make_spiking_node",
    "set_msf_record_v_flag",
]


def __getattr__(name: str):
    if name == "MSFNode":
        from .msf_neuron import MSFNode

        return MSFNode
    if name in ("MSFRes2Net", "build_model", "make_spiking_node", "set_msf_record_v_flag"):
        from . import res2net_msf as _rm

        return getattr(_rm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
