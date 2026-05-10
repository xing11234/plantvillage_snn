"""
Torchvision **ResNet-18** ANN for fair comparison with `sj_resnet18` SNN (same canonical backbone family).

Input `[B, 3, H, W]`, logits `[B, num_classes]`. Final `fc` is replaced for the dataset class count.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from config import TrainConfig


def build_tv_resnet18_ann(
    num_classes: int,
    *,
    imagenet_pretrained: bool = False,
) -> nn.Module:
    """
    Official ``torchvision.models.resnet18`` with a linear head for ``num_classes``.

    Parameters
    ----------
    num_classes
        Dataset class count (e.g. PlantVillage color = 38).
    imagenet_pretrained
        If True, load ImageNet-1K weights and replace only ``fc``. If False, random init
        (typical setting when comparing to from-scratch SNN).
    """
    from torchvision.models import resnet18

    model: nn.Module
    if imagenet_pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                model = resnet18(pretrained=True)  # type: ignore[call-arg]
            except TypeError:
                model = resnet18(weights=None)  # type: ignore[call-arg]
    else:
        try:
            model = resnet18(weights=None)
        except TypeError:
            model = resnet18(pretrained=False)  # type: ignore[call-arg]

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(num_classes))
    return model


def build_tv_resnet18_ann_from_config(
    cfg: "TrainConfig",
    *,
    imagenet_pretrained: bool = False,
) -> nn.Module:
    """Same as :func:`build_tv_resnet18_ann` using ``cfg.num_classes``."""
    return build_tv_resnet18_ann(int(cfg.num_classes), imagenet_pretrained=imagenet_pretrained)
