"""
Utility package.

`train_utils` depends on SpikingJelly. Import it only when needed, e.g.
``from utils.train_utils import SpikeCounter``.

ANN-only code can use ``from utils.lr_schedule import cosine_lr`` and
``from utils.seed import set_seed``; loading ``utils`` no longer pulls SpikingJelly.
"""
from __future__ import annotations

from .lr_schedule import cosine_lr
from .seed import set_seed

__all__ = [
    "cosine_lr",
    "set_seed",
    "SpikeCounter",
    "first_spike_coding",
    "denormalize_to_01",
]


def __getattr__(name: str):
    if name == "SpikeCounter":
        from .train_utils import SpikeCounter

        return SpikeCounter
    if name == "first_spike_coding":
        from .train_utils import first_spike_coding

        return first_spike_coding
    if name == "denormalize_to_01":
        from .train_utils import denormalize_to_01

        return denormalize_to_01
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
