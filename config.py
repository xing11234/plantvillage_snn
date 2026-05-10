"""
Central configuration for PlantVillage MSF-Res2Net experiments.

Toggle USE_MSF / USE_ATTENTION for ablation studies.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional


@dataclass
class TrainConfig:
    # --- experiment switches (ablation) ---
    use_msf: bool = True
    """If False, conv-BN stacks use standard LIFNode instead of MSF."""

    use_attention: bool = True
    """If False, CSA blocks in residual paths are identity (no extra params)."""

    # --- time & MSF ---
    T: int = 8
    """Number of simulation time steps (matches first-spike coding length)."""

    msf_D: int = 4
    """Number of thresholds in MSF (multi-level surrogate / multi-threshold)."""

    msf_decay: float = 0.25
    """Membrane decay factor per step: v <- decay * v + input."""

    msf_v_threshold: float = 1.0
    """Base firing threshold (scaled across D levels)."""

    msf_surrogate: str = "rect"
    """One of: rect | sigmoid | arctan | gaussian."""

    msf_surrogate_alpha: float = 1.0
    """
    Width / scale for **smooth** surrogates in ``MSFNode`` backward (sigmoid, arctan, gaussian).
    ``rect`` only uses α as the flat window half-width; typical rect runs ignore fine-tuning α.

    Tuning (empirical): smooth surrogates often need **larger α than 1** (e.g. arctan 2–5) so gradients
    near threshold are not too weak vs ``rect``. **Sigmoid** can vanish if α is too large (narrow band)
    or too small (saturated tails); try **0.5–2** jointly with **learning rate** (e.g. slightly higher lr
    than rect when mean surrogate magnitude is lower).
    """

    msf_reset_mode: str = "hard"
    """hard: subtract threshold on spike; soft: multiplicative leak only."""

    lif_tau: float = 2.0
    """Membrane time constant used when use_msf=False (LIFNode)."""

    # --- Res2Net-29_32w_4s style ---
    base_width: int = 32
    scale: int = 4
    layers: tuple = (2, 2, 2, 2)
    """Four stages; (2,2,2,2) + stem matches common Res2Net-29 naming."""

    num_classes: int = 38
    """PlantVillage color split class count (update if using another config)."""

    # --- data ---
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    dataset_name: str = "mohanty/PlantVillage"
    dataset_config: str = "color"

    # --- optimizer & schedule ---
    epochs: int = 100
    lr: float = 0.1  # SGD peak; sigmoid/arctan often need slightly higher lr than rect for same training budget
    weight_decay: float = 5e-4
    momentum: float = 0.9
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip_max_norm: float = 5.0

    label_smoothing: float = 0.0
    """Torch cross-entropy label smoothing (0 disables). Typical 0.05–0.1 to reduce overfitting."""

    # --- system ---
    amp: bool = True
    device: str = "cuda"
    seed: int = 42

    # --- logging & checkpoints ---
    log_dir: str = "runs/plantvillage_snn"
    checkpoint_dir: str = "checkpoints"
    save_name: str = "best_msf_res2net.pt"

    # --- spike statistics ---
    spike_counter_enabled: bool = True

    # --- visualization (optional hooks) ---
    viz_enabled: bool = False
    viz_dir: str = "viz_out"
    viz_every_n_epochs: int = 1

    batch_log_interval: int = 0
    """If >0, `train_one_epoch` prints running loss/acc every N batches (useful in notebooks)."""

    def surrogate_choices(self) -> List[str]:
        return ["rect", "sigmoid", "arctan", "gaussian"]

    def validate(self) -> None:
        if self.msf_surrogate not in self.surrogate_choices():
            raise ValueError(f"msf_surrogate must be one of {self.surrogate_choices()}")
        if self.T < 2:
            raise ValueError("T must be >= 2 for first-spike coding.")
        if self.msf_D < 1:
            raise ValueError("msf_D must be >= 1.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.msf_surrogate_alpha <= 0:
            raise ValueError("msf_surrogate_alpha must be > 0.")

    def tag(self) -> str:
        return f"MSF{int(self.use_msf)}_ATT{int(self.use_attention)}_T{self.T}_D{self.msf_D}"

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        return d


# Preset profiles for quick switching in train.py / test.py
PRESETS: Dict[str, TrainConfig] = {
    "default": TrainConfig(),
    "no_msf": TrainConfig(use_msf=False),
    "no_attention": TrainConfig(use_attention=False),
    "ablation_baseline": TrainConfig(use_msf=False, use_attention=False),
}


def get_config(preset: Optional[str] = None, **overrides: Any) -> TrainConfig:
    base = PRESETS[preset] if preset and preset in PRESETS else TrainConfig()
    cfg = replace(base, **overrides)
    cfg.validate()
    return cfg
