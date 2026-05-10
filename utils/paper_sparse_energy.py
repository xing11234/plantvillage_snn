"""
Paper helpers: read TensorBoard spikes at best val epoch, build **activity / ops proxy** tables.

These metrics are **dimensionless proxies** derived from logged mean spike rates — not Joules.
For neuromorphic hardware one would map synaptic events to energy separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _load_scalars(logdir: Path, tag: str, max_step: int) -> List[Tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(int(s.step), float(s.value)) for s in ea.Scalars(tag) if int(s.step) < max_step]


def best_val_epoch(logdir: Path, max_step: int = 256) -> Tuple[Optional[int], Optional[float]]:
    """Epoch index and value where ``acc/val`` is maximal (TB step = epoch)."""
    pairs = _load_scalars(logdir, "acc/val", max_step)
    if not pairs:
        return None, None
    ep, v = max(pairs, key=lambda x: x[1])
    return ep, v


def spike_rate_at_epoch(
    logdir: Path,
    epoch: int,
    max_step: int = 256,
) -> Optional[float]:
    """Mean spike rate at given epoch; tries common TB tag names."""
    for tag in ("spike/val", "spike/val_global_rate"):
        pairs = _load_scalars(logdir, tag, max_step)
        if not pairs:
            continue
        m = {e: v for e, v in pairs}
        if epoch in m:
            return float(m[epoch])
        # nearest
        closest = min(m.keys(), key=lambda x: abs(x - epoch))
        return float(m[closest])
    return None


def build_ops_proxy_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a comparison table from **manual or scripted** rows.

    Each row dict may contain:
        ``method``, ``neuron`` (e.g. MSF / LIF / -), ``T`` (int), ``val_acc`` (float 0–100 or None),
        ``rho_spike`` (0–1 mean activity, optional), ``notes`` (str).

    Adds:
        - ``ops_activity_proxy`` = ``rho_spike * T`` (synaptic-event density vs one ANN forward)
        - ``dense_time_unfold`` = ``T`` (naïve upper bound: full dense compute each step)
        - ``activity_vs_time_unfold`` = ``rho_spike`` (fraction of dense temporal compute under linear proxy)

    **ANN convention:** set ``T=1``, ``rho_spike=1.0`` (activations always present; not event-sparse).
    """
    df = pd.DataFrame(rows)
    if "rho_spike" not in df.columns:
        df["rho_spike"] = None
    if "T" not in df.columns:
        df["T"] = 1
    df["T"] = df["T"].fillna(1).astype(int)
    rho = pd.to_numeric(df["rho_spike"], errors="coerce")
    T = df["T"].astype(float)
    # 线性代理：相对 ANN(T=1,ρ=1) 的「事件×时间」密度；非电路能耗
    df["ops_activity_proxy"] = rho * T
    df["dense_time_unfold"] = T
    df["spike_fraction_rho"] = rho
    return df


def row_from_tb(
    name: str,
    neuron: str,
    T: int,
    logdir: Path,
    max_step: int = 256,
) -> Dict[str, Any]:
    """One SNN row: best-val epoch and spike rate at that epoch (if logged)."""
    ep, vacc = best_val_epoch(logdir, max_step)
    rho = None
    if ep is not None:
        rho = spike_rate_at_epoch(logdir, ep, max_step)
    return {
        "method": name,
        "neuron": neuron,
        "T": int(T),
        "val_acc": float(vacc * 100) if vacc is not None else None,
        "rho_spike": rho,
        "tb_logdir": str(logdir),
        "best_epoch": ep,
    }
