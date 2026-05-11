"""
Visualization helpers: save per-timestep spike rasters and MSF membrane traces.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron

from models.msf_neuron import MSFNode
from models.res2net_msf import set_msf_record_v_flag

from .train_utils import reset_snn_state


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_spike_grid(spikes: torch.Tensor, out_path: str, title: str = "First-spike coded input") -> None:
    """
    Save a row of images for `spikes` shaped `[T, C, H, W]` (single sample, no batch dim).

    Collapses RGB channels by taking the max across `C` for display.
    """
    if spikes.dim() != 4:
        raise ValueError("Expected spikes [T,C,H,W]")
    T, _, _, _ = spikes.shape
    fig, axes = plt.subplots(1, T, figsize=(T * 2.0, 2.8))
    if T == 1:
        axes = [axes]
    fig.suptitle(title)
    for t in range(T):
        ax = axes[t]
        frame = spikes[t].float().max(dim=0).values.cpu().numpy()
        ax.imshow(frame, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"t={t}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def save_membrane_curves(v_traces: List[torch.Tensor], out_path: str, title: str = "MSF membrane (sample units)") -> None:
    """
    v_traces: list length T of tensors [B,C,H,W] — plot several channels at the image center.
    """
    if not v_traces:
        raise ValueError("Empty v_traces")
    T = len(v_traces)
    v0 = v_traces[0]
    _, c, h, w = v0.shape
    hh, ww = h // 2, w // 2
    chans = min(8, c)
    ts = torch.arange(T)
    plt.figure(figsize=(8, 4))
    for i in range(chans):
        vals = [float(v_traces[t][0, i, hh, ww].cpu()) for t in range(T)]
        plt.plot(ts.numpy(), vals, label=f"ch{i}")
    plt.xlabel("time step")
    plt.ylabel("membrane v")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def visualize_sample(
    model: nn.Module,
    spike_input: torch.Tensor,
    out_dir: str,
    tag: str = "viz",
) -> None:
    """
    Parameters
    ----------
    model : nn.Module
        Expects `spike_input` `[T,B,C,H,W]` (B>=1; index 0 is visualized).
    spike_input : Tensor
        Encoded spikes for one mini-batch.
    """
    _ensure_dir(out_dir)
    tin = spike_input[:, 0].detach().cpu()  # [T,C,H,W]
    save_spike_grid(tin, os.path.join(out_dir, f"{tag}_input_spikes.png"))

    set_msf_record_v_flag(model, True)
    first_msf: Optional[MSFNode] = None
    for m in model.modules():
        if isinstance(m, MSFNode):
            first_msf = m
            break
    with torch.no_grad():
        reset_snn_state(model)
        _ = model(spike_input)
    set_msf_record_v_flag(model, False)

    if first_msf is not None and first_msf.last_v_trace:
        save_membrane_curves(first_msf.last_v_trace, os.path.join(out_dir, f"{tag}_msf_membrane.png"))


def save_paper_panel(
    model: nn.Module,
    spike_input: torch.Tensor,
    out_path: str,
    title: str = "SNN dynamics (single sample)",
    dpi: int = 300,
) -> None:
    """
    One high-resolution figure for papers: input spike frames + MSF membrane traces (if any).
    """
    from matplotlib import gridspec

    parent = os.path.dirname(out_path)
    if parent:
        _ensure_dir(parent)
    tin = spike_input[:, 0].detach().cpu()
    T, _, _, _ = tin.shape

    set_msf_record_v_flag(model, True)
    first_msf: Optional[MSFNode] = None
    for m in model.modules():
        if isinstance(m, MSFNode):
            first_msf = m
            break
    with torch.no_grad():
        reset_snn_state(model)
        _ = model(spike_input)
    set_msf_record_v_flag(model, False)

    has_v = first_msf is not None and bool(first_msf.last_v_trace)
    fig = plt.figure(figsize=(7.2, 5.0 if has_v else 2.6), constrained_layout=False)
    if has_v:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.15, 1.0], hspace=0.35)
    else:
        gs = gridspec.GridSpec(1, 1)

    g0 = gridspec.GridSpecFromSubplotSpec(1, T, subplot_spec=gs[0, 0], wspace=0.08)
    for t in range(T):
        ax = fig.add_subplot(g0[0, t])
        frame = tin[t].float().max(dim=0).values.cpu().numpy()
        ax.imshow(frame, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(f"$t={t}$", fontsize=9)
        ax.axis("off")

    if has_v:
        ax1 = fig.add_subplot(gs[1, 0])
        traces = first_msf.last_v_trace  # type: ignore[union-attr]
        Tm = len(traces)
        v0 = traces[0]
        _, c, h, w = v0.shape
        hh, ww = h // 2, w // 2
        chans = min(6, c)
        ts = torch.arange(Tm)
        for i in range(chans):
            vals = [float(traces[tt][0, i, hh, ww].cpu()) for tt in range(Tm)]
            ax1.plot(ts.numpy(), vals, lw=1.5, label=f"ch {i}")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Membrane $v$")
        ax1.set_title("First MSF layer — membrane at image center")
        ax1.legend(ncol=3, fontsize=7, frameon=True)
        ax1.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _short_module_name(name: str) -> str:
    parts = name.split(".")
    tail = parts[-3:] if len(parts) > 3 else parts
    return "/".join(tail) if len(tail) > 1 else name


def forward_collect_spiking_outputs(
    model: nn.Module,
    spike_input: torch.Tensor,
) -> List[Tuple[str, torch.Tensor]]:
    """
    One forward pass; collect each ``MSFNode`` / ``LIFNode`` output in **forward order**.

    Returns
    -------
    list of (module_name, tensor)
        Each tensor is ``[T, B, C, H, W]`` on CPU (detached), typically binary spikes.
    """
    captured: List[Tuple[str, torch.Tensor]] = []
    hooks: List = []

    def make_hook(full_name: str):
        def _hook(_m, _inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 5:
                captured.append((full_name, out.detach().float().cpu()))

        return _hook

    for name, m in model.named_modules():
        if isinstance(m, (MSFNode, neuron.LIFNode, neuron.ParametricLIFNode)):
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        reset_snn_state(model)
        _ = model(spike_input)
    for h in hooks:
        h.remove()
    return captured


def save_cross_layer_spike_heatmap(
    model: nn.Module,
    spike_input: torch.Tensor,
    out_path: str,
    *,
    title: str = "Spike activity across depth (mean over C,H,W)",
    max_rows: int = 32,
    dpi: int = 300,
) -> None:
    """
    **Layer × time** heatmap: each row is one spiking module, each column is simulation step ``t``.

    Interprets "pulse wave" through the network (aggregate over channels and space).
    """
    rows = forward_collect_spiking_outputs(model, spike_input)
    if not rows:
        return
    if len(rows) > max_rows:
        idx = torch.linspace(0, len(rows) - 1, max_rows).long().tolist()
        rows = [rows[i] for i in idx]
    T = rows[0][1].shape[0]
    mat = torch.zeros(len(rows), T)
    labels: List[str] = []
    for i, (name, s) in enumerate(rows):
        labels.append(_short_module_name(name))
        mat[i] = s[:, 0].mean(dim=(1, 2, 3))

    parent = os.path.dirname(out_path)
    if parent:
        _ensure_dir(parent)

    fig, ax = plt.subplots(figsize=(min(12, 0.45 * T + 2), max(3.0, 0.22 * len(rows) + 1.2)))
    im = ax.imshow(mat.numpy(), aspect="auto", cmap="magma", interpolation="nearest", vmin=0.0, vmax=float(mat.max().clamp_min(1e-6)))
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Layer (forward order)")
    ax.set_xticks(range(0, T, max(1, T // 8)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mean spike rate")
    fig.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cross_layer_spike_raster(
    model: nn.Module,
    spike_input: torch.Tensor,
    out_path: str,
    *,
    title: str = "Subsampled spike raster (layers × time)",
    max_layers: int = 14,
    neurons_per_layer: int = 40,
    seed: int = 0,
    dpi: int = 300,
) -> None:
    """
    **Raster-style** figure: subsampled spatial–channel units per layer; dot at ``(t, y)`` if spike.

    ``y`` stacks layers bottom-to-top (early layers at bottom). Uses real activations from one
    forward pass (dataset image encoded with ``first_spike_coding``).
    """
    rows = forward_collect_spiking_outputs(model, spike_input)
    if not rows:
        return
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    if len(rows) > max_layers:
        idx = torch.linspace(0, len(rows) - 1, max_layers).long().tolist()
        rows = [rows[i] for i in idx]

    xs: List[float] = []
    ys: List[float] = []
    y_cursor = 0
    layer_ticks: List[Tuple[float, str]] = []

    for name, s in rows:
        T, b, c, h, w = s.shape
        assert b >= 1
        flat = s[:, 0].reshape(T, c * h * w)
        n = flat.shape[1]
        k = min(neurons_per_layer, n)
        perm = torch.randperm(n, generator=g)[:k]
        sub = flat[:, perm]
        for ti in range(T):
            fired = (sub[ti] > 0.5).nonzero(as_tuple=False).flatten()
            for local_i in fired.tolist():
                xs.append(float(ti))
                ys.append(float(y_cursor + local_i))
        mid_y = y_cursor + k / 2.0
        layer_ticks.append((mid_y, _short_module_name(name)))
        y_cursor += k + max(2, neurons_per_layer // 8)

    parent = os.path.dirname(out_path)
    if parent:
        _ensure_dir(parent)

    fig_h = min(14, max(4.0, y_cursor * 0.012))
    fig, ax = plt.subplots(figsize=(min(10, 0.35 * T + 2.5), fig_h))
    if xs:
        ax.scatter(xs, ys, s=2.0, c="black", alpha=0.65, linewidths=0, rasterized=True)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, y_cursor + 0.5)
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Subsampled units (stacked by layer, forward order top-down)")
    ax.set_title(title)
    for y0, _lab in layer_ticks[:: max(1, len(layer_ticks) // 12)]:
        ax.axhline(y0, color="0.85", lw=0.4, zorder=0)
    fig.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_layer_mean_spike_bar(
    model: nn.Module,
    spike_input: torch.Tensor,
    out_path: str,
    max_bars: int = 16,
    dpi: int = 300,
) -> None:
    """Bar chart of mean spike rate per spiking module (one forward)."""
    means: List[tuple[str, float]] = []
    hooks: List = []

    def make_hook(short_name: str):
        def _hook(_m, _inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 5:
                means.append((short_name, float(out.detach().float().mean().cpu())))

        return _hook

    for name, m in model.named_modules():
        if isinstance(m, (MSFNode, neuron.LIFNode, neuron.ParametricLIFNode)):
            hooks.append(m.register_forward_hook(make_hook(_short_module_name(name))))

    with torch.no_grad():
        reset_snn_state(model)
        _ = model(spike_input)
    for h in hooks:
        h.remove()

    means = means[:max_bars]
    if not means:
        return
    labels, vals = zip(*means)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.25 * len(labels))))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), vals, color="#2c7bb6", height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean spike activity")
    ax.set_title("Per-layer mean spike rate (sample forward)")
    ax.set_xlim(0, max(vals) * 1.15 if max(vals) > 0 else 1.0)
    fig.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
