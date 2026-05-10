#!/usr/bin/env python3
"""
ANN training report: curves from TensorBoard events (first N epochs), optional val-set F1/acc.

Usage (from repo root):
  python scripts/ann_report.py --checkpoint best_ann_res2net.pt \\
      --logdir path/to/runs_ann/kaggle \\
      --data_root path/to/plantvillage_root \\
      --max_epoch 15 \\
      --output_dir ann_report_out

- --logdir: folder containing ``events.out.tfevents.*`` (e.g. copy ``/kaggle/working/runs_ann/kaggle`` locally).
- --data_root: if set, builds val loader (same seed/val_ratio as Kaggle notebook) and evaluates F1.
- Energy proxy: parameter count + optional ``--benchmark`` throughput (images/s) on CUDA.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_scalars_from_tfevents(logdir: Path, max_epoch: int) -> Dict[str, List[Tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not logdir.is_dir():
        raise FileNotFoundError(f"logdir not found: {logdir}")
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in ("loss/train", "acc/train", "acc/val"):
        if tag not in tags:
            continue
        rows = [(s.step, float(s.value)) for s in ea.Scalars(tag) if int(s.step) < max_epoch]
        out[tag] = rows
    return out


def _plot_curves(series: Dict[str, List[Tuple[int, float]]], out_path: Path, max_epoch: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    titles = [("loss/train", "Train loss"), ("acc/train", "Train acc"), ("acc/val", "Val acc")]
    for ax, (key, title) in zip(axes, titles):
        pts = series.get(key, [])
        if not pts:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key.split("/")[-1])
        ax.set_title(title)
        ax.set_xlim(-0.5, max_epoch - 0.5)
        if "acc" in key:
            ax.set_ylim(0, 1)
    fig.suptitle(f"ANN metrics (epochs 0–{max_epoch - 1})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _cfg_from_dict(d: Dict[str, Any]):
    from dataclasses import fields, replace

    from config import TrainConfig

    names = {f.name for f in fields(TrainConfig)}
    kwargs = {k: v for k, v in d.items() if k in names}
    return replace(TrainConfig(), **kwargs)


def _build_ann_for_checkpoint(ckpt: Dict[str, Any], cfg) -> Any:
    """Rebuild ANN architecture to match ``ckpt['model']`` (keys from Kaggle notebook)."""
    backbone = ckpt.get("backbone", "Res2NetANN")
    if backbone == "torchvision_resnet18":
        from models.resnet18_tv_ann import build_tv_resnet18_ann_from_config

        return build_tv_resnet18_ann_from_config(cfg, imagenet_pretrained=False)
    from models.res2net_ann import build_ann_model

    return build_ann_model(cfg)


def _evaluate_val_f1(
    checkpoint: Path,
    cfg,
    data_root: Path,
    device: str,
    auto_find_subdir: bool,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    from data.kaggle_dataloader import get_kaggle_dataloaders

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, val_loader, num_classes = get_kaggle_dataloaders(
        str(data_root),
        image_size=cfg.image_size,
        batch_size=min(64, cfg.batch_size),
        num_workers=0,
        val_ratio=0.2,
        seed=cfg.seed,
        auto_find_subdir=auto_find_subdir,
    )
    if num_classes != cfg.num_classes:
        cfg.num_classes = num_classes
    model = _build_ann_for_checkpoint(ckpt, cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(dev)
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(dev)
            logits = model(images)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            ps.extend(pred)
            ys.extend(labels.numpy().tolist())

    target_names = [str(i) for i in range(num_classes)]
    report = classification_report(
        ys,
        ps,
        labels=list(range(num_classes)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    micro_f1 = float(f1_score(ys, ps, average="micro", zero_division=0))
    acc = float(np.mean(np.array(ys) == np.array(ps)))
    cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _plot_confusion(cm: List[List[int]], class_names: Optional[List[str]], out_path: Path, max_classes: int = 40) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cm_arr = np.array(cm, dtype=float)
    n = min(cm_arr.shape[0], max_classes)
    cm_arr = cm_arr[:n, :n]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if class_names:
        tick = class_names[:n]
    else:
        tick = [str(i) for i in range(n)]
    ax.set(xticks=np.arange(n), yticks=np.arange(n), xticklabels=tick, yticklabels=tick, ylabel="True", xlabel="Pred")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Confusion matrix (val, first N classes shown)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _benchmark_throughput(model, val_loader, device, max_batches: int = 30) -> float:
    import torch

    model.eval()
    n = 0
    if device.type != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            model(images)
            n += images.size(0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return n / max(dt, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="ANN report: TB curves + optional val F1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default=None, help="Directory with tfevents (e.g. runs_ann/kaggle)")
    parser.add_argument("--data_root", type=Path, default=None, help="PlantVillage ImageFolder root for val F1")
    parser.add_argument("--auto_find_subdir", action="store_true", help="Match Kaggle dataloader search")
    parser.add_argument("--max_epoch", type=int, default=15, help="Use epochs [0, max_epoch) from TB and naming")
    parser.add_argument("--output_dir", type=Path, default=Path("ann_report_out"))
    parser.add_argument("--benchmark", action="store_true", help="CUDA images/s throughput on val (proxy)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = _cfg_from_dict(cfg_dict) if isinstance(cfg_dict, dict) else None

    summary: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "stored_epoch": ckpt.get("epoch"),
        "stored_val_acc": ckpt.get("val_acc"),
        "epochs_used_in_plots": f"0..{args.max_epoch - 1}",
    }

    series: Dict[str, List[Tuple[int, float]]] = {}
    if args.logdir is not None:
        series = _load_scalars_from_tfevents(args.logdir, args.max_epoch)
        summary["tensorboard_tags"] = list(series.keys())
        if series:
            _plot_curves(series, args.output_dir / "curves_first_epochs.png", args.max_epoch)
            summary["plot"] = str(args.output_dir / "curves_first_epochs.png")
            # best val in window
            vals = [v for _, v in series.get("acc/val", [])]
            if vals:
                summary["best_val_acc_in_window"] = max(vals)
                summary["last_val_acc_in_window"] = vals[-1]

    if args.data_root is not None and cfg is not None:
        import torch as _torch

        dev = "cuda" if _torch.cuda.is_available() else "cpu"
        eval_out = _evaluate_val_f1(
            args.checkpoint,
            cfg,
            args.data_root,
            device=dev,
            auto_find_subdir=args.auto_find_subdir,
        )
        summary["val_eval"] = {
            "accuracy": eval_out["accuracy"],
            "macro_f1": eval_out["macro_f1"],
            "micro_f1": eval_out["micro_f1"],
            "weighted_f1": eval_out["weighted_f1"],
        }
        with open(args.output_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(eval_out["classification_report"], f, indent=2)
        _plot_confusion(
            eval_out["confusion_matrix"],
            None,
            args.output_dir / "confusion_matrix.png",
        )
        summary["confusion_matrix_plot"] = str(args.output_dir / "confusion_matrix.png")

        if args.benchmark and _torch.cuda.is_available():
            from data.kaggle_dataloader import get_kaggle_dataloaders as _gkd

            _, val_loader, _ = _gkd(
                str(args.data_root),
                image_size=cfg.image_size,
                batch_size=min(64, cfg.batch_size),
                num_workers=0,
                val_ratio=0.2,
                seed=cfg.seed,
                auto_find_subdir=args.auto_find_subdir,
            )
            model = _build_ann_for_checkpoint(ckpt, cfg)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(_torch.device("cuda"))
            ips = _benchmark_throughput(model, val_loader, _torch.device("cuda"))
            summary["throughput_images_per_s_cuda"] = ips
            summary["energy_note"] = (
                "ANN has no spike counts; use throughput as a rough inverse proxy for "
                "latency energy at fixed hardware, or plug in GPU power * time externally."
            )

    if cfg is not None:
        m = _build_ann_for_checkpoint(ckpt, cfg)
        summary["num_parameters"] = _param_count(m)
        summary["image_size"] = cfg.image_size

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
