"""
Train Res2Net-ANN on PlantVillage-style loaders (standard [B,C,H,W] images, no SNN encoding).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import amp as torch_amp

from config import TrainConfig


@torch.no_grad()
def evaluate_ann(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch_amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def train_one_epoch_ann(
    model: torch.nn.Module,
    loader,
    optimizer,
    scaler: torch_amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    log_every = int(getattr(cfg, "batch_log_interval", 0) or 0)
    n_batches = len(loader)
    if log_every > 0:
        print(f"  training: {n_batches} batches (log every {log_every})", flush=True)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch_amp.autocast(device_type="cuda", enabled=cfg.amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=float(cfg.label_smoothing))
        scaler.scale(loss).backward()
        if cfg.grad_clip_max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().cpu().item()) * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            print(
                f"    batch {batch_idx + 1}/{n_batches}  "
                f"loss={running_loss / max(total, 1):.4f}  acc={100.0 * correct / max(total, 1):.2f}%",
                flush=True,
            )

    return running_loss / max(total, 1), correct / max(total, 1)
