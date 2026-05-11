"""
Train MSF-Res2Net on PlantVillage with first-spike coding and SpikingJelly multi-step mode.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp as torch_amp
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig, get_config
from data.dataloader import get_dataloaders
from models.res2net_msf import build_model
from utils.seed import set_seed
from utils.train_utils import SpikeCounter, cosine_lr, denormalize_to_01, first_spike_coding, reset_snn_state
from utils.visualizer import visualize_sample


def encode_batch(images: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
    """ImageNet-normalized batch -> [0,1] -> first-spike tensor on device."""
    x01 = denormalize_to_01(images)
    spikes = first_spike_coding(x01, T=T)
    return spikes.to(device)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    T: int,
    device: torch.device,
    amp_enabled: bool,
    spike_counter: Optional[SpikeCounter] = None,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    if spike_counter is not None:
        spike_counter.reset_buffer()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch_amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
            x = encode_batch(images, T, device)
            reset_snn_state(model)
            logits_t = model(x)
            logits = logits_t.mean(dim=0)
            pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    val_spike = 0.0
    if spike_counter is not None:
        spike_counter.finalize_epoch()
        val_spike = spike_counter.global_spike_rate
    return correct / max(total, 1), val_spike


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    scaler: torch_amp.GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    spike_counter: Optional[SpikeCounter],
) -> tuple[float, float, float]:
    model.train()
    if spike_counter is not None:
        spike_counter.reset_buffer()
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
            x = encode_batch(images, cfg.T, device)
            reset_snn_state(model)
            logits_t = model(x)
            logits = logits_t.mean(dim=0)
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

    train_spike = 0.0
    if spike_counter is not None:
        spike_counter.finalize_epoch()
        train_spike = spike_counter.global_spike_rate
    return running_loss / max(total, 1), correct / max(total, 1), train_spike


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PlantVillage MSF-Res2Net SNN")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="config.PRESETS key: no_msf | plif_rn18 | no_attention | ablation_baseline (see config.py)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_spike_counter", action="store_true")
    parser.add_argument("--viz", action="store_true", help="save spike/membrane figures each viz_every_n_epochs")
    parser.add_argument(
        "--msf_surrogate",
        type=str,
        default=None,
        help="MSF surrogate name: rect|sigmoid|arctan|gaussian (default from TrainConfig)",
    )
    parser.add_argument(
        "--msf_surrogate_alpha",
        type=float,
        default=None,
        help="Smooth surrogate width α for MSFNode (>0). Tune with --lr when using arctan/sigmoid.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.device is not None:
        overrides["device"] = args.device
    if args.no_amp:
        overrides["amp"] = False
    if args.msf_surrogate is not None:
        overrides["msf_surrogate"] = args.msf_surrogate
    if args.msf_surrogate_alpha is not None:
        overrides["msf_surrogate_alpha"] = args.msf_surrogate_alpha

    cfg = get_config(args.preset, **overrides)
    if args.no_spike_counter:
        cfg.spike_counter_enabled = False
    if args.viz:
        cfg.viz_enabled = True

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    cfg.num_classes = num_classes

    model = build_model(cfg).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scaler = torch_amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, args.run_name))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    spike_counter: Optional[SpikeCounter] = None
    if cfg.spike_counter_enabled:
        spike_counter = SpikeCounter(model, enabled=True)

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        lr = cosine_lr(epoch, cfg.lr, cfg.warmup_epochs, cfg.epochs, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_acc, train_spike = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, spike_counter
        )
        val_acc, val_spike = evaluate(model, test_loader, cfg.T, device, cfg.amp, spike_counter=spike_counter)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("lr", lr, epoch)
        if spike_counter is not None:
            writer.add_scalar("spike/train", train_spike, epoch)
            writer.add_scalar("spike/val", val_spike, epoch)
            writer.add_scalar("spike/train_global_rate", train_spike, epoch)
            writer.add_scalar("spike/val_global_rate", val_spike, epoch)
            for i, (k, v) in enumerate(sorted(spike_counter.per_layer_spike_rate.items())):
                if i >= 8:
                    break
                writer.add_scalar(f"spike/layer_val/{k}", v, epoch)

        print(
            f"Epoch {epoch+1}/{cfg.epochs}  lr={lr:.5f}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  val_acc={val_acc*100:.2f}%"
            + (f"  spikes(train/val)={train_spike:.4f}/{val_spike:.4f}" if spike_counter is not None else "")
        )

        if cfg.viz_enabled and (epoch + 1) % max(cfg.viz_every_n_epochs, 1) == 0:
            images, _ = next(iter(train_loader))
            images = images[:1].to(device)
            x = encode_batch(images, cfg.T, device)
            visualize_sample(model, x, os.path.join(cfg.viz_dir, args.run_name), tag=f"ep{epoch+1}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg.to_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, cfg.save_name))
            print(f"  saved best checkpoint ({val_acc*100:.2f}%)")

    writer.close()
    if spike_counter is not None:
        spike_counter.remove()
    print(f"Done. Best val acc={best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
