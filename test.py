"""
Evaluate a saved checkpoint on the PlantVillage test split.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import amp as torch_amp

from config import TrainConfig
from data.dataloader import get_dataloaders
from models.res2net_msf import build_model
from utils.train_utils import SpikeCounter, denormalize_to_01, first_spike_coding, reset_snn_state


def encode_batch(images: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
    x01 = denormalize_to_01(images)
    spikes = first_spike_coding(x01, T=T)
    return spikes.to(device)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {}) or {}
    valid_keys = {f.name for f in fields(TrainConfig)}
    cfg = replace(TrainConfig(), **{k: v for k, v in cfg_dict.items() if k in valid_keys})
    if args.device:
        cfg.device = args.device
    cfg.batch_size = args.batch_size
    if args.no_amp:
        cfg.amp = False

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _, test_loader, num_classes = get_dataloaders(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    cfg.num_classes = num_classes

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    spike_counter = SpikeCounter(model, enabled=True)
    spike_counter.reset_buffer()

    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch_amp.autocast(device_type="cuda", enabled=cfg.amp and device.type == "cuda"):
            x = encode_batch(images, cfg.T, device)
            reset_snn_state(model)
            logits_t = model(x)
            logits = logits_t.mean(dim=0)
            pred = logits.argmax(dim=1)
        all_preds.extend(pred.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    spike_counter.finalize_epoch()
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"Global mean spike rate: {spike_counter.global_spike_rate:.6f}")
    target_names = [str(i) for i in range(cfg.num_classes)]
    print("\nClassification report (macro avg):")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(cfg.num_classes)),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    spike_counter.remove()


if __name__ == "__main__":
    main()
