"""
PlantVillage dataset via Hugging Face `datasets`.

Uses the official `mohanty/PlantVillage` card with leaf-safe train/test split.
See: https://huggingface.co/datasets/mohanty/PlantVillage
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    load_dataset = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class PlantVillageHFDataset(Dataset):
    """Wraps HF `Dataset` dict rows into (tensor[C,H,W], label)."""

    def __init__(self, hf_split, transform: Optional[Callable] = None):
        if _IMPORT_ERR is not None:
            raise ImportError("Please `pip install datasets`") from _IMPORT_ERR
        self.ds = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.ds[idx]
        img = row["image"]
        # HF may return RGB PIL; ensure RGB
        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def discover_num_classes(hf_train) -> int:
    """Infer class count from HF features when available."""
    feats = hf_train.features
    if "label" in feats and hasattr(feats["label"], "num_classes"):
        n = feats["label"].num_classes
        if n is not None:
            return int(n)
    # Fallback: scan labels (slower, only if needed)
    labels = hf_train["label"]
    return int(max(labels)) + 1


def get_dataloaders(
    dataset_name: str = "mohanty/PlantVillage",
    dataset_config: str = "color",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    if load_dataset is None:
        raise ImportError("Please install Hugging Face datasets: pip install datasets")

    dset = load_dataset(dataset_name, dataset_config)
    if "train" not in dset or "test" not in dset:
        raise KeyError("Expected splits 'train' and 'test' from the HF dataset card.")

    train_tf = build_transforms(image_size, train=True)
    eval_tf = build_transforms(image_size, train=False)

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = PlantVillageHFDataset(dset["train"], transform=train_tf)
    test_ds = PlantVillageHFDataset(dset["test"], transform=eval_tf)
    num_classes = discover_num_classes(dset["train"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, num_classes
