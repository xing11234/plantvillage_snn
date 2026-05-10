"""
Build train/val DataLoaders from a **local folder** (Kaggle `/kaggle/input/...`).

Supports:
- **Single ImageFolder root** (each subfolder is a class): random 80/20 split by default.
- **Pre-split** layout: `.../train/<class>/` and `.../val/<class>/` or `.../test/<class>/`.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from data.dataloader import build_transforms


def _subdir_has_images(path: str) -> bool:
    for name in os.listdir(path):
        lower = name.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")):
            return True
    return False


def is_imagefolder_root(path: str) -> bool:
    """True if `path` has >=2 subdirectories that look like class folders with images."""
    if not os.path.isdir(path):
        return False
    ok = 0
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and _subdir_has_images(sub):
            ok += 1
    return ok >= 2


def find_imagefolder_under(base: str, max_depth: int = 5) -> Optional[str]:
    """
    Search under `base` for a directory that qualifies as an ImageFolder root.
    Prefers directory names like ``color`` / ``rgb``.
    """
    if not os.path.isdir(base):
        return None
    preferred_names = {"color", "rgb", "segmented", "grayscale"}
    candidates: List[Tuple[int, str]] = []

    base = os.path.abspath(base)
    for root, dirnames, _files in os.walk(base):
        depth = root[len(base) :].count(os.sep)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if is_imagefolder_root(root):
            score = len(os.listdir(root))
            bn = os.path.basename(root).lower()
            if bn in preferred_names:
                score += 10_000
            candidates.append((score, root))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


class _IndexedPathsDataset(Dataset):
    """Load by index into a master `samples` list with an explicit transform."""

    def __init__(self, samples: List[Tuple[str, int]], indices: List[int], transform):
        self.samples = samples
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real = self.indices[idx]
        path, label = self.samples[real]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def get_kaggle_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.2,
    seed: int = 42,
    auto_find_subdir: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Parameters
    ----------
    data_root
        Kaggle input root or direct ImageFolder root (folder whose children are class names).
    auto_find_subdir
        If True and `data_root` is not an ImageFolder root, search under it for a valid root.
    """
    root = os.path.abspath(data_root)
    if auto_find_subdir and not is_imagefolder_root(root):
        found = find_imagefolder_under(root)
        if found is None:
            raise FileNotFoundError(
                f"No ImageFolder-style directory found under: {root}\n"
                "Set `data_root` to the folder that **directly contains class subfolders** "
                "(e.g. .../color), or disable auto_find_subdir."
            )
        root = found

    train_tf = build_transforms(image_size, train=True)
    eval_tf = build_transforms(image_size, train=False)

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    if os.path.isdir(train_dir) and (os.path.isdir(val_dir) or os.path.isdir(test_dir)):
        vdir = val_dir if os.path.isdir(val_dir) else test_dir
        train_ds = ImageFolder(train_dir, transform=train_tf)
        val_ds = ImageFolder(vdir, transform=eval_tf)
        if train_ds.classes != val_ds.classes:
            raise ValueError("train/val class folder names differ; check dataset layout.")
        num_classes = len(train_ds.classes)
    else:
        full = ImageFolder(root, transform=None)
        n_total = len(full.samples)
        n_val = max(int(round(n_total * val_ratio)), 1)
        n_train = max(n_total - n_val, 1)
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_ds = _IndexedPathsDataset(full.samples, train_idx, train_tf)
        val_ds = _IndexedPathsDataset(full.samples, val_idx, eval_tf)
        num_classes = len(full.classes)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, num_classes
