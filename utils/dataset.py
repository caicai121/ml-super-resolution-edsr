#!/usr/bin/env python3
"""Dataset for super-resolution training, validation, and testing."""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """Super-Resolution dataset.

    Returns lr (raw LR), lr_up (bicubic-upsampled), hr, name.
    - SRCNN uses lr_up as input
    - EDSR uses lr as input
    """

    def __init__(self, hr_dir, lr_dir, scale=2, patch_size=None, split="train"):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.split = split

        # Match HR and LR images by filename
        hr_names = sorted([f.name for f in self.hr_dir.glob("*.png")])
        lr_names = sorted([f.name for f in self.lr_dir.glob("*.png")])
        self.image_names = sorted(set(hr_names) & set(lr_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        hr = Image.open(self.hr_dir / name).convert("RGB")
        lr = Image.open(self.lr_dir / name).convert("RGB")

        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        if self.split == "train" and self.patch_size is not None:
            # Random crop from HR, corresponding crop from LR
            lr_patch = self.patch_size // self.scale
            lx = random.randint(0, lr_w - lr_patch)
            ly = random.randint(0, lr_h - lr_patch)
            lr = lr.crop((lx, ly, lx + lr_patch, ly + lr_patch))
            hr = hr.crop((
                lx * self.scale, ly * self.scale,
                (lx + lr_patch) * self.scale, (ly + lr_patch) * self.scale
            ))

        # Bicubic upsample LR to HR size
        hr_w, hr_h = hr.size
        lr_up = lr.resize((hr_w, hr_h), Image.BICUBIC)

        # Convert to tensors [0, 1]
        lr_np = np.array(lr).astype(np.float32) / 255.0
        lr_up = np.array(lr_up).astype(np.float32) / 255.0
        hr = np.array(hr).astype(np.float32) / 255.0

        # HWC -> CHW
        lr_np = lr_np.transpose(2, 0, 1)
        lr_up = lr_up.transpose(2, 0, 1)
        hr = hr.transpose(2, 0, 1)

        return {"lr": lr_np, "lr_up": lr_up, "hr": hr, "name": name}
