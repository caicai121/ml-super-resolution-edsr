#!/usr/bin/env python3
"""Split selected UC Merced classes into train/val/test with LR_x4 generation.

Usage:
    python scripts/split_selected.py --classes airplane baseballdiamond golfcourse runway tenniscourt
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def split_class_images(class_name, hr_source_dir, out_dirs, seed=42):
    """Split 100 images of one class into 80/10/10."""
    rng = np.random.RandomState(seed)

    class_images = sorted(hr_source_dir.glob(f"{class_name}_*.png"))
    if len(class_images) == 0:
        print(f"  WARNING: No images found for class {class_name}")
        return 0

    indices = rng.permutation(len(class_images))
    n_train = int(len(class_images) * 0.8)
    n_val = int(len(class_images) * 0.1)

    splits = {
        "train": [class_images[i] for i in indices[:n_train]],
        "val": [class_images[i] for i in indices[n_train:n_train + n_val]],
        "test": [class_images[i] for i in indices[n_train + n_val:]],
    }

    count = 0
    for split_name, img_list in splits.items():
        hr_dir = out_dirs[split_name]["hr"]
        for src in img_list:
            shutil.copy2(src, hr_dir / src.name)
            count += 1

    return count


def generate_lr(hr_dir, lr_dir, scale=4):
    """Generate LR images from HR using bicubic downsampling."""
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    lr_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(hr_dir.glob("*.png"))
    for img_path in images:
        hr = Image.open(img_path).convert("RGB")
        w, h = hr.size
        lr = hr.resize((w // scale, h // scale), Image.BICUBIC)
        lr.save(lr_dir / img_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+",
                        default=["airplane", "baseballdiamond", "golfcourse",
                                 "runway", "tenniscourt"])
    parser.add_argument("--hr_source", type=str,
                        default="data_experiments/ucmerced_all_classes/HR")
    parser.add_argument("--output_root", type=str,
                        default="data_final/ucmerced_selected")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hr_source = Path(args.hr_source)
    out_root = Path(args.output_root)

    out_dirs = {}
    for split in ["train", "val", "test"]:
        hr_dir = out_root / split / "HR"
        lr_dir = out_root / split / "LR_x4"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[split] = {"hr": hr_dir, "lr": lr_dir}

    print(f"Splitting {len(args.classes)} classes into train/val/test...")
    total = {"train": 0, "val": 0, "test": 0}

    for cls in args.classes:
        counts = {}
        for split in ["train", "val", "test"]:
            # We need to split across all splits at once
            pass

        # Do the split
        class_images = sorted(hr_source.glob(f"{cls}_*.png"))
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(len(class_images))
        n_train = int(len(class_images) * 0.8)
        n_val = int(len(class_images) * 0.1)

        split_imgs = {
            "train": [class_images[i] for i in indices[:n_train]],
            "val": [class_images[i] for i in indices[n_train:n_train + n_val]],
            "test": [class_images[i] for i in indices[n_train + n_val:]],
        }

        for split_name, img_list in split_imgs.items():
            hr_dir = out_dirs[split_name]["hr"]
            for src in img_list:
                shutil.copy2(src, hr_dir / src.name)
            total[split_name] += len(img_list)

        print(f"  {cls}: train={len(split_imgs['train'])}, "
              f"val={len(split_imgs['val'])}, test={len(split_imgs['test'])}")

    print(f"\nTotal: train={total['train']}, val={total['val']}, test={total['test']}")

    # Generate LR_x4
    print(f"\nGenerating LR_x4 (scale={args.scale})...")
    for split in ["train", "val", "test"]:
        hr_dir = out_dirs[split]["hr"]
        lr_dir = out_dirs[split]["lr"]
        generate_lr(hr_dir, lr_dir, args.scale)
        n_hr = len(list(hr_dir.glob("*.png")))
        n_lr = len(list(lr_dir.glob("*.png")))
        print(f"  {split}: HR={n_hr}, LR={n_lr}")

    print("\nDone!")


if __name__ == "__main__":
    main()
