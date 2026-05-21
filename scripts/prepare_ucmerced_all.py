#!/usr/bin/env python3
"""Prepare UC Merced Land Use all 21 classes for super-resolution experiments.

Reads from raw directory (class subdirectories with TIFF images),
converts to PNG, and saves to HR directory with class name in filename.
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str,
                        default="data_experiments/ucmerced_all_classes/raw/Images",
                        help="Path to raw UC Merced Images directory")
    parser.add_argument("--hr_dir", type=str,
                        default="data_experiments/ucmerced_all_classes/HR",
                        help="Output directory for HR PNG images")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    hr_dir = Path(args.hr_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes in {raw_dir}")

    total = 0
    class_stats = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = sorted(class_dir.glob("*.tif")) + sorted(class_dir.glob("*.tiff"))
        if not images:
            images = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.bmp"))

        count = 0
        for img_path in images:
            out_name = f"{class_name}_{img_path.stem}.png"
            out_path = hr_dir / out_name

            img = Image.open(img_path).convert("RGB")
            # Ensure consistent 256x256 size (some TIFFs vary slightly)
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.BICUBIC)
            img.save(out_path, "PNG")
            count += 1

        class_stats.append((class_name, count))
        total += count
        print(f"  {class_name}: {count} images")

    print(f"\nTotal: {total} images")
    if total != 2100:
        print(f"WARNING: Expected 2100, got {total}")

    # Save stats
    stats_path = hr_dir.parent / "class_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"UC Merced Land Use - All Classes\n")
        f.write(f"=" * 50 + "\n")
        for class_name, count in class_stats:
            f.write(f"{class_name}: {count}\n")
        f.write(f"\nTotal: {total}\n")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
