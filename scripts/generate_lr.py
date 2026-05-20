#!/usr/bin/env python3
"""Generate LR images by Bicubic downsampling from HR images."""

import argparse
from pathlib import Path
from PIL import Image


def generate_lr(hr_dir, lr_dir, scale):
    """Downsample HR images to LR using Bicubic interpolation."""
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    lr_dir.mkdir(parents=True, exist_ok=True)

    hr_images = sorted(hr_dir.glob("*.png"))
    print(f"Found {len(hr_images)} HR images in {hr_dir}")
    print(f"Scale: x{scale}")
    print(f"Output: {lr_dir}")

    for i, hr_path in enumerate(hr_images):
        hr_img = Image.open(hr_path).convert("RGB")
        w, h = hr_img.size
        lr_img = hr_img.resize((w // scale, h // scale), Image.BICUBIC)
        lr_img.save(lr_dir / hr_path.name)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(hr_images)}] {hr_path.name}: {w}x{h} -> {w//scale}x{h//scale}")

    print(f"Done. Generated {len(hr_images)} LR images.")


def main():
    parser = argparse.ArgumentParser(description="Generate LR images via Bicubic downsampling")
    parser.add_argument("--hr_dir", type=str, required=True, help="HR images directory")
    parser.add_argument("--lr_dir", type=str, required=True, help="Output LR directory")
    parser.add_argument("--scale", type=int, required=True, help="Downsampling scale factor")
    args = parser.parse_args()

    generate_lr(args.hr_dir, args.lr_dir, args.scale)


if __name__ == "__main__":
    main()
