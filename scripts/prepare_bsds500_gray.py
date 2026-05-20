#!/usr/bin/env python3
"""Convert BSDS500 images to 3-channel grayscale for RCAN compatibility."""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_size", type=int, default=128)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in raw_dir.rglob("*") if f.suffix.lower() in exts)
    print(f"Found {len(files)} images in {raw_dir}")

    count = 0
    for f in files:
        try:
            img = Image.open(f).convert("L")
            if min(img.size) < args.min_size:
                continue
            # Convert grayscale to 3-channel RGB (R=G=B) for RCAN compatibility
            gray = np.array(img)
            rgb = np.stack([gray, gray, gray], axis=-1)
            out_name = f"bsds_{count + 1:04d}.png"
            Image.fromarray(rgb).save(out_dir / out_name)
            count += 1
        except Exception as e:
            print(f"  Skip {f.name}: {e}")

    print(f"Converted {count} images to {out_dir}")


if __name__ == "__main__":
    main()
