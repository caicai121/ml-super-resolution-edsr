#!/usr/bin/env python3
"""
Scale Visual Comparison Demo (generalized).

Generate LR images from test/HR at arbitrary scale, upsample back with Bicubic,
and produce side-by-side comparison figures: LR / Bicubic / HR.

Supports --scale 4, --scale 8, etc.
This is a visual-only experiment for the report, not a training run.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_metrics


def ensure_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def bicubic_downsample(hr_path, lr_path, scale):
    img = Image.open(hr_path).convert("RGB")
    w, h = img.size
    lr_img = img.resize((w // scale, h // scale), Image.BICUBIC)
    lr_img.save(lr_path)
    return lr_img


def bicubic_upsample(lr_img, target_size):
    return lr_img.resize(target_size, Image.BICUBIC)


def create_comparison(lr_display, bicubic_img, hr_img, title_lr, title_bicubic, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(lr_display)
    axes[0].set_title(title_lr, fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(bicubic_img)
    axes[1].set_title(title_bicubic, fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(hr_img)
    axes[2].set_title("HR (Ground Truth)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Scale visual comparison demo")
    parser.add_argument("--hr_dir", type=str, default="data/test/HR")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir for comparisons (default: results/comparisons_x{scale}_visual)")
    parser.add_argument("--lr_dir", type=str, default=None,
                        help="Output dir for LR images (default: results/x{scale}_visual)")
    parser.add_argument("--report_dir", type=str, default=None,
                        help="Report figures dir (default: report_assets/figures/x{scale}_visual)")
    parser.add_argument("--metrics_dir", type=str, default=None,
                        help="Report tables dir (default: report_assets/tables/x{scale}_visual)")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--scale", type=int, default=8, choices=[2, 3, 4, 8])
    args = parser.parse_args()

    scale = args.scale
    tag = f"x{scale}"

    hr_dir = Path(args.hr_dir)
    out_dir = Path(args.out_dir or f"results/comparisons_{tag}_visual")
    lr_dir = Path(args.lr_dir or f"results/{tag}_visual")
    report_dir = Path(args.report_dir or f"report_assets/figures/{tag}_visual")
    metrics_dir = Path(args.metrics_dir or f"report_assets/tables/{tag}_visual")

    ensure_dirs(out_dir, lr_dir, report_dir, metrics_dir)

    # Select first N test images
    hr_images = sorted(hr_dir.glob("*.png"))[: args.num_images]
    print(f"Selected {len(hr_images)} images from {hr_dir} (scale={scale})")

    results = []

    for i, hr_path in enumerate(hr_images):
        name = hr_path.stem
        hr_img = Image.open(hr_path).convert("RGB")
        hr_w, hr_h = hr_img.size

        # Generate LR
        lr_path = lr_dir / f"lr_{tag}_{name}.png"
        lr_img = bicubic_downsample(hr_path, lr_path, scale=scale)
        lr_w, lr_h = lr_img.size

        # Bicubic upsample back to HR size
        bicubic_img = bicubic_upsample(lr_img, (hr_w, hr_h))

        # For display: upscale LR to same size so columns are equal (Nearest to show pixels)
        lr_display = lr_img.resize((hr_w, hr_h), Image.NEAREST)

        # Calculate metrics
        hr_arr = np.array(hr_img)
        bicubic_arr = np.array(bicubic_img)
        metrics = calculate_metrics(hr_arr, bicubic_arr)

        # Save comparison
        comp_name = f"compare_{tag}_{name}.png"
        comp_path = out_dir / comp_name
        create_comparison(
            lr_display,
            bicubic_arr,
            hr_arr,
            f"LR_x{scale} ({lr_w}x{lr_h})",
            f"Bicubic_x{scale} (upsampled)",
            comp_path,
        )

        # Copy to report assets
        shutil.copy2(comp_path, report_dir / comp_name)

        results.append({"image": name, "psnr": metrics["psnr"], "ssim": metrics["ssim"]})
        print(
            f"  [{i+1}/{len(hr_images)}] {name} - "
            f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}"
        )

    # Save metrics CSV
    metrics_csv = metrics_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])

    print("\n" + "=" * 50)
    print(f"x{scale} Visual Comparison - Bicubic_x{scale} Metrics")
    print("=" * 50)
    print(f"  Images: {len(results)}")
    print(f"  Avg PSNR: {avg_psnr:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim:.4f}")
    print(f"\n  Comparisons: {out_dir}")
    print(f"  Report figures: {report_dir}")
    print(f"  Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
