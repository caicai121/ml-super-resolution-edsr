#!/usr/bin/env python3
"""
Run Bicubic baseline for super-resolution.
Upsamples LR_x2 images to HR size using Bicubic interpolation,
then computes PSNR/SSIM against ground-truth HR.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_metrics

# Paths
DATA_ROOT = Path("data")
TEST_HR = DATA_ROOT / "test" / "HR"
TEST_LR = DATA_ROOT / "test" / "LR_x2"
RESULTS_DIR = Path("results")
BICUBIC_DIR = RESULTS_DIR / "bicubic" / "images"
COMPARISON_DIR = RESULTS_DIR / "comparisons"
METRICS_CSV = RESULTS_DIR / "bicubic" / "metrics.csv"


def ensure_dirs():
    BICUBIC_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def bicubic_upsample(lr_path, sr_path, scale=2):
    """Upsample LR image to HR size using Bicubic."""
    img = Image.open(lr_path).convert("RGB")
    w, h = img.size
    sr_img = img.resize((w * scale, h * scale), Image.BICUBIC)
    sr_img.save(sr_path)


def create_comparison(lr_img, sr_img, hr_img, save_path):
    """Create side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(lr_img)
    axes[0].set_title("LR (x2 downsampled)")
    axes[0].axis("off")

    axes[1].imshow(sr_img)
    axes[1].set_title("Bicubic (x2 upsampled)")
    axes[1].axis("off")

    axes[2].imshow(hr_img)
    axes[2].set_title("HR (Ground Truth)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    # Get all test images
    test_images = sorted([f.name for f in TEST_LR.glob("*.png")])
    print(f"Found {len(test_images)} test images")

    results = []

    for i, img_name in enumerate(test_images):
        lr_path = TEST_LR / img_name
        hr_path = TEST_HR / img_name
        sr_path = BICUBIC_DIR / img_name

        # Bicubic upsample
        bicubic_upsample(lr_path, sr_path)

        # Load images for metrics
        hr = np.array(Image.open(hr_path).convert("RGB"))
        sr = np.array(Image.open(sr_path).convert("RGB"))
        lr = np.array(Image.open(lr_path).convert("RGB"))

        # Calculate metrics
        metrics = calculate_metrics(hr, sr)
        results.append(
            {"image": img_name, "psnr": metrics["psnr"], "ssim": metrics["ssim"]}
        )

        # Generate comparison for first 3 images
        if i < 3:
            comp_path = COMPARISON_DIR / f"compare_{img_name}"
            create_comparison(lr, sr, hr, comp_path)
            print(
                f"  [{i+1}/{len(test_images)}] {img_name} - "
                f"PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f} "
                f"[comparison saved]"
            )
        else:
            print(
                f"  [{i+1}/{len(test_images)}] {img_name} - "
                f"PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
            )

    # Save metrics CSV
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])

    print("\n" + "=" * 50)
    print("Bicubic Baseline Results")
    print("=" * 50)
    print(f"  Images: {len(results)}")
    print(f"  Avg PSNR: {avg_psnr:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim:.4f}")
    print(f"\n  Metrics saved to: {METRICS_CSV}")
    print(f"  Comparison images: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
