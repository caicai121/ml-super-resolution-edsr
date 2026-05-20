#!/usr/bin/env python3
"""Full Bicubic x4 evaluation on all 50 test images with standard SR metrics."""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_metrics, calculate_metrics_standard


def main():
    hr_dir = Path("data/test/HR")
    lr_dir = Path("data/test/LR_x4")
    results_dir = Path("results/bicubic_x4")
    images_dir = results_dir / "images"
    compare_dir = Path("results/comparisons_bicubic_x4")
    report_dir = Path("report_assets/tables")

    images_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Check LR_x4 exists
    lr_images = sorted(lr_dir.glob("*.png"))
    if not lr_images:
        print(f"ERROR: No LR_x4 images found in {lr_dir}")
        print("Run generate_lr.py first.")
        sys.exit(1)

    print(f"Found {len(lr_images)} LR_x4 images")
    print(f"HR dir: {hr_dir}")
    print(f"Output: {results_dir}")

    results = []

    for i, lr_path in enumerate(lr_images):
        name = lr_path.name
        hr_path = hr_dir / name
        if not hr_path.exists():
            print(f"  WARNING: HR not found for {name}, skipping")
            continue

        hr = np.array(Image.open(hr_path).convert("RGB"))
        lr = np.array(Image.open(lr_path).convert("RGB"))
        hr_h, hr_w = hr.shape[:2]

        # Bicubic upsample LR to HR size
        lr_pil = Image.open(lr_path).convert("RGB")
        sr_pil = lr_pil.resize((hr_w, hr_h), Image.BICUBIC)
        sr_pil.save(images_dir / name)

        sr = np.array(sr_pil)

        m_rgb = calculate_metrics(hr, sr)
        m_std = calculate_metrics_standard(hr, sr, scale=4)

        results.append({
            "image": name,
            "psnr_rgb": m_rgb["psnr"],
            "ssim_rgb": m_rgb["ssim"],
            "psnr_y": m_std["psnr"],
            "ssim_y": m_std["ssim"],
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(lr_images)}] {name}: "
                  f"RGB({m_rgb['psnr']:.2f}/{m_rgb['ssim']:.4f}) "
                  f"Y({m_std['psnr']:.2f}/{m_std['ssim']:.4f})")

    # Save metrics CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y"])
        writer.writeheader()
        writer.writerows(results)

    # Copy to report_assets
    report_csv = report_dir / "standard_metrics_bicubic_x4.csv"
    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_psnr_rgb = np.mean([r["psnr_rgb"] for r in results])
    avg_ssim_rgb = np.mean([r["ssim_rgb"] for r in results])
    avg_psnr_y = np.mean([r["psnr_y"] for r in results])
    avg_ssim_y = np.mean([r["ssim_y"] for r in results])

    summary_lines = [
        "Bicubic x4 Full Test Results",
        "=" * 50,
        f"Images: {len(results)}",
        f"RGB:    PSNR = {avg_psnr_rgb:.2f} dB, SSIM = {avg_ssim_rgb:.4f}",
        f"Y+crop: PSNR = {avg_psnr_y:.2f} dB, SSIM = {avg_ssim_y:.4f}",
    ]
    summary = "\n".join(summary_lines)

    print(f"\n{summary}")

    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nMetrics: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_csv}")


if __name__ == "__main__":
    main()
