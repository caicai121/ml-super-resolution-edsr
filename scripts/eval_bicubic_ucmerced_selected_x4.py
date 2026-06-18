#!/usr/bin/env python3
"""Bicubic x4 evaluation on UC Merced selected test 70 images with standard SR metrics.

Output: results/bicubic_ucmerced_selected_x4/
- metrics.csv (per-image RGB + Y+crop PSNR/SSIM)
- summary.txt (average metrics)
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_metrics, calculate_metrics_standard


def main():
    hr_dir = Path("data_final/ucmerced_selected/test/HR")
    lr_dir = Path("data_final/ucmerced_selected/test/LR_x4")
    results_dir = Path("results/bicubic_ucmerced_selected_x4")
    images_dir = results_dir / "images"
    compare_dir = results_dir / "comparisons"

    for d in [results_dir, images_dir, compare_dir]:
        d.mkdir(parents=True, exist_ok=True)

    lr_images = sorted(lr_dir.glob("*.png"))
    if not lr_images:
        print(f"ERROR: No LR_x4 images found in {lr_dir}")
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
        hr_h, hr_w = hr.shape[:2]

        # Bicubic upsample LR to HR size
        lr_pil = Image.open(lr_path).convert("RGB")
        sr_pil = lr_pil.resize((hr_w, hr_h), Image.BICUBIC)
        sr_pil.save(images_dir / name)

        sr = np.array(sr_pil)

        # RGB metrics (full image, no crop)
        m_rgb = calculate_metrics(hr, sr)

        # Y+crop metrics (standard SR evaluation, crop_border=scale)
        m_std = calculate_metrics_standard(hr, sr, scale=4)

        results.append({
            "image": name,
            "rgb_psnr": round(m_rgb["psnr"], 4),
            "rgb_ssim": round(m_rgb["ssim"], 4),
            "y_psnr": round(m_std["psnr"], 4),
            "y_ssim": round(m_std["ssim"], 4),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(lr_images)}] {name}: "
                  f"RGB({m_rgb['psnr']:.2f}/{m_rgb['ssim']:.4f}) "
                  f"Y+crop({m_std['psnr']:.2f}/{m_std['ssim']:.4f})")

    # Save metrics CSV (use consistent column names with test.py)
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "rgb_psnr", "rgb_ssim", "y_psnr", "y_ssim"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_psnr_rgb = np.mean([r["rgb_psnr"] for r in results])
    avg_ssim_rgb = np.mean([r["rgb_ssim"] for r in results])
    avg_psnr_y = np.mean([r["y_psnr"] for r in results])
    avg_ssim_y = np.mean([r["y_ssim"] for r in results])

    summary = "\n".join([
        "Bicubic x4 Test Results (UC Merced selected 70)",
        "=" * 50,
        f"Images: {len(results)}",
        f"RGB PSNR:  {avg_psnr_rgb:.2f} dB | RGB SSIM:  {avg_ssim_rgb:.4f}",
        f"Y+crop PSNR: {avg_psnr_y:.2f} dB | Y+crop SSIM: {avg_ssim_y:.4f}",
    ])

    print(f"\n{summary}")

    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    print(f"\nMetrics CSV: {csv_path}")
    print(f"Summary:     {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
