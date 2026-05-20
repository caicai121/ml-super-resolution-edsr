#!/usr/bin/env python3
"""Recompute x4 Bicubic with standard SR evaluation."""
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import calculate_metrics, calculate_metrics_standard

hr_dir = Path("data/test/HR")
results = []

for hr_path in sorted(hr_dir.glob("*.png"))[:5]:
    name = hr_path.name
    hr = np.array(Image.open(hr_path).convert("RGB"))
    hr_w, hr_h = hr.shape[1], hr.shape[0]

    # x4 downsample then bicubic upsample
    lr = Image.open(hr_path).convert("RGB").resize((hr_w // 4, hr_h // 4), Image.BICUBIC)
    sr = np.array(lr.resize((hr_w, hr_h), Image.BICUBIC))

    m_rgb = calculate_metrics(hr, sr)
    m_std = calculate_metrics_standard(hr, sr, scale=4)

    results.append({
        "image": name,
        "psnr_rgb": m_rgb["psnr"],
        "ssim_rgb": m_rgb["ssim"],
        "psnr_y": m_std["psnr"],
        "ssim_y": m_std["ssim"],
    })
    print(f"  {name}: RGB({m_rgb['psnr']:.2f}/{m_rgb['ssim']:.4f}) Y({m_std['psnr']:.2f}/{m_std['ssim']:.4f})")

with open("report_assets/tables/standard_metrics_x4_bicubic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y"])
    w.writeheader()
    w.writerows(results)

avg_y = np.mean([r["psnr_y"] for r in results])
avg_ssim_y = np.mean([r["ssim_y"] for r in results])
print(f"\nx4 Bicubic (Y+crop): PSNR={avg_y:.2f} dB, SSIM={avg_ssim_y:.4f}")
