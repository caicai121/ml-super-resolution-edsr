#!/usr/bin/env python3
"""RCAN-pretrained x4 evaluation on UC Merced selected test 70 images.

Uses the official yulunzhang/RCAN pretrained weights with:
- [0, 255] float input
- DIV2K RGB mean subtraction (R=114.44, G=111.46, B=103.02)
- Standard Y+crop (crop=4) evaluation

Output: results/rcan_pretrained_ucmerced_selected_x4/
- metrics.csv (per-image RGB + Y+crop PSNR/SSIM)
- summary.txt (average metrics)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rcan import RCAN, load_pretrained_rcan
from utils.metrics import calculate_metrics, calculate_metrics_standard

# DIV2K RGB channel means (computed from training set)
# NOTE: These must match the pretrained model's training pipeline
RGB_MEAN = [114.444, 111.4605, 103.02]  # R, G, B


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    ckpt_path = Path("checkpoints/rcan_x4/pretrained_rcan_x4.pth")
    hr_dir = Path("data_final/ucmerced_selected/test/HR")
    lr_dir = Path("data_final/ucmerced_selected/test/LR_x4")
    results_dir = Path("results/rcan_pretrained_ucmerced_selected_x4")
    images_dir = results_dir / "images"
    compare_dir = results_dir / "comparisons"

    for d in [results_dir, images_dir, compare_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build model (full RCAN: 10 groups, 20 blocks, 64 features)
    model = RCAN(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_resgroups=10,
        num_resblocks=20,
        reduction=16,
        scale=4,
    )
    model = load_pretrained_rcan(ckpt_path, model, device)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Precompute mean tensor for input preprocessing
    mean = torch.tensor(RGB_MEAN, device=device).view(1, 3, 1, 1)

    lr_images = sorted(lr_dir.glob("*.png"))
    print(f"\nEvaluating {len(lr_images)} test images...")

    results = []
    with torch.no_grad():
        for i, lr_path in enumerate(lr_images):
            name = lr_path.name
            hr_path = hr_dir / name
            if not hr_path.exists():
                print(f"  WARNING: HR not found for {name}, skipping")
                continue

            hr = np.array(Image.open(hr_path).convert("RGB"))

            # Preprocess: [0, 255] float, subtract RGB channel means
            lr_np = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32)
            inp = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).to(device)
            inp = inp - mean

            # Forward pass
            sr = model(inp)

            # Postprocess: add back channel means, clamp to [0, 255]
            sr = sr + mean
            sr_np = sr[0].cpu().numpy().clip(0, 255).transpose(1, 2, 0).astype(np.uint8)

            # Save SR image
            Image.fromarray(sr_np).save(images_dir / name)

            # Compute metrics
            m_rgb = calculate_metrics(hr, sr_np)
            m_std = calculate_metrics_standard(hr, sr_np, scale=4)

            results.append({
                "image": name,
                "rgb_psnr": round(m_rgb["psnr"], 4),
                "rgb_ssim": round(m_rgb["ssim"], 4),
                "y_psnr": round(m_std["psnr"], 4),
                "y_ssim": round(m_std["ssim"], 4),
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(lr_images)}] {name}: "
                      f"RGB({m_rgb['psnr']:.2f}) Y+crop({m_std['psnr']:.2f})")

    # Save metrics CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "rgb_psnr", "rgb_ssim", "y_psnr", "y_ssim"])
        writer.writeheader()
        writer.writerows(results)

    # Compute averages
    avg_rgb = np.mean([r["rgb_psnr"] for r in results])
    avg_ssim_rgb = np.mean([r["rgb_ssim"] for r in results])
    avg_y = np.mean([r["y_psnr"] for r in results])
    avg_ssim_y = np.mean([r["y_ssim"] for r in results])

    # Summary
    summary = "\n".join([
        "RCAN-pretrained x4 Test Results (UC Merced selected 70)",
        "=" * 55,
        f"Images: {len(results)}",
        f"RGB PSNR:  {avg_rgb:.2f} dB | RGB SSIM:  {avg_ssim_rgb:.4f}",
        f"Y+crop PSNR: {avg_y:.2f} dB | Y+crop SSIM: {avg_ssim_y:.4f}",
        "",
        "Input preprocessing: [0,255] float - RGB mean [114.44, 111.46, 103.02]",
        f"Model params: {n_params:,}",
        "",
        f"30dB check (Y+crop): {'>= 30 dB! PASS' if avg_y >= 30.0 else 'NO - need %.2f more dB' % (30.0 - avg_y)}",
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
