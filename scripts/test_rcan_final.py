#!/usr/bin/env python3
"""RCAN x4 inference - final version with correct input preprocessing."""
import sys
sys.path.insert(0, "/root/Code/ml-super-resolution-edsr")
from pathlib import Path
import torch, csv
import numpy as np
from PIL import Image
from models.rcan import RCAN, load_pretrained_rcan
from utils.metrics import calculate_metrics, calculate_metrics_standard

# DIV2K RGB channel means (computed from training set)
RGB_MEAN = [114.444, 111.4605, 103.02]  # R, G, B means to subtract

def main():
    ckpt_path = Path("/root/Code/ml-super-resolution-edsr/checkpoints/rcan_x4/pretrained_rcan_x4.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RCAN(num_features=64, num_resgroups=10, num_resblocks=20, reduction=16, scale=4)
    model = load_pretrained_rcan(ckpt_path, model, device)
    model = model.to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    hr_dir = Path("/root/Code/ml-super-resolution-edsr/data/test/HR")
    lr_dir = Path("/root/Code/ml-super-resolution-edsr/data/test/LR_x4")
    results_dir = Path("/root/Code/ml-super-resolution-edsr/results/rcan_x4")
    compare_dir = Path("/root/Code/ml-super-resolution-edsr/results/comparisons_rcan_x4")
    report_dir = Path("/root/Code/ml-super-resolution-edsr/report_assets/tables")
    images_dir = results_dir / "images"
    for d in [results_dir, compare_dir, report_dir, images_dir]:
        d.mkdir(parents=True, exist_ok=True)

    lr_images = sorted(lr_dir.glob("*.png"))
    print(f"\nEvaluating {len(lr_images)} test images...")

    # Precompute mean tensor
    mean = torch.tensor(RGB_MEAN, device=device).view(1, 3, 1, 1)

    results = []
    with torch.no_grad():
        for i, lr_path in enumerate(lr_images):
            name = lr_path.name
            hr = np.array(Image.open(hr_dir / name).convert("RGB"))

            # Input: [0, 255] float, subtract RGB mean
            lr_np = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32)
            inp = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).to(device)
            inp = inp - mean  # sub_mean: subtract channel means

            sr = model(inp)

            # Output: add back channel means, clamp to [0, 255]
            sr = sr + mean
            sr_np = sr[0].cpu().numpy().clip(0, 255).transpose(1, 2, 0).astype(np.uint8)

            Image.fromarray(sr_np).save(images_dir / name)
            m_rgb = calculate_metrics(hr, sr_np)
            m_std = calculate_metrics_standard(hr, sr_np, scale=4)
            results.append({"image": name, "psnr_rgb": m_rgb["psnr"], "ssim_rgb": m_rgb["ssim"],
                           "psnr_y": m_std["psnr"], "ssim_y": m_std["ssim"]})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(lr_images)}] {name}: RGB({m_rgb['psnr']:.2f}) Y({m_std['psnr']:.2f})")

    # Save metrics
    for p in [results_dir / "metrics.csv", report_dir / "standard_metrics_rcan_x4.csv"]:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image","psnr_rgb","ssim_rgb","psnr_y","ssim_y"])
            w.writeheader()
            w.writerows(results)

    avg_rgb = np.mean([r["psnr_rgb"] for r in results])
    avg_ssim_rgb = np.mean([r["ssim_rgb"] for r in results])
    avg_y = np.mean([r["psnr_y"] for r in results])
    avg_ssim_y = np.mean([r["ssim_y"] for r in results])

    summary = "\n".join([
        "RCAN x4 Test Results (pretrained, RGB mean subtraction)", "=" * 55,
        f"Images: {len(results)}",
        f"RGB:    PSNR = {avg_rgb:.2f} dB, SSIM = {avg_ssim_rgb:.4f}",
        f"Y+crop: PSNR = {avg_y:.2f} dB, SSIM = {avg_ssim_y:.4f}",
        "",
        "Comparison (Y+crop):",
        f"  Bicubic x4:  PSNR = 28.04 dB, SSIM = 0.7773",
        f"  RCAN x4:     PSNR = {avg_y:.2f} dB, SSIM = {avg_ssim_y:.4f}",
        f"  vs Bicubic:  PSNR {avg_y - 28.04:+.2f} dB",
        "",
        "30dB check (Y+crop): " + (">= 30 dB! PASS" if avg_y >= 30.0 else f"NO - need {30.0 - avg_y:.2f} more dB"),
    ])
    print(f"\n{summary}")
    with open(results_dir / "summary.txt", "w") as f:
        f.write(summary)

    # Save comparison images
    from utils.plot_utils import create_comparison
    for i in range(min(3, len(lr_images))):
        name = lr_images[i].name
        lr_img = np.array(Image.open(lr_images[i]).convert("RGB"))
        hr_img = np.array(Image.open(hr_dir / name).convert("RGB"))
        sr_img = np.array(Image.open(images_dir / name).convert("RGB"))
        hr_h, hr_w = hr_img.shape[:2]
        bicubic_img = np.array(Image.fromarray(lr_img).resize((hr_w, hr_h), Image.BICUBIC))
        create_comparison(lr_img, bicubic_img, sr_img, hr_img,
                         compare_dir / f"compare_rcan_x4_{name}",
                         title=f"RCAN x4 Y+crop={results[i]['psnr_y']:.2f}dB")

if __name__ == "__main__":
    main()
