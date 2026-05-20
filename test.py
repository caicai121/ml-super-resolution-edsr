#!/usr/bin/env python3
"""Testing script for SRCNN."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.srcnn import SRCNN
from utils.dataset import SRDataset
from utils.metrics import calculate_metrics
from utils.image_utils import tensor_to_np, save_image
from utils.plot_utils import create_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/srcnn.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scale = cfg["scale"]
    channels = cfg["channels"]
    eval_dir = Path(cfg["eval"]["save_dir"])
    compare_dir = Path(cfg["eval"]["compare_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = Path(cfg["train"]["save_dir"]) / "best_srcnn.pth"

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Model
    model = SRCNN(channels=channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint from epoch {checkpoint['epoch']}, "
          f"Val PSNR: {checkpoint['psnr']:.2f}, Val SSIM: {checkpoint['ssim']:.4f}")

    # Test dataset
    test_set = SRDataset(
        cfg["data"]["test_hr"], cfg["data"]["test_lr"],
        scale=scale, split="test"
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    print(f"Test images: {len(test_set)}")

    # Evaluate
    results = []
    compare_count = 0

    with torch.no_grad():
        for batch in test_loader:
            lr_up = batch["lr_up"].to(device)
            hr = batch["hr"].to(device)
            name = batch["name"][0]

            sr = model(lr_up)

            sr_np = tensor_to_np(sr[0])
            hr_np = tensor_to_np(hr[0])
            lr_up_np = tensor_to_np(lr_up[0])

            metrics = calculate_metrics(hr_np, sr_np)
            results.append({"image": name, "psnr": metrics["psnr"], "ssim": metrics["ssim"]})

            # Generate comparison images (first 3)
            if compare_count < 3:
                # Load original LR for comparison
                lr_path = Path(cfg["data"]["test_lr"]) / name
                lr_img = np.array(Image.open(lr_path).convert("RGB"))

                # Bicubic upsample for comparison
                lr_pil = Image.open(lr_path).convert("RGB")
                hr_w, hr_h = hr_np.shape[1], hr_np.shape[0]
                bicubic_np = np.array(lr_pil.resize((hr_w, hr_h), Image.BICUBIC))

                comp_path = compare_dir / f"compare_srcnn_{name}"
                create_comparison(lr_img, bicubic_np, sr_np, hr_np, comp_path,
                                  title=f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                compare_count += 1

            print(f"  {name} - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")

    # Save metrics CSV
    csv_path = eval_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])

    summary = (
        "SRCNN Test Results\n"
        "=" * 50 + "\n"
        f"Images: {len(results)}\n"
        f"Avg PSNR: {avg_psnr:.2f} dB\n"
        f"Avg SSIM: {avg_ssim:.4f}\n"
        f"\nBicubic Baseline:\n"
        f"Avg PSNR: 30.90 dB\n"
        f"Avg SSIM: 0.8975\n"
        f"\nSRCNN vs Bicubic:\n"
        f"PSNR {'improved' if avg_psnr > 30.90 else 'not improved'} "
        f"({avg_psnr - 30.90:+.2f} dB)\n"
        f"SSIM {'improved' if avg_ssim > 0.8975 else 'not improved'} "
        f"({avg_ssim - 0.8975:+.4f})\n"
    )

    print(f"\n{summary}")

    summary_path = eval_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"Metrics saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Comparison images: {compare_dir}")


if __name__ == "__main__":
    main()
