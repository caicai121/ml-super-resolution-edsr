#!/usr/bin/env python3
"""Test RCAN pretrained on all 21 UC Merced classes with per-class stats.

Generates per-class metrics, ranking, and comparison figures.

Usage:
    python scripts/test_rcan_all_classes.py --config configs/rcan_x4_ucmerced_all_classes.yaml \
        --checkpoint checkpoints/rcan_x4/pretrained_rcan_x4.pth
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rcan import RCAN, load_pretrained_rcan
from utils.dataset import SRDataset
from utils.metrics import calculate_metrics, calculate_metrics_standard
from utils.image_utils import tensor_to_np
from utils.plot_utils import create_comparison


def get_class_from_name(name):
    stem = Path(name).stem
    return stem.rstrip("0123456789")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scale = cfg["scale"]
    eval_dir = Path(cfg["eval"]["save_dir"])
    compare_dir = Path(cfg["eval"]["compare_dir"])
    crop_border = cfg["eval"].get("crop_border", scale)
    eval_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Build and load model
    mp = cfg.get("model_params", {})
    model = RCAN(
        in_channels=mp.get("in_channels", 3),
        out_channels=mp.get("out_channels", 3),
        num_features=mp.get("num_features", 64),
        num_resgroups=mp.get("num_resgroups", 10),
        num_resblocks=mp.get("num_resblocks", 20),
        reduction=mp.get("reduction", 16),
        scale=scale,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_pretrained_rcan(args.checkpoint, model, device)
    model = model.to(device)
    model.eval()

    # Test dataset - use HR dir and LR dir directly (not SRDataset for pretrained)
    hr_dir = Path(cfg["data"]["test_hr"])
    lr_dir = Path(cfg["data"]["test_lr"])

    # Pretrained RCAN input: [0,255] float, subtract RGB mean
    rgb_mean = np.array([114.44, 111.46, 103.02], dtype=np.float32)

    images = sorted(lr_dir.glob("*.png"))
    print(f"Test images: {len(images)}")

    results = []
    class_results = defaultdict(list)
    class_compare_count = defaultdict(int)

    with torch.no_grad():
        for i, lr_path in enumerate(images):
            name = lr_path.name
            hr_path = hr_dir / name

            # Load LR and HR
            lr = Image.open(lr_path).convert("RGB")
            hr = Image.open(hr_path).convert("RGB")
            hr_np = np.array(hr)
            lr_np = np.array(lr)

            # Preprocess: [0,255] float - mean
            lr_tensor = torch.from_numpy(lr_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            lr_tensor = lr_tensor - torch.from_numpy(rgb_mean).view(1, 3, 1, 1)
            lr_tensor = lr_tensor.to(device)

            # Forward
            sr_tensor = model(lr_tensor)

            # Postprocess: add mean back, clamp to [0,255]
            sr_np = (sr_tensor[0].cpu() + torch.from_numpy(rgb_mean).view(3, 1, 1))
            sr_np = sr_np.permute(1, 2, 0).numpy()
            sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)

            # RGB metrics
            rgb_m = calculate_metrics(hr_np, sr_np)
            # Y+crop metrics
            y_m = calculate_metrics_standard(hr_np, sr_np, scale=crop_border)

            class_name = get_class_from_name(name)
            result = {
                "image": name,
                "class": class_name,
                "rgb_psnr": rgb_m["psnr"],
                "rgb_ssim": rgb_m["ssim"],
                "y_psnr": y_m["psnr"],
                "y_ssim": y_m["ssim"],
            }
            results.append(result)
            class_results[class_name].append(result)

            # Generate comparison for top 2 per class
            if class_compare_count[class_name] < 2:
                h, w = hr_np.shape[:2]
                bicubic_np = np.array(lr.resize((w, h), Image.BICUBIC))

                comp_path = compare_dir / f"compare_{name}"
                create_comparison(
                    lr_np, bicubic_np, sr_np, hr_np, comp_path,
                    title=f"{class_name} - Y={y_m['psnr']:.2f}dB"
                )
                class_compare_count[class_name] += 1

            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(images)}")

    # Save per-image metrics
    csv_path = eval_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "class", "rgb_psnr", "rgb_ssim", "y_psnr", "y_ssim"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Per-class stats
    class_stats = []
    for cls in sorted(class_results.keys()):
        vals = class_results[cls]
        class_stats.append({
            "class": cls,
            "num_images": len(vals),
            "rgb_psnr_mean": np.mean([v["rgb_psnr"] for v in vals]),
            "rgb_ssim_mean": np.mean([v["rgb_ssim"] for v in vals]),
            "y_psnr_mean": np.mean([v["y_psnr"] for v in vals]),
            "y_ssim_mean": np.mean([v["y_ssim"] for v in vals]),
        })

    # Print summary
    all_rgb_psnr = [r["rgb_psnr"] for r in results]
    all_y_psnr = [r["y_psnr"] for r in results]

    print(f"\n{'='*60}")
    print(f"RCAN x4 Pretrained - UC Merced All Classes")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Overall RGB PSNR: {np.mean(all_rgb_psnr):.2f} dB")
    print(f"Overall Y+crop PSNR: {np.mean(all_y_psnr):.2f} dB")
    print(f"\nPer class (sorted by Y+crop PSNR):")
    for s in sorted(class_stats, key=lambda x: -x["y_psnr_mean"]):
        print(f"  {s['class']:25s} Y={s['y_psnr_mean']:.2f}  "
              f"RGB={s['rgb_psnr_mean']:.2f}  "
              f"n={s['num_images']}")

    # Save summary
    summary_lines = [
        "RCAN x4 Pretrained - UC Merced All Classes",
        "=" * 50,
        f"Total images: {len(results)}",
        f"Overall RGB PSNR: {np.mean(all_rgb_psnr):.2f} dB",
        f"Overall Y+crop PSNR: {np.mean(all_y_psnr):.2f} dB",
        "",
        "Per class (sorted by Y+crop PSNR):",
    ]
    for s in sorted(class_stats, key=lambda x: -x["y_psnr_mean"]):
        summary_lines.append(
            f"  {s['class']:25s} Y={s['y_psnr_mean']:.2f}  "
            f"RGB={s['rgb_psnr_mean']:.2f}"
        )
    summary = "\n".join(summary_lines)

    summary_path = eval_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nMetrics saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Comparison images: {compare_dir}")


if __name__ == "__main__":
    main()
