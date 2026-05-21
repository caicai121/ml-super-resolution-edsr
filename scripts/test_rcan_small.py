#!/usr/bin/env python3
"""Test RCAN-small (from scratch) with per-class stats and comparison figures.

Usage:
    python scripts/test_rcan_small.py --config configs/rcan_small_x4_ucmerced_selected.yaml \
        --checkpoint checkpoints/rcan_small_x4/best_rcan_small_x4.pth
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

from models.rcan import RCAN
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
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Optional: pretrained RCAN checkpoint for comparison")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scale = cfg["scale"]
    mp = cfg.get("model_params", {})
    eval_dir = Path(cfg["eval"]["save_dir"])
    compare_dir = eval_dir / "comparisons"
    eval_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Build model (same as rcan_small - just RCAN with smaller params)
    model = RCAN(
        in_channels=mp.get("in_channels", 3),
        out_channels=mp.get("out_channels", 3),
        num_features=mp.get("num_features", 64),
        num_resgroups=mp.get("num_resgroups", 3),
        num_resblocks=mp.get("num_resblocks", 5),
        reduction=mp.get("reduction", 16),
        scale=scale,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint from epoch {checkpoint['epoch']}, "
          f"Val PSNR: {checkpoint.get('psnr', 0):.2f}")

    # Load pretrained model for comparison if provided
    pretrained_model = None
    if args.pretrained_checkpoint:
        from models.rcan import load_pretrained_rcan
        pretrained_model = RCAN(
            in_channels=3, out_channels=3, num_features=64,
            num_resgroups=10, num_resblocks=20, reduction=16, scale=scale,
        )
        pretrained_model = load_pretrained_rcan(
            args.pretrained_checkpoint, pretrained_model, device)
        pretrained_model = pretrained_model.to(device)
        pretrained_model.eval()
        print("Loaded pretrained RCAN for comparison")

    # Load Bicubic baseline if exists
    bicubic_results = {}
    bicubic_csv = Path("data_final/ucmerced_selected/results/bicubic_x4/metrics.csv")
    if bicubic_csv.exists():
        with open(bicubic_csv) as f:
            for row in csv.DictReader(f):
                bicubic_results[row["image"]] = float(row.get("y_psnr", row.get("psnr", 0)))

    # Test dataset
    test_set = SRDataset(
        cfg["data"]["test_hr"], cfg["data"]["test_lr"],
        scale=scale, split="test"
    )
    print(f"Test images: {len(test_set)}")

    results = []
    class_results = defaultdict(list)
    class_compare_count = defaultdict(int)

    with torch.no_grad():
        for i in range(len(test_set)):
            batch = test_set[i]
            lr = torch.from_numpy(batch["lr"]).unsqueeze(0).to(device)
            hr = torch.from_numpy(batch["hr"]).unsqueeze(0).to(device)
            name = batch["name"]

            sr = model(lr)

            sr_np = tensor_to_np(sr[0])
            hr_np = tensor_to_np(hr[0])

            rgb_m = calculate_metrics(hr_np, sr_np)
            y_m = calculate_metrics_standard(hr_np, sr_np, scale=scale)

            class_name = get_class_from_name(name)
            bicubic_y = bicubic_results.get(name, 0)

            result = {
                "image": name,
                "class": class_name,
                "rgb_psnr": rgb_m["psnr"],
                "rgb_ssim": rgb_m["ssim"],
                "y_psnr": y_m["psnr"],
                "y_ssim": y_m["ssim"],
                "bicubic_y_psnr": bicubic_y,
                "gain_vs_bicubic": y_m["psnr"] - bicubic_y if bicubic_y else 0,
            }
            results.append(result)
            class_results[class_name].append(result)

            # Generate comparison
            if class_compare_count[class_name] < 1:
                lr_path = Path(cfg["data"]["test_lr"]) / name
                lr_img = np.array(Image.open(lr_path).convert("RGB"))
                h, w = hr_np.shape[:2]
                bicubic_np = np.array(
                    Image.open(lr_path).convert("RGB").resize((w, h), Image.BICUBIC)
                )

                if pretrained_model is not None:
                    # 5-column: LR / Bicubic / RCAN-small / RCAN-pretrained / HR
                    # Pretrained RCAN: raw LR [0,255] float - RGB mean
                    rgb_mean = np.array([114.44, 111.46, 103.02], dtype=np.float32)
                    lr_raw = np.array(Image.open(lr_path).convert("RGB")).astype(np.float32)
                    lr_pretrained = torch.from_numpy(lr_raw).permute(2, 0, 1).unsqueeze(0)
                    lr_pretrained = lr_pretrained - torch.from_numpy(rgb_mean).view(1, 3, 1, 1)
                    lr_pretrained = lr_pretrained.to(device)
                    sr_pretrained = pretrained_model(lr_pretrained)
                    sr_p = (sr_pretrained[0].cpu() + torch.from_numpy(rgb_mean).view(3, 1, 1))
                    pretrained_np = np.clip(sr_p.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)

                    from utils.plot_utils import create_comparison_5col
                    comp_path = compare_dir / f"compare_{name}"
                    create_comparison_5col(
                        lr_img, bicubic_np, sr_np, pretrained_np, hr_np,
                        comp_path,
                        title=f"Small={y_m['psnr']:.2f}dB",
                        scale=scale,
                        model_a_name="RCAN-small",
                        model_b_name="RCAN-pretrained"
                    )
                else:
                    comp_path = compare_dir / f"compare_{name}"
                    create_comparison(
                        lr_img, bicubic_np, sr_np, hr_np, comp_path,
                        title=f"{class_name} - Y={y_m['psnr']:.2f}dB",
                        scale=scale
                    )

                class_compare_count[class_name] += 1

    # Save per-image metrics
    csv_path = eval_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "class", "rgb_psnr", "rgb_ssim",
            "y_psnr", "y_ssim", "bicubic_y_psnr", "gain_vs_bicubic"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Per-class summary
    all_rgb_psnr = [r["rgb_psnr"] for r in results]
    all_y_psnr = [r["y_psnr"] for r in results]

    print(f"\n{'='*60}")
    print(f"RCAN-small Test Results")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Overall RGB PSNR: {np.mean(all_rgb_psnr):.2f} dB")
    print(f"Overall Y+crop PSNR: {np.mean(all_y_psnr):.2f} dB")

    summary_lines = [
        "RCAN-small Test Results",
        "=" * 50,
        f"Model: RCAN-small (n_resgroups={mp.get('num_resgroups', 3)}, "
        f"n_resblocks={mp.get('num_resblocks', 5)}, n_feats={mp.get('num_features', 64)})",
        f"Checkpoint: epoch {checkpoint['epoch']}",
        f"Total images: {len(results)}",
        f"Overall RGB PSNR: {np.mean(all_rgb_psnr):.2f} dB",
        f"Overall Y+crop PSNR: {np.mean(all_y_psnr):.2f} dB",
    ]

    summary = "\n".join(summary_lines)
    summary_path = eval_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Comparisons: {compare_dir}")


if __name__ == "__main__":
    main()
