#!/usr/bin/env python3
"""
Recompute all model results using standard SR evaluation (Y-channel + border crop).

Usage:
  python scripts/recompute_standard.py --all          # all models
  python scripts/recompute_standard.py --model bicubic # single model
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_metrics, calculate_metrics_standard


def eval_from_images(sr_dir, hr_dir, scale, model_name):
    """Evaluate from saved SR images."""
    hr_dir = Path(hr_dir)
    sr_dir = Path(sr_dir)

    sr_images = sorted(sr_dir.glob("*.png"))
    if not sr_images:
        print(f"  No images found in {sr_dir}")
        return None

    results = []
    for sr_path in sr_images:
        name = sr_path.name
        hr_path = hr_dir / name
        if not hr_path.exists():
            continue

        hr = np.array(Image.open(hr_path).convert("RGB"))
        sr = np.array(Image.open(sr_path).convert("RGB"))

        m_rgb = calculate_metrics(hr, sr)
        m_std = calculate_metrics_standard(hr, sr, scale=scale)

        results.append({
            "image": name,
            "psnr_rgb": m_rgb["psnr"],
            "ssim_rgb": m_rgb["ssim"],
            "psnr_y": m_std["psnr"],
            "ssim_y": m_std["ssim"],
        })

    return results


def eval_srcnn(hr_dir, lr_dir, scale, device):
    """Load SRCNN checkpoint and evaluate."""
    from models.srcnn import SRCNN

    ckpt_path = Path("checkpoints/srcnn/best_srcnn.pth")
    if not ckpt_path.exists():
        print(f"  SRCNN checkpoint not found: {ckpt_path}")
        return None

    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)

    model = SRCNN(channels=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    lr_images = sorted(lr_dir.glob("*.png"))

    with torch.no_grad():
        for lr_path in lr_images:
            name = lr_path.name
            hr_path = hr_dir / name
            if not hr_path.exists():
                continue

            hr = np.array(Image.open(hr_path).convert("RGB"))
            lr = Image.open(lr_path).convert("RGB")
            hr_h, hr_w = hr.shape[:2]

            # SRCNN takes bicubic-upsampled LR as input
            lr_up = lr.resize((hr_w, hr_h), Image.BICUBIC)
            lr_up_np = np.array(lr_up, dtype=np.float32) / 255.0
            inp = torch.from_numpy(lr_up_np).permute(2, 0, 1).unsqueeze(0).to(device)

            sr = model(inp)
            sr_np = (sr[0].cpu().numpy().clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)

            m_rgb = calculate_metrics(hr, sr_np)
            m_std = calculate_metrics_standard(hr, sr_np, scale=scale)

            results.append({
                "image": name,
                "psnr_rgb": m_rgb["psnr"],
                "ssim_rgb": m_rgb["ssim"],
                "psnr_y": m_std["psnr"],
                "ssim_y": m_std["ssim"],
            })

    return results


def eval_light_edsr(hr_dir, lr_dir, scale, device):
    """Load Light-EDSR checkpoint and evaluate."""
    from models.edsr import LightEDSR
from models.rcan import RCAN

    ckpt_path = Path("checkpoints/light_edsr/best_light_edsr.pth")
    if not ckpt_path.exists():
        print(f"  Light-EDSR checkpoint not found: {ckpt_path}")
        return None

    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)

    model = LightEDSR(
        in_channels=3, out_channels=3,
        num_features=64, num_res_blocks=8,
        res_scale=0.1, scale=scale,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    lr_images = sorted(lr_dir.glob("*.png"))

    with torch.no_grad():
        for lr_path in lr_images:
            name = lr_path.name
            hr_path = hr_dir / name
            if not hr_path.exists():
                continue

            hr = np.array(Image.open(hr_path).convert("RGB"))
            lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
            inp = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).to(device)

            sr = model(inp)
            sr_np = (sr[0].cpu().numpy().clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)

            m_rgb = calculate_metrics(hr, sr_np)
            m_std = calculate_metrics_standard(hr, sr_np, scale=scale)

            results.append({
                "image": name,
                "psnr_rgb": m_rgb["psnr"],
                "ssim_rgb": m_rgb["ssim"],
                "psnr_y": m_std["psnr"],
                "ssim_y": m_std["ssim"],
            })

    return results



def eval_rcan_x4(hr_dir, lr_dir, scale, device):
    """Load RCAN x4 checkpoint and evaluate."""
    ckpt_path = Path("checkpoints/rcan_x4/best_rcan_x4.pth")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/rcan_x4/pretrained_rcan_x4.pth")
    if not ckpt_path.exists():
        print(f"  RCAN x4 checkpoint not found")
        return None

    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)

    model = RCAN(
        in_channels=3, out_channels=3,
        num_features=64, num_resgroups=10,
        num_resblocks=20, reduction=16, scale=4,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    results = []
    lr_images = sorted(lr_dir.glob("*.png"))

    with torch.no_grad():
        for lr_path in lr_images:
            name = lr_path.name
            hr_path = hr_dir / name
            if not hr_path.exists():
                continue

            hr = np.array(Image.open(hr_path).convert("RGB"))
            lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32) / 255.0
            inp = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0).to(device)

            sr = model(inp)
            sr_np = (sr[0].cpu().numpy().clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)

            m_rgb = calculate_metrics(hr, sr_np)
            m_std = calculate_metrics_standard(hr, sr_np, scale=scale)

            results.append({
                "image": name,
                "psnr_rgb": m_rgb["psnr"],
                "ssim_rgb": m_rgb["ssim"],
                "psnr_y": m_std["psnr"],
                "ssim_y": m_std["ssim"],
            })

    return results

def save_results(results, save_path, model_name):
    """Save metrics CSV and print summary."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y"])
        writer.writeheader()
        writer.writerows(results)

    avg_psnr_rgb = np.mean([r["psnr_rgb"] for r in results])
    avg_ssim_rgb = np.mean([r["ssim_rgb"] for r in results])
    avg_psnr_y = np.mean([r["psnr_y"] for r in results])
    avg_ssim_y = np.mean([r["ssim_y"] for r in results])

    print(f"\n  {model_name} ({len(results)} images):")
    print(f"    RGB:  PSNR = {avg_psnr_rgb:.2f} dB, SSIM = {avg_ssim_rgb:.4f}")
    print(f"    Y+crop: PSNR = {avg_psnr_y:.2f} dB, SSIM = {avg_ssim_y:.4f}")

    return avg_psnr_y, avg_ssim_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=["bicubic", "srcnn", "light_edsr", "x4_bicubic", "rcan_x4", "bicubic_x4_full"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = []
    if args.all or args.model == "bicubic":
        models.append("bicubic")
    if args.all or args.model == "srcnn":
        models.append("srcnn")
    if args.all or args.model == "light_edsr":
        models.append("light_edsr")
    if args.all or args.model == "x4_bicubic":
        models.append("x4_bicubic")
    if args.all or args.model == "rcan_x4":
        models.append("rcan_x4")
    if args.all or args.model == "bicubic_x4_full":
        models.append("bicubic_x4_full")

    summary = {}

    for name in models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        if name == "bicubic":
            results = eval_from_images(
                "results/bicubic/images", "data/test/HR", scale=2, model_name="Bicubic"
            )
            if results:
                summary["Bicubic (x2)"] = save_results(
                    results, "report_assets/tables/standard_metrics_bicubic.csv", "Bicubic (x2)"
                )

        elif name == "srcnn":
            results = eval_srcnn("data/test/HR", "data/test/LR_x2", scale=2, device=device)
            if results:
                summary["SRCNN (x2)"] = save_results(
                    results, "report_assets/tables/standard_metrics_srcnn.csv", "SRCNN (x2)"
                )

        elif name == "light_edsr":
            results = eval_light_edsr("data/test/HR", "data/test/LR_x2", scale=2, device=device)
            if results:
                summary["Light-EDSR (x2)"] = save_results(
                    results, "report_assets/tables/standard_metrics_light_edsr.csv", "Light-EDSR (x2)"
                )

        elif name == "x4_bicubic":
            results = eval_from_images(
                "results/x4_visual", "data/test/HR", scale=4, model_name="Bicubic (x4)"
            )
            if results:
                summary["Bicubic (x4)"] = save_results(
                    results, "report_assets/tables/standard_metrics_x4_bicubic.csv", "Bicubic (x4)"
                )


        elif name == "bicubic_x4_full":
            results = eval_from_images(
                "results/bicubic_x4/images", "data/test/HR", scale=4, model_name="Bicubic (x4 full)"
            )
            if results:
                summary["Bicubic (x4)"] = save_results(
                    results, "report_assets/tables/standard_metrics_bicubic_x4.csv", "Bicubic (x4 full)"
                )

        elif name == "rcan_x4":
            results = eval_rcan_x4("data/test/HR", "data/test/LR_x4", scale=4, device=device)
            if results:
                summary["RCAN (x4)"] = save_results(
                    results, "report_assets/tables/standard_metrics_rcan_x4.csv", "RCAN (x4)"
                )

    # Print comparison table
    if summary:
        print(f"\n{'='*60}")
        print("Standard SR Evaluation Summary (Y-channel + border crop)")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'PSNR (dB)':<12} {'SSIM':<12}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")
        for model, (psnr, ssim) in summary.items():
            print(f"{model:<20} {psnr:<12.2f} {ssim:<12.4f}")


if __name__ == "__main__":
    main()
