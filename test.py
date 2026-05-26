#!/usr/bin/env python3
"""Testing script for super-resolution models (SRCNN, Light-EDSR, RCAN)."""

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
from models.edsr import LightEDSR
from models.rcan import RCAN
from utils.dataset import SRDataset
from utils.metrics import calculate_metrics, calculate_metrics_standard
from utils.image_utils import tensor_to_np, save_image
from utils.plot_utils import create_comparison, create_comparison_5col


def build_model(cfg):
    """Build model from config."""
    model_name = cfg["model"]
    scale = cfg["scale"]

    if model_name == "srcnn":
        channels = cfg.get("channels", 3)
        model = SRCNN(channels=channels)
        input_key = "lr_up"
        ckpt_default = "best_srcnn.pth"
    elif model_name == "light_edsr":
        mp = cfg.get("model_params", {})
        model = LightEDSR(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_res_blocks=mp.get("num_res_blocks", 8),
            res_scale=mp.get("res_scale", 0.1),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_light_edsr.pth"
    elif model_name == "rcan":
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
        input_key = "lr"
        ckpt_default = "best_rcan_x4.pth"
    elif model_name == "rcan_small":
        mp = cfg.get("model_params", {})
        model = RCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 3),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_rcan_small_x4.pth"
    elif model_name == "ms_rcan_small":
        from models.rcan import MSRCAN
        mp = cfg.get("model_params", {})
        model = MSRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 3),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_ms_rcan_small_x4.pth"
    elif model_name == "msr_rcan_small":
        from models.rcan import MSRRCAN
        mp = cfg.get("model_params", {})
        model = MSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 3),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_msr_rcan_small_x4.pth"
    elif model_name == "msr_rcan_small_v2":
        from models.rcan import MSRRCANV2
        mp = cfg.get("model_params", {})
        model = MSRRCANV2(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 3),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_msr_rcan_small_v2_x4.pth"
    elif model_name == "dmsr_rcan_small":
        from models.rcan import DMSRRCAN
        mp = cfg.get("model_params", {})
        model = DMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 3),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_dmsr_rcan_small_x4.pth"
    elif model_name == "msr_rcan_mid":
        from models.rcan import MSRRCANV2
        mp = cfg.get("model_params", {})
        model = MSRRCANV2(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 5),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_msr_rcan_mid_x4.pth"
    elif model_name == "eg_msr_rcan":
        from models.rcan import EGMSRRCAN
        mp = cfg.get("model_params", {})
        model = EGMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 5),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
            edge_mid_channels=mp.get("edge_mid_channels", 32),
            edge_num_layers=mp.get("edge_num_layers", 3),
        )
        input_key = "lr"
        ckpt_default = "best_eg_msr_rcan_x4.pth"
    elif model_name == "amsr_rcan_mid":
        from models.rcan import AMSRRCAN
        mp = cfg.get("model_params", {})
        model = AMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 5),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_amsr_rcan_mid_x4.pth"
    elif model_name == "rdr_msr_rcan_mid":
        from models.rcan import RDRMSRRCAN
        mp = cfg.get("model_params", {})
        model = RDRMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 5),
            num_resblocks=mp.get("num_resblocks", 5),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_rdr_msr_rcan_mid_x4.pth"
    elif model_name == "msr_rcan_large":
        from models.rcan import MSRRCANV2
        mp = cfg.get("model_params", {})
        model = MSRRCANV2(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 8),
            num_resblocks=mp.get("num_resblocks", 8),
            reduction=mp.get("reduction", 16),
            scale=scale,
        )
        input_key = "lr"
        ckpt_default = "best_msr_rcan_large50_cosine_x4.pth"




    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, input_key, ckpt_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/srcnn.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--srcnn-results", type=str, default=None,
                        help="Path to SRCNN metrics.csv for 5-col comparison")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = cfg["model"]
    scale = cfg["scale"]
    eval_dir = Path(cfg["eval"]["save_dir"])
    compare_dir = Path(cfg["eval"]["compare_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    model, input_key, ckpt_default = build_model(cfg)
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = Path(cfg["train"]["save_dir"]) / ckpt_default

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint from epoch {checkpoint['epoch']}, "
          f"Val PSNR: {checkpoint.get('psnr', 0):.2f}")

    # Load SRCNN results for 5-column comparison if available
    srcnn_results = {}
    if args.srcnn_results:
        srcnn_path = Path(args.srcnn_results)
    else:
        srcnn_path = Path("results/srcnn/metrics.csv")
    if srcnn_path.exists():
        with open(srcnn_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                srcnn_results[row["image"]] = {
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                }
        print(f"Loaded SRCNN results from {srcnn_path} ({len(srcnn_results)} images)")

    # Test dataset
    test_set = SRDataset(
        cfg["data"]["test_hr"], cfg["data"]["test_lr"],
        scale=scale, split="test"
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    print(f"Test images: {len(test_set)}")

    # Load SRCNN model for 5-column comparison if results exist
    srcnn_model = None
    if srcnn_results:
        srcnn_ckpt_path = Path("checkpoints/srcnn/best_srcnn.pth")
        if srcnn_ckpt_path.exists():
            srcnn_model = SRCNN(channels=3).to(device)
            srcnn_ckpt = torch.load(srcnn_ckpt_path, map_location=device, weights_only=False)
            srcnn_model.load_state_dict(srcnn_ckpt["model_state_dict"])
            srcnn_model.eval()
            print("Loaded SRCNN model for 5-column comparison")

    # Evaluate - both RGB and Y+crop
    results = []
    compare_count = 0

    with torch.no_grad():
        for batch in test_loader:
            inp = batch[input_key].to(device)
            hr = batch["hr"].to(device)
            name = batch["name"][0]

            sr = model(inp)

            sr_np = tensor_to_np(sr[0])
            hr_np = tensor_to_np(hr[0])

            rgb_m = calculate_metrics(hr_np, sr_np)
            y_m = calculate_metrics_standard(hr_np, sr_np, scale=scale)

            results.append({
                "image": name,
                "rgb_psnr": rgb_m["psnr"],
                "rgb_ssim": rgb_m["ssim"],
                "y_psnr": y_m["psnr"],
                "y_ssim": y_m["ssim"],
            })

            # Generate comparison images (first 3)
            if compare_count < 3:
                lr_path = Path(cfg["data"]["test_lr"]) / name
                lr_img = np.array(Image.open(lr_path).convert("RGB"))

                hr_w, hr_h = hr_np.shape[1], hr_np.shape[0]
                lr_pil = Image.open(lr_path).convert("RGB")
                bicubic_np = np.array(lr_pil.resize((hr_w, hr_h), Image.BICUBIC))

                if srcnn_model is not None:
                    lr_up_tensor = batch["lr_up"].to(device)
                    srcnn_sr = srcnn_model(lr_up_tensor)
                    srcnn_np = tensor_to_np(srcnn_sr[0])

                    comp_path = compare_dir / f"compare_{model_name}_{name}"
                    create_comparison_5col(
                        lr_img, bicubic_np, srcnn_np, sr_np, hr_np,
                        comp_path,
                        title=f"Y={y_m['psnr']:.2f}dB"
                    )
                else:
                    comp_path = compare_dir / f"compare_{model_name}_{name}"
                    create_comparison(
                        lr_img, bicubic_np, sr_np, hr_np, comp_path,
                        title=f"Y={y_m['psnr']:.2f}dB, RGB={rgb_m['psnr']:.2f}dB"
                    )

                compare_count += 1

            print(f"  {name} - RGB: {rgb_m['psnr']:.2f}, Y+crop: {y_m['psnr']:.2f}")

    # Save metrics CSV with both metrics
    csv_path = eval_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "rgb_psnr", "rgb_ssim", "y_psnr", "y_ssim"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_rgb_psnr = np.mean([r["rgb_psnr"] for r in results])
    avg_rgb_ssim = np.mean([r["rgb_ssim"] for r in results])
    avg_y_psnr = np.mean([r["y_psnr"] for r in results])
    avg_y_ssim = np.mean([r["y_ssim"] for r in results])

    summary_lines = [
        f"{model_name.upper()} Test Results",
        "=" * 50,
        f"Images: {len(results)}",
        f"RGB PSNR:  {avg_rgb_psnr:.2f} dB | RGB SSIM:  {avg_rgb_ssim:.4f}",
        f"Y+crop PSNR: {avg_y_psnr:.2f} dB | Y+crop SSIM: {avg_y_ssim:.4f}",
    ]

    summary = "\n".join(summary_lines)
    print(f"\n{summary}")

    summary_path = eval_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nMetrics saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Comparison images: {compare_dir}")


if __name__ == "__main__":
    main()
