#!/usr/bin/env python3
"""Training script for super-resolution models (SRCNN, Light-EDSR, RCAN)."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.srcnn import SRCNN
from models.edsr import LightEDSR
from models.rcan import RCAN
from utils.dataset import SRDataset
from utils.metrics import calculate_metrics, calculate_metrics_standard
from utils.image_utils import tensor_to_np
from utils.plot_utils import plot_loss_curve


def build_model(cfg):
    """Build model from config."""
    model_name = cfg["model"]
    scale = cfg["scale"]

    if model_name == "srcnn":
        channels = cfg.get("channels", 3)
        model = SRCNN(channels=channels)
        input_key = "lr_up"
        ckpt_name = "best_srcnn.pth"
        last_name = "last_srcnn.pth"
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
        ckpt_name = "best_light_edsr.pth"
        last_name = "last_light_edsr.pth"
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
        ckpt_name = "best_rcan_x4.pth"
        last_name = "last_rcan_x4.pth"
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
        ckpt_name = "best_rcan_small_x4.pth"
        last_name = "last_rcan_small_x4.pth"
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
        ckpt_name = "best_ms_rcan_small_x4.pth"
        last_name = "last_ms_rcan_small_x4.pth"
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
        ckpt_name = "best_msr_rcan_small_x4.pth"
        last_name = "last_msr_rcan_small_x4.pth"
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
        ckpt_name = "best_msr_rcan_small_v2_x4.pth"
        last_name = "last_msr_rcan_small_v2_x4.pth"
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
        ckpt_name = "best_dmsr_rcan_small_x4.pth"
        last_name = "last_dmsr_rcan_small_x4.pth"
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
        ckpt_name = "best_msr_rcan_mid_x4.pth"
        last_name = "last_msr_rcan_mid_x4.pth"
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
        ckpt_name = "best_eg_msr_rcan_x4.pth"
        last_name = "last_eg_msr_rcan_x4.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, input_key, ckpt_name, last_name


def validate(model, val_loader, device, input_key, scale):
    """Run validation and return both RGB and Y+crop metrics."""
    model.eval()
    rgb_psnrs, rgb_ssims = [], []
    y_psnrs, y_ssims = [], []

    with torch.no_grad():
        for batch in val_loader:
            inp = batch[input_key].to(device)
            hr = batch["hr"].to(device)
            sr = model(inp)

            for i in range(sr.shape[0]):
                sr_np = tensor_to_np(sr[i])
                hr_np = tensor_to_np(hr[i])

                rgb_m = calculate_metrics(hr_np, sr_np)
                rgb_psnrs.append(rgb_m["psnr"])
                rgb_ssims.append(rgb_m["ssim"])

                y_m = calculate_metrics_standard(hr_np, sr_np, scale=scale)
                y_psnrs.append(y_m["psnr"])
                y_ssims.append(y_m["ssim"])

    model.train()
    return {
        "rgb_psnr": np.mean(rgb_psnrs),
        "rgb_ssim": np.mean(rgb_ssims),
        "y_psnr": np.mean(y_psnrs),
        "y_ssim": np.mean(y_ssims),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/srcnn.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = cfg["model"]
    scale = cfg["scale"]
    epochs = args.epochs if args.epochs else cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    patch_size = cfg["train"]["patch_size"]
    lr = cfg["train"]["lr"]
    num_workers = cfg["train"]["num_workers"]
    save_dir = Path(cfg["train"]["save_dir"])
    eval_dir = Path(cfg["eval"]["save_dir"])

    save_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    augment = cfg.get("train", {}).get("augment", False)
    train_set = SRDataset(
        cfg["data"]["train_hr"], cfg["data"]["train_lr"],
        scale=scale, patch_size=patch_size, split="train", augment=augment
    )
    val_set = SRDataset(
        cfg["data"]["val_hr"], cfg["data"]["val_lr"],
        scale=scale, split="val"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True)

    print(f"Model: {model_name}")
    print(f"Train images: {len(train_set)}, Val images: {len(val_set)}")

    # Model
    model, input_key, ckpt_name, last_name = build_model(cfg)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input key: {input_key}")

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = None
    scheduler_cfg = cfg.get("train", {}).get("scheduler")
    if scheduler_cfg == "cosine":
        min_lr = cfg["train"].get("min_lr", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
        print(f"Scheduler: CosineAnnealingLR (min_lr={min_lr})")
    elif scheduler_cfg == "step":
        step_size = cfg["train"].get("step_size", 20)
        gamma = cfg["train"].get("gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        print(f"Scheduler: StepLR (step={step_size}, gamma={gamma})")

    # Training loop
    best_y_psnr = 0.0
    train_losses = []
    val_metrics_history = []
    first_batch = True

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inp = batch[input_key].to(device)
            hr = batch["hr"].to(device)

            if first_batch:
                print(f"\n[Shape Check - Epoch {epoch}, Batch 1]")
                if "lr" in batch:
                    print(f"  lr shape:    {batch['lr'].shape}")
                print(f"  input shape: {inp.shape}")
                print(f"  hr shape:    {hr.shape}")
                first_batch = False

            sr = model(inp)

            if num_batches == 0 and epoch == 1:
                print(f"  sr shape:    {sr.shape}")

            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Validation - both RGB and Y+crop
        val_m = validate(model, val_loader, device, input_key, scale)
        val_metrics_history.append(val_m)

        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f} | "
              f"RGB: {val_m['rgb_psnr']:.2f}/{val_m['rgb_ssim']:.4f} | "
              f"Y+crop: {val_m['y_psnr']:.2f}/{val_m['y_ssim']:.4f} | "
              f"Best Y: {best_y_psnr:.2f}")

        # Save best (by Y+crop PSNR)
        if val_m["y_psnr"] > best_y_psnr:
            best_y_psnr = val_m["y_psnr"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "psnr": val_m["rgb_psnr"],
                "ssim": val_m["rgb_ssim"],
                "y_psnr": val_m["y_psnr"],
                "y_ssim": val_m["y_ssim"],
                "loss": avg_loss,
                "model_name": model_name,
            }, save_dir / ckpt_name)
            print(f"  -> New best model saved (Y+crop PSNR: {best_y_psnr:.2f})")

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "psnr": val_m["rgb_psnr"],
            "ssim": val_m["rgb_ssim"],
            "y_psnr": val_m["y_psnr"],
            "y_ssim": val_m["y_ssim"],
            "loss": avg_loss,
            "model_name": model_name,
        }, save_dir / last_name)

    # Save training log with both metrics
    log_path = eval_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss",
                         "val_rgb_psnr", "val_rgb_ssim",
                         "val_y_psnr", "val_y_ssim"])
        for i in range(epochs):
            m = val_metrics_history[i]
            writer.writerow([i + 1, train_losses[i],
                             m["rgb_psnr"], m["rgb_ssim"],
                             m["y_psnr"], m["y_ssim"]])

    # Save loss curve
    plot_loss_curve(train_losses,
                    [m["y_psnr"] for m in val_metrics_history],
                    eval_dir / "loss_curve.png")

    print(f"\nTraining complete. Best Val Y+crop PSNR: {best_y_psnr:.2f} dB")
    print(f"Checkpoints: {save_dir}")
    print(f"Training log: {log_path}")
    print(f"Loss curve: {eval_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
