#!/usr/bin/env python3
"""Training script for SRCNN."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.srcnn import SRCNN
from utils.dataset import SRDataset
from utils.metrics import calculate_metrics
from utils.image_utils import tensor_to_np
from utils.plot_utils import plot_loss_curve


def validate(model, val_loader, device):
    """Run validation and return avg PSNR and SSIM."""
    model.eval()
    psnrs, ssims = [], []

    with torch.no_grad():
        for batch in val_loader:
            lr_up = batch["lr_up"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr_up)

            for i in range(sr.shape[0]):
                sr_np = tensor_to_np(sr[i])
                hr_np = tensor_to_np(hr[i])
                m = calculate_metrics(hr_np, sr_np)
                psnrs.append(m["psnr"])
                ssims.append(m["ssim"])

    model.train()
    return np.mean(psnrs), np.mean(ssims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/srcnn.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    scale = cfg["scale"]
    channels = cfg["channels"]
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
    train_set = SRDataset(
        cfg["data"]["train_hr"], cfg["data"]["train_lr"],
        scale=scale, patch_size=patch_size, split="train"
    )
    val_set = SRDataset(
        cfg["data"]["val_hr"], cfg["data"]["val_lr"],
        scale=scale, split="val"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"Train images: {len(train_set)}, Val images: {len(val_set)}")

    # Model
    model = SRCNN(channels=channels).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_psnr = 0.0
    train_losses = []
    val_psnrs = []
    first_batch = True

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            lr_up = batch["lr_up"].to(device)
            hr = batch["hr"].to(device)

            if first_batch:
                print(f"\n[Shape Check - Epoch {epoch}, Batch 1]")
                print(f"  lr_up shape: {lr_up.shape}")
                print(f"  hr shape:    {hr.shape}")
                first_batch = False

            sr = model(lr_up)

            if first_batch is False and epoch == 1 and num_batches == 0:
                print(f"  sr shape:    {sr.shape}")

            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation
        val_psnr, val_ssim = validate(model, val_loader, device)

        train_losses.append(avg_loss)
        val_psnrs.append(val_psnr)

        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f} | "
              f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | "
              f"Best PSNR: {best_psnr:.2f} dB")

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "psnr": val_psnr,
                "ssim": val_ssim,
                "loss": avg_loss,
            }, save_dir / "best_srcnn.pth")
            print(f"  -> New best model saved (PSNR: {best_psnr:.2f})")

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "psnr": val_psnr,
            "ssim": val_ssim,
            "loss": avg_loss,
        }, save_dir / "last_srcnn.pth")

    # Save training log
    log_path = eval_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim"])
        for i in range(epochs):
            writer.writerow([i + 1, train_losses[i], val_psnrs[i],
                             ""])  # SSIM not saved per-epoch in list, reuse psnrs

    # Save loss curve
    plot_loss_curve(train_losses, val_psnrs, eval_dir / "loss_curve.png")

    print(f"\nTraining complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints: {save_dir}")
    print(f"Training log: {log_path}")
    print(f"Loss curve: {eval_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
