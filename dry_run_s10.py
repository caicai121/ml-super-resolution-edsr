#!/usr/bin/env python3
"""Dry-run for Cascade-MSR-RCAN Stage2-10: verify shapes and residual stats."""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "/root/Code/ml-super-resolution-edsr")

from models.rcan import CascadeMSRRCAN
from utils.dataset import SRDataset
from torch.utils.data import DataLoader


def main():
    config_path = "configs/cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scale = cfg["scale"]

    # Dataset
    train_set = SRDataset(
        cfg["data"]["train_hr"], cfg["data"]["train_lr"],
        scale=scale, patch_size=cfg["train"]["patch_size"],
        split="train", augment=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True)

    # Model
    mp = cfg.get("model_params", {})
    model = CascadeMSRRCAN(
        in_channels=mp.get("in_channels", 3),
        out_channels=mp.get("out_channels", 3),
        num_features=mp.get("num_features", 64),
        num_resgroups=mp.get("num_resgroups", 8),
        num_resblocks=mp.get("num_resblocks", 8),
        reduction=mp.get("reduction", 16),
        scale=scale,
        cascade_num_blocks=mp.get("cascade_num_blocks", 10),
        cascade_mid_channels=mp.get("cascade_mid_channels", 64),
        cascade_residual_scale=mp.get("cascade_residual_scale", 0.1),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    refine_params = sum(p.numel() for p in model.refine.parameters())
    cascade_params = sum(p.numel() for p in model.cascade.parameters())
    print(f"\nModel: CascadeMSRRCAN (Stage2-10)")
    print(f"  Total params:    {total_params:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Refine params:   {refine_params:,}")
    print(f"  Cascade params:  {cascade_params:,}")

    # Dry-run one batch
    batch = next(iter(train_loader))
    lr = batch["lr"].to(device)
    hr = batch["hr"].to(device)

    print(f"\n--- Dry-run ---")
    print(f"LR shape:          {lr.shape}")
    print(f"LR range:          [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"HR shape:          {hr.shape}")
    print(f"HR range:          [{hr.min():.4f}, {hr.max():.4f}]")

    # Stage 1
    sr_initial = model.backbone(lr)
    sr_stage1 = model.refine(sr_initial)
    print(f"SR_stage1 shape:   {sr_stage1.shape}")
    print(f"SR_stage1 range:   [{sr_stage1.min():.4f}, {sr_stage1.max():.4f}]")

    # Stage 2 cascade (get residual before adding)
    feat = model.cascade.entry(sr_stage1)
    feat = model.cascade.blocks(feat)
    residual_stage2 = model.cascade.exit_conv(feat)
    scaled_residual = model.cascade.residual_scale * residual_stage2
    print(f"residual_stage2 shape: {residual_stage2.shape}")
    print(f"residual_stage2 mean:  {residual_stage2.mean().item():.6f}")
    print(f"residual_stage2 std:   {residual_stage2.std().item():.6f}")
    print(f"scaled_residual mean:  {scaled_residual.mean().item():.6f}")
    print(f"scaled_residual std:   {scaled_residual.std().item():.6f}")

    # Full forward
    sr_final = model(lr)
    print(f"SR_final shape:    {sr_final.shape}")
    print(f"SR_final range:    [{sr_final.min():.4f}, {sr_final.max():.4f}]")

    # Loss
    criterion = torch.nn.L1Loss()
    loss = criterion(sr_final, hr)
    print(f"\nL1 loss:           {loss.item():.6f}")

    if torch.cuda.is_available():
        print(f"GPU memory:        {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU reserved:      {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Verify shapes
    assert sr_stage1.shape == hr.shape, \
        f"Shape mismatch: sr_stage1 {sr_stage1.shape} vs hr {hr.shape}"
    assert sr_final.shape == hr.shape, \
        f"Shape mismatch: sr_final {sr_final.shape} vs hr {hr.shape}"
    assert residual_stage2.shape == hr.shape, \
        f"Shape mismatch: residual {residual_stage2.shape} vs hr {hr.shape}"

    print(f"\n--- All checks passed ---")


if __name__ == "__main__":
    main()
