#!/usr/bin/env python3
"""Dry-run for Learnable-Scale-Cascade-MSR-RCAN: verify shapes, alpha."""

import sys
import torch
import yaml

sys.path.insert(0, "/root/Code/ml-super-resolution-edsr")

from models.rcan import LearnableScaleCascadeMSRRCAN
from utils.dataset import SRDataset
from torch.utils.data import DataLoader


def main():
    config_path = "configs/learnable_scale_cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    scale = cfg["scale"]

    train_set = SRDataset(
        cfg["data"]["train_hr"], cfg["data"]["train_lr"],
        scale=scale, patch_size=cfg["train"]["patch_size"],
        split="train", augment=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True)

    mp = cfg.get("model_params", {})
    model = LearnableScaleCascadeMSRRCAN(
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
    print(f"\nModel: LearnableScaleCascadeMSRRCAN")
    print(f"  Total params:    {total_params:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Refine params:   {refine_params:,}")
    print(f"  Cascade params:  {cascade_params:,}")
    print(f"  residual_alpha:  {model.residual_alpha.item():.4f} (learnable)")

    batch = next(iter(train_loader))
    lr = batch["lr"].to(device)
    hr = batch["hr"].to(device)

    print(f"\n--- Dry-run ---")
    print(f"LR shape:          {lr.shape}")
    print(f"HR shape:          {hr.shape}")

    # Stage 1
    sr_initial = model.backbone(lr)
    sr_stage1 = model.refine(sr_initial)
    print(f"SR_stage1 shape:   {sr_stage1.shape}")

    # Get residual
    feat = model.cascade.entry(sr_stage1)
    feat = model.cascade.blocks(feat)
    residual_stage2 = model.cascade.exit_conv(feat)
    print(f"residual_stage2 shape: {residual_stage2.shape}")

    # Full forward
    sr_final = model(lr)
    print(f"SR_final shape:    {sr_final.shape}")
    print(f"\nalpha value:       {model.residual_alpha.item():.6f}")

    criterion = torch.nn.L1Loss()
    loss = criterion(sr_final, hr)
    print(f"L1 loss:           {loss.item():.6f}")

    if torch.cuda.is_available():
        print(f"GPU memory:        {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU reserved:      {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Shape checks
    assert sr_stage1.shape == hr.shape
    assert sr_final.shape == hr.shape
    assert residual_stage2.shape == hr.shape
    assert not torch.isnan(model.residual_alpha), "Alpha is NaN!"

    print(f"\n--- All checks passed ---")


if __name__ == "__main__":
    main()
