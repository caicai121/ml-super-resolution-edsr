#!/usr/bin/env python3
"""Dry-run for BP-Cascade-MSR-RCAN: verify shapes and LR consistency."""

import sys
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "/root/Code/ml-super-resolution-edsr")

from models.rcan import BPCascadeMSRRCAN
from utils.dataset import SRDataset
from torch.utils.data import DataLoader


def main():
    config_path = "configs/bp_cascade_msr_rcan_large50_cosine_x4_ucmerced_selected.yaml"
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
    model = BPCascadeMSRRCAN(
        in_channels=mp.get("in_channels", 3),
        out_channels=mp.get("out_channels", 3),
        num_features=mp.get("num_features", 64),
        num_resgroups=mp.get("num_resgroups", 8),
        num_resblocks=mp.get("num_resblocks", 8),
        reduction=mp.get("reduction", 16),
        scale=scale,
        cascade_num_blocks=mp.get("cascade_num_blocks", 6),
        cascade_mid_channels=mp.get("cascade_mid_channels", 64),
        cascade_residual_scale=mp.get("cascade_residual_scale", 0.1),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    refine_params = sum(p.numel() for p in model.refine.parameters())
    cascade_params = sum(p.numel() for p in model.cascade.parameters())
    print(f"\nModel: BPCascadeMSRRCAN")
    print(f"  Total params:    {total_params:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Refine params:   {refine_params:,}")
    print(f"  Cascade params:  {cascade_params:,}")

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

    # Back-projection
    lr_recon = F.interpolate(sr_stage1, scale_factor=1.0/scale,
                             mode='bicubic', align_corners=False,
                             recompute_scale_factor=False)
    print(f"LR_recon shape:    {lr_recon.shape}")
    print(f"LR_recon range:    [{lr_recon.min():.4f}, {lr_recon.max():.4f}]")

    lr_error = lr - lr_recon
    print(f"LR_error shape:    {lr_error.shape}")
    print(f"LR_error mean:     {lr_error.mean().item():.6f}")
    print(f"LR_error std:      {lr_error.std().item():.6f}")

    hr_error = F.interpolate(lr_error, scale_factor=scale,
                             mode='bicubic', align_corners=False,
                             recompute_scale_factor=False)
    print(f"HR_error shape:    {hr_error.shape}")
    print(f"HR_error mean:     {hr_error.mean().item():.6f}")
    print(f"HR_error std:      {hr_error.std().item():.6f}")

    # Stage2 input
    stage2_input = torch.cat([sr_stage1, hr_error], dim=1)
    print(f"Stage2 input shape: {stage2_input.shape}")

    # Full forward
    sr_final = model(lr)
    print(f"SR_final shape:    {sr_final.shape}")
    print(f"SR_final range:    [{sr_final.min():.4f}, {sr_final.max():.4f}]")

    criterion = torch.nn.L1Loss()
    loss = criterion(sr_final, hr)
    print(f"\nL1 loss:           {loss.item():.6f}")

    if torch.cuda.is_available():
        print(f"GPU memory:        {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU reserved:      {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    assert sr_stage1.shape == hr.shape
    assert sr_final.shape == hr.shape
    assert lr_recon.shape == lr.shape
    assert lr_error.shape == lr.shape
    assert hr_error.shape == hr.shape
    assert stage2_input.shape == (lr.shape[0], 6, hr.shape[2], hr.shape[3])

    print(f"\n--- All checks passed ---")


if __name__ == "__main__":
    main()
