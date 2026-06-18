#!/usr/bin/env python3
"""Dry-run for Gated-Cascade-MSR-RCAN: verify shapes, gate stats."""

import sys
import torch
import yaml

sys.path.insert(0, "/root/Code/ml-super-resolution-edsr")

from models.rcan import GatedCascadeMSRRCAN
from utils.dataset import SRDataset
from torch.utils.data import DataLoader


def main():
    config_path = "configs/gated_cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml"
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
    model = GatedCascadeMSRRCAN(
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
    gate_params = sum(p.numel() for p in model.gate.parameters())
    print(f"\nModel: GatedCascadeMSRRCAN")
    print(f"  Total params:    {total_params:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Refine params:   {refine_params:,}")
    print(f"  Cascade params:  {cascade_params:,}")
    print(f"  Gate params:     {gate_params:,}")

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
    print(f"SR_stage1 range:   [{sr_stage1.min():.4f}, {sr_stage1.max():.4f}]")

    # Get residual from cascade
    feat = model.cascade.entry(sr_stage1)
    feat = model.cascade.blocks(feat)
    residual_stage2 = model.cascade.exit_conv(feat)
    print(f"residual_stage2 shape: {residual_stage2.shape}")
    print(f"residual_stage2 mean:  {residual_stage2.mean().item():.6f}")
    print(f"residual_stage2 std:   {residual_stage2.std().item():.6f}")

    # Gate
    gate_input = torch.cat([sr_stage1, residual_stage2], dim=1)
    gate = model.gate(gate_input)
    print(f"gate shape:        {gate.shape}")
    print(f"gate min:          {gate.min().item():.4f}")
    print(f"gate max:          {gate.max().item():.4f}")
    print(f"gate mean:         {gate.mean().item():.4f}")

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

    # Shape checks
    assert sr_stage1.shape == hr.shape
    assert sr_final.shape == hr.shape
    assert residual_stage2.shape == hr.shape
    assert gate.shape == (lr.shape[0], 3, hr.shape[2], hr.shape[3])
    assert not torch.isnan(gate).any(), "Gate contains NaN!"

    print(f"\n--- All checks passed ---")


if __name__ == "__main__":
    main()
