#!/usr/bin/env python3
"""Dry-run for GC-MSR-RCAN: verify shapes and value ranges."""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.rcan import GCMSRRCAN
from utils.dataset import SRDataset
from torch.utils.data import DataLoader


def main():
    config_path = "configs/gc_msr_rcan_large50_cosine_x4_ucmerced_selected.yaml"
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
    model = GCMSRRCAN(
        in_channels=mp.get("in_channels", 3),
        out_channels=mp.get("out_channels", 3),
        num_features=mp.get("num_features", 64),
        num_resgroups=mp.get("num_resgroups", 8),
        num_resblocks=mp.get("num_resblocks", 8),
        reduction=mp.get("reduction", 16),
        scale=scale,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    gc_params = sum(p.numel() for p in model.gc_refine.parameters())
    refine_params = sum(p.numel() for p in model.refine.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"\nModel: GCMSRRCAN")
    print(f"  Total params:    {total_params:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  GC refine params:{gc_params:,}")
    print(f"  Deep refine params: {refine_params:,}")

    # Dry-run one batch
    batch = next(iter(train_loader))
    lr = batch["lr"].to(device)
    hr = batch["hr"].to(device)

    print(f"\n--- Dry-run ---")
    print(f"LR shape:          {lr.shape}")
    print(f"LR range:          [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"HR shape:          {hr.shape}")
    print(f"HR range:          [{hr.min():.4f}, {hr.max():.4f}]")

    # Forward through backbone
    sr_initial = model.backbone(lr)
    print(f"SR_initial shape:  {sr_initial.shape}")
    print(f"SR_initial range:  [{sr_initial.min():.4f}, {sr_initial.max():.4f}]")

    # Forward through GC refine
    gc_feat = model.gc_refine.up(sr_initial)
    print(f"GC feature shape:  {gc_feat.shape}")

    # Get attention mask
    B, C, H, W = gc_feat.shape
    mask = model.gc_refine.gc.mask_conv(gc_feat)
    mask_flat = mask.view(B, 1, -1)
    mask_softmax = torch.softmax(mask_flat, dim=-1)
    mask_sum = mask_softmax.sum(dim=-1)
    print(f"Attention mask shape: {mask.shape}")
    print(f"Attention mask softmax sum: {mask_sum.mean().item():.6f} (should be 1.0)")

    sr_gc = model.gc_refine(sr_initial)
    print(f"SR_gc shape:       {sr_gc.shape}")
    print(f"SR_gc range:       [{sr_gc.min():.4f}, {sr_gc.max():.4f}]")

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
    assert sr_final.shape == hr.shape, \
        f"Shape mismatch: sr_final {sr_final.shape} vs hr {hr.shape}"
    assert abs(mask_sum.mean().item() - 1.0) < 1e-4, \
        f"Attention mask sum should be 1.0, got {mask_sum.mean().item()}"

    print(f"\n--- All checks passed ---")


if __name__ == "__main__":
    main()
