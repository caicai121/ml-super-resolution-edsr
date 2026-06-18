#!/usr/bin/env python3
"""Dry-run: one batch through teacher + student, verify shapes and value ranges."""

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.rcan import RCAN, load_pretrained_rcan
from models.rcan import MSRRCANV2
from utils.dataset import SRDataset
from torch.utils.data import DataLoader


def main():
    config_path = "configs/msr_rcan_large50_cosine_distill_a01_x4_ucmerced_selected.yaml"
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

    # Student
    mp = cfg.get("model_params", {})
    student = MSRRCANV2(
        in_channels=mp.get("in_channels", 3),
        out_channels=mp.get("out_channels", 3),
        num_features=mp.get("num_features", 64),
        num_resgroups=mp.get("num_resgroups", 8),
        num_resblocks=mp.get("num_resblocks", 8),
        reduction=mp.get("reduction", 16),
        scale=scale,
    ).to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\nStudent: MSRRCANV2 8g8b, params={student_params:,}")

    # Teacher
    teacher = RCAN(
        in_channels=3, out_channels=3,
        num_features=64,
        num_resgroups=10, num_resblocks=20,
        reduction=16, scale=scale,
    )
    teacher = load_pretrained_rcan(
        cfg["distillation"]["teacher_checkpoint"], teacher, device
    )
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: RCAN pretrained, params={teacher_params:,}")

    # Dry-run one batch
    batch = next(iter(train_loader))
    lr = batch["lr"].to(device)
    hr = batch["hr"].to(device)

    print(f"\n--- Dry-run ---")
    print(f"LR shape:          {lr.shape}")
    print(f"LR range:          [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"HR shape:          {hr.shape}")
    print(f"HR range:          [{hr.min():.4f}, {hr.max():.4f}]")

    with torch.no_grad():
        sr_teacher = teacher(lr)

    sr_student = student(lr)

    print(f"SR_student shape:  {sr_student.shape}")
    print(f"SR_student range:  [{sr_student.min():.4f}, {sr_student.max():.4f}]")
    print(f"SR_teacher shape:  {sr_teacher.shape}")
    print(f"SR_teacher range:  [{sr_teacher.min():.4f}, {sr_teacher.max():.4f}]")

    criterion = torch.nn.L1Loss()
    alpha = cfg["distillation"]["alpha"]

    l1_loss = criterion(sr_student, hr)
    distill_loss = criterion(sr_student, sr_teacher)
    total_loss = l1_loss + alpha * distill_loss

    print(f"\nL1 loss:           {l1_loss.item():.6f}")
    print(f"Distill loss:      {distill_loss.item():.6f}")
    print(f"Total loss:        {total_loss.item():.6f} (alpha={alpha})")

    print(f"\nTeacher eval mode: {not teacher.training}")
    print(f"Teacher frozen:    {all(not p.requires_grad for p in teacher.parameters())}")

    if torch.cuda.is_available():
        print(f"GPU memory:        {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU reserved:      {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Verify shapes match
    assert sr_student.shape == sr_teacher.shape, \
        f"Shape mismatch: student {sr_student.shape} vs teacher {sr_teacher.shape}"
    assert sr_student.shape == hr.shape, \
        f"Shape mismatch: student {sr_student.shape} vs hr {hr.shape}"

    print(f"\n--- All checks passed ---")


if __name__ == "__main__":
    main()
