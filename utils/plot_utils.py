#!/usr/bin/env python3
"""Plotting utilities for training curves and comparison figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(train_losses, val_psnrs, save_path):
    """Plot training loss and validation PSNR curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_psnrs, "r-", label="Val PSNR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Validation PSNR")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_comparison(lr_img, bicubic_img, sr_img, hr_img, save_path, title="", scale=4):
    """Create 4-column comparison: LR / Bicubic / SR / HR."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(lr_img)
    axes[0].set_title(f"LR (x{scale} down)")
    axes[0].axis("off")

    axes[1].imshow(bicubic_img)
    axes[1].set_title("Bicubic")
    axes[1].axis("off")

    axes[2].imshow(sr_img)
    axes[2].set_title("SR")
    axes[2].axis("off")

    axes[3].imshow(hr_img)
    axes[3].set_title("HR (GT)")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_comparison_5col(lr_img, bicubic_img, model_a_img, model_b_img, hr_img,
                           save_path, title="", scale=4,
                           model_a_name="Model A", model_b_name="Model B"):
    """Create 5-column comparison: LR / Bicubic / Model A / Model B / HR."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(lr_img)
    axes[0].set_title(f"LR (x{scale} down)")
    axes[0].axis("off")

    axes[1].imshow(bicubic_img)
    axes[1].set_title("Bicubic")
    axes[1].axis("off")

    axes[2].imshow(model_a_img)
    axes[2].set_title(model_a_name)
    axes[2].axis("off")

    axes[3].imshow(model_b_img)
    axes[3].set_title(model_b_name)
    axes[3].axis("off")

    axes[4].imshow(hr_img)
    axes[4].set_title("HR (GT)")
    axes[4].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
