#!/usr/bin/env python3
import sys
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

BASE = Path("/root/Code/ml-super-resolution-edsr")
FIG_OUT = BASE / "report_assets" / "figures" / "final_report"
SR_DIR = FIG_OUT / "sr_images"
HR_DIR = BASE / "data_final" / "ucmerced_selected" / "test" / "HR"
LR_DIR = BASE / "data_final" / "ucmerced_selected" / "test" / "LR_x4"

# Load candidate info
candidates = []
with open(BASE / "report_assets" / "tables" / "final_report" / "selected_visual_candidates.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        candidates.append(row)

fig_idx = 3
for candidate in candidates:
    target = candidate["class"]
    img_name = candidate["image"]
    y_psnr = float(candidate["y_psnr"])
    y_ssim = float(candidate["y_ssim"])
    
    print(f"Processing {target}: {img_name}")
    
    paths = {
        'LR (x4)': LR_DIR / img_name,
        'Bicubic': SR_DIR / "bicubic" / img_name,
        'RCAN-small': SR_DIR / "rcan_small" / img_name,
        'Cascade-10': SR_DIR / "cascade_10" / img_name,
        'HR': HR_DIR / img_name,
    }
    
    all_exist = True
    images = []
    labels = []
    for label, path in paths.items():
        if path.exists():
            images.append(Image.open(path).convert('RGB'))
            labels.append(label)
        else:
            print(f"  MISSING: {label} -> {path}")
            all_exist = False
    
    if not all_exist or len(images) < 5:
        print(f"  SKIP: images missing")
        continue
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        ax.axis('off')
    
    fig.suptitle(f'{target.title()} - Cascade-10: Y+crop PSNR={y_psnr:.2f} dB, SSIM={y_ssim:.4f}',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    fig_path = FIG_OUT / f'fig{fig_idx}_{target}_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")
    fig_idx += 1

print("\nDONE!")
