#!/usr/bin/env python3
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/root/Code/ml-super-resolution-edsr")
RESULTS = BASE / "results"
FIG_OUT = BASE / "report_assets" / "figures" / "final_report"
TABLE_OUT = BASE / "report_assets" / "tables" / "final_report"
FIG_OUT.mkdir(parents=True, exist_ok=True)
TABLE_OUT.mkdir(parents=True, exist_ok=True)

def get_field(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"None of {names} found in {list(row.keys())}")

# STEP 1: Already done in previous run, skip to save time
# Just read existing table1
with open(TABLE_OUT / "final_table1_metrics.csv") as f:
    table1 = list(csv.DictReader(f))
print("Loaded table1:", len(table1), "models")

print("\n" + "=" * 60)
print("STEP 2: Select best visual comparison samples")
print("=" * 60)

cascade_metrics = RESULTS / "cascade_msr_rcan_large_s10_50_cosine_x4" / "metrics.csv"
samples = []
with open(cascade_metrics) as f:
    reader = csv.DictReader(f)
    for row in reader:
        y_psnr = float(get_field(row, "y_psnr", "psnr_y"))
        if y_psnr > 31.0:
            cls = row["image"].split("_")[0] if "_" in row["image"] else "unknown"
            samples.append({
                "image": row["image"], "class": cls,
                "rgb_psnr": float(get_field(row, "rgb_psnr", "psnr_rgb")),
                "rgb_ssim": float(get_field(row, "rgb_ssim", "ssim_rgb")),
                "y_psnr": y_psnr,
                "y_ssim": float(get_field(row, "y_ssim", "ssim_y")),
            })

samples.sort(key=lambda x: x["y_psnr"], reverse=True)
by_class = defaultdict(list)
for s in samples:
    by_class[s["class"]].append(s)

candidates = []
for target in ["runway", "tenniscourt", "airplane"]:
    if target in by_class:
        best = by_class[target][0]
        candidates.append({
            "class": target, "image": best["image"],
            "y_psnr": best["y_psnr"], "y_ssim": best["y_ssim"],
            "rgb_psnr": best["rgb_psnr"], "recommended": True
        })
        print(f"{target}: {best['image']} y_psnr={best['y_psnr']:.2f} [RECOMMENDED]")
    else:
        alt = [s for s in samples if target in s["image"].lower()]
        if alt:
            best = max(alt, key=lambda x: x["y_psnr"])
            candidates.append({
                "class": target, "image": best["image"],
                "y_psnr": best["y_psnr"], "y_ssim": best["y_ssim"],
                "rgb_psnr": best["rgb_psnr"], "recommended": False
            })
            print(f"{target}: {best['image']} y_psnr={best['y_psnr']:.2f} [ALT]")
        else:
            print(f"{target}: NOT FOUND")

with open(TABLE_OUT / "selected_visual_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["class", "image", "y_psnr", "y_ssim", "rgb_psnr", "recommended"])
    w.writeheader()
    w.writerows(candidates)
print(f"Saved: {TABLE_OUT / 'selected_visual_candidates.csv'}")

print("\n" + "=" * 60)
print("STEP 3: Generate plots")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["model"] for r in table1]
    values = [float(r["y_psnr"]) if r["y_psnr"] != "N/A" else 0 for r in table1]
    colors = ['gray', '#3498db', '#2ecc71', '#27ae60', '#e67e22', '#e74c3c', '#9b59b6']
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=30, color='red', linestyle='--', linewidth=1, label='Course requirement (30 dB)')
    bars[5].set_edgecolor('gold')
    bars[5].set_linewidth(2)
    ax.set_ylabel('Y+crop PSNR / dB', fontsize=12)
    ax.set_title('Super-Resolution Model Comparison', fontsize=14)
    ax.set_ylim([28, 33.5])
    ax.legend(loc='upper left')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fig2_path = FIG_OUT / 'fig2_ycrop_psnr_bar.png'
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig2_path}")

    train_log = RESULTS / "cascade_msr_rcan_large_s10_50_cosine_x4" / "train_log.csv"
    if train_log.exists():
        epochs, val_y_psnr = [], []
        with open(train_log) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                val_y_psnr.append(float(row["val_y_psnr"]))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, val_y_psnr, 'b-', linewidth=1.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Y+crop PSNR / dB', fontsize=12)
        ax.set_title('Cascade-MSR-RCAN Stage2-10 Validation PSNR', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig7_path = FIG_OUT / 'fig7_val_psnr_curve.png'
        plt.savefig(fig7_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig7_path}")

src_loss = RESULTS / "cascade_msr_rcan_large_s10_50_cosine_x4" / "loss_curve.png"
if src_loss.exists():
    dst_loss = FIG_OUT / 'fig6_loss_curve.png'
    shutil.copy(src_loss, dst_loss)
    print(f"Copied: {dst_loss}")

print("\n" + "=" * 60)
print("STEP 4: Generate visual comparisons")
print("=" * 60)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: PIL not available")

if HAS_PIL and HAS_MPL:
    model_dirs = {
        'Bicubic': RESULTS / 'bicubic_x4' / 'images',
        'RCAN-small': RESULTS / 'rcan_small_x4' / 'images',
        'Cascade-10': RESULTS / 'cascade_msr_rcan_large_s10_50_cosine_x4' / 'images',
    }
    hr_dir = BASE / 'data_final' / 'ucmerced_selected' / 'test' / 'HR'
    lr_dir = BASE / 'data_final' / 'ucmerced_selected' / 'test' / 'LR_x4'

    fig_idx = 3
    for target in ['runway', 'tenniscourt', 'airplane']:
        sel = [c for c in candidates if c["class"] == target]
        if not sel:
            print(f"SKIP: {target} not found")
            continue
        img_name = sel[0]["image"]
        print(f"Processing {target}: {img_name}")

        paths = {
            'LR (x4)': lr_dir / img_name,
            'Bicubic': model_dirs['Bicubic'] / img_name,
            'RCAN-small': model_dirs['RCAN-small'] / img_name,
            'Cascade-10': model_dirs['Cascade-10'] / img_name,
            'HR': hr_dir / img_name,
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
            print(f"  SKIP: images missing for {img_name}")
            continue

        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]
        for ax, img, label in zip(axes, images, labels):
            ax.imshow(img)
            ax.set_title(label, fontsize=12)
            ax.axis('off')

        info = sel[0]
        fig.suptitle(f'{target.title()} - Cascade-10: Y+crop PSNR={info["y_psnr"]:.2f} dB, SSIM={info["y_ssim"]:.4f}',
                     fontsize=14, y=0.98)
        plt.tight_layout()
        fig_path = FIG_OUT / f'fig{fig_idx}_{target}_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {fig_path}")
        fig_idx += 1

print("\n" + "=" * 60)
print("STEP 5: Generate manifest")
print("=" * 60)

manifest = []
for f in sorted(FIG_OUT.glob('*')):
    manifest.append(f"FIG: {f.name} ({f.stat().st_size / 1024:.1f} KB)")
for f in sorted(TABLE_OUT.glob('*')):
    manifest.append(f"TABLE: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

manifest_path = BASE / "report_assets" / "final_report_manifest.txt"
with open(manifest_path, "w") as f:
    f.write("\n".join(manifest))
print("\n".join(manifest))
print(f"\nSaved: {manifest_path}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
