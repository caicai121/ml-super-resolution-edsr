#!/usr/bin/env python3
"""Update CSV tables and plan with Cascade ablation results."""

import csv
import os

# 1. Update final_model_comparison.csv
csv_path = "report_assets/tables/final_model_comparison.csv"
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

# Remove old GC row and add Cascade results
new_rows = []
for row in rows:
    if "GC-MSR-RCAN" in row["model"]:
        continue  # remove GC
    new_rows.append(row)

# Add Cascade results before RCAN-pretrained
cascade_rows = [
    {"model": "Cascade-MSR-RCAN-large-6", "y_psnr_y": "31.35", "rgb_psnr": "29.72", "rgb_ssim": "", "params_m": "13.67", "epochs": "50", "notes": "Stage2 ResBlock x6"},
    {"model": "Cascade-MSR-RCAN-large-10", "y_psnr_y": "31.40", "rgb_psnr": "29.77", "rgb_ssim": "", "params_m": "13.76", "epochs": "50", "notes": "Stage2 ResBlock x10 (BEST 50e)"},
    {"model": "BP-Cascade-MSR-RCAN-large", "y_psnr_y": "31.34", "rgb_psnr": "29.72", "rgb_ssim": "", "params_m": "13.47", "epochs": "50", "notes": "Back-Projection error feedback"},
    {"model": "Gated-Cascade-MSR-RCAN-large-10", "y_psnr_y": "31.37", "rgb_psnr": "29.74", "rgb_ssim": "", "params_m": "13.77", "epochs": "50", "notes": "GateNet spatial gating"},
    {"model": "Learnable-Scale-Cascade-10", "y_psnr_y": "31.39", "rgb_psnr": "29.76", "rgb_ssim": "", "params_m": "13.76", "epochs": "50", "notes": "Learnable alpha=0.1048"},
]

# Insert before RCAN-pretrained
final_rows = []
for row in new_rows:
    if "RCAN-pretrained" in row["model"]:
        for cr in cascade_rows:
            final_rows.append(cr)
    final_rows.append(row)

# Update MSR-RCAN-large note
for row in final_rows:
    if "MSR-RCAN-large-50-cosine" in row["model"]:
        row["notes"] = "8g8b+MSRCAB+Deep Refine+Cosine LR"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)
print(f"Updated {csv_path}")

# 2. Create cascade_ablation_summary.csv
cascade_csv = "report_assets/tables/cascade_ablation_summary.csv"
cascade_fieldnames = ["model", "stage2_blocks", "modification", "y_psnr_y", "rgb_psnr", "delta_vs_baseline", "conclusion"]
cascade_data = [
    {"model": "MSR-RCAN-large (baseline)", "stage2_blocks": "0", "modification": "none", "y_psnr_y": "31.28", "rgb_psnr": "29.66", "delta_vs_baseline": "0.00", "conclusion": "baseline"},
    {"model": "Cascade-6", "stage2_blocks": "6", "modification": "basic cascade", "y_psnr_y": "31.35", "rgb_psnr": "29.72", "delta_vs_baseline": "+0.07", "conclusion": "effective"},
    {"model": "Cascade-10", "stage2_blocks": "10", "modification": "basic cascade", "y_psnr_y": "31.40", "rgb_psnr": "29.77", "delta_vs_baseline": "+0.12", "conclusion": "BEST"},
    {"model": "BP-Cascade", "stage2_blocks": "6", "modification": "back-projection error", "y_psnr_y": "31.34", "rgb_psnr": "29.72", "delta_vs_baseline": "+0.06", "conclusion": "no improvement vs Cascade-6"},
    {"model": "Gated-Cascade-10", "stage2_blocks": "10", "modification": "GateNet spatial gating", "y_psnr_y": "31.37", "rgb_psnr": "29.74", "delta_vs_baseline": "+0.09", "conclusion": "gate suppresses useful residual"},
    {"model": "Learnable-Scale-Cascade-10", "stage2_blocks": "10", "modification": "learnable alpha (0.1->0.1048)", "y_psnr_y": "31.39", "rgb_psnr": "29.76", "delta_vs_baseline": "+0.11", "conclusion": "fixed 0.1 near optimal"},
]

with open(cascade_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cascade_fieldnames)
    writer.writeheader()
    writer.writerows(cascade_data)
print(f"Created {cascade_csv}")

print("DONE")
