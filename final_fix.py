#!/usr/bin/env python3
"""Final fix: unify all numbers to selected 7-class test set (70 images)."""

# 1. Fix plan_mser_rcan.md Phase 3
print("=== 1. Fix plan_mser_rcan.md Phase 3 ===")
with open("plan_mser_rcan.md", "r") as f:
    plan = f.read()

old_phase3 = """## Phase 3: DMSR-RCAN-small (Dilated Multi-Scale + Deep Refine)

**Status**: PLANNED

**Goal**: Replace 5x5 branch with 3x3 dilation=2 in MSRCAB for larger receptive field with fewer parameters.
**Keep**: Deep Refine v2 unchanged.
**Expected improvement**: Push Y+crop PSNR toward 30.90~31.00 dB."""

new_phase3 = """## Phase 3: DMSR-RCAN-small (Dilated Multi-Scale + Deep Refine)

**Status**: COMPLETED - NOT EFFECTIVE

**What changed**: Replace 5x5 branch with 3x3 dilation=2 in MSRCAB for larger receptive field with fewer parameters.
**Keep**: Deep Refine v2 unchanged.

**Results**:
- DMSR-RCAN: Y+crop 30.77 dB (vs MSR-RCAN-v2 30.81 dB, -0.04)
- Parameters: 2.38M (reduced from 3.36M)
- Dilated 3x3(d=2) achieves similar receptive field to 5x5 with fewer params, but slight performance drop

**Conclusion**: Dilated convolution reduces parameters but does not improve performance.
Not effective for this task. MSRCAB with 5x5 branch retained."""

plan = plan.replace(old_phase3, new_phase3)

with open("plan_mser_rcan.md", "w") as f:
    f.write(plan)
print("  Fixed Phase 3: PLANNED -> COMPLETED - NOT EFFECTIVE")

# 2. Fix README.md
print("\n=== 2. Fix README.md ===")
with open("README.md", "r") as f:
    readme = f.read()

# Fix Bicubic numbers: 28.81/27.41 -> 29.75/28.04
readme = readme.replace("| Bicubic ×4 | 28.81 | 27.41 |", "| Bicubic ×4 | 29.75 | 28.04 |")

# Fix RCAN-pretrained: 32.08 -> 32.52
readme = readme.replace("| RCAN-pretrained | 32.08 | 30.48 |", "| RCAN-pretrained | 32.52 | 30.48 |")

# Fix invalid ablation numbers
old_invalid = """| 实验方向 | 结果 | 结论 |
|----------|------|------|
| Teacher Distillation (α=0.1) | 31.16 dB (-0.12) | 师生结构分布不匹配 |
| Global Context Block | 31.22 dB (-0.06) | 空间注意力无额外收益 |
| Edge Branch + Edge Loss | 31.22 dB (-0.06) | 边缘监督干扰主任务 |
| AMSRCAB 多尺度注意力 | 31.22 dB (-0.06) | 与 MSRCAB 功能重叠 |
| RDRB 递归残差 | 31.25 dB (-0.03) | 容量不足 |"""

new_invalid = """| 实验方向 | 结果 | 结论 |
|----------|------|------|
| Edge Branch (EG-MSR-RCAN) | 31.07 dB (-0.08) | 边缘分支特征冗余 |
| Edge Branch + Edge Loss | 30.93 dB (-0.22) | 边缘损失破坏像素精度 |
| AMSRCAB 自适应多尺度 | 31.09 dB (-0.06) | 与 MSRCAB 功能重叠 |
| RDRB 密集细化模块 | 31.12 dB (-0.03) | 容量不足 |
| Teacher Distillation (α=0.1) | 31.16 dB (-0.12) | 师生结构分布不匹配 |
| Global Context Block | 31.22 dB (-0.06) | 空间注意力无额外收益 |
| DMSR-RCAN (Dilated) | 30.77 dB (-0.04) | 参数减少但性能下降 |"""

readme = readme.replace(old_invalid, new_invalid)

with open("README.md", "w") as f:
    f.write(readme)
print("  Fixed Bicubic: 28.81 -> 29.75")
print("  Fixed RCAN-pretrained: 32.08 -> 32.52")
print("  Fixed invalid ablation table: 7 rows with correct numbers")

# 3. Verify final_model_comparison.csv Bicubic row
print("\n=== 3. Verify final_model_comparison.csv ===")
import csv
with open("report_assets/tables/final_model_comparison.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    if "Bicubic" in row["model"]:
        print(f"  Bicubic: y_psnr={row['y_psnr_y']}, rgb={row['rgb_psnr']}")
        if row["y_psnr_y"] != "29.75":
            print("  WARNING: Bicubic Y+crop should be 29.75!")
    if "RCAN-pretrained" in row["model"]:
        print(f"  RCAN-pretrained: y_psnr={row['y_psnr_y']}, rgb={row['rgb_psnr']}")
        if row["y_psnr_y"] != "32.52":
            print("  WARNING: RCAN-pretrained should be 32.52!")

print("\nDONE")
