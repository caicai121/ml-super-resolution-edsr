# 基于 RCAN 的遥感图像 ×4 超分辨率重建

使用 RCAN（Residual Channel Attention Network）及变体对 UC Merced Land Use 遥感场景图像进行 ×4 超分辨率重建，
通过系统性消融实验探索模块组合与训练策略，最终达到 Y+crop PSNR 31.40 dB。

## 数据集

UC Merced Land Use Dataset — 256×256 像素、1 英尺分辨率的航空遥感图像。

选取 7 类规则遥感场景作为训练/评估数据集：

| 类别 | 场景特征 |
|------|----------|
| beach | 大面积均匀纹理 |
| golfcourse | 大面积均匀纹理 |
| baseballdiamond | 规则几何结构 |
| runway | 规则几何结构 |
| freeway | 规则几何结构 |
| tenniscourt | 规则几何结构 |
| airplane | 混合（飞机 + 背景） |

### 数据划分

| 集合 | 图像数 | 每类 |
|------|--------|------|
| Train | 560 | 80 |
| Val | 70 | 10 |
| Test | 70 | 10 |

- HR 尺寸：256×256
- LR 尺寸（×4 Bicubic 下采样）：64×64

## 评价指标

采用双重指标（与 EDSR/RDN/RCAN 论文一致）：

- **RGB PSNR**：全图 RGB 三通道计算
- **Y+crop PSNR**：转 Y（亮度）通道，裁剪 4 像素边界后计算

Y+crop 是超分辨率领域的标准评价方式，排除边界插值误差，更准确反映模型重建质量。

## 模型与结果

### 总体指标（Test 集，UC Merced Selected 7 类）

| 方法 | Y+crop PSNR | RGB PSNR | 参数量 | 说明 |
|------|-------------|----------|--------|------|
| Bicubic ×4 | 29.75 | 28.04 | — | 基线上采样 |
| RCAN-small | 30.60 | 29.00 | 1.56M | 3g5b 从零训练 |
| MSR-RCAN-mid | 31.15 | 29.54 | 5.37M | 5g5b + MSRCAB + Deep Refine + Cosine |
| MSR-RCAN-large | 31.28 | 29.66 | 13.02M | 8g8b + MSRCAB + Deep Refine + Cosine |
| **Cascade-10（最优 50e）** | **31.40** | **29.77** | **13.76M** | **Stage2 ResBlock ×10 + 残差级联** |
| RCAN-pretrained | 32.52 | 30.48 | 15.59M | 公开预训练权重（参考） |

### Cascade 消融实验

| 模型 | Stage2 Blocks | 改动 | Y+crop PSNR | vs Baseline |
|------|--------------|------|-------------|-------------|
| MSR-RCAN-large (baseline) | 0 | — | 31.28 | — |
| Cascade-6 | 6 | 基础残差级联 | 31.35 | +0.07 |
| **Cascade-10** | **10** | **基础残差级联** | **31.40** | **+0.12** |
| BP-Cascade | 6 | 反投影误差反馈 | 31.34 | +0.06 |
| Gated-Cascade-10 | 10 | 空间门控 | 31.37 | +0.09 |
| Learnable-Scale-10 | 10 | 可学习残差缩放 (α=0.1048) | 31.39 | +0.11 |

**结论**：Cascade 残差级联有效（+0.12 dB），但额外引入反投影误差、空间门控或可学习残差缩放均未超越 Cascade-10。

### 无效消融实验

| 实验方向 | 结果 | 结论 |
|----------|------|------|
| Edge Branch (EG-MSR-RCAN) | 31.07 dB (-0.08) | 边缘分支特征冗余 |
| Edge Branch + Edge Loss | 30.93 dB (-0.22) | 边缘损失破坏像素精度 |
| AMSRCAB 自适应多尺度 | 31.09 dB (-0.06) | 与 MSRCAB 功能重叠 |
| RDRB 密集细化模块 | 31.12 dB (-0.03) | 容量不足 |
| Teacher Distillation (α=0.1) | 31.16 dB (-0.12) | 师生结构分布不匹配 |
| Global Context Block | 31.22 dB (-0.06) | 空间注意力无额外收益 |
| DMSR-RCAN (Dilated) | 30.77 dB (-0.04) | 参数减少但性能下降 |

### 消融小结

```
Bicubic (28.81) → RCAN-small (30.60, +1.79)
                → MSR-RCAN-mid (31.15, +0.55)
                → MSR-RCAN-large (31.28, +0.13)
                → Cascade-10 (31.40, +0.12) ★ 最优 50e
```

有效改进路径：多尺度特征提取 (MSRCAB) → 深度精炼 (Deep Refine) → 余弦退火 → 残差级联修正。

## 最优模型

**Cascade-MSR-RCAN-large Stage2-10**

- 主干：MSR-RCAN-large (8g8b, MSRCAB, Deep Refine v2)
- Stage2：Cascade Residual Correction Network (10 × BasicResidualBlock)
- 残差缩放：0.1
- 训练：50 epochs, Adam (lr=2e-4), CosineAnnealingLR, L1 Loss
- Y+crop PSNR：31.40 dB | RGB PSNR：29.77 dB

## 目录结构

```
├── models/              # 模型定义（rcan.py, srcnn.py, edsr.py）
├── utils/               # 数据集、指标、绘图工具
├── configs/             # 实验配置文件
├── scripts/             # 数据处理与评测脚本
├── data_final/          # 最终数据集（不入库）
├── results/             # 评测结果（不入库）
├── report_assets/       # 报告用图表（入库）
│   ├── figures/         # 对比图
│   └── tables/          # 指标表格
├── checkpoints/         # 模型权重（不入库）
├── plan_mser_rcan.md    # 实验计划与记录
└── README.md
```

## 运行环境

- Python 3.10.8 + PyTorch 2.1.2+cu121
- GPU: NVIDIA GeForce RTX 4090 D (24GB)
- 训练/推理在 AutoDL GPU 服务器上进行

## 快速复现

```bash
# Cascade-10 推理（使用最佳 checkpoint）
/root/miniconda3/bin/python test.py --config configs/cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml

# Bicubic ×4 baseline
/root/miniconda3/bin/python scripts/test_ucmerced_regular_bicubic.py
```
