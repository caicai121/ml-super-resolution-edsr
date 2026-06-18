# 基于 RCAN 的遥感图像超分辨率重建（×2 / ×4）

使用 RCAN（Residual Channel Attention Network）及自研 Cascade 变体对 UC Merced Land Use 遥感场景图像进行超分辨率重建，通过系统性消融实验探索模块组合与训练策略。

**核心贡献**：提出 Cascade-MSR-RCAN，通过两级残差级联修正主干网络输出，在 ×4 任务达到 Y+crop PSNR 31.40 dB（50 epoch）；×2 任务达到 37.29 dB。

## 数据集

UC Merced Land Use Dataset — 256×256 像素、1 英尺分辨率航空遥感图像。选取 7 类规则遥感场景：

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

- HR：256×256；LR ×4：64×64（Bicubic 下采样）

## 评价指标

- **RGB PSNR/SSIM**：全图三通道计算
- **Y+crop PSNR/SSIM**：Y 通道 + crop_border=scale（超分辨率标准评价）

## 模型与结果（×4，50 epoch 主表）

| 方法 | Y+crop PSNR | RGB PSNR | 参数量 | 说明 |
|------|-------------|----------|--------|------|
| Bicubic ×4 | 29.75 | 28.04 | — | 基线上采样 |
| RCAN-small | 30.60 | 29.00 | 1.56M | 3g5b 从零训练 |
| MSR-RCAN-mid | 31.15 | 29.54 | 5.37M | 5g5b + MSRCAB + Deep Refine + Cosine |
| MSR-RCAN-large | 31.28 | 29.66 | 13.02M | 8g8b + MSRCAB + Deep Refine + Cosine |
| **Cascade-10（最优）** | **31.40** | **29.77** | **13.76M** | Stage2 ResBlock ×10 + 残差级联 |

## Cascade 消融实验（×4, 50 epoch）

| 模型 | Stage2 | Y+crop PSNR | vs Baseline |
|------|--------|-------------|-------------|
| MSR-RCAN-large (baseline) | 0 | 31.28 | — |
| Cascade-6 | 6 | 31.35 | +0.07 |
| **Cascade-10** | **10** | **31.40** | **+0.12** |
| BP-Cascade (反投影) | 6 | 31.34 | +0.06 |
| Gated-Cascade | 10 | 31.37 | +0.09 |
| Learnable-Scale | 10 | 31.39 | +0.11 |

**结论**：Cascade 残差级联有效（+0.12 dB），但 BP、Gated、Learnable Scale 等变体未超越基础级联。

## 扩展训练：200 epoch 上限实验

为观察 Cascade-10 ×4 在更长训练预算下的性能上限，额外训练 200 epoch（**不替代 50 epoch 主表结果**）：

| 训练设置 | RGB PSNR | RGB SSIM | Y+crop PSNR | Y+crop SSIM | vs 50e |
|----------|----------|----------|-------------|-------------|--------|
| Stage2-10 ×4 50e | 29.77 | 0.7931 | 31.40 | 0.8247 | — |
| Stage2-10 ×4 200e | **30.27** | **0.8079** | **31.86** | **0.8374** | **+0.46 dB** |

模型在 200 epoch 后仍在缓慢提升，但收益递减明显（后 150 epoch 仅贡献 +0.46 dB）。

## ×2 补充实验

### ×2 多模型对比（50 epoch）

| 模型 | Y+crop PSNR | RGB PSNR |
|------|-------------|----------|
| Bicubic ×2 | 34.80 | 33.26 |
| RCAN-small ×2 | 36.87 | — |
| MSR-RCAN-mid ×2 | 37.11 | — |
| MSR-RCAN-large ×2 | 37.25 | — |
| Cascade-S6 ×2 | 37.23 | — |
| **Cascade-S10 ×2** | **37.29** | **35.37** |

### ×2 vs ×4 对比

| 模型 | Scale | Y+crop PSNR |
|------|-------|-------------|
| Bicubic | ×4 / ×2 | 29.75 / 34.80 |
| Cascade-10 | ×4 / ×2 | 31.40 / 37.29 |

×2 任务信息丢失更少（4× vs 16× 像素压缩），PSNR 显著高于 ×4。Cascade 残差级联在两种倍率下均有效。

## 无效消融方向（×4, 50 epoch）

| 方向 | Y+crop PSNR | vs Baseline | 结论 |
|------|-------------|-------------|------|
| Edge Branch (EG-MSR-RCAN) | 31.07 | -0.08 | 特征冗余 |
| Edge Branch + Edge Loss | 30.93 | -0.22 | 破坏像素精度 |
| AMSRCAB (自适应多尺度) | 31.09 | -0.06 | 与 MSRCAB 重叠 |
| RDRB (密集细化) | 31.12 | -0.03 | 容量不足 |
| DMSR-RCAN (Dilated) | 30.77 | -0.04 | 参数减少性能降 |
| Cosine + Augment | 31.00 | -0.15 | 规则场景不宜增强 |

## 最优模型

**Cascade-MSR-RCAN-large Stage2-10**（50 epoch）

- 主干：MSR-RCAN-large (8g8b, MSRCAB, Deep Refine v2)
- Stage2：Cascade Residual Correction (10 × BasicResidualBlock, α=0.1)
- 训练：Adam (lr=2e-4), CosineAnnealingLR (→1e-6), L1 Loss
- ×4 Y+crop PSNR：31.40 dB | ×2 Y+crop PSNR：37.29 dB

## 目录结构

```
├── models/              # 模型定义（rcan.py, srcnn.py, edsr.py）
├── utils/               # 数据集、指标、绘图工具
├── configs/             # 实验配置文件（×2/×4/消融）
├── scripts/             # 数据预处理、评测脚本
├── tools/               # 分析工具
├── data_final/          # 最终数据集（不入库）
├── results/             # 评测结果（不入库）
├── checkpoints/         # 模型权重（不入库）
├── report_assets/       # 报告用图表（不入库）
└── README.md
```

## 运行环境

- Python 3.10.8 + PyTorch 2.1.2+cu121
- GPU: NVIDIA GeForce RTX 4090 D (24GB)
- 训练/推理在 AutoDL GPU 服务器上进行

## 快速复现

```bash
# 安装依赖
pip install -r requirements.txt

# Cascade-10 ×4 训练（50 epoch）
python train.py --config configs/cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml

# Cascade-10 ×4 测试
python test.py --config configs/cascade_msr_rcan_large_s10_50_cosine_x4_ucmerced_selected.yaml

# Cascade-10 ×2 测试
python test.py --config configs/cascade_msr_rcan_large_s10_50_cosine_x2_ucmerced_selected.yaml
```

## 引用

```bibtex
@misc{ml-sr-edsr,
  author = {caicai121},
  title  = {Cascade-MSR-RCAN for Remote Sensing Image Super-Resolution},
  year   = {2026},
  url    = {https://github.com/caicai121/ml-super-resolution-edsr}
}
```
