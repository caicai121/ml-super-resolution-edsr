# 基于 RCAN 的遥感图像 ×4 超分辨率重建

使用 RCAN（Residual Channel Attention Network）对 UC Merced Land Use 遥感场景图像进行 ×4 超分辨率重建，
对比 Bicubic 插值基线与从零训练的 RCAN-small 模型，并以 Y 通道 PSNR/SSIM 作为标准评价指标。

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

### 总体指标（Test 集）

| 方法 | RGB PSNR | RGB SSIM | Y+crop PSNR | Y+crop SSIM |
|------|----------|----------|-------------|-------------|
| Bicubic ×4 | 27.41 | 0.7358 | 28.81 | 0.7675 |
| **RCAN-small（从零训练）** | **29.00** | **0.7729** | **30.60** | **0.8060** |
| vs Bicubic | +1.59 | +0.0371 | +1.79 | +0.0385 |
| RCAN ×4 pretrained | 30.48 | 0.8166 | 32.08 | 0.8473 |
| vs Bicubic | +3.07 | +0.0808 | +3.27 | +0.0798 |

- Bicubic 基线来自同一批图像的双三次插值上采样
- RCAN ×4 pretrained 使用预训练权重，在 regular v2（5 类）上评测

### 按类别 Y+crop PSNR（Test 集）

| 类别 | Bicubic | RCAN-small | Gain |
|------|---------|------------|------|
| beach | 35.74 | 36.28 | +0.53 |
| golfcourse | 32.16 | 32.90 | +0.73 |
| baseballdiamond | 31.35 | 32.15 | +0.80 |
| runway | 28.21 | 29.67 | +1.47 |
| tenniscourt | 27.07 | 27.79 | +0.72 |
| freeway | 26.97 | 27.75 | +0.78 |
| airplane | 26.76 | 27.67 | +0.91 |

runway 类增益最大（+1.47 dB），说明 RCAN-small 对规则线性结构的恢复能力明显优于 Bicubic。

## RCAN-small 模型参数

| 参数 | 值 |
|------|----|
| num_features | 64 |
| num_resgroups | 3 |
| num_resblocks | 5 |
| reduction | 16 |
| 总参数量 | ~1M |
| 训练轮数 | 50 |
| 最佳 checkpoint | epoch 49 |
| val Y+crop PSNR | 30.12 dB |

## 目录结构

```
├── models/              # 模型定义（rcan.py, srcnn.py, edsr.py）
├── utils/               # 数据集、指标、绘图工具
├── configs/             # 实验配置文件
├── scripts/             # 数据处理与评测脚本
├── data_final/          # 最终数据集（不入库）
├── data_experiments/    # 其他实验数据（不入库）
├── results/             # 评测结果（不入库）
├── report_assets/       # 报告用图表（入库）
│   ├── figures/         # 对比图
│   └── tables/          # 指标表格
├── checkpoints/         # 模型权重（不入库）
└── README.md
```

## 运行环境

- Python 3.10.8 + PyTorch 2.1.2+cu121
- GPU: NVIDIA GeForce RTX 4090 D (24GB)
- 训练/推理在 AutoDL GPU 服务器上进行

## 快速复现

```bash
# RCAN-small 推理（使用最佳 checkpoint）
/root/miniconda3/bin/python scripts/test_rcan_small.py

# Bicubic ×4 baseline
/root/miniconda3/bin/python scripts/test_ucmerced_regular_bicubic.py
```

## 后续计划

计划引入 MSER-RCAN（多尺度增强残差通道注意力网络），在 RCAN 基础上改进特征提取能力，
进一步提升遥感场景的超分辨率重建质量。
