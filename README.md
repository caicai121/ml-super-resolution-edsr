# 基于 RCAN 的规则遥感场景图像 ×4 超分辨率重建系统

机器学习大作业：使用 RCAN（Residual Channel Attention Network）对规则遥感场景图像进行 ×4 超分辨率重建。

## 项目目标

在 UC Merced Land Use 数据集的规则场景子集上，使用预训练 RCAN 模型完成 ×4 超分辨率任务，
通过 Bicubic baseline 对比验证深度学习方法的有效性，并以 Y 通道 PSNR/SSIM 作为标准评价指标。

## 技术路线

1. **Baseline — Bicubic 插值**：作为最基本的对照。
2. **主干模型 — RCAN ×4 pretrained**：预训练权重来自 [yulunzhang/RCAN](https://github.com/yulunzhang/RCAN)，Residual Channel Attention Network，约 1560 万参数。
3. **可选改进 — Self-Ensemble**：测试是否能进一步提升指标。

## 数据集

**最终数据集：UC Merced Land Use — 规则遥感场景子集 (regular v2)**

| 类别 | 数量 | 说明 |
|------|------|------|
| airplane | 100 | 机场场景 |
| baseball_diamond | 100 | 棒球场 |
| golf_course | 100 | 高尔夫球场 |
| runway | 100 | 跑道 |
| tennis_court | 100 | 网球场 |

- 总计：500 张
- HR 尺寸：256×256
- LR 尺寸（×4 Bicubic 下采样）：64×64
- 主题统一，场景结构规则，适合展示 ×4 超分辨率效果

## 最终结果

| 方法 | RGB PSNR | RGB SSIM | Y+crop PSNR | Y+crop SSIM |
|------|----------|----------|-------------|-------------|
| Bicubic ×4 | 27.79 | 0.7392 | 29.20 | 0.7730 |
| **RCAN ×4 pretrained** | **30.48** | **0.8166** | **32.08** | **0.8473** |
| vs Bicubic | +2.69 | +0.0775 | +2.88 | +0.0743 |

### 按类别 RCAN Y+crop PSNR

| 类别 | PSNR |
|------|------|
| baseball_diamond | 33.81 dB |
| golf_course | 33.67 dB |
| runway | 32.23 dB |
| airplane | 30.39 dB |
| tennis_court | 30.32 dB |

## 目录结构

```
├── models/              # 模型定义（rcan.py, srcnn.py, edsr.py）
├── utils/               # 数据集、指标、绘图工具
├── configs/             # 实验配置文件
├── scripts/             # 数据处理与评测脚本
├── data/                # DIV2K 数据（symlink）
├── data_experiments/    # 实验数据（UC Merced 等，不入库）
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

```bash
cd ~/Code/ml-super-resolution-edsr
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
python scripts/check_env.py
```

## 评测说明

标准评测协议（与 EDSR/RDN/RCAN 论文一致）：
- 将 RGB 转为 Y（亮度）通道
- 裁剪边界像素（crop_border = scale = 4）
- 计算 Y 通道 PSNR 和 SSIM

## 快速复现

```bash
# RCAN ×4 推理（使用预训练权重）
/root/miniconda3/bin/python scripts/test_rcan_final.py

# Bicubic ×4 baseline
/root/miniconda3/bin/python scripts/test_ucmerced_regular_bicubic.py
```
