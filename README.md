# ml-super-resolution-edsr

机器学习大作业：基于 EDSR 的单图像超分辨率重建（Single Image Super-Resolution, SISR）。

## 项目目标

在 DIV2K 数据集上，实现并训练一个轻量化的 EDSR 模型用于 ×4 超分辨率任务，
与经典 baseline 进行系统对比，并通过自定义改进模块进一步提升 PSNR / SSIM。

## 技术路线

按 baseline → 主干 → 改进的顺序逐步推进，每个阶段保留独立的实验记录。

1. **Baseline 1 — Bicubic 插值**：作为最基本的对照。
2. **Baseline 2 — SRCNN**：经典三层 CNN 超分模型。
3. **主干模型 — Light-EDSR**：参数量受限的 EDSR 变体（减少残差块数量与通道数）。
4. **改进模型 — MSRA-EDSR**：在 Light-EDSR 基础上引入注意力机制 / 边缘损失等改进。
5. **消融实验**：逐一验证改进模块的贡献。

评测指标：PSNR、SSIM；同时输出可视化对比图。

## 阶段 TODO

- [ ] M1 工程脚手架与数据流水线（DIV2K 读取、LR/HR 配对、patch 采样）
- [ ] M2 跑通 Bicubic + SRCNN baseline，产出指标与对比图
- [ ] M3 实现并训练 Light-EDSR
- [ ] M4 加入改进模块，完成消融实验
- [ ] M5 整理实验结果、撰写技术报告

## 运行环境

PyTorch 项目。Mac 本地仅用于代码开发与调试；模型训练统一在 GPU 服务器（AutoDL）上进行。

详细的环境安装命令将在 M1 阶段补充到本节。

## 目录说明

仓库当前为初始化状态，目录结构会随各阶段推进逐步添加。规划中的主要目录：

- `models/` 模型定义
- `utils/` 数据集、指标、绘图等工具
- `configs/` 各实验的配置文件
- `scripts/` 一键训练 / 评测脚本
- `results/` 评测指标、loss 曲线、对比图（大文件不入库）
- `checkpoints/` 模型权重（不入库）

## 数据与权重

DIV2K 数据集与训练得到的模型权重体积较大，**不**纳入 Git 仓库管理。
数据集放置路径与权重保存路径将在 M1 阶段在 README 中明确给出。

## Server Environment Setup

**Current server uses conda base environment (PyTorch 2.1.2+cu121 pre-installed).**

The `.venv` directory exists but is not currently used. To use the pre-configured conda base environment:

```bash
cd ~/Code/ml-super-resolution-edsr
# Ensure you are NOT in .venv; use conda base directly
which python  # should show /root/miniconda3/bin/python
pip install -r requirements.txt
python scripts/check_env.py
```

### Environment Check

```bash
python scripts/check_env.py
```

Expected output includes:
- `CUDA available: True`
- `GPU name: NVIDIA GeForce RTX 4090 D`
- `environment check passed`
