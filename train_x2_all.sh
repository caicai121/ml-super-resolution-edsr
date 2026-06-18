#!/bin/bash
# Batch x2 training script - runs all 4 missing x2 models sequentially
# Each model: 50 epochs

set -e
cd /root/Code/ml-super-resolution-edsr
PYTHON=/root/miniconda3/bin/python

echo "=========================================="
echo "X2 Training Pipeline Started: $(date)"
echo "=========================================="

# Priority 1: RCAN-small x2 (fastest, baseline)
echo ""
echo "===== [1/4] RCAN-small x2 ====="
echo "Start: $(date)"
$PYTHON train.py --config configs/rcan_small_x2_ucmerced_selected.yaml 2>&1 | tee train_rcan_small_x2.log
echo "Done: $(date)"

# Priority 1: MSR-RCAN-large-50-cosine x2 (main backbone)
echo ""
echo "===== [2/4] MSR-RCAN-large-50-cosine x2 ====="
echo "Start: $(date)"
$PYTHON train.py --config configs/msr_rcan_large50_cosine_x2_ucmerced_selected.yaml 2>&1 | tee train_msr_rcan_large_x2.log
echo "Done: $(date)"

# Priority 1: Cascade-MSR-RCAN Stage2-6 x2
echo ""
echo "===== [3/4] Cascade-MSR-RCAN Stage2-6 x2 ====="
echo "Start: $(date)"
$PYTHON train.py --config configs/cascade_msr_rcan_large50_cosine_x2_ucmerced_selected.yaml 2>&1 | tee train_cascade_s6_x2.log
echo "Done: $(date)"

# Priority 2: MSR-RCAN-mid-50-cosine x2
echo ""
echo "===== [4/4] MSR-RCAN-mid-50-cosine x2 ====="
echo "Start: $(date)"
$PYTHON train.py --config configs/msr_rcan_mid50_cosine_x2_ucmerced_selected.yaml 2>&1 | tee train_msr_rcan_mid_x2.log
echo "Done: $(date)"

echo ""
echo "=========================================="
echo "X2 Training Pipeline Complete: $(date)"
echo "=========================================="
