#!/bin/bash
# Batch x2 training script
cd /root/Code/ml-super-resolution-edsr
PY=/root/miniconda3/bin/python

echo "=========================================="
echo "X2 Training Started: $(date)"
echo "=========================================="

# [1/4] RCAN-small x2
echo "[1/4] RCAN-small x2  Start: $(date)"
$PY train.py --config configs/rcan_small_x2_ucmerced_selected.yaml
echo "[1/4] RCAN-small x2  Done:  $(date)"

# [2/4] MSR-RCAN-large-50-cosine x2
echo "[2/4] MSR-RCAN-large x2  Start: $(date)"
$PY train.py --config configs/msr_rcan_large50_cosine_x2_ucmerced_selected.yaml
echo "[2/4] MSR-RCAN-large x2  Done:  $(date)"

# [3/4] Cascade-MSR-RCAN Stage2-6 x2
echo "[3/4] Cascade S6 x2  Start: $(date)"
$PY train.py --config configs/cascade_msr_rcan_large50_cosine_x2_ucmerced_selected.yaml
echo "[3/4] Cascade S6 x2  Done:  $(date)"

# [4/4] MSR-RCAN-mid-50-cosine x2
echo "[4/4] MSR-RCAN-mid x2  Start: $(date)"
$PY train.py --config configs/msr_rcan_mid50_cosine_x2_ucmerced_selected.yaml
echo "[4/4] MSR-RCAN-mid x2  Done:  $(date)"

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
