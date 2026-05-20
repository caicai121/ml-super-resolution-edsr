#!/usr/bin/env bash
set -e
cd /root/Code/ml-super-resolution-edsr
/root/miniconda3/bin/python test.py --config configs/srcnn.yaml --checkpoint checkpoints/srcnn/best_srcnn.pth
