#!/bin/bash
set -e
cd /root/Code/ml-super-resolution-edsr
/root/miniconda3/bin/python test.py --config configs/rcan_x4.yaml --checkpoint checkpoints/rcan_x4/best_rcan_x4.pth
