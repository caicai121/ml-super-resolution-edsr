#!/usr/bin/env bash
set -e
cd /root/Code/ml-super-resolution-edsr
/root/miniconda3/bin/python test.py --config configs/light_edsr.yaml --checkpoint checkpoints/light_edsr/best_light_edsr.pth
