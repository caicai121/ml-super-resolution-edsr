#!/usr/bin/env bash
set -e
cd /root/Code/ml-super-resolution-edsr
/root/miniconda3/bin/python train.py --config configs/light_edsr.yaml
