#!/bin/bash
cd /root/Code/ml-super-resolution-edsr
PY=/root/miniconda3/bin/python

echo "===== X2 Test All Models ====="
echo "Start: $(date)"

# 1. Bicubic x2 (run separately via script)
echo "[Bicubic x2] Testing..."
$PY -c "
import sys; sys.path.insert(0,'.')
from utils.metrics import calculate_metrics_standard
from pathlib import Path
import numpy as np, cv2, csv

hr_dir = Path('data_final/ucmerced_selected/test/HR')
lr_dir = Path('data_final/ucmerced_selected/test/LR_x2')
out_dir = Path('results/bicubic_x2_ucmerced_selected')
out_dir.mkdir(parents=True, exist_ok=True)

results = []
for hr_path in sorted(hr_dir.glob('*.png')):
    lr_path = lr_dir / hr_path.name
    hr = cv2.imread(str(hr_path))
    lr = cv2.imread(str(lr_path))
    bicubic = cv2.resize(lr, (256,256), interpolation=cv2.INTER_CUBIC)
    # Convert BGR to RGB for metrics
    hr_rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    sr_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
    m = calculate_metrics_standard(hr_rgb, sr_rgb, scale=2)
    results.append({'image': hr_path.name, 'rgb_psnr': 0, 'rgb_ssim': 0, 'y_psnr': m['psnr'], 'y_ssim': m['ssim']})

# Compute RGB metrics
from utils.metrics import calculate_metrics
for r, hr_path in zip(results, sorted(hr_dir.glob('*.png'))):
    hr = cv2.imread(str(hr_path))
    hr_rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    lr = cv2.imread(str(lr_dir / hr_path.name))
    bicubic = cv2.resize(lr, (256,256), interpolation=cv2.INTER_CUBIC)
    sr_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
    rgb_m = calculate_metrics(hr_rgb, sr_rgb)
    r['rgb_psnr'] = rgb_m['psnr']
    r['rgb_ssim'] = rgb_m['ssim']

with open(out_dir / 'metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image','rgb_psnr','rgb_ssim','y_psnr','y_ssim'])
    w.writeheader(); w.writerows(results)

avg_rgb_psnr = np.mean([r['rgb_psnr'] for r in results])
avg_rgb_ssim = np.mean([r['rgb_ssim'] for r in results])
avg_y_psnr = np.mean([r['y_psnr'] for r in results])
avg_y_ssim = np.mean([r['y_ssim'] for r in results])
summary = f'''Bicubic x2 Test Results
==================================================
Images: {len(results)}
RGB PSNR:  {avg_rgb_psnr:.2f} dB | RGB SSIM:  {avg_rgb_ssim:.4f}
Y+crop PSNR: {avg_y_psnr:.2f} dB | Y+crop SSIM: {avg_y_ssim:.4f}'''
with open(out_dir / 'summary.txt', 'w') as f:
    f.write(summary)
print(summary)
"
echo "[Bicubic x2] Done"

# 2. RCAN-small x2
echo "[RCAN-small x2] Testing..."
$PY test.py --config configs/rcan_small_x2_ucmerced_selected.yaml
echo "[RCAN-small x2] Done"

# 3. MSR-RCAN-mid x2
echo "[MSR-RCAN-mid x2] Testing..."
$PY test.py --config configs/msr_rcan_mid50_cosine_x2_ucmerced_selected.yaml
echo "[MSR-RCAN-mid x2] Done"

# 4. MSR-RCAN-large x2
echo "[MSR-RCAN-large x2] Testing..."
$PY test.py --config configs/msr_rcan_large50_cosine_x2_ucmerced_selected.yaml
echo "[MSR-RCAN-large x2] Done"

# 5. Cascade S6 x2
echo "[Cascade S6 x2] Testing..."
$PY test.py --config configs/cascade_msr_rcan_large50_cosine_x2_ucmerced_selected.yaml
echo "[Cascade S6 x2] Done"

# 6. Cascade S10 x2 (re-test to confirm)
echo "[Cascade S10 x2] Testing..."
$PY test.py --config configs/cascade_msr_rcan_large_s10_50_cosine_x2_ucmerced_selected.yaml
echo "[Cascade S10 x2] Done"

echo "===== ALL DONE: $(date) ====="
