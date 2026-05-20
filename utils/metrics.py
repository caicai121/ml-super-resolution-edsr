#!/usr/bin/env python3
"""
Metrics for super-resolution evaluation.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rgb_to_y(img):
    """Convert RGB image to Y channel (ITU-R BT.601).

    Args:
        img: numpy array, shape (H, W, 3), dtype uint8 or float

    Returns:
        Y channel, shape (H, W), dtype uint8
    """
    img = img.astype(np.float64)
    y = 16.0 + (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2]) / 255.0
    return np.clip(y, 16, 235).astype(np.uint8)


def crop_border(img, border):
    """Crop pixels from each border.

    Args:
        img: numpy array, shape (H, W) or (H, W, C)
        border: number of pixels to crop from each side

    Returns:
        Cropped image
    """
    if border == 0:
        return img
    if img.ndim == 3:
        return img[border:-border, border:-border, :]
    return img[border:-border, border:-border]


def calculate_psnr(img1, img2, data_range=255.0):
    """Calculate PSNR between two images.

    Args:
        img1: numpy array, HR image
        img2: numpy array, SR image
        data_range: maximum pixel value (255 for uint8)

    Returns:
        PSNR value in dB
    """
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=255.0):
    """Calculate SSIM between two images.

    Args:
        img1: numpy array, HR image
        img2: numpy array, SR image
        data_range: maximum pixel value (255 for uint8)

    Returns:
        SSIM value
    """
    if img1.ndim == 3:
        channel_axis = -1
    else:
        channel_axis = None
    return structural_similarity(
        img1, img2, data_range=data_range, channel_axis=channel_axis
    )


def calculate_metrics(img1, img2, data_range=255.0):
    """Calculate both PSNR and SSIM on RGB full image.

    Returns:
        dict with 'psnr' and 'ssim' keys
    """
    return {
        "psnr": calculate_psnr(img1, img2, data_range),
        "ssim": calculate_ssim(img1, img2, data_range),
    }


def calculate_metrics_standard(hr_rgb, sr_rgb, scale=2):
    """Standard SR evaluation: Y-channel, border crop.

    Follows the evaluation protocol used in EDSR/RDN/RCAN and most SR papers:
    1. Convert RGB to Y channel (ITU-R BT.601)
    2. Crop `scale` pixels from each border
    3. Compute PSNR/SSIM on the cropped Y channel

    Args:
        hr_rgb: numpy array, shape (H, W, 3), uint8, ground truth
        sr_rgb: numpy array, shape (H, W, 3), uint8, super-resolved
        scale: super-resolution scale factor (used for border crop)

    Returns:
        dict with 'psnr' and 'ssim' keys
    """
    hr_y = rgb_to_y(hr_rgb)
    sr_y = rgb_to_y(sr_rgb)

    hr_y = crop_border(hr_y, scale)
    sr_y = crop_border(sr_y, scale)

    return {
        "psnr": calculate_psnr(hr_y, sr_y, data_range=255.0),
        "ssim": calculate_ssim(hr_y, sr_y, data_range=255.0),
    }
