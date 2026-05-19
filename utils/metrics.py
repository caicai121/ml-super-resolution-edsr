#!/usr/bin/env python3
"""
Metrics for super-resolution evaluation.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2, data_range=255.0):
    """
    Calculate PSNR between two images.

    Args:
        img1: numpy array, HR image
        img2: numpy array, SR image
        data_range: maximum pixel value (255 for uint8)

    Returns:
        PSNR value in dB
    """
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=255.0):
    """
    Calculate SSIM between two images.

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
    """
    Calculate both PSNR and SSIM.

    Returns:
        dict with 'psnr' and 'ssim' keys
    """
    return {
        "psnr": calculate_psnr(img1, img2, data_range),
        "ssim": calculate_ssim(img1, img2, data_range),
    }
