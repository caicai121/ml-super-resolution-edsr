#!/usr/bin/env python3
"""Image utility functions."""

import numpy as np
from PIL import Image


def tensor_to_np(tensor):
    """Convert CHW tensor [0,1] to HWC numpy uint8 [0,255]."""
    img = tensor.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def save_image(img_array, path):
    """Save numpy array (HWC uint8) to file."""
    Image.fromarray(img_array).save(path)
