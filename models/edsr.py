#!/usr/bin/env python3
"""Light-EDSR model for image super-resolution."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with res_scale."""

    def __init__(self, num_features, res_scale=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class LightEDSR(nn.Module):
    """Lightweight EDSR for super-resolution.

    Input: LR image [B, 3, H, W]
    Output: SR image [B, 3, H*scale, W*scale]
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_res_blocks=8, res_scale=0.1, scale=2):
        super().__init__()
        self.scale = scale

        # Head: feature extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body: residual blocks
        body = [ResBlock(num_features, res_scale) for _ in range(num_res_blocks)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampling + output
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x
