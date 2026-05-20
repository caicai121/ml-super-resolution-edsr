#!/usr/bin/env python3
"""SRCNN model for image super-resolution."""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network.

    Input: Bicubic-upsampled LR image (same size as HR)
    Output: Super-resolved image (same size as input)
    """

    def __init__(self, channels=3, num_filters=(64, 32)):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, num_filters[0], kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_filters[1], channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
