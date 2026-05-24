"""Sobel Edge Loss for super-resolution training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """L1 loss on Sobel edge maps.

    L_edge = L1(Sobel(SR), Sobel(HR))
    Sobel kernels are fixed, not trainable.
    For RGB images, computes Sobel on each channel then averages.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Register as buffers (not parameters, moves with .to(device))
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_edges(self, x):
        """Compute Sobel edge magnitude. x: (B, C, H, W)"""
        b, c, h, w = x.shape
        # Reshape sobel for grouped conv: apply same kernel to each channel
        kx = self.sobel_x.expand(c, 1, 3, 3)
        ky = self.sobel_y.expand(c, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, sr, hr):
        edge_sr = self._compute_edges(sr)
        edge_hr = self._compute_edges(hr)
        return F.l1_loss(edge_sr, edge_hr)
