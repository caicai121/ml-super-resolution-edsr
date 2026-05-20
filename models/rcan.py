#!/usr/bin/env python3
"""RCAN (Residual Channel Attention Network) for image super-resolution.

Paper: "Image Super-Resolution Using Very Deep Residual Channel Attention Networks"
       Zhang et al., ECCV 2018
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention (CA) module."""

    def __init__(self, num_features, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, num_features, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """Residual Group: multiple RCABs + one conv."""

    def __init__(self, num_features, num_resblocks, reduction=16):
        super().__init__()
        body = [RCAB(num_features, reduction) for _ in range(num_resblocks)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return x + self.body(x)


class RCAN(nn.Module):
    """Residual Channel Attention Network.

    Args:
        in_channels: input channels (default: 3 for RGB)
        out_channels: output channels (default: 3 for RGB)
        num_features: number of feature maps (default: 64)
        num_resgroups: number of residual groups (default: 10)
        num_resblocks: number of RCABs per group (default: 20)
        reduction: channel attention reduction ratio (default: 16)
        scale: upscaling factor (default: 4)
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=10, num_resblocks=20, reduction=16, scale=4):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body: residual groups
        body = [ResidualGroup(num_features, num_resblocks, reduction)
                for _ in range(num_resgroups)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampling + output
        if scale == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            )
        elif scale == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unsupported scale: {scale}")

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.tail(x)
        return x


def load_pretrained_rcan(ckpt_path, model, device="cpu"):
    """Load pretrained RCAN weights with exact key mapping.

    Handles naming differences between original RCAN repo and our model:
    - head.0.weight -> head.weight
    - conv_du.N -> attention.M (index shift)
    - tail.0.0 -> tail.0, tail.0.2 -> tail.2, tail.1 -> tail.4
    - sub_mean / add_mean ignored (not in our model)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt_sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        ckpt_sd = ckpt["params"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt_sd = ckpt["state_dict"]
    else:
        ckpt_sd = ckpt

    model_sd = model.state_dict()
    new_sd = {}
    skipped = []

    for target_key, target_val in model_sd.items():
        target_shape = target_val.shape

        # 1. Direct match
        if target_key in ckpt_sd and ckpt_sd[target_key].shape == target_shape:
            new_sd[target_key] = ckpt_sd[target_key]
            continue

        # 2. head: head.weight -> head.0.weight
        if target_key == "head.weight" and "head.0.weight" in ckpt_sd:
            new_sd[target_key] = ckpt_sd["head.0.weight"]
            continue
        if target_key == "head.bias" and "head.0.bias" in ckpt_sd:
            new_sd[target_key] = ckpt_sd["head.0.bias"]
            continue

        # 3. attention.1 -> conv_du.0, attention.3 -> conv_du.2
        if "attention.1." in target_key:
            src_key = target_key.replace("attention.1.", "conv_du.0.")
            if src_key in ckpt_sd and ckpt_sd[src_key].shape == target_shape:
                new_sd[target_key] = ckpt_sd[src_key]
                continue
        if "attention.3." in target_key:
            src_key = target_key.replace("attention.3.", "conv_du.2.")
            if src_key in ckpt_sd and ckpt_sd[src_key].shape == target_shape:
                new_sd[target_key] = ckpt_sd[src_key]
                continue

        # 4. tail structure: original tail.0.0, tail.0.2, tail.1
        #    our model: tail.0, tail.2, tail.4 (two-step ×2 upsample)
        #    OR our model: tail.0, tail.2, tail.4 (same for single-step)
        tail_remap = {
            "tail.0.weight": ["tail.0.0.weight"],
            "tail.0.bias": ["tail.0.0.bias"],
            "tail.2.weight": ["tail.0.2.weight"],
            "tail.2.bias": ["tail.0.2.bias"],
        }
        if target_key in tail_remap:
            found = False
            for src_key in tail_remap[target_key]:
                if src_key in ckpt_sd and ckpt_sd[src_key].shape == target_shape:
                    new_sd[target_key] = ckpt_sd[src_key]
                    found = True
                    break
            if found:
                continue

        # tail.4 (final conv): try tail.1 or tail.2 depending on structure
        if target_key == "tail.4.weight":
            for src_key in ["tail.1.weight", "tail.2.weight"]:
                if src_key in ckpt_sd and ckpt_sd[src_key].shape == target_shape:
                    new_sd[target_key] = ckpt_sd[src_key]
                    break
            if target_key in new_sd:
                continue
        if target_key == "tail.4.bias":
            for src_key in ["tail.1.bias", "tail.2.bias"]:
                if src_key in ckpt_sd and ckpt_sd[src_key].shape == target_shape:
                    new_sd[target_key] = ckpt_sd[src_key]
                    break
            if target_key in new_sd:
                continue

        skipped.append(target_key)

    # Load
    msg = model.load_state_dict(new_sd, strict=False)
    mapped = len(model_sd) - len(skipped)
    print(f"Weight loading: {mapped}/{len(model_sd)} keys mapped")
    if skipped:
        print(f"  Unmapped ({len(skipped)}):")
        for k in skipped[:10]:
            print(f"    {k}: {model_sd[k].shape}")

    return model
