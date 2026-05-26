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

    def __init__(self, num_features, num_resblocks, reduction=16, rcab_class=RCAB):
        super().__init__()
        body = [rcab_class(num_features, reduction) for _ in range(num_resblocks)]
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


class MSRCAB(nn.Module):
    """Multi-Scale Residual Channel Attention Block.

    Replaces the single 3x3 conv in RCAB with three parallel branches
    (1x1, 3x3, 5x5) followed by 1x1 fusion, ReLU, 3x3 conv, and CA.
    """

    def __init__(self, num_features, reduction=16):
        super().__init__()
        # Multi-scale branches
        self.branch_1x1 = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0)
        self.branch_3x3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.branch_5x5 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2)
        # Fusion
        self.fusion = nn.Conv2d(num_features * 3, num_features, kernel_size=1, padding=0)
        # Post-fusion body (same as original RCAB tail)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)
        fused = self.fusion(torch.cat([b1, b3, b5], dim=1))
        return x + self.body(fused)


class MSRCAN(nn.Module):
    """Multi-Scale RCAN: RCAN with MSRCAB blocks.

    Same architecture as RCAN, only RCAB is replaced by MSRCAB.
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=10, num_resblocks=20, reduction=16, scale=4):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body: residual groups with MSRCAB
        body = [ResidualGroup(num_features, num_resblocks, reduction, rcab_class=MSRCAB)
                for _ in range(num_resgroups)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampling + output (identical to RCAN)
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


class OutputRefineBlock(nn.Module):
    """Output Refine Block: residual refinement at the output end."""

    def __init__(self, num_channels=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class MSRRCAN(nn.Module):
    """MSR-RCAN: MS-RCAN-small with Output Refine Block."""

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=3, num_resblocks=5, reduction=16, scale=4):
        super().__init__()
        self.backbone = MSRCAN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_resgroups=num_resgroups,
            num_resblocks=num_resblocks,
            reduction=reduction,
            scale=scale,
        )
        self.refine = OutputRefineBlock(out_channels)

    def forward(self, x):
        sr_initial = self.backbone(x)
        sr_final = self.refine(sr_initial)
        return sr_final


class DeepOutputRefineBlock(nn.Module):
    """Deep Output Refine Block: 3->32->32->3 residual refinement.

    Stronger than simple 2-layer refine, more parameters but still lightweight.
    """

    def __init__(self, in_channels=3, mid_channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class MSRRCANV2(nn.Module):
    """MSR-RCAN V2: MS-RCAN-small with Deep Output Refine Block.

    Phase 2b: enhanced refine with 3->32->32->3 channels.
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=3, num_resblocks=5, reduction=16, scale=4):
        super().__init__()
        self.backbone = MSRCAN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_resgroups=num_resgroups,
            num_resblocks=num_resblocks,
            reduction=reduction,
            scale=scale,
        )
        self.refine = DeepOutputRefineBlock(out_channels, mid_channels=32)

    def forward(self, x):
        sr_initial = self.backbone(x)
        sr_final = self.refine(sr_initial)
        return sr_final


class DilatedMSRCAB(nn.Module):
    """Dilated Multi-Scale Residual Channel Attention Block.

    Replaces the 5x5 branch with 3x3 dilation=2 for equivalent 5x5 receptive field
    but fewer parameters (9 vs 25 weights per channel).
    """

    def __init__(self, num_features, reduction=16):
        super().__init__()
        # Multi-scale branches
        self.branch_1x1 = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0)
        self.branch_3x3_d1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, dilation=1)
        self.branch_3x3_d2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=2, dilation=2)
        # Fusion
        self.fusion = nn.Conv2d(num_features * 3, num_features, kernel_size=1, padding=0)
        # Post-fusion body
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3_d1(x)
        b5 = self.branch_3x3_d2(x)
        fused = self.fusion(torch.cat([b1, b3, b5], dim=1))
        return x + self.body(fused)


class DMSRRCAN(nn.Module):
    """DMSR-RCAN: Dilated Multi-Scale RCAN with Deep Refine Block.

    Phase 3: replaces 5x5 in MSRCAB with 3x3 dilation=2.
    Keeps Deep Output Refine v2.
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=3, num_resblocks=5, reduction=16, scale=4):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body: residual groups with DilatedMSRCAB
        body = [ResidualGroup(num_features, num_resblocks, reduction, rcab_class=DilatedMSRCAB)
                for _ in range(num_resgroups)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampling + output (identical to RCAN)
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

        # Deep Refine v2
        self.refine = DeepOutputRefineBlock(out_channels, mid_channels=32)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        sr_initial = self.tail(x)
        sr_final = self.refine(sr_initial)
        return sr_final


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
        #    our model: tail.0, tail.2, tail.4 (two-step x2 upsample)
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
# EG-MSR-RCAN classes to append to rcan.py


class EdgeBranch(nn.Module):
    """Edge Feature Extraction Branch.

    Extracts edge structure features from SR_initial image.
    Uses learnable 3x3 convolutions (not fixed Sobel filters)
    so the network can learn task-specific edge patterns.
    """

    def __init__(self, in_channels=3, mid_channels=32, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class EGMSRRCAN(nn.Module):
    """EG-MSR-RCAN: Edge-Guided Multi-Scale Refined RCAN.

    Dual-branch architecture:
    - Main branch: MSR-RCAN backbone (MSRCAB + Deep Refine)
    - Edge branch: lightweight edge feature extraction from SR_initial
    - Fusion: concat + 1x1 conv before final refine

    Compared to MSRRCANV2:
    + Edge Branch for structure-aware reconstruction
    + Fusion layer before refine
    Same backbone, same training config
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=5, num_resblocks=5, reduction=16, scale=4,
                 edge_mid_channels=32, edge_num_layers=3):
        super().__init__()
        self.scale = scale

        # Main branch: reuse existing MSRCAN backbone
        self.backbone = MSRCAN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_resgroups=num_resgroups,
            num_resblocks=num_resblocks,
            reduction=reduction,
            scale=scale,
        )

        # Edge branch: extract edge features from SR_initial
        self.edge_branch = EdgeBranch(
            in_channels=out_channels,
            mid_channels=edge_mid_channels,
            num_layers=edge_num_layers,
        )

        # Fusion: concat(3 + edge_mid) -> out_channels
        self.fusion = nn.Conv2d(out_channels + edge_mid_channels, out_channels, kernel_size=1)

        # Final refine
        self.refine = DeepOutputRefineBlock(out_channels, mid_channels=32)

    def forward(self, x):
        sr_initial = self.backbone(x)
        edge_feat = self.edge_branch(sr_initial)
        fused = torch.cat([sr_initial, edge_feat], dim=1)
        fused = self.fusion(fused)
        sr_final = self.refine(fused)
        return sr_final

# AMSR-RCAN classes: Adaptive Multi-Scale fusion

class AMSRCAB(nn.Module):
    """Adaptive Multi-Scale Residual Channel Attention Block.

    Replaces concat + 1x1 fusion in MSRCAB with adaptive weighted fusion.
    The model learns per-scale importance weights via GAP + small MLP.
    """

    def __init__(self, num_features, reduction=16):
        super().__init__()
        # Multi-scale branches (same as MSRCAB)
        self.branch_1x1 = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0)
        self.branch_3x3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.branch_5x5 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2)
        # Adaptive scale weight generator
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale_fc = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 3, kernel_size=1),
        )
        # Post-fusion body (same as original RCAB tail)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction),
        )

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b3 = self.branch_3x3(x)
        b5 = self.branch_5x5(x)

        # Adaptive weight: context -> GAP -> MLP -> softmax -> [B, 3, 1, 1]
        context = b1 + b3 + b5
        gap = self.gap(context)
        scale_weights = self.scale_fc(gap)
        w = torch.softmax(scale_weights, dim=1)

        # Weighted sum: [B, C, H, W]
        fused = w[:, 0:1] * b1 + w[:, 1:2] * b3 + w[:, 2:3] * b5

        return x + self.body(fused)


class AMSRRCAN(nn.Module):
    """AMSR-RCAN: Adaptive Multi-Scale Refined RCAN.

    Same architecture as MSRRCANV2, only MSRCAB replaced by AMSRCAB.
    Head, tail, Deep Refine v2 all identical to MSRRCANV2.
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=5, num_resblocks=5, reduction=16, scale=4):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body: residual groups with AMSRCAB
        body = [ResidualGroup(num_features, num_resblocks, reduction, rcab_class=AMSRCAB)
                for _ in range(num_resgroups)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampling + output (identical to MSRRCANV2)
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

        # Deep Refine v2 (same as MSRRCANV2)
        self.refine = DeepOutputRefineBlock(out_channels, mid_channels=32)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        sr_initial = self.tail(x)
        sr_final = self.refine(sr_initial)
        return sr_final


# RDR-MSR-RCAN classes: Residual Dense Refinement

class RDRB(nn.Module):
    """Residual Dense Refinement Block.

    Dense connectivity: each layer sees SR_initial + all previous features.
    Stronger than DeepOutputRefineBlock (sequential convs).
    """

    def __init__(self, in_channels=3, mid_channels=32):
        super().__init__()
        # f1: in_channels -> mid
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        # f2: in_channels + mid -> mid
        self.conv2 = nn.Conv2d(in_channels + mid_channels, mid_channels, kernel_size=3, padding=1)
        # f3: in_channels + 2*mid -> mid
        self.conv3 = nn.Conv2d(in_channels + 2 * mid_channels, mid_channels, kernel_size=3, padding=1)
        # fusion: in_channels + 3*mid -> mid
        self.fusion = nn.Conv2d(in_channels + 3 * mid_channels, mid_channels, kernel_size=1)
        # residual: mid -> in_channels
        self.residual = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.relu(self.conv1(x))
        f2 = self.relu(self.conv2(torch.cat([x, f1], dim=1)))
        f3 = self.relu(self.conv3(torch.cat([x, f1, f2], dim=1)))
        fused = self.relu(self.fusion(torch.cat([x, f1, f2, f3], dim=1)))
        residual = self.residual(fused)
        return x + residual


class RDRMSRRCAN(nn.Module):
    """RDR-MSR-RCAN: MSR-RCAN-mid with Residual Dense Refinement Block.

    Same backbone as MSRRCANV2 (MSRCAB + Deep Refine v2),
    but replaces DeepOutputRefineBlock with RDRB.
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=5, num_resblocks=5, reduction=16, scale=4):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Body: residual groups with MSRCAB (original concat+1x1 fusion)
        body = [ResidualGroup(num_features, num_resblocks, reduction, rcab_class=MSRCAB)
                for _ in range(num_resgroups)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail: upsampling + output (identical to MSRRCANV2)
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

        # RDRB (replaces Deep Refine v2)
        self.refine = RDRB(out_channels, mid_channels=32)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        sr_initial = self.tail(x)
        sr_final = self.refine(sr_initial)
        return sr_final

