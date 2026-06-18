#!/usr/bin/env python3
"""Append BP-Cascade classes to rcan.py and register in train.py/test.py."""

import subprocess

# 0. Add F import if missing
with open("models/rcan.py", "r") as f:
    rcan_content = f.read()

if "import torch.nn.functional as F" not in rcan_content:
    rcan_content = rcan_content.replace(
        "import torch.nn as nn",
        "import torch.nn as nn\nimport torch.nn.functional as F",
    )
    with open("models/rcan.py", "w") as f:
        f.write(rcan_content)
    print("Added 'import torch.nn.functional as F' to rcan.py")
else:
    print("F import already exists")

# 1. Append BP-Cascade classes to rcan.py
bp_code = '''


class BPCascadeCorrectionNet(nn.Module):
    """Back-Projection Cascade Correction Network (Stage 2).

    Input: concat(SR_stage1, HR_error) [B, 6, H, W]
    Output: correction [B, 3, H, W]

    SR_final = SR_stage1 + residual_scale * correction
    """

    def __init__(self, in_channels=6, out_channels=3, mid_channels=64,
                 num_blocks=6, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[BasicResidualBlock(mid_channels, res_scale=0.1)
              for _ in range(num_blocks)]
        )

        self.exit_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, sr_stage1, hr_error):
        x = torch.cat([sr_stage1, hr_error], dim=1)  # [B, 6, H, W]
        feat = self.entry(x)
        feat = self.blocks(feat)
        correction = self.exit_conv(feat)
        return sr_stage1 + self.residual_scale * correction


class BPCascadeMSRRCAN(nn.Module):
    """BP-Cascade-MSR-RCAN: Back-Projection Cascade Residual Correction.

    Stage 1: MSR-RCAN-large backbone + Deep Refine v2
    Stage 2: BP Cascade Correction with LR consistency error

    Flow:
    LR -> backbone -> SR_stage1
    LR_recon = bicubic_downsample(SR_stage1, scale=4)
    LR_error = LR - LR_recon
    HR_error = bicubic_upsample(LR_error, scale=4)
    SR_final = BPCascadeCorrectionNet(SR_stage1, HR_error)
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=8, num_resblocks=8, reduction=16, scale=4,
                 cascade_num_blocks=6, cascade_mid_channels=64,
                 cascade_residual_scale=0.1):
        super().__init__()
        self.scale = scale

        # Stage 1
        self.backbone = MSRCAN(
            in_channels=in_channels, out_channels=out_channels,
            num_features=num_features, num_resgroups=num_resgroups,
            num_resblocks=num_resblocks, reduction=reduction, scale=scale,
        )
        self.refine = DeepOutputRefineBlock(out_channels, mid_channels=32)

        # Stage 2: BP Cascade (6 channels input: SR_stage1 + HR_error)
        self.cascade = BPCascadeCorrectionNet(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            mid_channels=cascade_mid_channels,
            num_blocks=cascade_num_blocks,
            residual_scale=cascade_residual_scale,
        )

    def forward(self, x):
        # Stage 1
        sr_initial = self.backbone(x)
        sr_stage1 = self.refine(sr_initial)

        # Back-projection: compute LR consistency error
        lr_recon = F.interpolate(sr_stage1, scale_factor=1.0/self.scale,
                                 mode='bicubic', align_corners=False,
                                 recompute_scale_factor=False)
        lr_error = x - lr_recon
        hr_error = F.interpolate(lr_error, scale_factor=self.scale,
                                 mode='bicubic', align_corners=False,
                                 recompute_scale_factor=False)

        # Stage 2: BP Cascade correction
        sr_final = self.cascade(sr_stage1, hr_error)
        return sr_final
'''

with open("models/rcan.py", "a") as f:
    f.write(bp_code)
print("Appended BPCascadeCorrectionNet, BPCascadeMSRRCAN to rcan.py")

# 2. Add bp_cascade_msr_rcan_large to train.py
with open("train.py", "r") as f:
    train_content = f.read()

bp_train_case = '''    elif model_name == "bp_cascade_msr_rcan_large":
        from models.rcan import BPCascadeMSRRCAN
        mp = cfg.get("model_params", {})
        model = BPCascadeMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 8),
            num_resblocks=mp.get("num_resblocks", 8),
            reduction=mp.get("reduction", 16),
            scale=scale,
            cascade_num_blocks=mp.get("cascade_num_blocks", 6),
            cascade_mid_channels=mp.get("cascade_mid_channels", 64),
            cascade_residual_scale=mp.get("cascade_residual_scale", 0.1),
        )
        input_key = "lr"
        ckpt_name = "best_bp_cascade_msr_rcan_large50_cosine_x4.pth"
        last_name = "last_bp_cascade_msr_rcan_large50_cosine_x4.pth"
'''

train_content = train_content.replace(
    '        last_name = "last_cascade_msr_rcan_large50_cosine_x4.pth"\n',
    '        last_name = "last_cascade_msr_rcan_large50_cosine_x4.pth"\n' + bp_train_case,
)
with open("train.py", "w") as f:
    f.write(train_content)
print("Added bp_cascade_msr_rcan_large to train.py")

# 3. Add bp_cascade_msr_rcan_large to test.py
with open("test.py", "r") as f:
    test_content = f.read()

bp_test_case = '''    elif model_name == "bp_cascade_msr_rcan_large":
        from models.rcan import BPCascadeMSRRCAN
        mp = cfg.get("model_params", {})
        model = BPCascadeMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 8),
            num_resblocks=mp.get("num_resblocks", 8),
            reduction=mp.get("reduction", 16),
            scale=scale,
            cascade_num_blocks=mp.get("cascade_num_blocks", 6),
            cascade_mid_channels=mp.get("cascade_mid_channels", 64),
            cascade_residual_scale=mp.get("cascade_residual_scale", 0.1),
        )
        input_key = "lr"
        ckpt_default = "best_bp_cascade_msr_rcan_large50_cosine_x4.pth"
'''

test_content = test_content.replace(
    '        ckpt_default = "best_cascade_msr_rcan_large50_cosine_x4.pth"\n',
    '        ckpt_default = "best_cascade_msr_rcan_large50_cosine_x4.pth"\n' + bp_test_case,
)
with open("test.py", "w") as f:
    f.write(test_content)
print("Added bp_cascade_msr_rcan_large to test.py")

# 4. Verify
print("\\n=== Verify ===")
subprocess.run(["grep", "-n", "bp_cascade_msr_rcan_large", "train.py", "test.py"])
subprocess.run(["grep", "-n", "BPCascade", "models/rcan.py"])
print("DONE")
