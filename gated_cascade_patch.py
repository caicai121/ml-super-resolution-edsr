#!/usr/bin/env python3
"""Append Gated-Cascade classes to rcan.py and register in train.py/test.py."""

import subprocess

# 1. Append GateNet + GatedCascadeMSRRCAN to rcan.py
gated_code = '''


class GateNet(nn.Module):
    """Gate network for gated cascade correction.

    Input: concat(SR_stage1, residual_stage2) [B, 6, H, W]
    Output: gate [B, 3, H, W] in range [0, 1]
    """

    def __init__(self, in_channels=6, out_channels=3, mid_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class GatedCascadeMSRRCAN(nn.Module):
    """Gated-Cascade-MSR-RCAN: Cascade with learned gating.

    Stage 1: MSR-RCAN-large backbone + Deep Refine v2
    Stage 2: Cascade Residual Correction with GateNet

    SR_final = SR_stage1 + gate * residual_scale * residual_stage2
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=8, num_resblocks=8, reduction=16, scale=4,
                 cascade_num_blocks=10, cascade_mid_channels=64,
                 cascade_residual_scale=0.1):
        super().__init__()
        self.scale = scale

        # Stage 1: MSR-RCAN-large backbone + Deep Refine v2
        self.backbone = MSRCAN(
            in_channels=in_channels, out_channels=out_channels,
            num_features=num_features, num_resgroups=num_resgroups,
            num_resblocks=num_resblocks, reduction=reduction, scale=scale,
        )
        self.refine = DeepOutputRefineBlock(out_channels, mid_channels=32)

        # Stage 2: Cascade Residual Correction (reuse existing class)
        self.cascade = CascadeResidualCorrectionNet(
            in_channels=out_channels,
            mid_channels=cascade_mid_channels,
            num_blocks=cascade_num_blocks,
            residual_scale=cascade_residual_scale,
        )

        # GateNet: decides per-pixel correction strength
        self.gate = GateNet(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            mid_channels=32,
        )

    def forward(self, x):
        # Stage 1
        sr_initial = self.backbone(x)
        sr_stage1 = self.refine(sr_initial)

        # Get residual from cascade (before adding to sr_stage1)
        feat = self.cascade.entry(sr_stage1)
        feat = self.cascade.blocks(feat)
        residual_stage2 = self.cascade.exit_conv(feat)

        # Gate: concat sr_stage1 and residual, predict gate
        gate_input = torch.cat([sr_stage1, residual_stage2], dim=1)
        gate = self.gate(gate_input)

        # Gated correction
        sr_final = sr_stage1 + gate * self.cascade.residual_scale * residual_stage2
        return sr_final
'''

with open("models/rcan.py", "a") as f:
    f.write(gated_code)
print("Appended GateNet, GatedCascadeMSRRCAN to rcan.py")

# 2. Add gated_cascade_msr_rcan_large to train.py
with open("train.py", "r") as f:
    train_content = f.read()

gated_train_case = '''    elif model_name == "gated_cascade_msr_rcan_large":
        from models.rcan import GatedCascadeMSRRCAN
        mp = cfg.get("model_params", {})
        model = GatedCascadeMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 8),
            num_resblocks=mp.get("num_resblocks", 8),
            reduction=mp.get("reduction", 16),
            scale=scale,
            cascade_num_blocks=mp.get("cascade_num_blocks", 10),
            cascade_mid_channels=mp.get("cascade_mid_channels", 64),
            cascade_residual_scale=mp.get("cascade_residual_scale", 0.1),
        )
        input_key = "lr"
        ckpt_name = "best_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"
        last_name = "last_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"
'''

train_content = train_content.replace(
    '        last_name = "last_bp_cascade_msr_rcan_large50_cosine_x4.pth"\n',
    '        last_name = "last_bp_cascade_msr_rcan_large50_cosine_x4.pth"\n' + gated_train_case,
)
with open("train.py", "w") as f:
    f.write(train_content)
print("Added gated_cascade_msr_rcan_large to train.py")

# 3. Add gated_cascade_msr_rcan_large to test.py
with open("test.py", "r") as f:
    test_content = f.read()

gated_test_case = '''    elif model_name == "gated_cascade_msr_rcan_large":
        from models.rcan import GatedCascadeMSRRCAN
        mp = cfg.get("model_params", {})
        model = GatedCascadeMSRRCAN(
            in_channels=mp.get("in_channels", 3),
            out_channels=mp.get("out_channels", 3),
            num_features=mp.get("num_features", 64),
            num_resgroups=mp.get("num_resgroups", 8),
            num_resblocks=mp.get("num_resblocks", 8),
            reduction=mp.get("reduction", 16),
            scale=scale,
            cascade_num_blocks=mp.get("cascade_num_blocks", 10),
            cascade_mid_channels=mp.get("cascade_mid_channels", 64),
            cascade_residual_scale=mp.get("cascade_residual_scale", 0.1),
        )
        input_key = "lr"
        ckpt_default = "best_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"
'''

test_content = test_content.replace(
    '        ckpt_default = "best_bp_cascade_msr_rcan_large50_cosine_x4.pth"\n',
    '        ckpt_default = "best_bp_cascade_msr_rcan_large50_cosine_x4.pth"\n' + gated_test_case,
)
with open("test.py", "w") as f:
    f.write(test_content)
print("Added gated_cascade_msr_rcan_large to test.py")

# 4. Verify
print("\n=== Verify ===")
subprocess.run(["grep", "-n", "gated_cascade_msr_rcan_large", "train.py", "test.py"])
subprocess.run(["grep", "-n", "class Gate", "models/rcan.py"])
print("DONE")
