#!/usr/bin/env python3
"""Append LearnableScaleCascadeMSRRCAN to rcan.py and register in train.py/test.py."""


# 1. Append new class to rcan.py
ls_code = '''


class LearnableScaleCascadeMSRRCAN(nn.Module):
    """Learnable-Scale Cascade-MSR-RCAN.

    Same as CascadeMSRRCAN but with learnable residual scale alpha.
    SR_final = SR_stage1 + alpha * residual_stage2
    alpha initialized to 0.1, learned freely.
    """

    def __init__(self, in_channels=3, out_channels=3, num_features=64,
                 num_resgroups=8, num_resblocks=8, reduction=16, scale=4,
                 cascade_num_blocks=10, cascade_mid_channels=64,
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

        # Stage 2: Cascade (uses fixed residual_scale internally for ResBlocks)
        self.cascade = CascadeResidualCorrectionNet(
            in_channels=out_channels,
            mid_channels=cascade_mid_channels,
            num_blocks=cascade_num_blocks,
            residual_scale=cascade_residual_scale,
        )

        # Learnable alpha: controls overall correction strength
        self.residual_alpha = nn.Parameter(torch.tensor(float(cascade_residual_scale)))

    def forward(self, x):
        sr_initial = self.backbone(x)
        sr_stage1 = self.refine(sr_initial)

        # Get raw residual from cascade (bypass cascade's internal scaling)
        feat = self.cascade.entry(sr_stage1)
        feat = self.cascade.blocks(feat)
        residual_stage2 = self.cascade.exit_conv(feat)

        # Apply learnable alpha
        sr_final = sr_stage1 + self.residual_alpha * residual_stage2
        return sr_final
'''

with open("models/rcan.py", "a") as f:
    f.write(ls_code)
print("Appended LearnableScaleCascadeMSRRCAN to rcan.py")

# 2. Add to train.py
with open("train.py", "r") as f:
    train_content = f.read()

ls_train_case = '''    elif model_name == "learnable_scale_cascade_msr_rcan_large":
        from models.rcan import LearnableScaleCascadeMSRRCAN
        mp = cfg.get("model_params", {})
        model = LearnableScaleCascadeMSRRCAN(
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
        ckpt_name = "best_learnable_scale_cascade_msr_rcan_large_s10_50_cosine_x4.pth"
        last_name = "last_learnable_scale_cascade_msr_rcan_large_s10_50_cosine_x4.pth"
'''

train_content = train_content.replace(
    '        last_name = "last_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"\n',
    '        last_name = "last_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"\n' + ls_train_case,
)
with open("train.py", "w") as f:
    f.write(train_content)
print("Added learnable_scale_cascade_msr_rcan_large to train.py")

# 3. Add to test.py
with open("test.py", "r") as f:
    test_content = f.read()

ls_test_case = '''    elif model_name == "learnable_scale_cascade_msr_rcan_large":
        from models.rcan import LearnableScaleCascadeMSRRCAN
        mp = cfg.get("model_params", {})
        model = LearnableScaleCascadeMSRRCAN(
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
        ckpt_default = "best_learnable_scale_cascade_msr_rcan_large_s10_50_cosine_x4.pth"
'''

test_content = test_content.replace(
    '        ckpt_default = "best_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"\n',
    '        ckpt_default = "best_gated_cascade_msr_rcan_large_s10_50_cosine_x4.pth"\n' + ls_test_case,
)
with open("test.py", "w") as f:
    f.write(test_content)
print("Added learnable_scale_cascade_msr_rcan_large to test.py")

# 4. Verify
import subprocess
print("\n=== Verify ===")
subprocess.run(["grep", "-n", "learnable_scale_cascade_msr_rcan_large", "train.py", "test.py"])
subprocess.run(["grep", "-n", "LearnableScaleCascade", "models/rcan.py"])
print("DONE")
