#!/usr/bin/env python3
"""Append Phase 7 Cascade ablation to plan_mser_rcan.md."""

phase7 = """

## Phase 7: Cascade Residual Correction Ablation

**Status**: COMPLETED

**Setup**:
- MSR-RCAN-large backbone + Deep Refine v2 (Stage 1)
- Cascade Residual Correction Network (Stage 2): learns residual error between SR_stage1 and HR
- SR_final = SR_stage1 + residual_scale * residual_stage2
- Residual scale: 0.1 (fixed)

**Ablation Results**:

| Model | Stage2 Blocks | Modification | Y+crop PSNR | Delta vs Baseline |
|-------|--------------|--------------|-------------|-------------------|
| MSR-RCAN-large (baseline) | 0 | none | 31.28 | 0.00 |
| Cascade-6 | 6 | basic cascade | 31.35 | +0.07 |
| Cascade-10 | 10 | basic cascade | **31.40** | **+0.12** |
| BP-Cascade | 6 | back-projection error | 31.34 | +0.06 |
| Gated-Cascade-10 | 10 | GateNet spatial gating | 31.37 | +0.09 |
| Learnable-Scale-Cascade-10 | 10 | learnable alpha (0.1->0.1048) | 31.39 | +0.11 |

**Key Findings**:
1. Cascade residual correction is effective: +0.07 dB (6 blocks) and +0.12 dB (10 blocks)
2. Back-Projection error feedback (LR consistency) did not help: 31.34 < 31.35
3. Spatial gating suppressed useful residual: 31.37 < 31.40
4. Learnable residual scale converged to alpha=0.1048, confirming fixed 0.1 is near-optimal
5. Additional modules stacked on Cascade (BP, Gate, Learnable Scale) all failed to beat Cascade-10

**Conclusion**: Cascade-10 (Stage2 ResBlock x10, residual_scale=0.1) is the optimal configuration
within the Cascade direction. Further micro-optimizations on Cascade are not productive.

**Best 50-epoch model**: Cascade-MSR-RCAN-large-10 (31.40 dB)
"""

with open("plan_mser_rcan.md", "a") as f:
    f.write(phase7)
print("Appended Phase 7 to plan_mser_rcan.md")
