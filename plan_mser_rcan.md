# MS-RCAN Experiment Plan

## Phase 1: MS-RCAN-small (Multi-Scale RCAB only)

**Status**: COMPLETED

**What changed**: RCAB → MSRCAB (1x1 + 3x3 + 5x5 parallel branches)
**What did NOT change**: No Refine, No Edge Loss, same dataset, same training settings

**Results**:
- Parameters: 1.56M → 3.35M (+114%)
- Test Y+crop PSNR: 30.60 → 30.67 dB (+0.07 dB)
- Test RGB PSNR: 29.00 → 29.06 dB (+0.06 dB)

**Conclusion**: Multi-scale RCAB is effective but the gain is small.

## Phase 2: MSR-RCAN-small (Multi-Scale RCAB + Output Refine)

### Phase 2a: Simple Refine (3→3, 168 params)

**Status**: COMPLETED - NOT EFFECTIVE

- Test Y+crop PSNR: 30.65 dB (-0.02 vs MS-RCAN-small)
- Simple 2-layer refine is too weak to make a difference.

### Phase 2b: Deep Refine (3→32→32→3, +11K params)

**Status**: COMPLETED - EFFECTIVE

- Test Y+crop PSNR: **30.81 dB** (+0.14 vs MS-RCAN-small)
- Test RGB PSNR: 29.18 dB
- Test RGB SSIM: 0.7772
- Parameters: 3.36M
- Deep Refine Block with intermediate channels learns meaningful residual corrections.

**Current best model**: MSR-RCAN-small v2

## Phase 3: DMSR-RCAN-small (Dilated Multi-Scale + Deep Refine)

**Status**: PLANNED

**Goal**: Replace 5x5 branch with 3x3 dilation=2 in MSRCAB for larger receptive field with fewer parameters.
**Keep**: Deep Refine v2 unchanged.
**Expected improvement**: Push Y+crop PSNR toward 30.90~31.00 dB.

## Phase 4: Capacity & Training Strategy Experiments

**Status**: COMPLETED

Scaling and training strategy experiments (full results in memory):
- MSR-RCAN-mid-50 (5g5b): 31.01 dB
- MSR-RCAN-mid-50-cosine (Cosine LR): 31.15 dB (+0.14)
- MSR-RCAN-large-50-cosine (8g8b): 31.28 dB (+0.13)
- MSR-RCAN-mid-100 (extended training): 31.44 dB

**Effective**: Cosine LR, larger capacity, more epochs
**Ineffective**: data augmentation, Edge Branch/Loss, AMSRCAB, RDRB

**Current 50-epoch best**: MSR-RCAN-large-50-cosine (31.28 dB, 13.02M params)

## Phase 5: Teacher Distillation Ablation

**Status**: COMPLETED - NOT EFFECTIVE

**Setup**:
- Teacher: RCAN-pretrained (Y+crop 32.52 dB)
- Student: MSR-RCAN-large-50-cosine (Y+crop 31.28 dB)
- Loss: L_total = L1(SR_student, HR) + 0.1 * L1(SR_student, SR_teacher)
- Teacher frozen, not trained

**Results**:
- Distill alpha=0.1: 31.16 dB (-0.12 vs baseline 31.28)
- Training best (val): 30.75 dB at epoch 48

**Conclusion**: Direct L1 distillation on final SR output did not improve student
performance. Teacher (standard RCAN 10g20b) and student (MSRRCANV2 8g8b +
MSRCAB + Deep Refine) have structural differences leading to distribution
mismatch. Soft supervision conflicts with HR hard supervision.
Alpha=0.2 not pursued.

**Best 50-epoch model remains**: MSR-RCAN-large-50-cosine (31.28 dB)

## Phase 6: Global Context Block Ablation

**Status**: COMPLETED - NOT EFFECTIVE

**Setup**:
- MSR-RCAN-large + GlobalContextRefineBlock (between backbone and Deep Refine v2)
- GC Block: spatial attention mask + weighted context pooling + transform
- GC refine params: 2,348 (very lightweight)

**Results**:
- GC-MSR-RCAN-large-50-cosine: 31.22 dB (-0.06 vs baseline 31.28)
- RGB PSNR: 29.59 dB, SSIM: 0.7880

**Conclusion**: Global Context Block did not improve performance.
Spatial attention + global context pooling at image level does not provide
useful information beyond local MSRCAB features for this task.

**Best 50-epoch model remains**: MSR-RCAN-large-50-cosine (31.28 dB)


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
