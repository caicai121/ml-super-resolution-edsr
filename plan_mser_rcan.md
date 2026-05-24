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
