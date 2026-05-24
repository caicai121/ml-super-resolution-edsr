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
The cost-benefit ratio is not great, but the direction is validated.

## Phase 2: MSR-RCAN (MS-RCAN-small + Output Refine Block)

**Status**: PLANNED

**Goal**: Add a Refine module at the output end of MS-RCAN-small.
**Expected improvement**: Push Y+crop PSNR toward 30.8~31.0 dB.

**Refine design (TBD)**:
- Input: SR output from MS-RCAN-small (before final conv)
- Structure: residual refinement with skip connection
- Purpose: refine edge details and reduce artifacts

**Target**: Test Y+crop PSNR > 30.67 dB
