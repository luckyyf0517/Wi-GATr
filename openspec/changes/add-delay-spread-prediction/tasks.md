# Tasks: Add Delay Spread Prediction Support

## Overview

This document tracks the implementation tasks for adding delay spread prediction support to Wi-GATr.

## Task List

### Phase 1: Core Functionality

- [x] **T1.1**: Add `get_delay_spread_from_mpc()` function to `src/wigatr/data/utils.py`
  - Compute RMS delay spread from MPC data
  - Formula: τ_rms = sqrt(Σ P_i · (τ_i - τ_mean)² / Σ P_i)
  - Use mpc[0] for path_strength_db, mpc[1] for delays
  - Return scalar tensor in nanoseconds
  - **Validation**: Unit test with known MPC data ✓

- [x] **T1.2**: Add `compute_delay_spread_from_mpc()` target function wrapper
  - Decorate with `@target_data_function`
  - Call `get_delay_spread_from_mpc(data["mpc"])`
  - **Validation**: Verify it's registered in TARGET_DATA_FCTS ✓

- [x] **T1.3**: Remove restrictive assertion in `load_wiinsim_dataset()`
  - Replace `assert target_data_fct_name == "compute_non_coherent_total_power"`
  - Add proper validation with helpful error message
  - **Validation**: Test that both RSRP and delay spread functions work ✓

### Phase 2: Configuration & Tooling

- [x] **T2.1**: Create statistics computation script
  - New file: `scripts/compute_target_stats.py`
  - Load dataset samples and compute target statistics
  - Output: mean, std, min, max for normalization
  - **Validation**: Run on Wi3R data, verify reasonable values ✓

- [x] **T2.2**: Compute delay spread statistics for Wi3R
  - Run `compute_target_stats.py` with `target_data_fct=compute_delay_spread_from_mpc`
  - Sample ~10,000 examples
  - Document results in design.md
  - **Validation**: Statistics make physical sense (0-100ns range) ✓
  - Result: mean=1.34ns, std=0.43ns, range=[0.16, 2.60]ns

- [x] **T2.3**: Create Wi3R delay spread config
  - New file: `config/data/wi3r_delay_spread.yaml`
  - Inherit from `wi3r` via `defaults`
  - Override `target_data_fct` and `target_scaling`
  - **Validation**: Config loads without errors ✓

- [x] **T2.4**: Create WiPTR delay spread config
  - New file: `config/data/wiptr_delay_spread.yaml`
  - Follow same pattern as Wi3R config
  - Use WiPTR-specific statistics
  - **Validation**: Config loads without errors ✓

- [x] **T2.5**: Create test overfit config for delay spread
  - New file: `config/data/wi3r_delay_spread_test.yaml`
  - Use minimal data (floors 0-1, tx 0,2, rx 0-30)
  - For quick validation during development
  - **Validation**: Matches pattern of `wi3r_test_overfit` ✓

### Phase 3: Validation & Documentation

- [ ] **T3.1**: Run quick test training
  - Train with new config for 1001 steps
  - Command: `uv run python scripts/train.py --config-name=wigatr_wi3r data=wi3r_delay_spread_test ...`
  - **Validation**: Training completes, loss decreases

- [ ] **T3.2**: Verify RSRP training still works
  - Run existing quick test: `data=wi3r_test_overfit`
  - Compare results with baseline
  - **Validation**: No regression, same MAE/RMSE

- [x] **T3.3**: Update documentation
  - Add delay spread training example to README.md
  - Document new configs in GETTING_STARTED.md
  - Add note about target_data_fct parameter
  - **Validation**: Documentation is clear and accurate

- [x] **T3.4**: Add inline comments and docstrings
  - Document MPC data structure in `get_delay_spread_from_mpc()`
  - Explain delay spread formula and units
  - **Validation**: Code is self-documenting ✓

## Dependencies

```
T1.1 → T1.2 → T1.3
T1.3 → T2.1 → T2.2
T2.2 → T2.3, T2.4
T2.3 → T2.5
T2.5 → T3.1
T3.1 → T3.2 → T3.3 → T3.4
```

## Parallelizable Work

These tasks can be done in parallel:
- (T2.3, T2.4, T2.5) - Config creation can happen after T2.2
- Documentation (T3.3, T3.4) can be written alongside implementation

## Definition of Done

A task is complete when:
1. Code is written and follows project conventions
2. File has proper copyright header
3. Function has docstring (Google style)
4. Code passes existing linting (black, isort, pylint)
5. Validation criteria are met
6. No regressions in existing functionality

## Estimated Effort

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: Core | T1.1-T1.3 | 2-3 hours |
| Phase 2: Config/Tools | T2.1-T2.5 | 3-4 hours |
| Phase 3: Validation | T3.1-T3.4 | 2-3 hours |
| **Total** | **14 tasks** | **7-10 hours** |
