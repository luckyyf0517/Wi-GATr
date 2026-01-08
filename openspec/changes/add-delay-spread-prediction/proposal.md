# Add Delay Spread Prediction Support

## Summary

Enable Wi-GATr training pipeline to support RMS delay spread prediction in addition to the existing RSRP (received power) prediction. This change allows the model to predict a key channel characteristic mentioned in the ICLR 2025 paper but not currently functional in the open-source codebase.

## Motivation

The Wi-GATr paper claims support for predicting multiple scalar channel characteristics including:
- Non-coherent received power (RSRP) - currently supported
- Band-limited received power - not implemented
- Band-limited delay spread - **not implemented but has stub code**

Currently, the codebase has:
1. A registered but unused `extract_delay_spread` function that expects a `delay_spread` field in the data
2. An assertion in `load_wiinsim_dataset()` that prevents using any target function other than `compute_non_coherent_total_power`
3. The HDF5 datasets contain only raw MPC data, not pre-computed delay spread values

This change will enable delay spread prediction by computing it from MPC data during training.

## Proposed Solution

### Phase 1: Core Functionality
1. Add `compute_delay_spread_from_mpc()` function to calculate RMS delay spread from MPC paths
2. Remove the restrictive assertion in `load_wiinsim_dataset()`
3. Add proper validation for target_data_fct selection

### Phase 2: Configuration & Tooling
4. Create new data configs for delay spread training (Wi3R and WiPTR)
5. Add a statistics computation script to determine proper normalization parameters
6. Update documentation

### Phase 3: Validation
7. Add test configuration to validate delay spread training works end-to-end

## Design Considerations

### MPC Data Structure
Based on HDF5 inspection, the MPC paths array has 7 features:
- Index 0: path_strength_db (dB)
- Index 1: delay (nanoseconds)
- Index 2-3: AoA (azimuth, elevation)
- Index 4-5: AoD (azimuth, elevation)
- Index 6: valid_flag

### Delay Spread Formula
RMS delay spread: τ_rms = sqrt(Σ P_i · (τ_i - τ_mean)² / Σ P_i)
where P_i is linear power and τ_i is delay for path i

### Normalization Requirements
Delay spread has different statistical properties than RSRP:
- RSRP: ~ -75 to -20 dB (power domain)
- Delay spread: likely 0 to 100+ ns (time domain)
Separate target_scaling parameters will be needed.

## Alternatives Considered

1. **Pre-compute delay spread in dataset**: Rejected because it requires modifying HDF5 files and increases storage
2. **Add to existing compute_non_coherent_total_power**: Rejected because it would break single-responsibility and complicate the codebase
3. **Wait for upstream WiInSim support**: Rejected because WiInSim is external and we can't control its timeline

## Dependencies

- Requires WiInSim datasets with MPC data (already available)
- No new external dependencies
- Does not modify GATr library or model architecture

## Impact

### User-Visible Changes
- New training configs: `config/data/wi3r_delay_spread.yaml`, `config/data/wiptr_delay_spread.yaml`
- New utility script: `scripts/compute_target_stats.py`
- Can train models to predict delay spread instead of RSRP

### Breaking Changes
- None - this is purely additive
- Existing RSRP training continues to work unchanged

## Success Criteria

1. `compute_delay_spread_from_mpc()` correctly computes RMS delay spread from MPC
2. Training with `data=wi3r_delay_spread` completes without errors
3. Model predictions show reasonable delay spread values
4. No existing RSRP training functionality is broken

## Open Questions

1. Should we also add band-limited power prediction?
   - Decision: Deferred to future change to keep this focused

2. Should target_scaling parameters be computed from a data sample or hardcoded?
   - Decision: Provide a script to compute them, document typical values

3. Should delay spread and power be predicted simultaneously (multi-task)?
   - Decision: Start with single-task, multi-task can be a separate enhancement
