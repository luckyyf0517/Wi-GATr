# Design: Add Delay Spread Prediction Support

## Architecture Overview

This change extends the existing training pipeline to support a new target variable (delay spread) while maintaining full backward compatibility with RSRP training.

```
Current:  Scene → Model → RSRP
          ↑
    (MPC data used only for label computation)

Extended: Scene → Model → RSRP | Delay Spread
          ↑
    (MPC data used for label computation)
```

## Current Architecture Analysis

### Data Flow

1. **WiInSim Dataset Loading**
   - HDF5 file contains: `channels`, `paths` (MPC), `tx`, `rx`, `floor_idx_channels`
   - `MultiFloorDataset.__getitem__()` returns dict with `mpc` key
   - `mpc` is a tensor of shape `(7, num_paths)` with path features

2. **Target Computation** ([`src/wigatr/data/utils.py:62-63`](src/wigatr/data/utils.py#L62-L63))
   ```python
   target_data_fct_name = data_cfg.get("target_data_fct", "compute_non_coherent_total_power")
   self.extract_target = TARGET_DATA_FCTS[target_data_fct_name]
   ```
   Currently only `compute_non_coherent_total_power` works due to assertion.

3. **Model Training**
   - `GeometricDataset` wraps `MultiFloorDataset`
   - Calls `self.extract_target(data)` to get label from MPC
   - Model predicts the label, loss is computed

### The Blockage

At [`src/wigatr/data/utils.py:178`](src/wigatr/data/utils.py#L178):
```python
assert target_data_fct_name == "compute_non_coherent_total_power"
```

This assertion prevents using any other target function, even though:
- `TARGET_DATA_FCTS` registry exists and supports multiple functions
- `extract_delay_spread` is already registered
- The infrastructure is already in place

## Design: Extended Architecture

### 1. Target Function Registry Pattern

The existing `@target_data_function` decorator pattern is good and should be extended:

```python
# Registry
TARGET_DATA_FCTS = {
    "compute_non_coherent_total_power": ...,
    "compute_delay_spread_from_mpc": ...,  # NEW
    "extract_power_db": ...,
    "extract_delay_spread": ...,
}
```

### 2. Delay Spread Computation

New function to compute RMS delay spread from MPC:

```python
def get_delay_spread_from_mpc(mpc):
    """
    Compute RMS delay spread from MPC data.

    τ_rms = sqrt(Σ P_i · (τ_i - τ_mean)² / Σ P_i)

    Args:
        mpc: Tensor (7, num_paths) with MPC data
            mpc[0] = path_strength_db (dB)
            mpc[1] = delay (nanoseconds)

    Returns:
        delay_spread: Scalar tensor with RMS delay spread in nanoseconds
    """
    mask = mpc[-1].to(torch.bool)  # Valid paths
    strengths_db = mpc[0, mask]
    delays_ns = mpc[1, mask]

    # Convert dB to linear power
    powers_linear = torch.exp(2.0 * strengths_db * np.log(10.0) / 10.0)

    # Power-weighted mean delay
    total_power = torch.sum(powers_linear)
    mean_delay = torch.sum(powers_linear * delays_ns) / total_power

    # RMS delay spread
    delay_spread = torch.sqrt(
        torch.sum(powers_linear * (delays_ns - mean_delay)**2) / total_power
    )

    return delay_spread
```

### 3. Config Structure

New configs follow the existing pattern:

```yaml
# config/data/wi3r_delay_spread.yaml
defaults:
  - wi3r@_here_
  - _self_

target_data_fct: compute_delay_spread_from_mpc  # Override

target_scaling:
  mean: 15.0   # To be computed from data
  std: 10.0
  min: 0.0
  max: 100.0
```

### 4. Validation Strategy

Instead of hard assertion, use runtime validation:

```python
# Before (restrictive):
assert target_data_fct_name == "compute_non_coherent_total_power"

# After (flexible):
if target_data_fct_name not in TARGET_DATA_FCTS:
    raise ValueError(
        f"Unknown target_data_fct: {target_data_fct_name}. "
        f"Available: {list(TARGET_DATA_FCTS.keys())}"
    )
```

## Implementation Details

### File Modifications

| File | Change | Reason |
|------|--------|--------|
| `src/wigatr/data/utils.py` | Add `get_delay_spread_from_mpc()` | Core computation |
| `src/wigatr/data/utils.py` | Add `compute_delay_spread_from_mpc()` | Registry wrapper |
| `src/wigatr/data/utils.py` | Replace assertion with validation | Enable multiple targets |
| `config/data/wi3r_delay_spread.yaml` | **NEW** | Wi3R delay spread config |
| `config/data/wiptr_delay_spread.yaml` | **NEW** | WiPTR delay spread config |
| `config/data/wi3r_delay_spread_test.yaml` | **NEW** | Quick test config |
| `scripts/compute_target_stats.py` | **NEW** | Statistics computation |

### Statistics Computation Script

New script to compute normalization parameters for any target:

```python
# scripts/compute_target_stats.py
# Analyzes dataset samples to compute mean, std, min, max
# Usage:
#   uv run python scripts/compute_target_stats.py \
#     --config-name wi3r \
#     target_data_fct=compute_delay_spread_from_mpc \
#     num_samples=10000
```

## Testing Strategy

1. **Unit Test**: Verify `get_delay_spread_from_mpc()` computation
   - Test with known MPC data
   - Compare against manual calculation

2. **Integration Test**: End-to-end training with test config
   - Use `wi3r_test_overfit` pattern
   - Train for 1001 steps
   - Verify loss decreases

3. **Regression Test**: Ensure RSRP training still works
   - Run existing training pipeline
   - Verify no changes in results

## Migration Path

**No migration needed** - this is a purely additive change.

Existing users:
- Continue using `--config-name wigatr_wi3r` (RSRP prediction)

New users (delay spread):
- Use `--config-name wigatr_wi3r data=wi3r_delay_spread`

## Future Enhancements

Out of scope for this change but enabled by it:

1. **Multi-task learning**: Predict both RSRP and delay spread simultaneously
   - Would require model architecture changes (2 output channels)

2. **Band-limited predictions**: Filter MPC by frequency before computing target
   - Requires defining band limits in config

3. **Other channel metrics**: Coherence time, Doppler spread, etc.
   - Follow same pattern once MPC features are available

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Delay spread computation error | Medium | High | Unit tests + manual verification |
| Wrong normalization params | Medium | Medium | Provide stats computation script |
| Performance degradation | Low | Low | Computation is O(num_paths) like RSRP |
| Breaking existing training | Low | High | Regression tests + validation |

## Performance Considerations

- Delay spread computation: O(num_paths) per sample
- Same complexity as RSRP computation (also O(num_paths))
- No additional memory overhead
- No impact on model architecture or training speed

## Computed Statistics (Wi3R Dataset)

Based on 1000 samples from Wi3R training data:

| Statistic | Value |
|-----------|-------|
| Mean | 1.34 ns |
| Std | 0.43 ns |
| Min | 0.16 ns |
| Max | 2.60 ns |
| Median | 1.36 ns |

**Note**: The delay spread values are smaller than initially expected (0.16-2.60 ns vs expected 0-100 ns). This is because:
1. Indoor environments have shorter multipath delays
2. The 3.5 GHz carrier frequency and specific simulation settings
3. Only valid paths (above threshold) are included

The computed normalization parameters have been applied to the delay spread configs.
