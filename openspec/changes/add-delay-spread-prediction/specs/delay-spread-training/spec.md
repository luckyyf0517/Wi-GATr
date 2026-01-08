## ADDED Requirements

### Requirement: Target Function Extensibility

The training pipeline SHALL support multiple target functions beyond just RSRP prediction, allowing users to train models to predict different channel characteristics from the same MPC data. The system MUST validate the selected target function against a registry of available functions and provide helpful error messages when an invalid function is specified.

**Rationale**: The paper claims support for multiple predictions (RSRP, delay spread, band-limited power) but the code currently restricts to RSRP only via a hard assertion. Extensibility is needed to fulfill the paper's claims and enable future research.

#### Scenario: Train model to predict delay spread

- **GIVEN** a Wi3R or WiPTR dataset with MPC data
- **WHEN** a user sets `target_data_fct: compute_delay_spread_from_mpc` in the config
- **THEN** the training pipeline SHALL:
  - Accept the target function without assertion errors
  - Compute RMS delay spread from MPC paths for each sample
  - Use the computed delay spread as the training label
  - Train the model to predict delay spread values

#### Scenario: Unknown target function is specified

- **GIVEN** any data config
- **WHEN** a user sets `target_data_fct: nonexistent_function`
- **THEN** the system SHALL:
  - Raise a clear `ValueError` listing available target functions
  - Not crash with an assertion error
  - Provide helpful error message suggesting valid options

#### Scenario: Default target function behavior

- **GIVEN** a data config without `target_data_fct` specified
- **WHEN** the training pipeline loads the config
- **THEN** the system SHALL:
  - Default to `compute_non_coherent_total_power` (RSRP)
  - Maintain backward compatibility with existing configs
  - Not require changes to any existing training scripts

---

### Requirement: Delay Spread Computation from MPC

The system MUST compute RMS (Root Mean Square) delay spread from raw MPC (Multi-Path Component) data during training, using the standard power-weighted formula: τ_rms = sqrt(Σ P_i · (τ_i - τ_mean)² / Σ P_i), where P_i is linear power and τ_i is delay for path i.

**Rationale**: Delay spread is a key channel characteristic that affects inter-symbol interference and equalizer complexity. Computing it from MPC (rather than storing pre-computed values) maintains flexibility and reduces storage requirements.

#### Scenario: Compute delay spread for single path

- **GIVEN** MPC data with a single valid path where path_strength_db = -20 dB and delay = 50 ns
- **WHEN** `get_delay_spread_from_mpc()` is called
- **THEN** the result MUST be:
  - Delay spread: 0 ns (single path has no spread)
  - Shape: scalar tensor
  - Device/dtype: matches input MPC tensor

#### Scenario: Compute delay spread for multiple paths

- **GIVEN** MPC data with three paths:
  - Path 0: strength = -20 dB, delay = 50 ns
  - Path 1: strength = -23 dB, delay = 60 ns
  - Path 2: strength = -26 dB, delay = 70 ns
- **WHEN** `get_delay_spread_from_mpc()` is called
- **THEN** the function MUST:
  - Convert dB to linear power
  - Compute power-weighted mean delay
  - Compute RMS delay spread using the standard formula
  - Return value approximately 8-10 ns (based on manual calculation)

#### Scenario: Handle invalid paths in MPC data

- **GIVEN** MPC data where the last row is a validity mask with some paths marked as invalid (mask = 0) and some valid (mask = 1 or 2)
- **WHEN** `get_delay_spread_from_mpc()` is called
- **THEN** the function MUST:
  - Use only valid paths where mask > 0
  - Ignore invalid paths in all computations
  - Handle zero valid paths gracefully (return 0 or raise clear error)

---

### Requirement: Delay Spread Training Configuration

The system MUST allow users to configure and train models to predict delay spread using standard Hydra configs, with appropriate normalization parameters for the delay spread target variable. The system SHALL support creating new configs that inherit from existing base configs and override the target function and scaling parameters.

**Rationale**: Delay spread has different statistical properties (range: 0-100+ ns) compared to RSRP (range: -75 to -20 dB), requiring separate normalization and configuration.

#### Scenario: Configure delay spread training for Wi3R

- **GIVEN** the Wi3R dataset
- **WHEN** a user creates a config with:
  ```yaml
  defaults:
    - wi3r@_here_
    - _self_
  target_data_fct: compute_delay_spread_from_mpc
  target_scaling:
    mean: 15.0
    std: 10.0
    min: 0.0
    max: 100.0
  ```
- **THEN** the system SHALL:
  - Inherit all Wi3R settings (paths, splits, transforms)
  - Override target computation to use delay spread
  - Apply delay spread-specific normalization
  - Successfully train a model predicting delay spread

#### Scenario: Quick test with delay spread config

- **GIVEN** a `wi3r_delay_spread_test` config with minimal data (floors 0-1, tx 0,2, rx 0-30, training steps: 1001)
- **WHEN** user runs training with this config
- **THEN** the training MUST:
  - Complete in approximately 10 minutes
  - Show decreasing loss
  - Produce MAE/RMSE < 5 ns threshold
  - Save model to experiments directory
  - Not use more than 12GB GPU memory

---

### Requirement: Target Statistics Computation Tool

The system MUST provide a utility script to compute normalization parameters (mean, std, min, max) for any target variable from the actual dataset, enabling proper configuration for new target functions. The script SHALL support configurable sample sizes and output results in a format suitable for direct copy-paste into config files.

**Rationale**: Normalization is crucial for training stability and convergence. The statistics for delay spread are unknown until computed from actual data samples.

#### Scenario: Compute statistics for delay spread on Wi3R

- **GIVEN** the Wi3R dataset with 5M channels
- **WHEN** user runs:
  ```bash
  uv run python scripts/compute_target_stats.py \
    --config-name wi3r \
    target_data_fct=compute_delay_spread_from_mpc \
    num_samples=10000
  ```
- **THEN** the script SHALL:
  - Load 10,000 random samples from dataset
  - Compute delay spread for each sample using `get_delay_spread_from_mpc()`
  - Calculate and display: mean, std, min, max, median
  - Output in format suitable for copying to config
  - Complete in less than 5 minutes

#### Scenario: Output format for config integration

- **GIVEN** the statistics computation script
- **WHEN** it completes successfully
- **THEN** it MUST output in the format:
  ```yaml
  target_scaling:
    mean: 14.234
    std: 9.876
    min: 1.234
    max: 89.012
  ```
  With clear instructions to copy these values to the config file.

---

### Requirement: Backward Compatibility

The addition of delay spread support MUST NOT break any existing RSRP training functionality, configs, or scripts. All existing training commands and configurations SHALL continue to work without modification.

**Rationale**: Users rely on existing training pipelines. Breaking changes would disrupt workflows and require migration.

#### Scenario: Existing RSRP training unchanged

- **GIVEN** the current codebase with RSRP training
- **WHEN** the delay spread changes are applied
- **THEN** existing training MUST work:
  - Command: `uv run python scripts/train.py --config-name=wigatr_wi3r data_root_dir=~/datasets`
  - No config modifications needed
  - Same MAE/RMSE results as before
  - No changes to model architecture
  - No changes to training hyperparameters

#### Scenario: Existing quick test still works

- **GIVEN** the `wi3r_test_overfit` config
- **WHEN** user runs the quick test
- **THEN** results MUST match:
  - MAE < 1.0 for all splits
  - RMSE < 1.0 for all splits
  - Training completes in approximately 10 minutes
  - Same output format

---

## MODIFIED Requirements

### Requirement: Target Function Selection Validation

The system MUST validate the selected target function against the TARGET_DATA_FCTS registry and provide helpful error messages, rather than using a restrictive assertion that prevents extensibility. The validation SHALL occur at dataset load time and list all available functions when an invalid function is specified.

**Before**: Hard assertion `assert target_data_fct_name == "compute_non_coherent_total_power"` that only allows RSRP
**After**: Runtime validation checking if function exists in TARGET_DATA_FCTS registry

#### Scenario: Invalid target function with helpful error

- **GIVEN** a config with `target_data_fct: invalid_function`
- **WHEN** the dataset is loaded
- **THEN** the system MUST raise a ValueError with message:
  ```
  Unknown target_data_fct: 'invalid_function'.
  Available options: ['compute_non_coherent_total_power', 'compute_delay_spread_from_mpc', 'extract_power_db', 'extract_delay_spread']
  ```

#### Scenario: Valid target function accepted

- **GIVEN** a config with `target_data_fct: compute_delay_spread_from_mpc`
- **WHEN** the dataset is loaded
- **THEN** the system MUST:
  - Not raise any assertion error
  - Find the function in TARGET_DATA_FCTS registry
  - Complete dataset loading successfully
  - Use the target function for all samples
