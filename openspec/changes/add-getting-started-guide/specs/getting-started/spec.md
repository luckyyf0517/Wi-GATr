## ADDED Requirements

### Requirement: Getting Started Documentation

The project SHALL provide comprehensive documentation for first-time users to set up their environment, configure data, and run their first experiment.

#### Scenario: User needs to know where to place downloaded data

- **GIVEN** user has downloaded Wi3R or WiPTR dataset
- **WHEN** user reads the getting started guide
- **THEN** the guide specifies the exact directory structure required

#### Scenario: User needs to verify data placement is correct

- **GIVEN** user has placed data in a directory
- **WHEN** user follows the verification steps
- **THEN** they can run commands to confirm the data structure matches expectations

#### Scenario: User wants to test the setup before full training

- **GIVEN** user has completed environment and data setup
- **WHEN** user runs the quick test overfit command
- **THEN** the test completes in ~10 minutes with expected MAE/RMSE values

#### Scenario: User encounters common setup issues

- **GIVEN** user is following the getting started guide
- **WHEN** they encounter an error during setup
- **THEN** the troubleshooting section covers common issues and solutions

### Requirement: Data Directory Structure Documentation

The documentation SHALL specify the required data directory structure for both Wi3R and WiPTR datasets.

#### Scenario: Wi3R data structure

- **GIVEN** user has extracted Wi3R.zip
- **WHEN** data_root_dir is set to the parent directory
- **THEN** `<data_root_dir>/Wi3R/main_5tx/` exists with training data
- **AND** `<data_root_dir>/Wi3R/grid/` exists for visualization
- **AND** `<data_root_dir>/Wi3R/outofdistribution/` exists for OOD splits

#### Scenario: WiPTR data structure

- **GIVEN** user has extracted WiPTR.zip
- **WHEN** data_root_dir is set to the parent directory
- **THEN** `<data_root_dir>/WiPTR/train/` exists
- **AND** `<data_root_dir>/WiPTR/val/` exists
- **AND** `<data_root_dir>/WiPTR/test/` exists

### Requirement: Environment Setup Instructions

The documentation SHALL provide step-by-step environment setup instructions using uv.

#### Scenario: Fresh environment setup

- **GIVEN** user has Python 3.10 installed
- **WHEN** user follows the environment setup steps
- **THEN** uv is installed
- **AND** `uv sync` completes successfully
- **AND** dependencies are installed in `.venv/`

### Requirement: Quick Test Validation

The documentation SHALL include a quick overfit test to validate the entire setup.

#### Scenario: Quick test overfit

- **GIVEN** user has completed environment and data setup
- **WHEN** user runs the quick test command with `data=wi3r_test_overfit`
- **THEN** training completes in approximately 10 minutes
- **AND** uses less than 12GB GPU memory
- **AND** MAE and RMSE are below 1.0 for all splits
- **AND** model is saved to `experiments/wi3r_test/wi-gatr/`

### Requirement: Troubleshooting Common Issues

The documentation SHALL include solutions to common setup problems.

#### Scenario: Data not found error

- **GIVEN** user gets a "data not found" error
- **WHEN** user checks the troubleshooting section
- **THEN** they find instructions to verify data_root_dir path
- **AND** they find commands to list directory contents

#### Scenario: GPU out of memory

- **GIVEN** user gets CUDA out of memory error
- **WHEN** user checks the troubleshooting section
- **THEN** they find instructions to reduce batch size in config
