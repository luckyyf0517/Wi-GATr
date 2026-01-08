# Getting Started with Wi-GATr

This guide will help you set up the Wi-GATr project from scratch and run your first experiment. It assumes you have already downloaded the Wi3R or WiPTR datasets.

## Prerequisites

- **Python**: 3.10.x (required for dependency compatibility)
- **GPU**: NVIDIA GPU with ~12GB memory for training
- **OS**: Linux (tested on Ubuntu) or macOS with Docker support

## 1. Environment Setup

We recommend using `uv` for dependency management as it's faster and more reliable than traditional pip.

### Step 1.1: Install uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on macOS with Homebrew
brew install uv
```

### Step 1.2: Clone the repository

```bash
git clone https://github.com/Qualcomm-AI-Research/Wi-GATr
cd Wi-GATr
```

### Step 1.3: Install dependencies

```bash
# Sync the repository - this installs all dependencies in .venv/
uv sync
```

This will create a `.venv/` directory with all required packages including PyTorch, GATr, and WiInSim.

### Step 1.4: Verify installation

```bash
# Check that Python and packages are installed correctly
uv run python --version  # Should print Python 3.10.x
uv run python -c "import torch; print(torch.__version__)"  # Should print 2.0.1
uv run python -c "import gatr; print('GATr imported successfully')"
```

## 2. Data Setup

### Step 2.1: Download the datasets

Download one or both datasets:

- **Wi3R**: [Download Link](https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Wireless_Indoor_Simulations/Wi3R/Wi3R.zip)
- **WiPTR**: [Download Link](https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Wireless_Indoor_Simulations/WiPTR/WiPTR.zip)

### Step 2.2: Extract the datasets

Extract the downloaded zip files to a directory of your choice. This directory will be your `<PATH-TO-DATA>`.

```bash
# Example: extracting to ~/datasets
unzip Wi3R.zip -d ~/datasets/
unzip WiPTR.zip -d ~/datasets/
```

### Step 2.3: Verify data directory structure

After extraction, your data directory should look like this:

#### For Wi3R:

```
<PATH-TO-DATA>/
└── Wi3R/
    ├── main_5tx/
    │   ├── floor_0/
    │   ├── floor_1/
    │   └── ... (floor_4500 to floor_4999)
    ├── grid/
    │   ├── floor_0/
    │   └── ...
    └── outofdistribution/
        ├── scaled/
        ├── fourrooms/
        └── squared/
```

#### For WiPTR:

```
<PATH-TO-DATA>/
└── WiPTR/
    ├── train/
    │   ├── floor_0/
    │   └── ...
    ├── val/
    ├── test/
    ├── train_grid/
    ├── val_grid/
    └── val_ood/
```

### Step 2.4: Verify with commands

Run these commands to verify your data structure:

```bash
# Replace ~/datasets with your actual data directory
DATA_DIR=~/datasets

# Check Wi3R structure
ls -la "$DATA_DIR/Wi3R/main_5tx/"   # Should show floor_0, floor_1, etc.
ls -la "$DATA_DIR/Wi3R/grid/"        # Should show grid data floors

# Check WiPTR structure
ls -la "$DATA_DIR/WiPTR/train/"      # Should show training floors
ls -la "$DATA_DIR/WiPTR/val/"        # Should show validation floors
ls -la "$DATA_DIR/WiPTR/test/"       # Should show test floors
```

## 3. Quick Test: Overfit on Training Data

Before running full training, verify your setup with a quick overfit test. This should take about 10 minutes and use less than 12GB GPU memory.

### Step 3.1: Run the quick test

```bash
# Replace ~/datasets with your actual data directory
uv run python scripts/train.py \
  --config-name=wigatr_wi3r \
  data_root_dir=~/datasets \
  data=wi3r_test_overfit \
  exp_name=wi3r_test \
  training.steps=1001 \
  training.eval_batchsize=16 \
  training.batchsize=16 \
  training.log_every_n_steps=25
```

### Step 3.2: Expected output

At the end of training, you should see:
- MAE and RMSE below 1.0 for all splits (train, val, eval_rx_gen, eval_floor_gen, eval_rotation, eval_translation, eval_reciprocity)
- Model saved to `experiments/wi3r_test/wi-gatr/models/model_final.pt`

Example output:
```
Final Results:
- train MAE: 0.12, RMSE: 0.15
- val MAE: 0.10, RMSE: 0.13
- eval_rx_gen MAE: 0.08, RMSE: 0.11
- eval_floor_gen MAE: 0.09, RMSE: 0.12
- eval_rotation MAE: 0.11, RMSE: 0.14
- eval_translation MAE: 0.10, RMSE: 0.13
- eval_reciprocity MAE: 0.09, RMSE: 0.12
```

### Step 3.3: Evaluate the trained model

```bash
uv run python scripts/eval_regression.py --config-dir experiments/wi3r_test/wi-gatr/
```

The output should match the final results from training.

### Step 3.4: Test Tx inference (optional)

```bash
uv run python scripts/infer_tx.py \
  --config-name infer_tx_wi3r.yaml \
  exp_dir=experiments/wi3r_test/wi-gatr \
  scene=-4750 \
  steps=20
```

Expected: Tx error < 1.0 for all number of measurements.

## 4. Full Training

Once the quick test passes, you can proceed to full training.

### Step 4.1: Train on Wi3R

```bash
uv run python scripts/train.py \
  --config-name=wigatr_wi3r \
  data_root_dir=~/datasets
```

This will train for 500,000 steps and take several hours depending on your GPU.

### Step 4.2: Train on WiPTR

```bash
uv run python scripts/train.py \
  --config-name=wigatr_wiptr \
  data_root_dir=~/datasets
```

### Step 4.3: Available configs

- `wigatr_wi3r` - Full Wi3R training with Wi-GATr (RSRP prediction)
- `wigatr_wiptr` - Full WiPTR training with Wi-GATr (RSRP prediction)
- `transformer_wi3r` - Full Wi3R training with standard Transformer
- `transformer_wiptr` - Full WiPTR training with standard Transformer

### Step 4.4: Delay Spread Prediction

You can also train models to predict RMS delay spread instead of RSRP:

```bash
# Wi3R delay spread training
uv run python scripts/train.py \
  --config-name=wigatr_wi3r \
  data=wi3r_delay_spread \
  data_root_dir=~/datasets

# WiPTR delay spread training
uv run python scripts/train.py \
  --config-name=wigatr_wiptr \
  data=wiptr_delay_spread \
  data_root_dir=~/datasets
```

**Note**: The `target_scaling` parameters in delay spread configs are approximate. For optimal results, run the statistics computation script first:

```bash
# Compute accurate normalization parameters for delay spread
uv run python scripts/compute_target_stats.py \
  --config-name=wigatr_wi3r \
  data=wi3r_delay_spread \
  num_samples=10000
```

Then update the `target_scaling` values in the config file with the computed statistics.

## 5. Troubleshooting

### "Data not found" or "No such file or directory" errors

**Problem**: The script cannot find the data files.

**Solution**: Check that `data_root_dir` is set correctly:
```bash
# List what's in your data directory
ls -la ~/datasets/

# Verify the Wi3R or WiPTR folder exists
ls -la ~/datasets/Wi3R/
ls -la ~/datasets/WiPTR/

# Use absolute path if relative path doesn't work
uv run python scripts/train.py \
  --config-name=wigatr_wi3r \
  data_root_dir=/absolute/path/to/datasets
```

### CUDA out of memory

**Problem**: GPU runs out of memory during training.

**Solution**: Reduce batch size:
```bash
uv run python scripts/train.py \
  --config-name=wigatr_wi3r \
  data_root_dir=~/datasets \
  training.batchsize=32 \
  training.eval_batchsize=64
```

### Python version mismatch

**Problem**: Import errors or version conflicts.

**Solution**: Ensure you're using Python 3.10:
```bash
uv run python --version  # Should be 3.10.x

# If wrong version, you may need to reinstall dependencies
uv sync --reinstall
```

### ModuleNotFoundError: No module named 'gatr' or 'wiinsim'

**Problem**: External dependencies not installed.

**Solution**: These are Git dependencies. Run `uv sync` again:
```bash
uv sync
```

### Training takes too long

**Problem**: Full training is taking too much time.

**Solution**: Use the test overfit config for quick validation:
```bash
# Use wi3r_test_overfit for quick testing
uv run python scripts/train.py \
  --config-name=wigatr_wi3r \
  data_root_dir=~/datasets \
  data=wi3r_test_overfit \
  training.steps=1001
```

## 6. Next Steps

After successful training:

1. **Evaluate your model**:
   ```bash
   uv run python scripts/eval_regression.py --config-dir experiments/<exp_name>/<run_name>/
   ```

2. **Run Tx localization inference**:
   ```bash
   uv run python scripts/infer_tx.py \
     --config-name infer_tx_wi3r.yaml \
     exp_dir=experiments/<exp_name>/<run_name> \
     scene=0
   ```

3. **Explore the codebase**:
   - `config/` - Hydra configuration files
   - `src/wigatr/` - Source code
   - `scripts/` - Entry point scripts

## 7. Additional Resources

- **Full README**: See [README.md](README.md) for complete documentation
- **Paper**: [Differentiable and Learnable Wireless Simulation with Geometric Transformers](https://arxiv.org/abs/2410.23676) (ICLR 2025)
- **WiInSim Repository**: [github.com/Qualcomm-AI-Research/WiInSim](https://github.com/Qualcomm-AI-Research/WiInSim) - Data loading library
- **GATr Repository**: [github.com/Qualcomm-AI-Research/geometric-algebra-transformer](https://github.com/Qualcomm-AI-Research/geometric-algebra-transformer) - Core GATr implementation
