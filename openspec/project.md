# Project Context

## Purpose

Wi-GATr (Wireless Geometric Algebra Transformer) is a fully-learnable neural simulation surrogate designed to predict wireless channel observations based on scene primitives (surface meshes, antenna positions, orientations). The project was published at ICLR 2025 in the paper "Differentiable and Learnable Wireless Simulation with Geometric Transformers".

Key capabilities:
- Predict signal strength (RSRP) and delay spread in wireless environments
- Receiver localization and geometry reconstruction
- Equivariant to SE(3) transformations (rotations, translations, reflections)
- Sample-efficient and robust compared to traditional ray tracing simulators
- Transferable to real-world measurements

## Tech Stack

- **Python**: 3.10
- **Deep Learning Framework**: PyTorch 2.0.1
- **Geometric Deep Learning**:
  - `gatr` (Geometric Algebra Transformer) - custom fork from Qualcomm AI Research
  - `e3nn` (Euclidean Neural Networks)
  - `torch-geometric` (Geometric Deep Learning)
- **Scientific Computing**:
  - NumPy 1.24.4
  - SciPy 1.13.1
  - Pandas 1.5.1
- **Visualization**:
  - Matplotlib 3.9.2
  - Seaborn 0.13.2
- **Configuration**: Hydra 1.3.2 with OmegaConf
- **Testing**: pytest 7.4.3
- **Code Quality**: black 23.12.0, isort 5.13.2, pylint 3.0.3
- **Data Loading**: WiInSim (custom library from Qualcomm AI Research)
- **Optimization**: xformers 0.0.20 (memory-efficient attention)
- **Algebraic**: clifford 1.4.0, einops 0.8.0

## Project Conventions

### Code Style

- **Line length**: 100 characters (black, isort)
- **Import order**: FUTURE, STDLIB, THIRDPARTY, FIRSTPARTY, LOCALFOLDER (isort with black profile)
- **Naming**:
  - Classes: `PascalCase` (e.g., `RSRPRegressionGATr`)
  - Functions/variables: `snake_case` (e.g., `embed_into_ga`)
  - Constants: `UPPER_SNAKE_CASE`
- **Docstrings**: Google style for functions and classes
- **Type hints**: Required for function signatures

### Architecture Patterns

- **Experiment-based architecture**: All training/evaluation logic in `src/wigatr/experiments/`
- **Model wrapping**: Main models wrap GATr from the external library with domain-specific embeddings
- **Geometric Algebra embeddings**: Scene primitives (points, planes, translations) embedded into multivectors
  - Points: `embed_point()`
  - Oriented planes: `embed_oriented_plane()`
  - Translations: `embed_translation()`
- **Hybrid scalar-multivector representation**: Models use both multivector (GA) and scalar channels
- **Config-driven**: Hydra configs in `config/` directory
- **Token types**: Links (0), Tx (1), Rx (2), Mesh faces (3) - encoded via one-hot in `inputs.types`

### Testing Strategy

- **Location**: `tests/` directory
- **Framework**: pytest
- **Key tests**: Equivariance tests under rotations/reflections in `tests/gatr4wi/models/test_equivariance.py`
- **Test overfitting**: Small data configs (`wi3r_test_overfit`, `wiptr_test`) for quick validation
- **Evaluation splits**:
  - Standard: train, val, test
  - Generalization: `eval_rx_gen`, `eval_floor_gen`, `eval_rotation`, `eval_translation`, `eval_reciprocity`

### Git Workflow

- **Main branch**: `main`
- **License**: BSD-3-Clause-Clear (Qualcomm Technologies)
- **Copyright headers**: Required in all source files
- **Commit style**: Conventional commits preferred

## Domain Context

### Geometric Algebra for Wireless

The core innovation is representing wireless scene elements as geometric algebra objects:

- **Tx/Rx positions**: Points in 3D space
- **Mesh faces**: Triangles represented as 3 vertices + oriented plane (normal)
- **Tx-Rx links**: Tx point + Rx point + translation vector
- **Antenna orientation**: Oriented plane through antenna position (for directional antennas)

This representation ensures SE(3) equivariance - the model predictions transform consistently with scene transformations.

### Datasets

- **Wi3R**: Wireless Indoor Ray tracing dataset with simulated scenarios
- **WiPTR**: Wireless Indoor Path and Training dataset
- Both loaded via WiInSim library

### Inverse Problems

The model supports "overrides" for inverse problems like transmitter localization:
- Override Tx/Rx positions during inference via `overrides` parameter
- Optimize overriden positions to match observed channel measurements

## Important Constraints

- **Python version**: Must be exactly 3.10.x (due to dependency compatibility)
- **GPU memory**: Training requires ~12GB GPU memory (tested with test configs)
- **External dependencies**: `gatr`, `wiinsim` are Git dependencies, not on PyPI
- **License**: BSD-3-Clause-Clear - must preserve copyright headers
- **Reproducibility**: Seeds fixed in configs (e.g., seed: 1833)

## External Dependencies

### Git Dependencies (not on PyPI)

- **gatr**: https://github.com/Qualcomm-AI-research/geometric-algebra-transformer
  - Commit: 965a157d139de1131354a3bcc3a1e5c0d58590c1
  - Core Geometric Algebra Transformer implementation

- **wiinsim**: https://github.com/Qualcomm-AI-research/WiInSim
  - Commit: 86e7199fc68879490380830d95c8a0a46035e790
  - Data loading for Wi3R and WiPTR datasets

- **opt-einsum**: https://github.com/dgasmith/opt_einsum
  - Commit: 1a984b7b75f3e532e7129f6aa13f7ddc3da66e10
  - Optimized einsum for PyTorch

### Key APIs

- **Hydra**: Configuration composition via `defaults` lists
- **PyTorch Geometric**: Attention masks via `build_pyg_attention_mask()`
- **GATr interface**: `embed_point()`, `embed_oriented_plane()`, `embed_translation()`
