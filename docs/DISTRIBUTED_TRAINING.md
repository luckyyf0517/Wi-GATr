# Distributed GPU Training

This document describes how to use distributed GPU training with Wi-GATr.

## Overview

Wi-GATr supports distributed GPU training using PyTorch's `DistributedDataParallel` (DDP) and the `torchrun` launcher. This enables training across multiple GPUs on a single machine (single-node multi-GPU) or across multiple machines (multi-node).

## Prerequisites

- PyTorch with CUDA support
- Multiple GPUs available on the system
- For multi-node training: network connectivity between nodes

## Single-Node Multi-GPU Training

### Basic Usage

Train on 4 GPUs on a single machine:

```bash
torchrun --nproc_per_node=4 scripts/train.py
```

Train on all available GPUs:

```bash
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/train.py
```

### Custom Config

```bash
torchrun --nproc_per_node=4 scripts/train.py --config-name wigatr_wiptr
```

### How It Works

- Each GPU runs a separate process
- Data is automatically partitioned across GPUs using `DistributedSampler`
- Gradients are synchronized via all-reduce during backward pass
- Only rank 0 (GPU 0) performs logging, checkpointing, and visualization

## Multi-Node Training

### Basic Usage

On the master node (rank 0):

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  scripts/train.py
```

On worker nodes, run the same command (they will connect to the master).

### Environment Variables

Set these before running torchrun:

- `MASTER_ADDR`: IP address or hostname of the master node
- `MASTER_PORT`: Port for rendezvous (default: 29500)

Example:

```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  scripts/train.py
```

## Environment Variables Set by torchrun

When you run `torchrun`, it automatically sets these environment variables:

- `RANK`: Global process rank (0 to world_size-1)
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Local rank within a node (0 to nproc_per_node-1)
- `MASTER_ADDR`: Master node address (for multi-node)
- `MASTER_PORT`: Port for rendezvous (for multi-node)

These are auto-detected by the training script - no configuration needed.

## Batch Size Scaling

When using multiple GPUs, the **effective batch size** scales with the number of GPUs:

- Config `batchsize: 64` with 4 GPUs = effective batch size of 256
- Each GPU processes 64 samples per iteration
- Gradients are averaged across all GPUs

To keep the same effective batch size when increasing GPUs, reduce the config batchsize accordingly.

## Performance Expectations

Expected speedup (ideal conditions):

- 2 GPUs: ~1.8x faster than 1 GPU
- 4 GPUs: ~3.5x faster than 1 GPU
- 8 GPUs: ~6.5x faster than 1 GPU

Actual speedup may vary due to:
- Data loading bottlenecks (mitigated by `num_workers` in DataLoader)
- Communication overhead (more significant for multi-node)
- Model architecture and batch size

## Logging and Checkpointing

In distributed mode:

- **Logging**: Only rank 0 writes to stdout and log files
- **Checkpoints**: Only rank 0 saves model checkpoints
- **Visualization**: Only rank 0 generates plots
- **Metrics CSV**: Only rank 0 writes evaluation metrics

This prevents file system contention and duplicate output.

## Troubleshooting

### NCCL Timeout Error

If you encounter NCCL timeouts, try:

```bash
export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node=4 scripts/train.py
```

Or increase timeout:

```bash
export NCCL_BLOCKING_WAIT=1
torchrun --nproc_per_node=4 scripts/train.py
```

### Out of Memory

If a GPU runs out of memory:

1. Reduce `batchsize` in the config
2. Reduce model size (fewer layers, smaller hidden dimensions)
3. Enable mixed precision training (`float16: true` in config)

### Uneven Data Partitions

`DistributedSampler` automatically handles uneven batch sizes. The last batch on each GPU may be smaller if the dataset size isn't evenly divisible.

### Port Already in Use

If the default port is in use:

```bash
export MASTER_PORT=29501
torchrun --nproc_per_node=4 scripts/train.py
```

## Backward Compatibility

Single-GPU training still works exactly as before:

```bash
# Both commands work identically
python scripts/train.py
torchrun --nproc_per_node=1 scripts/train.py
```

No configuration changes are needed for existing workflows.

## Examples

### Quick Test with 2 GPUs

```bash
torchrun --nproc_per_node=2 scripts/train.py training.steps=1000
```

### Full Training on 8 GPUs

```bash
torchrun --nproc_per_node=8 scripts/train.py --config-name wigatr_wi3r
```

### Multi-Node Training (2 nodes, 4 GPUs each)

On master node (192.168.1.100):

```bash
export MASTER_ADDR=192.168.1.100
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=456 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.1.100:29500 \
  scripts/train.py
```

On worker node (192.168.1.101):

```bash
export MASTER_ADDR=192.168.1.100
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=456 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.1.100:29500 \
  scripts/train.py
```

## Implementation Details

The distributed training implementation uses:

1. **DistributedDataParallel (DDP)**: Wraps the model for distributed training
2. **DistributedSampler**: Partitions data across GPUs
3. **NCCL Backend**: GPU-to-GPU communication (optimized for NVIDIA GPUs)
4. **Process Groups**: Automatically initialized by torchrun

Key features:
- Minimal changes to the training framework
- Automatic gradient synchronization
- Rank-aware I/O operations
- Backward compatible with single-GPU training

For more details, see the implementation in:
- [scripts/train.py](../scripts/train.py) - DDP initialization
- [src/wigatr/experiments/base_experiment.py](../src/wigatr/experiments/base_experiment.py) - Distributed training logic
