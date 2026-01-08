# Design: Distributed GPU Training

## Architecture Overview

The distributed training design follows PyTorch's DDP pattern with minimal changes to the existing architecture:

```
┌─────────────────────────────────────────────────────────┐
│                     torchrun launcher                    │
│                 (spawns N processes)                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              scripts/train.py (each rank)                │
│  - Initialize process group                             │
│  - Set device based on local_rank                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         BaseExperiment (modified for DDP)                │
│  - Wrap model in DistributedDataParallel                │
│  - Use DistributedSampler for data                      │
│  - Rank checks for I/O operations                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Training Loop (unchanged logic)             │
│  - Forward pass                                         │
│  - Backward pass (DDP handles all-reduce)               │
│  - Optimizer step                                       │
└─────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Entry Point Changes

**File**: [scripts/train.py](scripts/train.py)

Add DDP initialization before Hydra main:

```python
import torch.distributed as dist

def setup_ddp():
    """Initialize DDP if running under torchrun"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return rank, int(os.environ['WORLD_SIZE']), local_rank
    return 0, 1, 0
```

### 2. BaseExperiment Modifications

**File**: [src/wigatr/experiments/base_experiment.py](src/wigatr/experiments/base_experiment.py)

#### New Attributes

```python
self.rank: int           # Global process rank
self.world_size: int     # Total number of processes
self.is_rank_0: bool     # Convenience flag
```

#### Model Wrapping

```python
def create_model(self):
    self.model = self._create_model()

    # Wrap in DDP if distributed
    if self.world_size > 1:
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )

    self.optim, self.scheduler = self.create_optimizer_and_scheduler()
```

#### Distributed Data Sampling

```python
def _make_data_loader(self, dataset, batch_size, shuffle):
    if self.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, ...)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, ...)
```

#### Rank-Aware Operations

All I/O operations wrapped in rank checks:

```python
# Logging
if self.is_rank_0:
    logger.info("...")

# Checkpointing
if self.is_rank_0:
    self.save_model(...)

# Visualization
if self.is_rank_0:
    self.visualize(...)
```

### 3. Configuration Schema

Add to training config:

```yaml
training:
  # Existing parameters...
  distributed: false          # Enable DDP
  distributed_backend: nccl    # nccl for GPU, gloo for CPU
```

Note: `world_size` and `rank` are auto-detected from environment variables set by torchrun.

## Data Flow

### Distributed Sampling

1. Each epoch: `sampler.set_epoch(epoch)` ensures different shuffling per epoch
2. Each rank receives different subset of data
3. Gradients automatically synchronized via all-reduce during backward pass
4. All ranks have identical model weights after optimizer step

### Batch Size Scaling

Default behavior: Scale batch size by world_size

```python
effective_batch_size = cfg.training.batchsize * world_size
per_gpu_batch_size = cfg.training.batchsize  # Config specifies per-GPU batch size
```

## Synchronization Points

### Synchronized Operations
- Forward/backward pass (automatic via DDP)
- Batch normalization (automatic via DDP)

### Single-Rank Operations (Rank 0 Only)
- Logging to stdout/file
- Checkpoint saving
- Visualization/plotting
- Metric CSV writing

### All-Rank Operations
- Model evaluation (each rank evaluates its subset, metrics averaged)
- Validation (same as evaluation)

## Launching Distributed Training

### Single Node, Multiple GPUs

```bash
torchrun --nproc_per_node=4 scripts/train.py
```

### Multiple Nodes

```bash
# On master node (rank 0)
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:29500 \
  scripts/train.py
```

### Environment Variable Detection

No config changes needed - torchrun sets:
- `RANK`: Global process rank
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Local rank within node
- `MASTER_ADDR`: Master node address
- `MASTER_PORT`: Port for rendezvous

## Error Handling

### Common Issues Addressed

1. **NCCL timeouts**: Increase timeout via environment variable if needed
2. **Uneven batch sizes**: DistributedSampler handles this
3. **Random seed synchronization**: Each rank uses same seed + rank offset
4. **File system contention**: Only rank 0 writes checkpoints/logs

## Backward Compatibility

Single-GPU training requires zero changes:

```bash
# Original command still works
python scripts/train.py

# Or explicitly single-process
torchrun --nproc_per_node=1 scripts/train.py
```

The `world_size` defaults to 1 when not running under torchrun, so all existing code paths remain unchanged.

## Performance Considerations

### Expected Scaling
- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.5x speedup
- 8 GPUs: ~6.5x speedup

Linear scaling may be reduced by:
- Data loading bottlenecks (mitigated by multiple workers)
- Communication overhead (more significant for multi-node)

### Memory Considerations
- DDP adds minimal overhead (~1-2% GPU memory)
- Each GPU loads full model replica
- Batch size scales linearly with GPUs

## Migration Path

1. Add DDP support behind feature flag
2. Test with 2 GPUs on existing experiments
3. Validate checkpoint loading/saving
4. Update documentation
5. No changes required to existing workflows