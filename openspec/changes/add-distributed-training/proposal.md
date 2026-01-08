# Proposal: Add Distributed GPU Training Support

## Summary

Add distributed GPU training support using PyTorch's `DistributedDataParallel` (DDP) and `torchrun` launcher. This will enable training across multiple GPUs and nodes with minimal changes to the existing framework architecture.

## Motivation

Current limitations:
- Training is limited to single GPU, restricting model scale and dataset size
- No support for multi-GPU or multi-node training
- Training time is a bottleneck for experimentation

The proposed solution will:
- Enable training across multiple GPUs on a single machine (single-node multi-GPU)
- Support scaling to multiple machines (multi-node) if needed
- Maintain backward compatibility with single-GPU training
- Preserve the existing experiment-based architecture

## Proposed Approach

### Core Strategy

Use PyTorch's `DistributedDataParallel` (DDP) with the `torchrun` launcher. This is the recommended approach for distributed training in PyTorch and provides:

1. **Minimal framework changes**: Wrap existing model in DDP wrapper
2. **Automatic gradient synchronization**: No manual gradient averaging needed
3. **Efficient all-reduce**: Optimized communication backend
4. **Standard tooling**: `torchrun` is built into PyTorch

### Implementation Scope

**In Scope:**
- Single-node multi-GPU training (primary use case)
- Multi-node training (secondary use case)
- Distributed data sampling via `DistributedSampler`
- Proper checkpointing for rank 0 only
- Logging only on rank 0
- `torchrun` launcher script

**Out of Scope:**
- Model parallelism (training large models that don't fit on one GPU)
- Custom communication primitives
- Load balancing optimization
- Automatic mixed precision for distributed training (can be added later)

### Key Design Decisions

1. **Configuration-based enable/disable**: Add `training.distributed` boolean flag to config
2. **Minimal code changes**: Wrap model in DDP when distributed mode is enabled
3. **Backward compatible**: Single-GPU training continues to work as before
4. **Standard sampler**: Use `DistributedSampler` for data partitioning
5. **Rank 0 operations**: Only rank 0 performs logging, checkpointing, and visualization

## Architecture Impact

### Changes to Existing Code

1. **[scripts/train.py](scripts/train.py)**: Add DDP initialization and launcher support
2. **[src/wigatr/experiments/base_experiment.py](src/wigatr/experiments/base_experiment.py)**:
   - Add `rank`, `world_size`, `local_rank` attributes
   - Wrap model in `DistributedDataParallel` when distributed
   - Replace `DataLoader` with distributed-aware version
   - Add rank checks for logging/checkpointing operations
3. **[src/wigatr/experiments/regression.py](src/wigatr/experiments/regression.py)**: Update DataLoader creation for distributed sampling
4. **Config files**: Add optional distributed training parameters

### No Changes Required

- Model definitions ([src/wigatr/models/](src/wigatr/models/))
- Data loading logic ([src/wigatr/data/](src/wigatr/data/))
- Forward pass implementations
- Loss functions and metrics
- Visualization code

## Alternatives Considered

### DataParallel (DP)
- **Rejected**: Simpler API but slower than DDP due to Python GIL and single-process bottlenecks
- DDP is the recommended approach for production training

### Custom distributed framework
- **Rejected**: Over-engineering for this use case
- DDP + torchrun provides all necessary functionality

### Third-party training frameworks (DeepSpeed, FairScale)
- **Rejected**: Unnecessary complexity for current requirements
- Can be added later if needed for specific optimizations

## Open Questions

1. **How many GPUs per node to target?** - Assumption: 4-8 GPUs per node is typical
2. **Multi-node training priority?** - Assumption: Single-node multi-GPU is primary, multi-node is nice-to-have
3. **Batch size scaling?** - Assumption: Linear scaling (batch_size × num_gpus) with optional config override

## Success Criteria

- Training runs successfully on 2+ GPUs
- Training speed scales approximately linearly with number of GPUs
- Checkpoint loading/saving works correctly in distributed mode
- All logs/metrics/plots generated only on rank 0
- Single-GPU training continues to work without changes
- Existing tests pass