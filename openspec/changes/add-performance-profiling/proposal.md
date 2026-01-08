# Performance Profiling and Debugging Tools

## Motivation

Current training shows severe performance bottlenecks:
- **GPU utilization**: Only 10-20% (should be 80%+)
- **Training speed**: ~3.64 it/s (evaluation), potentially much slower during training
- **CPU bottleneck**: Data loading is 97% of the training step time

**Problem**: We lack visibility into where time is being spent in the training pipeline. Without detailed profiling, we cannot:
1. Identify the exact bottleneck (HDF5 loading? Tokenization? Transforms?)
2. Measure the impact of optimizations
3. Make data-driven decisions about where to focus optimization efforts

## Proposed Solution

Add comprehensive performance profiling instrumentation to the training pipeline:

1. **Data Loading Profiling**: Measure time spent in each stage of `__getitem__`
2. **Training Step Profiling**: Break down time spent in data loading, forward pass, backward pass, optimizer step
3. **GPU Utilization Tracking**: Monitor GPU utilization during training
4. **Automated Reports**: Generate summary reports showing bottleneck distribution

## Success Criteria

1. Can identify top 3 time-consuming operations in data loading
2. Can measure GPU utilization vs idle time during training
3. Can generate a performance report showing % time spent in each pipeline stage
4. Overhead of profiling < 5% of training time

## Scope

**In Scope**:
- Profiling hooks in `GeometricDataset.__getitem__`
- Training step timing in `BaseExperiment._step`
- GPU utilization monitoring
- Summary report generation

**Out of Scope**:
- Automatic optimization (only measurement)
- Production profiling (this is a debugging tool)
- Memory profiling (focus on time bottlenecks)

## Related Changes

None (standalone debugging capability)