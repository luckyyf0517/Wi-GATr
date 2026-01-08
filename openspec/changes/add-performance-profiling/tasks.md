# Task Breakdown

## Phase 1: Core Profiling Infrastructure

### 1. Create profiler utility class
- [ ] Create `src/wigatr/utils/profiler.py`
- [ ] Implement `Profiler` class with context manager
- [ ] Add timing aggregation (mean, p50, p95, p99)
- [ ] Add report generation (text format)

### 2. Add profiling configuration
- [ ] Add `profiling` section to config files
- [ ] Add `enabled` flag
- [ ] Add `level` parameter (0-3)
- [ ] Add `output_dir` and `report_every_n_steps`

## Phase 2: Data Loading Profiling

### 3. Instrument GeometricDataset.__getitem__
- [ ] Wrap `_dataset[idx]` load with profiler
- [ ] Wrap `extract_target()` with profiler
- [ ] Wrap `_get_mesh()` with profiler
- [ ] Wrap `transform()` with profiler
- [ ] Wrap `tokenize_scene()` with profiler
- [ ] Add timing for E(2) augmentation
- [ ] Add timing for canonicalization

### 4. Add data loader metrics
- [ ] Track batch assembly time
- [ ] Track data transfer to GPU time
- [ ] Calculate samples/second throughput

## Phase 3: Training Loop Profiling

### 5. Instrument BaseExperiment._step
- [ ] Wrap data preparation with profiler
- [ ] Wrap forward pass with profiler
- [ ] Wrap backward pass with profiler
- [ ] Wrap optimizer step with profiler
- [ ] Add GPU synchronization points
- [ ] Calculate step time breakdown

### 6. Add validation profiling
- [ ] Profile `_compute_metrics()`
- [ ] Profile entire validation loop
- [ ] Track validation throughput

## Phase 4: GPU Monitoring

### 7. Implement GPU utilization monitor
- [ ] Create `GPUMonitor` class
- [ ] Query nvidia-smi for utilization stats
- [ ] Track SM utilization, memory, power, temperature
- [ ] Record at 1 Hz frequency during training

### 8. Correlate GPU with CPU timings
- [ ] Align GPU measurements with profiler stages
- [ ] Calculate GPU idle time
- [ ] Identify data loading vs computation time

## Phase 5: Reporting and Visualization

### 9. Generate summary reports
- [ ] Create text-based summary report
- [ ] Show percentage breakdown by stage
- [ ] Identify bottlenecks
- [ ] Provide optimization recommendations

### 10. Save profiling data
- [ ] Save timings to CSV for analysis
- [ ] Save GPU utilization data
- [ ] Save per-step summary

## Phase 6: Documentation and Testing

### 11. Document profiling usage
- [ ] Add profiling section to documentation
- [ ] Create example: How to profile data loading
- [ ] Create example: How to interpret reports
- [ ] Add configuration guide

### 12. Test and validate
- [ ] Test profiling overhead < 5%
- [ ] Test report generation
- [ ] Validate accuracy of measurements
- [ ] Test on both single-GPU and multi-GPU

## Dependencies

- Phase 1 must be completed before Phases 2-4
- Phases 2-4 can be done in parallel
- Phase 5 requires Phases 2-4
- Phase 6 requires Phase 5

## Validation Criteria

Each phase should be validated as follows:
- [ ] Code changes follow existing style conventions
- [ ] Profiling overhead is < 5% when enabled at level 2
- [ ] Profiling overhead is < 1% when enabled at level 1
- [ ] Reports accurately identify bottlenecks
- [ ] No performance impact when profiling is disabled
- [ ] Works correctly in distributed training (DDP)

## Summary

**Total Tasks**: 38

**Estimated Complexity**:
- Phase 1: Low (infrastructure)
- Phase 2: Medium (data pipeline instrumentation)
- Phase 3: Medium (training loop instrumentation)
- Phase 4: High (GPU monitoring)
- Phase 5: Low (reporting)
- Phase 6: Low (documentation)

**Critical Path**: Phase 1 → Phases 2-4 (parallel) → Phase 5 → Phase 6
