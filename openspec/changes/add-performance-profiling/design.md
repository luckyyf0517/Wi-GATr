# Performance Profiling Design

## Overview

This document describes the design for adding performance profiling instrumentation to identify bottlenecks in the training pipeline.

## Architecture

### Profiling Levels

```
Level 0: Off (no overhead)
Level 1: Basic (step-level timing, <1% overhead)
Level 2: Detailed (operation-level timing, 2-5% overhead)
Level 3: Verbose (line-by-line profiling, >5% overhead)
```

### Components

#### 1. Data Loading Profiler

**Location**: `src/wigatr/data/geometric.py`

**Instrumentation Points**:
```python
class GeometricDataset:
    def __getitem__(self, idx):
        with profiler.record("raw_load"):
            data = self._dataset[idx]

        with profiler.record("target_compute"):
            target = self.extract_target(data)

        with profiler.record("mesh_load"):
            mesh, materials = self._get_mesh(data["floor_idx"])

        with profiler.record("transforms"):
            tx, rx, mesh = transform(...)

        with profiler.record("tokenize"):
            data = tokenize_scene(...)

        return data
```

**Metrics**:
- Time per operation (mean, p50, p95, p99)
- Total time per sample
- Bottleneck identification

#### 2. Training Step Profiler

**Location**: `src/wigatr/experiments/base_experiment.py`

**Instrumentation Points**:
```python
def _step(self, data, step, val_data, val_loader):
    with profiler.record("data_load"):
        data = self._prep_data(data, device=self.device)

    with profiler.record("forward"):
        loss, metrics = self._forward(*data)

    with profiler.record("backward"):
        loss.backward()

    with profiler.record("optimizer"):
        self.optim.step()

    with profiler.record("gpu_sync"):
        torch.cuda.synchronize()
```

**Metrics**:
- Time per stage (data load, forward, backward, optimizer)
- GPU utilization during each stage
- Throughput (samples/second)
- Step time breakdown (pie chart)

#### 3. GPU Utilization Monitor

**Location**: `src/wigatr/utils/profiler.py`

**Metrics**:
- GPU SM utilization (%)
- GPU memory utilization (%)
- Power draw (W)
- Temperature (°C)

**Frequency**: 1 Hz (every second during training)

#### 4. Summary Report Generator

**Location**: `src/wigatr/utils/profiler.py`

**Output**:
```
Performance Report - Step 1000
================================

Data Loading Breakdown:
  Raw HDF5 Load:    45.2% (890 ms)
  Tokenization:     30.1% (593 ms)
  Transforms:       15.3% (301 ms)
  Target Compute:    5.2% (102 ms)
  Mesh Load:         4.2% ( 83 ms)

Training Step Breakdown:
  Data Load:        82.3% (1971 ms)
  Forward Pass:      8.5% ( 204 ms)
  Backward Pass:     7.8% ( 187 ms)
  Optimizer Step:    1.4% (  34 ms)

GPU Utilization:
  Average: 15.2%
  Peak: 22.1%
  Idle Time: 84.8%

Bottleneck: Data loading (82.3% of step time)
Recommendation: Increase num_workers or preprocess data
```

## Implementation Details

### Configuration

Add to `config/wigatr_wi3r.yaml`:

```yaml
profiling:
  enabled: true
  level: 2  # 0=off, 1=basic, 2=detailed, 3=verbose
  output_dir: ${exp_dir}/profiling
  report_every_n_steps: 1000
  gpu_monitoring: true
```

### Profiler Class

```python
# src/wigatr/utils/profiler.py

class Profiler:
    def __init__(self, level, output_dir):
        self.level = level
        self.timings = defaultdict(list)
        self.gpu_monitor = GPUMonitor()

    @contextmanager
    def record(self, name):
        """Context manager for timing operations"""
        if self.level == 0:
            yield
            return

        start = time.perf_counter()
        if self.level >= 2:
            torch.cuda.synchronize()
        yield
        if self.level >= 2:
            torch.cuda.synchronize()
        end = time.perf_counter()

        self.timings[name].append(end - start)

    def summarize(self, step):
        """Generate summary report"""
        report = self._generate_report()
        self._save_report(step, report)
        self.timings.clear()  # Reset for next period
```

### Integration Points

1. **Dataset**: Wrap operations in `with profiler.record(...)`
2. **Training Loop**: Add timing in `_step()` and `_compute_metrics()`
3. **Initialization**: Create profiler instance in `BaseExperiment.__init__`
4. **Reporting**: Call `profiler.summarize(step)` every N steps

## Trade-offs

### Profiling Overhead

| Level | Overhead | Granularity |
|-------|----------|-------------|
| 0 (off) | 0% | None |
| 1 (basic) | <1% | Step-level |
| 2 (detailed) | 2-5% | Operation-level |
| 3 (verbose) | >5% | Line-level |

**Recommendation**: Default to level 1, use level 2 for debugging.

### Memory Overhead

- Timing data: ~1 MB per 1000 steps (negligible)
- GPU monitoring: ~10 KB per sample (negligible)

## Alternative Approaches Considered

### 1. PyTorch Profiler (rejected)

**Pros**:
- Built-in to PyTorch
- Detailed kernel-level profiling

**Cons**:
- High overhead (>10%)
- Complex output
- Doesn't profile custom Python code well

**Decision**: Use custom lightweight profiler for Python-level profiling.

### 2. Py-Spy (rejected)

**Pros**:
- No code modification needed
- Low overhead

**Cons**:
- External dependency
- Doesn't work well with multiprocessing
- Requires root access

**Decision**: Build custom profiling into the codebase.

## Dependencies

None (pure Python implementation using standard libraries).

## Migration Path

1. Add profiler class
2. Instrument data loading
3. Instrument training loop
4. Add GPU monitoring
5. Generate reports
6. (Optional) Add visualization
