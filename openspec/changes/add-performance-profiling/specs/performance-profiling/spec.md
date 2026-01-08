# Performance Profiling Specification

## ADDED Requirements

### Requirement: Profiling Levels

The system SHALL support multiple profiling levels with different overhead and granularity trade-offs.

#### Scenario: Developer enables basic profiling during training

**Given** the developer is training the model
**When** they set `profiling.level=1` in the config
**Then** the system should record step-level timing with <1% overhead
**And** report training step breakdown (data load, forward, backward, optimizer)

#### Scenario: Developer enables detailed profiling for debugging

**Given** the developer needs to identify specific bottlenecks
**When** they set `profiling.level=2` in the config
**Then** the system should record operation-level timing with 2-5% overhead
**And** report data loading breakdown (HDF5 load, tokenize, transforms)
**And** report GPU utilization statistics

#### Scenario: Developer disables profiling for production training

**Given** the developer is running production training
**When** they set `profiling.enabled=false` or `profiling.level=0`
**Then** the system should have zero profiling overhead
**And** no profiling data should be collected

---

### Requirement: Data Loading Profiling

The system SHALL measure and report time spent in each stage of the data loading pipeline.

#### Scenario: Profile data loading bottleneck

**Given** profiling is enabled at level 2 or higher
**When** a sample is loaded by `GeometricDataset.__getitem__`
**Then** the system should measure time for each operation:
- Raw data load from HDF5
- Target computation
- Mesh loading
- Geometric transforms
- Tokenization
- E(2) augmentation
- Canonicalization
**And** report the percentage of total time per operation
**And** identify the top 3 bottlenecks

#### Scenario: Aggregate statistics over multiple steps

**Given** profiling is enabled
**When** training completes N steps (configured by `profiling.report_every_n_steps`)
**Then** the system should report statistics (mean, p50, p95, p99) for each operation
**And** generate a summary report

---

### Requirement: Training Step Profiling

The system SHALL measure and report time spent in each stage of the training step.

#### Scenario: Profile training step breakdown

**Given** profiling is enabled at level 1 or higher
**When** a training step is executed
**Then** the system should measure time for each stage:
- Data preparation (transfer to GPU)
- Forward pass
- Backward pass
- Optimizer step
**And** include GPU synchronization in measurements
**And** report the percentage of total step time per stage

#### Scenario: Calculate throughput metrics

**Given** profiling is enabled
**When** training is in progress
**Then** the system should calculate and report:
- Samples per second
- Steps per second
- Batch processing time
**And** display these metrics in the summary report

---

### Requirement: GPU Utilization Monitoring

The system SHALL monitor and report GPU utilization during training.

#### Scenario: Monitor GPU utilization during training

**Given** profiling is enabled and `profiling.gpu_monitoring=true`
**When** training is in progress
**Then** the system should query GPU statistics every 1 second:
- SM utilization (%)
- Memory utilization (%)
- Power draw (W)
- Temperature (°C)
**And** record these statistics with timestamps

#### Scenario: Correlate GPU utilization with training stages

**Given** GPU monitoring is enabled
**When** a profiling report is generated
**Then** the system should correlate GPU utilization with training stages
**And** calculate GPU idle time percentage
**And** report if GPU is the bottleneck

---

### Requirement: Summary Reports

The system SHALL generate human-readable summary reports identifying bottlenecks.

#### Scenario: Generate profiling summary report

**Given** profiling is enabled
**When** `profiling.report_every_n_steps` steps have completed
**Then** the system should generate a text report containing:
- Data loading breakdown (percentage and absolute time)
- Training step breakdown
- GPU utilization statistics
- Bottleneck identification
- Optimization recommendations
**And** save the report to `profiling.output_dir`

#### Scenario: Save profiling data for analysis

**Given** profiling is enabled
**When** a report is generated
**Then** the system should save raw timing data to CSV files
**And** save GPU utilization data to a separate CSV file
**And** include step numbers for correlation

---

### Requirement: Configuration

The profiling system SHALL be configurable via YAML configuration.

#### Scenario: Configure profiling in config file

**Given** a training config file
**When** the developer adds a `profiling` section
**Then** they should be able to configure:
- `enabled`: boolean to enable/disable profiling
- `level`: integer 0-3 for profiling detail level
- `output_dir`: directory for profiling reports
- `report_every_n_steps`: frequency of report generation
- `gpu_monitoring`: boolean to enable GPU monitoring

#### Scenario: Override profiling config via command line

**Given** a default config file
**When** the developer runs training with `profiling.enabled=true`
**Then** profiling should be enabled regardless of config file setting
**And** all other profiling parameters should be overridable

---

### Requirement: Distributed Training Support

The profiling system SHALL work correctly in distributed training (DDP) mode.

#### Scenario: Profile only on rank 0

**Given** distributed training is enabled with 4 GPUs
**When** profiling is enabled
**Then** only rank 0 should collect and report profiling data
**And** non-rank-0 processes should skip profiling to avoid overhead

#### Scenario: Aggregate metrics across GPUs

**Given** distributed training is enabled
**When** profiling report is generated
**Then** the report should indicate it's from distributed training
**And** show effective batch size (batchsize × num_gpus)
**And** show combined throughput across all GPUs

---

### Requirement: Minimal Overhead

The profiling system SHALL have minimal overhead when enabled.

#### Scenario: Level 1 profiling has < 1% overhead

**Given** profiling is enabled at level 1
**When** training is run
**Then** the profiling overhead should be < 1% of total training time
**And** should not significantly impact training speed

#### Scenario: Level 2 profiling has < 5% overhead

**Given** profiling is enabled at level 2
**When** training is run
**Then** the profiling overhead should be < 5% of total training time
**And** detailed operation timing should be available

#### Scenario: Disabled profiling has zero overhead

**Given** profiling is disabled (level 0 or enabled=false)
**When** training is run
**Then** there should be no measurable profiling overhead
**And** no timing code should execute
