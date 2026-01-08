# Spec: Distributed Training Support

## ADDED Requirements

### Requirement: DDP Initialization

The training system MUST support initialization via torchrun launcher with automatic rank detection.

#### Scenario: Launch training with torchrun

**Given** the user has multiple GPUs available
**When** executing `torchrun --nproc_per_node=4 scripts/train.py`
**Then** the system should:
- Detect `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables
- Initialize the distributed process group using NCCL backend
- Assign each process to a unique GPU based on `LOCAL_RANK`
- Set global rank, world size, and local rank attributes on the experiment

#### Scenario: Single-GPU training without torchrun

**Given** the user is running standard single-GPU training
**When** executing `python scripts/train.py`
**Then** the system should:
- Set rank=0, world_size=1, local_rank=0
- NOT initialize any distributed process groups
- Train normally without any DDP overhead

### Requirement: DistributedDataParallel Model Wrapping

The system MUST wrap the model in DistributedDataParallel when training with multiple GPUs.

#### Scenario: Multi-GPU model creation

**Given** world_size > 1
**When** `create_model()` is called
**Then** the system should:
- Create the base model as usual
- Wrap the model in `torch.nn.parallel.DistributedDataParallel`
- Set device_ids to the local_rank GPU
- Configure `find_unused_parameters=False` for efficiency

#### Scenario: Single-GPU model creation

**Given** world_size == 1
**When** `create_model()` is called
**Then** the system should NOT wrap the model in DDP

### Requirement: Distributed Data Sampling

The system MUST use DistributedSampler to partition data across GPUs.

#### Scenario: Distributed training data loading

**Given** world_size > 1 and training is active
**When** `_make_data_loader()` is called with shuffle=True
**Then** the system should:
- Create a `DistributedSampler` with num_replicas=world_size and rank=self.rank
- Pass the sampler to the DataLoader instead of shuffle=True
- Call `sampler.set_epoch(epoch)` at the start of each epoch
- Ensure each GPU receives a unique subset of training data

#### Scenario: Validation data loading

**Given** world_size > 1 and validation is active
**When** `_make_data_loader()` is called with shuffle=False
**Then** the system should:
- Create a `DistributedSampler` with shuffle=False
- Ensure validation data is consistently partitioned across GPUs

### Requirement: Rank-Aware I/O Operations

The system MUST restrict logging, checkpointing, and visualization to rank 0 only.

#### Scenario: Distributed training logging

**Given** world_size > 1
**When** logging messages during training
**Then** the system should:
- Only output logs from rank 0
- Suppress all log output from non-zero ranks

#### Scenario: Distributed checkpoint saving

**Given** world_size > 1
**When** `save_model()` is called
**Then** the system should:
- Only save checkpoints on rank 0
- Skip checkpoint saving on all non-zero ranks

#### Scenario: Distributed visualization

**Given** world_size > 1
**When** `visualize()` is called
**Then** the system should:
- Only generate and save plots on rank 0
- Skip visualization on all non-zero ranks

### Requirement: Distributed Configuration

The system MUST support optional distributed training configuration via Hydra config.

#### Scenario: Enable distributed via config

**Given** a config file with `training.distributed: true`
**When** the config is loaded
**Then** the system should:
- Enable distributed mode when world_size > 1
- Support `training.distributed_backend` option (default: nccl)

#### Scenario: Default behavior

**Given** a config file without distributed settings
**When** the config is loaded
**Then** the system should:
- Default to single-GPU training
- NOT require any distributed configuration for standard use

### Requirement: Backward Compatibility

The system MUST maintain full backward compatibility with existing single-GPU training workflows.

#### Scenario: Existing training command

**Given** an existing training script or config
**When** running `python scripts/train.py` without torchrun
**Then** the system should:
- Train exactly as before without any changes
- NOT require any modifications to existing configs
- Produce identical results to the current implementation

#### Scenario: Existing config files

**Given** any existing config file in config/
**When** used with the new training code
**Then** the system should:
- Work without modifications
- Ignore distributed settings (default to disabled)
- Maintain all existing functionality

### Requirement: Distributed Training Launch

The system MUST support launching distributed training via standard torchrun commands.

#### Scenario: Single-node multi-GPU launch

**Given** a machine with 4 GPUs
**When** executing `torchrun --nproc_per_node=4 scripts/train.py`
**Then** the system should:
- Launch 4 training processes
- Assign each process to a different GPU
- Train using DDP with world_size=4

#### Scenario: Multi-node launch

**Given** multiple machines with GPUs
**When** executing torchrun with appropriate node configuration
**Then** the system should:
- Launch processes across all nodes
- Establish rendezvous between nodes
- Train using DDP across all GPUs

## MODIFIED Requirements

None - all changes are additive.

## REMOVED Requirements

None.
