# Tasks: Add Distributed GPU Training Support

## Task Breakdown

### 1. Add DDP initialization to train.py
- [ ] Import `torch.distributed` and `os` modules
- [ ] Create `setup_ddp()` function to detect torchrun environment
- [ ] Initialize process group with NCCL backend when distributed
- [ ] Set CUDA device based on LOCAL_RANK
- [ ] Pass rank, world_size, local_rank to Experiment initialization

### 2. Modify BaseExperiment for distributed support
- [ ] Add `rank`, `world_size`, `local_rank`, and `is_rank_0` attributes to `__init__`
- [ ] Store distributed parameters passed from train.py
- [ ] Update `_init_backend()` to handle distributed device assignment

### 3. Implement DDP model wrapping
- [ ] Modify `create_model()` to wrap model in `DistributedDataParallel` when world_size > 1
- [ ] Configure DDP with appropriate device_ids and find_unused_parameters=False
- [ ] Ensure EMA wrapper works with DDP if applicable

### 4. Add distributed data sampling
- [ ] Modify `_make_data_loader()` to use `DistributedSampler` when world_size > 1
- [ ] Pass rank, world_size, and shuffle to DistributedSampler
- [ ] Update training loop to call `sampler.set_epoch(epoch)` each epoch

### 5. Add rank-aware I/O operations
- [ ] Wrap logger calls in `if self.is_rank_0:` checks throughout BaseExperiment
- [ ] Update `save_model()` to only save on rank 0
- [ ] Update `visualize()` to only run on rank 0
- [ ] Update `_initialize_logger()` to only add handlers on rank 0
- [ ] Update `_initialize_experiment_folder()` to only warn on rank 0

### 6. Update RegressionExperiment for distributed training
- [ ] Modify `_make_data_loader()` in RegressionExperiment to handle distributed sampler
- [ ] Ensure torch_geometric DataLoader works with DistributedSampler

### 7. Add distributed configuration schema
- [ ] Add optional `training.distributed` boolean flag (default: false)
- [ ] Add optional `training.distributed_backend` option (default: nccl)
- [ ] Update example configs if needed (though not required)

### 8. Create torchrun launcher documentation
- [ ] Add example commands for single-node multi-GPU training
- [ ] Add example commands for multi-node training
- [ ] Document environment variables set by torchrun
- [ ] Add troubleshooting section for common issues

### 9. Test distributed training
- [ ] Test 2-GPU training on Wi3R config
- [ ] Verify checkpoint saving only on rank 0
- [ ] Verify logging only on rank 0
- [ ] Test checkpoint loading in distributed mode
- [ ] Verify training speedup scales with GPU count
- [ ] Test that single-GPU training still works

### 10. Update documentation
- [ ] Add distributed training section to README or GETTING_STARTED.md
- [ ] Document how to launch with torchrun
- [ ] Document configuration options
- [ ] Add notes on batch size scaling

## Dependencies

- Tasks 1-2 must be completed before 3-6
- Task 3 and 4 can be done in parallel after 1-2
- Task 5 depends on understanding I/O patterns (can be done alongside 3-4)
- Tasks 7-8 can be done in parallel after 1-6
- Task 9 depends on all implementation tasks (1-7)
- Task 10 can be done alongside or after implementation

## Parallelizable Work

The following can be worked on in parallel:
- Task 5 (rank-aware I/O) - independent of model wrapping logic
- Task 6 (RegressionExperiment update) - independent of BaseExperiment DDP logic
- Task 7 (config schema) - independent of implementation
- Task 8 (documentation) - can be drafted during implementation

## Validation Criteria

Each task should be validated as follows:
- Code changes follow existing style conventions
- No changes to model definitions or data loading logic
- Single-GPU training continues to work without modifications
- All existing tests pass
- Distributed training completes successfully on 2+ GPUs
