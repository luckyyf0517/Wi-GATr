# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#!/usr/bin/env python3.10
"""
Entrypoint to train a regression model. The model and dataset are
specified via the configuration file.
"""

import logging
import os

import hydra
import torch


def setup_ddp():
    """Initialize DDP if running under torchrun.

    Returns
    -------
    rank : int
        Global process rank.
    world_size : int
        Total number of processes.
    local_rank : int
        Local rank within node.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Initialize distributed process group
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        # Set device for this process
        torch.cuda.set_device(local_rank)
        return rank, int(os.environ["WORLD_SIZE"]), local_rank
    # Single GPU training
    return 0, 1, 0


@hydra.main(config_path="../config", config_name="wigatr_wi3r", version_base=None)
def main(cfg):
    """Entry point for the training. Functionality lives in Experiment classes."""
    # Setup distributed training if running under torchrun
    rank, world_size, local_rank = setup_ddp()

    # Suppress logging on non-rank-0 processes to avoid duplicate output
    if world_size > 1 and rank != 0:
        # Set all loggers to WARNING level for non-rank-0 processes
        logging.getLogger().setLevel(logging.WARNING)
        # Also suppress common library loggers
        for name in ["wiinsim", "wigatr", "torch", "hydra"]:
            logging.getLogger(name).setLevel(logging.WARNING)

    # Keeping the target config separate to use global config as argument
    target_cfg = {"_target_": cfg.experiment_target}
    exp = hydra.utils.instantiate(target_cfg, cfg, rank=rank, world_size=world_size, local_rank=local_rank)
    exp(train=True, evaluate=True)

    # Clean up distributed training if needed
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
