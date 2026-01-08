# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#!/usr/bin/env python3.10
"""
Compute statistics (mean, std, min, max) for a target variable over dataset samples.

This script is useful for determining appropriate normalization parameters
(target_scaling) when adding new target functions like delay spread.
"""

import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from wigatr.data.utils import TARGET_DATA_FCTS

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="wigatr_wi3r", version_base=None)
def main(cfg: DictConfig):
    """Compute target statistics over dataset samples.

    Args:
        cfg: Hydra config with dataset settings. Override target_data_fct to
             compute statistics for different targets.
    """
    # Get target function name from config
    target_data_fct_name = cfg.data.get("target_data_fct", "compute_non_coherent_total_power")

    if target_data_fct_name not in TARGET_DATA_FCTS:
        logger.error(
            f"Unknown target_data_fct: '{target_data_fct_name}'. "
            f"Available: {list(TARGET_DATA_FCTS.keys())}"
        )
        raise ValueError(f"Invalid target_data_fct: {target_data_fct_name}")

    target_fct = TARGET_DATA_FCTS[target_data_fct_name]

    logger.info(f"Computing statistics for target: {target_data_fct_name}")
    logger.info(f"Dataset: {cfg.data.wiinsim.name}")

    # Import here to avoid issues before hydra initializes
    from wiinsim import MultiFloorDataset
    from wigatr.data.utils import load_wiinsim_dataset

    # Load dataset (only train split for statistics)
    num_samples = cfg.get("num_samples", 10000)
    logger.info(f"Sampling {num_samples} examples from dataset...")

    dataset, num_tx = load_wiinsim_dataset(
        cfg.data, "train", max_floor_plans=cfg.data.get("max_floor_plans", None)
    )

    # Sample indices
    dataset_size = len(dataset)
    if num_samples > dataset_size:
        logger.warning(
            f"Requested {num_samples} samples but dataset only has {dataset_size}. "
            f"Using all {dataset_size} samples."
        )
        num_samples = dataset_size

    indices = random.sample(range(dataset_size), num_samples)

    # Compute target values
    targets = []
    logger.info("Computing target values...")
    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]
        target = target_fct(sample)
        targets.append(target.item() if isinstance(target, torch.Tensor) else target)

    targets = np.array(targets)

    # Compute statistics
    mean_val = float(np.mean(targets))
    std_val = float(np.std(targets))
    min_val = float(np.min(targets))
    max_val = float(np.max(targets))
    median_val = float(np.median(targets))

    logger.info("\n" + "=" * 60)
    logger.info(f"Target Statistics for '{target_data_fct_name}'")
    logger.info("=" * 60)
    logger.info(f"  Samples:      {len(targets)}")
    logger.info(f"  Mean:         {mean_val:.4f}")
    logger.info(f"  Std:          {std_val:.4f}")
    logger.info(f"  Min:          {min_val:.4f}")
    logger.info(f"  Max:          {max_val:.4f}")
    logger.info(f"  Median:       {median_val:.4f}")
    logger.info("=" * 60)

    # Output in YAML format for easy copy-paste
    print("\n# Copy this to your config file:")
    print("target_scaling:")
    print(f"  mean: {mean_val:.4f}")
    print(f"  std:  {std_val:.4f}")
    print(f"  min:  {min_val:.4f}")
    print(f"  max:  {max_val:.4f}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
