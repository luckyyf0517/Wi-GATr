# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Performance profiling utilities for identifying training bottlenecks.
"""

import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


class Profiler:
    """
    Lightweight profiler for measuring training pipeline performance.

    Supports multiple profiling levels with different overhead:
    - Level 0: Off (no overhead)
    - Level 1: Basic (step-level timing, <1% overhead)
    - Level 2: Detailed (operation-level timing, 2-5% overhead)
    - Level 3: Verbose (line-by-line profiling, >5% overhead)
    """

    def __init__(self, level: int = 0, output_dir: str = None):
        """
        Initialize profiler.

        Parameters
        ----------
        level : int
            Profiling level (0=off, 1=basic, 2=detailed, 3=verbose)
        output_dir : str, optional
            Directory to save profiling reports
        """
        self.level = level
        self.output_dir = Path(output_dir) if output_dir else None
        self.timings = defaultdict(list)
        self.counts = defaultdict(int)
        self.enabled = level > 0

    @contextmanager
    def record(self, name: str, sync_gpu: bool = False):
        """
        Context manager for timing operations.

        Parameters
        ----------
        name : str
            Name of the operation being timed
        sync_gpu : bool
            Whether to synchronize GPU before/after timing
        """
        if not self.enabled or self.level == 0:
            yield
            return

        # Synchronize GPU if requested (for level 2+)
        if sync_gpu and self.level >= 2 and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        yield

        # Synchronize GPU if requested (for level 2+)
        if sync_gpu and self.level >= 2 and torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        elapsed = end - start

        self.timings[name].append(elapsed)
        self.counts[name] += 1

    def get_stats(self, name: str):
        """
        Get statistics for a timed operation.

        Parameters
        ----------
        name : str
            Name of the operation

        Returns
        -------
        stats : dict
            Dictionary with mean, p50, p95, p99, total, count
        """
        if name not in self.timings or len(self.timings[name]) == 0:
            return {}

        timings = np.array(self.timings[name])
        return {
            "mean": float(np.mean(timings)),
            "std": float(np.std(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "p99": float(np.percentile(timings, 99)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
            "total": float(np.sum(timings)),
            "count": len(timings),
        }

    def summarize(self, step: int) -> str:
        """
        Generate a human-readable summary report.

        Parameters
        ----------
        step : int
            Current training step

        Returns
        -------
        report : str
            Formatted summary report
        """
        if not self.enabled or len(self.timings) == 0:
            return "Profiling disabled or no data collected."

        lines = [
            f"Performance Report - Step {step}",
            "=" * 50,
            "",
        ]

        # Calculate total time
        total_time = sum(stats["total"] for stats in (self.get_stats(name) for name in self.timings))
        if total_time == 0:
            total_time = 1.0  # Avoid division by zero

        # Group by prefix (data_loading, training_step, etc.)
        groups = self._group_by_prefix()

        for group_name, operations in groups.items():
            lines.append(f"{group_name}:")
            group_total = sum(self.get_stats(op)["total"] for op in operations if op in self.timings)

            for op in operations:
                if op not in self.timings or len(self.timings[op]) == 0:
                    continue

                stats = self.get_stats(op)
                pct = (stats["total"] / total_time) * 100
                lines.append(
                    f"  {op:30s}: {pct:5.1f}% ({stats['mean']*1000:6.1f} ms, "
                    f"count={stats['count']})"
                )

            lines.append("")

        # Identify bottlenecks
        lines.append("Top 5 Bottlenecks:")
        bottlenecks = sorted(
            [(name, self.get_stats(name)["total"]) for name in self.timings if name in self.timings],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        for i, (name, total) in enumerate(bottlenecks, 1):
            pct = (total / total_time) * 100
            lines.append(f"  {i}. {name}: {pct:.1f}% of total time")

        lines.append("")

        # Recommendations
        lines.append("Recommendations:")
        data_loading_time = sum(
            self.get_stats(name)["total"]
            for name in self.timings
            if name.startswith("data_") and name in self.timings
        )
        if data_loading_time > 0:
            data_pct = (data_loading_time / total_time) * 100
            if data_pct > 50:
                lines.append(f"  - Data loading is {data_pct:.1f}% of step time")
                lines.append("    Consider: increasing num_workers or preprocessing data")

        return "\n".join(lines)

    def _group_by_prefix(self):
        """Group operations by prefix for organized reporting."""
        groups = defaultdict(list)

        for name in self.timings:
            if name.startswith("data_"):
                groups["Data Loading"].append(name)
            elif name.startswith("step_"):
                groups["Training Step"].append(name)
            elif name.startswith("gpu_"):
                groups["GPU Metrics"].append(name)
            else:
                groups["Other"].append(name)

        return dict(groups)

    def save_report(self, step: int):
        """
        Save profiling report to file.

        Parameters
        ----------
        step : int
            Current training step
        """
        if not self.enabled or self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save text report
        report = self.summarize(step)
        report_file = self.output_dir / f"profiling_step_{step}.txt"
        with open(report_file, "w") as f:
            f.write(report)

        # Save CSV data
        csv_file = self.output_dir / f"timings_step_{step}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["operation", "mean", "std", "p50", "p95", "p99", "min", "max", "total", "count"])

            for name in sorted(self.timings.keys()):
                stats = self.get_stats(name)
                if stats:
                    writer.writerow(
                        [
                            name,
                            f"{stats['mean']:.6f}",
                            f"{stats['std']:.6f}",
                            f"{stats['p50']:.6f}",
                            f"{stats['p95']:.6f}",
                            f"{stats['p99']:.6f}",
                            f"{stats['min']:.6f}",
                            f"{stats['max']:.6f}",
                            f"{stats['total']:.6f}",
                            stats['count'],
                        ]
                    )

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
        self.counts.clear()


class GPUMonitor:
    """
    Monitor GPU utilization during training.

    Requires nvidia-smi to be available.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize GPU monitor.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save GPU monitoring data
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.utilization_history = []
        self.enabled = torch.cuda.is_available()

    def sample(self) -> dict:
        """
        Sample current GPU utilization.

        Returns
        -------
        stats : dict
            Dictionary with GPU statistics
        """
        if not self.enabled:
            return {}

        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return {}

            values = result.stdout.strip().split(", ")
            if len(values) != 5:
                return {}

            stats = {
                "gpu_util": float(values[0]),
                "mem_util": float(values[1]),
                "mem_used": float(values[2]),
                "mem_total": float(values[3]),
                "power": float(values[4]),
            }

            self.utilization_history.append(stats)
            return stats

        except Exception:
            return {}

    def get_average_stats(self) -> dict:
        """
        Get average GPU utilization statistics.

        Returns
        -------
        avg_stats : dict
            Average GPU statistics over monitoring period
        """
        if not self.utilization_history:
            return {}

        avg_stats = {}
        for key in self.utilization_history[0].keys():
            values = [s[key] for s in self.utilization_history]
            avg_stats[f"avg_{key}"] = float(np.mean(values))
            avg_stats[f"max_{key}"] = float(np.max(values))

        return avg_stats

    def save_data(self, step: int):
        """
        Save GPU monitoring data to CSV.

        Parameters
        ----------
        step : int
            Current training step
        """
        if not self.enabled or self.output_dir is None or not self.utilization_history:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = self.output_dir / f"gpu_step_{step}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_util", "mem_util", "mem_used", "mem_total", "power"])

            for i, stats in enumerate(self.utilization_history):
                writer.writerow(
                    [
                        i,
                        stats.get("gpu_util", ""),
                        stats.get("mem_util", ""),
                        stats.get("mem_used", ""),
                        stats.get("mem_total", ""),
                        stats.get("power", ""),
                    ]
                )

    def reset(self):
        """Clear monitoring history."""
        self.utilization_history.clear()
