"""Benchmarking infrastructure for LBM solvers.

Provides wall-clock timing with warm-up, repetition, and statistical
reporting.  Designed for cross-validation: the same solver configuration
is timed across different implementations or parameter settings.

References
----------
* ASME V&V 20-2009, §5.3: "Numerical Uncertainty from Iterative Convergence"
* OpenFOAM bench: warm-up + N repetitions, report mean/std/min/max.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Parameters
    ----------
    n_warmup : int
        Number of warm-up iterations (not timed).
    n_repeats : int
        Number of timed repetitions.
    label : str
        Human-readable label for this benchmark.
    """
    n_warmup: int = 3
    n_repeats: int = 10
    label: str = ""


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes
    ----------
    label : str
        Benchmark label.
    times : list[float]
        Wall-clock times per repetition (seconds).
    mean : float
        Mean time.
    std : float
        Standard deviation.
    min : float
        Minimum time.
    max : float
        Maximum time.
    n_steps : int
        Number of LBM steps per repetition (if applicable).
    mlups : float or None
        Million lattice-site updates per second (if n_steps and grid size known).
    """
    label: str
    times: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    n_steps: int = 0
    mlups: Optional[float] = None

    def compute_stats(self) -> None:
        """Compute summary statistics from ``self.times``."""
        if not self.times:
            return
        self.mean = float(np.mean(self.times))
        self.std = float(np.std(self.times))
        self.min = float(np.min(self.times))
        self.max = float(np.max(self.times))

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"[{self.label}] "
            f"mean={self.mean:.4f}s std={self.std:.4f}s "
            f"min={self.min:.4f}s max={self.max:.4f}s "
            f"n={len(self.times)}"
            + (f" MLUPS={self.mlups:.2f}" if self.mlups else "")
        )


def run_benchmark(
    step_fn: Callable[[], Any],
    config: BenchmarkConfig,
    n_steps: int = 1,
    grid_size: Optional[int] = None,
) -> BenchmarkResult:
    """Run a benchmark: warm-up + timed repetitions.

    Parameters
    ----------
    step_fn : callable
        A function that performs one "unit of work" (e.g. one LBM step,
        or a full simulation).  Called ``n_steps`` times per repetition.
    config : BenchmarkConfig
        Benchmark configuration.
    n_steps : int
        Number of calls to ``step_fn`` per repetition.
    grid_size : int or None
        Total number of lattice sites (for MLUPS calculation).

    Returns
    -------
    BenchmarkResult
    """
    result = BenchmarkResult(label=config.label, n_steps=n_steps)

    # Warm-up
    for _ in range(config.n_warmup):
        for _ in range(n_steps):
            step_fn()

    # Timed repetitions
    for _ in range(config.n_repeats):
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step_fn()
        elapsed = time.perf_counter() - t0
        result.times.append(elapsed)

    result.compute_stats()

    # MLUPS
    if grid_size and n_steps > 0 and result.mean > 0:
        result.mlups = grid_size * n_steps / result.mean / 1e6

    return result
