"""Lattice definitions for von Neumann analysis.

Provides D2Q9 and D3Q19 with exact rational weights for high-precision analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LatticeBase:
    """Base class for lattice velocity sets."""

    name: str
    dim: int
    q: int  # number of discrete velocities
    c: np.ndarray  # (q, dim) lattice velocities
    w: np.ndarray  # (q,) weights
    cs2: float  # speed of sound squared

    @property
    def opposite(self) -> np.ndarray:
        """Return opposite direction indices."""
        raise NotImplementedError

    def check_isotropy(self, order: int = 4) -> dict:
        """Check lattice isotropy up to given order."""
        # For D2Q9/D3Q19, check that sum_q w_q c_q^a c_q^b = cs2 * delta_ab
        # and higher moments
        results = {}
        for a in range(self.dim):
            for b in range(self.dim):
                moment = np.sum(self.w * self.c[:, a] * self.c[:, b])
                expected = self.cs2 if a == b else 0.0
                results[f"c2_{a}{b}"] = {"value": moment, "expected": expected, "error": abs(moment - expected)}
        return results


class D2Q9Lattice(LatticeBase):
    """D2Q9 lattice: 9 velocities in 2D."""

    def __init__(self):
        c = np.array([
            [0, 0],   # rest
            [1, 0],   # cardinal
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],   # diagonal
            [-1, 1],
            [-1, -1],
            [1, -1],
        ], dtype=np.float64)

        # Exact rational weights
        w = np.array([
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ], dtype=np.float64)

        super().__init__(name="D2Q9", dim=2, q=9, c=c, w=w, cs2=1.0 / 3.0)

    @property
    def opposite(self) -> np.ndarray:
        return np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


class D3Q19Lattice(LatticeBase):
    """D3Q19 lattice: 19 velocities in 3D."""

    def __init__(self):
        c = np.array([
            [0, 0, 0],    # rest
            [1, 0, 0],    # face centers
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],    # edge centers
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 0, 1],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
        ], dtype=np.float64)

        # Exact rational weights
        w = np.array([
            1.0 / 3.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ], dtype=np.float64)

        super().__init__(name="D3Q19", dim=3, q=19, c=c, w=w, cs2=1.0 / 3.0)

    @property
    def opposite(self) -> np.ndarray:
        return np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)
