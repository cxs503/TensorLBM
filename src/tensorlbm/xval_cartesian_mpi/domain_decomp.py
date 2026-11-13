"""Cartesian domain decomposition emulating pylbm's MpiTopology.

This module implements the same domain decomposition strategy used by
pylbm/pylbm for cartesian-MPI parallelization, but emulated in a single
process.  The key elements copied from pylbm's source (mpi_topology.py):

1. ``compute_dims``-style load balancing: distribute N points across P
   processes as evenly as possible (N // P + (i < N % P) for rank i).

2. Cartesian topology: each subdomain has a (rank_x, rank_y) coordinate
   and knows its neighbours in each direction.

3. Halo exchange: each subdomain owns interior points plus a 1-cell halo
   on each face.  After collision, halo values are exchanged between
   neighbouring subdomains before streaming.

4. Region indices: ``get_region_indices`` computes the global index range
   owned by each subdomain along each axis.

Reference: pylbm/pylbm mpi_topology.py (BSD 3-clause)
    https://github.com/pylbm/pylbm
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


def compute_split(total: int, nproc: int) -> List[int]:
    """Compute per-process point counts, matching pylbm's Compute_dims + distribution.

    pylbm distributes N points across P processes as:
        count[i] = N // P + (1 if i < N % P else 0)

    This ensures the difference between any two subdomains is at most 1.
    """
    base = total // nproc
    remainder = total % nproc
    return [base + (1 if i < remainder else 0) for i in range(nproc)]


def compute_dims_2d(nprocs: int) -> Tuple[int, int]:
    """Find a 2D grid (px, py) such that px * py >= nprocs, px <= py.

    Matches pylbm's use of MPI.Compute_dims for 2D domains.
    """
    if nprocs == 1:
        return (1, 1)
    # Try to find a balanced decomposition
    best = (1, nprocs)
    for px in range(1, int(nprocs**0.5) + 1):
        if nprocs % px == 0:
            py = nprocs // px
            if abs(px - py) < abs(best[0] - best[1]):
                best = (px, py)
    return best


@dataclass
class SubdomainInfo:
    """Information about one subdomain in the cartesian decomposition."""
    rank: int
    coords: Tuple[int, int]  # (ix, iy) in the process grid
    x_start: int  # global x index start (inclusive)
    x_end: int    # global x index end (exclusive)
    y_start: int  # global y index start (inclusive)
    y_end: int    # global y index end (exclusive)
    nx_local: int  # local x size (including halo)
    ny_local: int  # local y size (including halo)
    # Neighbour ranks: (left, right, bottom, top)
    # None means no neighbour (physical boundary)
    neighbours: Tuple[int | None, int | None, int | None, int | None]


@dataclass
class CartesianDecomposition:
    """Cartesian domain decomposition emulating pylbm's MpiTopology.

    Parameters
    ----------
    nx, ny : int
        Global domain size in lattice points.
    npx, npy : int
        Number of subdomains in x and y directions.
    halo : int
        Halo width (1 for D2Q9).
    periodic : Tuple[bool, bool]
        Periodicity in x and y directions.
    """
    nx: int
    ny: int
    npx: int = 2
    npy: int = 2
    halo: int = 1
    periodic: Tuple[bool, bool] = (False, False)

    # Computed fields
    nprocs: int = field(init=False)
    subdomains: List[SubdomainInfo] = field(init=False, default_factory=list)
    x_counts: List[int] = field(init=False, default_factory=list)
    y_counts: List[int] = field(init=False, default_factory=list)
    x_regions: List[List[int]] = field(init=False, default_factory=list)
    y_regions: List[List[int]] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.nprocs = self.npx * self.npy
        self._compute_regions()
        self._build_subdomains()

    def _compute_regions(self):
        """Compute region indices for each subdomain, matching pylbm's get_region_indices_."""
        # x regions
        self.x_regions = [0]
        for i in range(self.npx):
            count = self.nx // self.npx + (1 if i < self.nx % self.npx else 0)
            self.x_regions.append(self.x_regions[-1] + count)
        self.x_counts = [self.x_regions[i + 1] - self.x_regions[i] for i in range(self.npx)]

        # y regions
        self.y_regions = [0]
        for i in range(self.npy):
            count = self.ny // self.npy + (1 if i < self.ny % self.npy else 0)
            self.y_regions.append(self.y_regions[-1] + count)
        self.y_counts = [self.y_regions[i + 1] - self.y_regions[i] for i in range(self.npy)]

    def _build_subdomains(self):
        """Build subdomain info for each rank."""
        self.subdomains = []
        for iy in range(self.npy):
            for ix in range(self.npx):
                rank = iy * self.npx + ix

                # Global index range (interior only, no halo)
                x_start = self.x_regions[ix]
                x_end = self.x_regions[ix + 1]
                y_start = self.y_regions[iy]
                y_end = self.y_regions[iy + 1]

                # Local size including halo
                nx_local = (x_end - x_start) + 2 * self.halo
                ny_local = (y_end - y_start) + 2 * self.halo

                # Neighbours (left, right, bottom, top)
                left = rank - 1 if ix > 0 else (None if not self.periodic[0] else iy * self.npx + self.npx - 1)
                right = rank + 1 if ix < self.npx - 1 else (None if not self.periodic[0] else iy * self.npx)
                bottom = rank - self.npx if iy > 0 else (None if not self.periodic[1] else (self.npy - 1) * self.npx + ix)
                top = rank + self.npx if iy < self.npy - 1 else (None if not self.periodic[1] else ix)

                self.subdomains.append(SubdomainInfo(
                    rank=rank,
                    coords=(ix, iy),
                    x_start=x_start,
                    x_end=x_end,
                    y_start=y_start,
                    y_end=y_end,
                    nx_local=nx_local,
                    ny_local=ny_local,
                    neighbours=(left, right, bottom, top),
                ))

    def get_subdomain(self, rank: int) -> SubdomainInfo:
        """Get subdomain info for a given rank."""
        return self.subdomains[rank]

    def get_region_indices(self) -> Tuple[List[int], List[int]]:
        """Return (x_regions, y_regions) matching pylbm's get_region_indices."""
        return self.x_regions, self.y_regions

    def get_region(self, rank: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return ((x_start, x_end), (y_start, y_end)) for a rank.

        Matches pylbm's get_region method.
        """
        sd = self.subdomains[rank]
        return ((sd.x_start, sd.x_end), (sd.y_start, sd.y_end))

    def global_to_local(self, rank: int, gx: int, gy: int) -> Tuple[int, int]:
        """Convert global indices to local indices (including halo offset)."""
        sd = self.subdomains[rank]
        lx = gx - sd.x_start + self.halo
        ly = gy - sd.y_start + self.halo
        return (lx, ly)

    def local_to_global(self, rank: int, lx: int, ly: int) -> Tuple[int, int]:
        """Convert local indices (including halo) to global indices."""
        sd = self.subdomains[rank]
        gx = lx - self.halo + sd.x_start
        gy = ly - self.halo + sd.y_start
        return (gx, gy)

    def interior_slice(self, rank: int) -> Tuple[slice, slice]:
        """Return the slice for interior points (excluding halo) in local coords."""
        h = self.halo
        return (slice(h, -h), slice(h, -h))

    def halo_slices(self, rank: int) -> dict:
        """Return halo slice definitions for exchange.

        Returns dict with keys: 'left', 'right', 'bottom', 'top'
        Each value is a (send_slice, recv_slice) tuple in local coordinates.
        """
        h = self.halo
        sd = self.subdomains[rank]
        nx_l = sd.nx_local
        ny_l = sd.ny_local

        return {
            'left': (
                (slice(h, -h) if ny_l > 2 * h else slice(h, h), slice(0, h)),  # send: left interior edge
                (slice(h, -h) if ny_l > 2 * h else slice(h, h), slice(0, h)),  # recv: left halo
            ),
            'right': (
                (slice(h, -h) if ny_l > 2 * h else slice(h, h), slice(-h, None)),
                (slice(h, -h) if ny_l > 2 * h else slice(h, h), slice(-h, None)),
            ),
            'bottom': (
                (slice(0, h), slice(h, -h) if nx_l > 2 * h else slice(h, h)),
                (slice(0, h), slice(h, -h) if nx_l > 2 * h else slice(h, h)),
            ),
            'top': (
                (slice(-h, None), slice(h, -h) if nx_l > 2 * h else slice(h, h)),
                (slice(-h, None), slice(h, -h) if nx_l > 2 * h else slice(h, h)),
            ),
        }
