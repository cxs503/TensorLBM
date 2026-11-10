#!/usr/bin/env python3
"""Scheme-order check on a *smooth* shallow-water solution (code verification).

The dam-break tests in this directory measure a solution with two
non-smooth features: the dry-bed front (moving singularity, unbounded depth
gradient) and the sonic point of the Ritter rarefaction.  Both are known to
floor any shock-capturing scheme at an observed order near one, so they cannot
by themselves prove whether the MUSCL-Hancock implementation in
:mod:`tensorlbm.shallow_water2d` is second order.

This script isolates the scheme: a small-amplitude standing gravity wave on a
constant depth (``h = 1 + 0.02 cos(2 pi x / L)``, ``u = 0``, reflective walls) is
integrated with the same solver, and the self-convergence order is measured in
the L1 norm against a fine-grid reference.  L1 is the standard norm for
convergence studies (pointwise norms are dominated by the O(dx) error of the
limiter at smooth extrema).

Usage::

    PYTHONPATH=src python benchmarks/dam_break/scheme_order_smooth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.shallow_water2d import SWEState, simulate  # noqa: E402

L = 4.0
G = 1.0
DEPTH = 1.0
AMPLITUDE = 0.02
T_END = 0.6
REFERENCE_NX = 1600
GRIDS = (50, 100, 200)
NY = 6


def initial_state(nx: int) -> SWEState:
    x = (np.arange(nx) + 0.5) * L / nx
    h = DEPTH + AMPLITUDE * np.cos(2.0 * np.pi * x / L)
    return SWEState(
        h=np.tile(h, (NY, 1)),
        hu=np.zeros((NY, nx)),
        hv=np.zeros((NY, nx)),
        dx=L / nx,
        dy=0.2,
        g=G,
    )


def main() -> int:
    reference = simulate(initial_state(REFERENCE_NX), t_end=T_END, cfl=0.4, order=2)
    x_ref = (np.arange(REFERENCE_NX) + 0.5) * L / REFERENCE_NX
    h_ref = reference.h[NY // 2]
    print(f"smooth standing wave: L={L} A={AMPLITUDE} t_end={T_END} reference nx={REFERENCE_NX}")
    for order in (1, 2):
        errors = []
        for nx in GRIDS:
            state = simulate(initial_state(nx), t_end=T_END, cfl=0.4, order=order)
            x = (np.arange(nx) + 0.5) * L / nx
            errors.append(float(np.mean(np.abs(np.interp(x, x_ref, h_ref) - state.h[NY // 2]))))
        orders = [
            float(np.log(errors[i] / errors[i + 1]) / np.log(GRIDS[i + 1] / GRIDS[i]))
            for i in range(len(errors) - 1)
        ]
        print(
            f"  order={order}: L1 errors {[f'{e:.3e}' for e in errors]} "
            f"observed p = {[f'{p:.2f}' for p in orders]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
