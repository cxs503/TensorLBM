"""Boundary-condition invariants of ``CavitySolver`` (CASE-CAVITY-RE5000 blocker).

The lid-driven cavity solver currently fails the properties *any* correct cavity
solver must satisfy, on a far easier case than Re = 5000 (Re = 100, n = 64):

* I1 -- after one step the macroscopic velocity on the lid row is
  (0, -0.032) instead of (u_lid, 0): the Zou/He wall solves for the wrong
  population group (3, 7, 8 instead of the incoming 4, 7, 8), so the population
  that streamed in from the *opposite* (periodically wrapped) wall is kept and a
  spurious wall-normal velocity is injected every step;
* I2 -- the solution therefore goes non-finite within ~50 steps;
* I4 -- the fluid mass drifts with it.

These tests are marked ``xfail(strict=True)``: they pin the defect (so a fix must
flip them, and removing the bug cannot go unnoticed) instead of turning the suite
red.  The measurements are taken by
``benchmarks/verified/cavity/re5000/diagnose_solver_bc.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "TensorLBM" / "src"))

from tensorlbm.verification.lid_driven_cavity.solver import (  # noqa: E402
    CavityConfig,
    CavitySolver,
)

N, RE, U_LID = 64, 100.0, 0.1


def _solver(steps: int = 400, threads: int = 2) -> CavitySolver:
    torch.set_num_threads(threads)
    return CavitySolver(CavityConfig(n=N, re=RE, u_lid=U_LID, max_steps=steps))


@pytest.mark.xfail(strict=True, reason="known defect: Zou/He lid solves for the wrong "
                                       "population group (see docs/CASE-CAVITY-RE5000-SOLVER-AUDIT.md)")
def test_lid_row_imposes_lid_velocity() -> None:
    s = _solver()
    s.step()
    _, ux, uy = s._macroscopic(s.f)
    assert abs(float(ux[N - 1, 2:-2].mean()) - U_LID) < 1e-9
    assert float(uy[N - 1, 2:-2].std()) < 1e-9


@pytest.mark.xfail(strict=True, reason="known defect: cavity solution goes non-finite at "
                                       "Re=100, n=64 within ~50 steps")
def test_solution_stays_finite_at_re100() -> None:
    s = _solver()
    for _ in range(400):
        s.step()
    _, ux, uy = s._macroscopic(s.f)
    assert bool(torch.all(torch.isfinite(ux))) and bool(torch.all(torch.isfinite(uy)))


@pytest.mark.xfail(strict=True, reason="known defect: no physical vortex (u_min/u_lid = -0.2109 "
                                       "at Re=100 per Ghia 1982 Table I)")
def test_primary_vortex_strength_re100() -> None:
    s = _solver()
    for _ in range(400):
        s.step()
    _, ux, uy = s._macroscopic(s.f)
    u_min = float(ux[:, N // 2].min()) / U_LID
    assert np.isfinite(u_min) and -0.45 < u_min < -0.10
