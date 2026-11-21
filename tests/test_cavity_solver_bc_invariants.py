"""Lid velocity, finite fields and primary-vortex regression at Re=100 and n=64."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.verification.lid_driven_cavity.solver import (  # noqa: E402
    CavityConfig,
    CavitySolver,
)

N, RE, U_LID = 64, 100.0, 0.1


def _solver(steps: int = 400, threads: int = 2) -> CavitySolver:
    torch.set_num_threads(threads)
    return CavitySolver(CavityConfig(n=N, re=RE, u_lid=U_LID, max_steps=steps))


def test_lid_row_imposes_lid_velocity() -> None:
    s = _solver()
    s.step()
    _, ux, uy = s._macroscopic(s.f)
    assert abs(float(ux[N - 1, 2:-2].mean()) - U_LID) < 1e-9
    assert float(uy[N - 1, 2:-2].std()) < 1e-9


def test_solution_stays_finite_at_re100() -> None:
    s = _solver()
    for _ in range(400):
        s.step()
    _, ux, uy = s._macroscopic(s.f)
    assert bool(torch.all(torch.isfinite(ux))) and bool(torch.all(torch.isfinite(uy)))


def test_primary_vortex_strength_re100() -> None:
    s = _solver()
    for _ in range(400):
        s.step()
    _, ux, uy = s._macroscopic(s.f)
    u_min = float(ux[:, N // 2].min()) / U_LID
    assert np.isfinite(u_min) and -0.45 < u_min < -0.10
