"""Anchor contracts for the 3D Taylor-Green cross-validation case.

The m18155 audit (finding 3) found the shipped reference exponent was
``2 nu k^2`` -- the *two-dimensional* Taylor-Green decay -- for a field whose
every velocity component carries a normal mode in all three axes.  These
contracts pin the corrected anchor so the defect cannot reappear:

1. the velocity amplitude decays with ``3 nu k^2`` (continuum) and with
   ``nu * 6 (1 - cos k)`` (lattice eigenvalue), and the lattice value tends to
   the continuum value as the grid is refined (gap ``k^2/12``);
2. a short free-decay run reproduces the analytic rate within 5 %, i.e. the
   viscosity implied by the measurement matches ``(tau - 1/2)/3`` -- the
   enforcement the audit asked for ("a regression test pinning the fitted nu").
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.xval_mpi_ghost_fused.campaign import stream_periodic
from tensorlbm.xval_mpi_ghost_fused.case import TGV, tgv_error


def test_velocity_amplitude_decay_exponent_is_3_not_2() -> None:
    """Continuum: |u| ~ exp(-3 nu k^2 t); the 2D exponent must be falsified."""
    case = TGV(n=64, u0=0.04, tau=0.65, dtype=torch.float64)
    nu, k = case.nu, case.k
    t = 400.0
    continuum_3d = math.exp(-3.0 * nu * k**2 * t)          # Stokes, 3D single mode
    lattice_3d = math.exp(-nu * 6.0 * (1.0 - math.cos(k)) * t)  # same, discrete lap
    rejected_2d = math.exp(-2.0 * nu * k**2 * t)           # the R1 defect
    assert case.decay(t) == pytest.approx(lattice_3d, rel=1e-15)
    # the shipped factor differs from the continuum 3D one only by the k^2/12 gap
    assert case.decay(t) == pytest.approx(continuum_3d, rel=1e-3)
    # and it is *not* the 2D exponent: the ratio is exp(-nu k^2 t) to 0.1 %
    assert case.decay(t) / rejected_2d == pytest.approx(math.exp(-nu * k**2 * t), rel=1e-3)
    # kinetic energy is the square of the amplitude factor
    assert case.energy_decay(t) == pytest.approx(lattice_3d**2, rel=1e-15)


def test_lattice_eigenvalue_tends_to_three_k_squared() -> None:
    """Discrete Laplacian eigenvalue 6(1-cos k) -> 3k^2, gap k^2/12 leading order."""
    gaps = []
    for n in (16, 32, 64, 128):
        case = TGV(n=n, dtype=torch.float64)
        discrete = case.laplacian_eigenvalue(discrete=True)
        continuum = case.laplacian_eigenvalue(discrete=False)
        assert discrete < continuum  # 2(1-cos k) <= k^2
        gap = 1.0 - discrete / continuum
        assert gap == pytest.approx(case.k**2 / 12.0, rel=0.02)
        gaps.append(gap)
    assert all(later < earlier for earlier, later in zip(gaps, gaps[1:]))


def test_free_decay_pins_the_viscosity_within_5_percent() -> None:
    """Short linear-regime run: fitted decay rate vs nu * |lap eigenvalue| < 5 %."""
    torch.set_num_threads(4)
    case = TGV(n=32, u0=0.0025, tau=0.65, dtype=torch.float64)
    steps = 100
    field = case.initial_field()
    for _ in range(steps):
        field = stream_periodic(case.collide(field))
    metrics = tgv_error(field, case, steps)
    assert metrics["decay_rate_rel_error"] == pytest.approx(0.0, abs=0.05)
    # the linear regime must also land inside the owner 5 % engineering target
    assert metrics["tgv_rel_l2"] < 0.05
    # Mach number and density bounds of the declared regime
    assert case.mach < 0.01
    assert 0.99 < metrics["rho_min"] < metrics["rho_max"] < 1.01


def test_nonlinear_regime_is_reported_not_hidden() -> None:
    """At U0 = 0.04 (re_k ~ 8) the linear anchor is outside its regime.

    The error must be visible and larger than in the linear regime: the case
    file may never tune the anchor to make the nonlinear run look good.
    """
    torch.set_num_threads(4)
    linear = TGV(n=32, u0=0.0025, tau=0.65, dtype=torch.float64)
    nonlinear = TGV(n=32, u0=0.04, tau=0.65, dtype=torch.float64)
    steps = 100
    errors = {}
    for name, case in (("linear", linear), ("nonlinear", nonlinear)):
        field = case.initial_field()
        for _ in range(steps):
            field = stream_periodic(case.collide(field))
        errors[name] = tgv_error(field, case, steps)["tgv_rel_l2"]
    assert linear.re_k < 1.0 < nonlinear.re_k
    assert errors["nonlinear"] > errors["linear"]