"""Unit + smoke tests for tensorlbm.oscillatory_cylinder_inertia.

The physics gate is in docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md (benchmark table);
these tests lock the *machinery*: geometry readings, harmonic orthogonality,
the coefficient normalisation, the same-mask potential reference, and that the
arm runs, is stable and produces two mutually consistent force estimators.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.oscillatory_cylinder_inertia import (
    CASES,
    circle_mask,
    geometry_readout,
    harmonic_decompose,
    inertia_coefficient,
    main,
    potential_reference,
    run_arm,
)


def test_geometry_readout_matches_circle():
    D = 16
    geo = geometry_readout(circle_mask(128, 128, D))
    assert abs(geo["D_area"] - D) / D < 0.03
    assert geo["D_area"] < geo["D_halfway"]
    perim_ideal = math.pi * D
    assert 0.75 < geo["perimeter_cells"] / perim_ideal < 1.30
    # halfway diameter must stay within one cell of the area diameter
    assert geo["D_halfway"] - geo["D_area"] < 1.5


def test_harmonic_decompose_recovers_synthetic():
    omega = 2 * math.pi / 100.0   # T = 100 steps -> integer-period windows
    t = np.arange(1, 401, dtype=float)
    y = 2.0 * np.cos(omega * t) + 0.5 * np.sin(omega * t) + 0.3
    d = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=2)
    assert abs(d["cos"] - 2.0) < 1e-9
    assert abs(d["sin"] - 0.5) < 1e-9
    assert abs(d["dc"] - 0.3) < 1e-9
    assert d["resid_rel"] < 1e-9
    assert abs(d["amp"] - math.hypot(2.0, 0.5)) < 1e-9


def test_harmonic_window_orthogonality():
    """A large DC offset must not leak into the first harmonic (integer periods)."""
    omega = 2 * math.pi / 100.0
    t = np.arange(1, 501, dtype=float)
    y = 1.25 * np.cos(omega * t) - 0.75 * np.sin(omega * t) + 100.0
    d = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=1)
    assert abs(d["cos"] - 1.25) < 1e-8
    assert abs(d["sin"] + 0.75) < 1e-8
    assert abs(d["dc"] - 100.0) < 1e-8


def test_harmonic_decompose_flags_nonlinear_residual():
    """Second-harmonic content must show up in the residual, not in C_I."""
    omega = 2 * math.pi / 100.0
    t = np.arange(1, 401, dtype=float)
    y = np.cos(omega * t) + 0.5 * np.cos(2 * omega * t)
    d1 = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=1)
    d2 = harmonic_decompose(t, y, omega, t0=100.0, t1=400.0, nh=2)
    assert d1["resid_rel"] > 0.2
    assert d2["resid_rel"] < 1e-9
    assert abs(d1["cos"] - 1.0) < 1e-8


def test_inertia_coefficient_normalisation():
    # F_cos = 1, a0 = 1, D = 2 -> V = pi -> C_I = 1/pi
    assert inertia_coefficient(1.0, 1.0, 2.0) == pytest.approx(1.0 / math.pi)
    assert inertia_coefficient(2.0, 1.0, 2.0, rho=2.0) == pytest.approx(1.0 / math.pi)


def test_potential_reference_bounds_and_conventions():
    pot = potential_reference(64, 64, 8.0)
    # same-mask potential added mass: below the unbounded 1.0 up to a few %
    assert 0.85 < pot["c_a_area"] < 1.35
    assert pot["c_a_halfway"] < pot["c_a_area"]
    assert pot["added_mass"] > 0


def test_quick_arm_is_stable_and_self_consistent():
    result, trace = run_arm(CASES["quick"], threads=2)
    assert result["stability_ok"]
    assert len(trace["t"]) == result["numerics"]["n_steps"]
    # the box-mean acceleration must track the drive within 25 %
    assert 0.75 < result["achieved"]["a0_over_nominal"] < 1.25
    kpi = result["kpi"]
    for tag in ("E1_mem_halfway", "E1_mem_area", "E2_trac_halfway", "E2_trac_area"):
        assert math.isfinite(kpi[tag]["C_I"])
    assert kpi["E1_mem_area"]["C_I"] > 0.0
    # the two independent estimators must bracket each other within 60 %
    assert result["estimator_spread_pct"] < 0.6 * kpi["E1_mem_area"]["C_I"]
    # reference must be reported for the error budget
    assert result["reference"]["potential_same_mask_area"] > 0
    assert math.isfinite(result["err_inphase_pct_area"])


def test_cli_list_runs(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "base" in out and "quick" in out


def test_interior_nodes_stay_inside_solid():
    """Zeroed core cells must not be able to stream into a fluid node.

    Regression: the 4-neighbour definition put solid cells that are only
    *diagonally* adjacent to fluid in the zeroed set, so a zero population was
    injected into the flow every step (max|rho-1| ~ 0.2, mass drift ~ -9 % per
    window).  Any solid cell with a fluid neighbour in the 8-point stencil must
    keep its populations.
    """
    from tensorlbm.oscillatory_cylinder_inertia import _interior_nodes

    for D, nx in ((8, 32), (16, 64), (24, 96)):
        mask = circle_mask(nx, nx, D)
        core = _interior_nodes(mask)
        assert not bool((core & ~mask).any()), "zeroed set must be inside the solid"
        fluid = ~mask
        leaks = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                leaks += int((core & torch.roll(torch.roll(fluid, dy, 0), dx, 1)).sum())
        assert leaks == 0, f"zeroed core cell streams into fluid (D={D}, {leaks} leaks)"
        assert int(core.sum()) > 0, "a solid core must exist for a cylinder"


def test_mass_budget_under_body_force():
    """Harmonic body force on a body-filled periodic box must conserve fluid mass.

    This is the invariant the interior-zeroing defect broke: the fluid mass and the
    near-wall density have to stay flat while the drive is on, no matter that the
    total momentum oscillates strongly.
    """
    from tensorlbm.boundary.boundaries import bounce_back_cells
    from tensorlbm.boussinesq import apply_guo_forcing
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.oscillatory_cylinder_inertia import _interior_nodes
    from tensorlbm.solver import stream

    torch.set_num_threads(1)
    D, nx, tau, omega, a0 = 8, 32, 0.7, 2 * math.pi / 100.0, 1e-3
    mask = circle_mask(nx, nx, D)
    fluid = ~mask
    core = _interior_nodes(mask)
    rho0 = torch.ones((nx, nx), dtype=torch.float64)
    zero = torch.zeros_like(rho0)
    f = bounce_back_cells(equilibrium(rho0, zero, zero), mask)
    Fx = torch.zeros((nx, nx), dtype=torch.float64)
    mass0 = None
    drho = 0.0
    for step in range(1, 201):
        Fx[fluid] = a0 * math.cos(omega * step)
        old = f[:, mask].clone()
        f = apply_guo_forcing(f, torch.stack([Fx, torch.zeros_like(Fx)]), tau)
        f[:, mask] = old
        f = stream(f)
        rho, _ux, _uy = macroscopic(f)
        f = bounce_back_cells(f, mask)
        f[:, core] = 0.0
        mass = float(rho[fluid].sum())
        mass0 = mass if mass0 is None else mass0
        drho = max(drho, float((rho[fluid] - 1.0).abs().max()))
    # an impulsive start legitimately rings the box acoustically (a few %); the
    # defect signature was ~0.28, i.e. 10x this band
    assert drho < 0.08, f"near-wall density excursion {drho:.4f} (should be O(Ma))"
    assert abs(mass - mass0) / mass0 < 1e-4, f"fluid mass drifted {abs(mass-mass0)/mass0:.2e}"


def test_stokes_cylinder_coefficients_limits_and_monotonicity():
    """Analytic unsteady-Stokes reference: potential limit 1, thick layer > 1."""
    from tensorlbm.oscillatory_cylinder_inertia import stokes_cylinder_coefficients

    c_hi = stokes_cylinder_coefficients(5000.0)
    assert abs(c_hi["Gamma_re"] - 1.0) < 0.07
    rows = [stokes_cylinder_coefficients(b) for b in (25.0, 50.0, 100.0, 400.0)]
    re = [r["Gamma_re"] for r in rows]
    assert all(a > b for a, b in zip(re, re[1:])), f"Gamma_re must fall with beta: {re}"
    assert all(r["C_I_viscous"] > 1.0 for r in rows)
    assert all(r["C_Q_viscous"] > 0.0 for r in rows)
    # beta = 50: Gamma = 1 + 4 K_1(z)/(z K_0(z)) with z = (1+i) sqrt(pi beta/2);
    # the value is pinned against the sqrt(2)/(delta/R) identity in
    # tests/test_stokes_layer_thickness.py (an earlier build quoted 1.4525, which is
    # what k = 1/(delta/R) with a sqrt(2)-too-thick delta gives).
    assert abs(rows[1]["Gamma_re"] - 1.2258) < 1e-3
