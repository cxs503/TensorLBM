"""Tests for partitioned-FSI added-mass stability calibration (campaign P96-N2).

Covers the machine-checkable criteria: 1D analytic anchor vs numerical error
sequences (D1), the light-body negative control and IQN-ILS convergence (D2),
physical invariance against the verified free-decay benchmark and across
stabilisation schemes (D3), the measured stability map, and the 2D D2Q9 LBM
realisation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm import fsi_partitioned_stability as fps

REPO = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "benchmarks" / "verified" / "floating_body_free_decay" / "result.json"


def _rel(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference) if reference != 0 else abs(value)


# --------------------------------------------------------------------------- #
# D1 - analytic anchor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 4.0])
def test_analytic_factor_matches_quoted_formula(ratio):
    """rho_dd must equal |1 - c*rho_f/rho_s| exactly (campaign formula)."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=ratio, added_mass_coefficient=1.0)
    assert model.rho_dd_explicit(1.0) == pytest.approx(abs(1.0 - 1.0 / ratio), abs=1e-12)
    assert model.chi == pytest.approx(1.0 / ratio, rel=1e-12)


def test_critical_density_ratio_and_optimal_relaxation():
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.5)
    assert model.critical_density_ratio_explicit(1.0) == pytest.approx(0.5, rel=1e-12)
    assert model.critical_density_ratio_explicit(0.5) == pytest.approx(0.25, rel=1e-12)
    # the optimal relaxation annihilates the error in one pass
    assert model.rho_dd_explicit(model.optimal_relaxation_explicit()) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("omega", [0.25, 0.5, 1.0])
def test_measured_explicit_rate_matches_analytic_within_5pct(ratio, omega):
    """D1: executed 1D error sequences vs |1 - omega*chi| (<= 5%)."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=ratio)
    run = fps.simulate_explicit_piston(model, n_steps=40, omega=omega)
    assert run["relative_error"] <= 0.05


@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0])
def test_measured_subiteration_rate_matches_analytic_within_5pct(ratio):
    """D1: sub-iteration residual ratios vs |1 - omega*(1+chi)| (<= 5%)."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=ratio)
    omega = 1.0
    _, rep = fps.solve_piston_step_subiterations(
        model, m_s=1.0, dt=1.0, f_ext=1.0, scheme="fixed",
        omega=omega, tol=1e-14, max_iter=25,
    )
    hist = np.asarray(rep.residual_history, dtype=float)
    ratios = hist[1:] / hist[:-1]
    measured = float(np.median(ratios))
    analytic = model.rho_dd_subiteration(omega)
    assert _rel(measured, analytic) <= 0.05


def test_numeric_critical_density_ratio_within_10pct():
    """D1: bisected divergence threshold vs c*omega/2 (<= 10%)."""
    for omega in (0.5, 1.0):
        numeric = fps.estimate_critical_density_ratio(omega=omega, n_steps=40)
        analytic = 0.5 * omega
        assert _rel(numeric, analytic) <= 0.10


# --------------------------------------------------------------------------- #
# D2 - negative control and stabilisation gains
# --------------------------------------------------------------------------- #
def test_negative_control_light_body_explicit_diverges():
    """D2: rho_s/rho_f = 0.1 must diverge with a positive growth exponent."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.1)
    run = fps.simulate_explicit_piston(model, n_steps=12, omega=1.0)
    assert run["diverged"] is True
    assert run["growth_exponent"] > 0.0
    assert run["growth_exponent"] == pytest.approx(math.log(9.0), rel=0.02)


def test_negative_control_subiteration_fixed_relaxation_diverges():
    """D2: constant relaxation cannot converge the loop at chi = 10."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.1)
    _, rep = fps.solve_piston_step_subiterations(
        model, m_s=1.0, dt=1.0, f_ext=1.0, scheme="fixed", omega=1.0,
        tol=1e-6, max_iter=20,
    )
    assert rep.converged is False
    hist = np.asarray(rep.residual_history)
    assert hist[-1] > hist[0] * 1e3 or rep.diverged


@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0])
def test_iqn_ils_converges_within_20_subiterations(ratio):
    """D2: IQN-ILS (5-vector history) reaches 1e-6 in <= 20 sub-iterations."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=ratio)
    _, rep = fps.solve_piston_step_subiterations(
        model, m_s=1.0, dt=1.0, f_ext=1.0, scheme="iqn_ils", max_iter=20, tol=1e-6,
    )
    assert rep.converged is True
    assert rep.n_iterations <= 20
    assert rep.final_residual <= 1e-6


@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0])
def test_aitken_stabilises_light_bodies(ratio):
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=ratio)
    _, rep = fps.solve_piston_step_subiterations(
        model, m_s=1.0, dt=1.0, f_ext=1.0, scheme="aitken", max_iter=20, tol=1e-6,
    )
    assert rep.converged is True
    assert rep.final_residual <= 1e-6


def test_iqn_ils_history_is_bounded_to_five_vectors():
    qn = fps.IQNILS(history_size=5)
    x = 0.0
    for k in range(12):
        r = 10.0 - 3.0 * x  # arbitrary linear coupling map
        x = float(qn.step(x, r)[0])
        qn._prev_x = np.atleast_1d(np.asarray(x, dtype=float))
    assert len(qn._dR) <= 5
    assert len(qn._dX) <= 5


# --------------------------------------------------------------------------- #
# D3 - physical invariance
# --------------------------------------------------------------------------- #
def test_structural_free_decay_matches_verified_benchmark():
    """D3: structural sub-solver vs benchmarks/verified/floating_body_free_decay."""
    reference = json.loads(VERIFIED.read_text())
    coarse = next(c for c in reference["cases"] if c["level"] == "coarse")
    mine = fps.structural_free_decay_frequency(
        m_eff=coarse["m_eff"], stiffness=coarse["stiffness"], dt=coarse["dt"],
        n_periods=16, amplitude=coarse["amplitude"],
    )
    assert _rel(mine["omega_meas"], coarse["omega_meas"]) <= 0.05
    assert _rel(mine["omega_meas"], coarse["omega_n_analytic"]) <= 0.05


def test_coupling_invariance_aitken_vs_iqn_ils():
    """D3: converged stabilisers must reproduce the same coupled physics."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=1.0)
    kwargs = dict(n_steps=200, max_subiter=20, zeta=0.0, tau=1.5, omega_n=0.02)
    a = fps.run_lbm_fsi_decay(model=model, scheme="aitken", tol=1e-6, **kwargs)
    b = fps.run_lbm_fsi_decay(model=model, scheme="iqn_ils", tol=1e-6, **kwargs)
    ref = fps.run_lbm_fsi_decay(model=model, scheme="iqn_ils", tol=1e-12, **kwargs)
    for run in (a, b):
        assert run["diverged"] is False
        assert run["unconverged_steps"] == 0
        hist = np.asarray(run["displacement_history"])
        base = np.asarray(ref["displacement_history"])
        assert hist.size == base.size
        denom = max(float(np.max(np.abs(base))), 1e-12)
        assert float(np.max(np.abs(hist - base))) / denom <= 0.05


# --------------------------------------------------------------------------- #
# 2D LBM realisation
# --------------------------------------------------------------------------- #
def test_lbm_rest_state_is_undisturbed():
    lbm = fps.HeavingBodyLBM()
    for _ in range(50):
        lbm.collide_stream()
        lbm.impose(lbm.wall_mask(), 0.0, 0.0)
        lbm.impose(lbm.mask(lbm.y0), 0.0, 0.0)
    rho, ux, uy = lbm.macroscopic()
    assert np.allclose(rho, 1.0, atol=1e-12)
    assert np.max(np.abs(ux)) < 1e-12
    assert np.max(np.abs(uy)) < 1e-12


def test_lbm_added_mass_equals_body_mass():
    """chi = m_a/m_s must reproduce c*rho_f/rho_s for the filled-body mask."""
    lbm = fps.HeavingBodyLBM(rho_f=1.0)
    assert lbm.added_mass(lbm.mask(lbm.y0)) == pytest.approx(lbm.n_body, rel=1e-12)


def test_lbm_negative_control_explicit_diverges():
    """D2 in the lattice solver: explicit coupling diverges at chi = 10."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.1)
    run = fps.run_lbm_fsi_decay(model=model, scheme="explicit", n_steps=200)
    assert run["diverged"] is True
    assert run["growth_exponent"] > 0.0
    # measured growth vs the analytic ln(chi) of the discrete added-mass law
    assert _rel(run["growth_exponent"], math.log(model.chi)) <= 0.05


def test_lbm_iqn_ils_stable_at_light_body():
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.1)
    run = fps.run_lbm_fsi_decay(model=model, scheme="iqn_ils", n_steps=100)
    assert run["diverged"] is False
    assert run["mean_subiterations"] <= 20
    assert run["final_residual"] <= 1e-6
    assert run["chi_measured"] == pytest.approx(model.chi, rel=0.02)


def test_lbm_fixed_relaxation_fails_at_light_body():
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.1)
    run = fps.run_lbm_fsi_decay(model=model, scheme="fixed", n_steps=50)
    assert run["diverged"] is True


# --------------------------------------------------------------------------- #
# Stability map
# --------------------------------------------------------------------------- #
def test_stability_map_classification_matches_analytic_boundary():
    ratios = [0.2, 0.4, 0.6, 1.0, 2.0]
    omegas = [0.25, 0.5, 1.0, 1.5]
    rows = fps.stability_map(density_ratios=ratios, omegas=omegas, n_steps=60)
    assert len(rows) == len(ratios) * len(omegas)
    for row in rows:
        assert row["diverged"] is not row["analytic_stable"]
        assert _rel(row["measured_factor"], row["analytic_factor"]) <= 0.05


def test_stability_map_boundary_is_relaxation_dependent():
    """Under-relaxation extends the stable region to lighter bodies."""
    model = fps.AddedMassPistonModel(rho_f=1.0, rho_s=0.3)
    assert model.is_linearly_stable_explicit(1.0) is False
    assert model.is_linearly_stable_explicit(0.5) is True