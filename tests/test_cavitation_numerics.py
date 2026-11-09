"""Regression tests for the P40W3B-N02 cavitation numerics layer.

Covers the audit relations of :mod:`tensorlbm.cavitation_numerics` (pressure
level, Brinkman slip, phase-change time scales, closure growth) plus the
verification status of the two flow engines:

* panel method: **verified** against the analytic cylinder solution;
* lifting (Kutta) branch of the panel method: **known gap**, encoded as a strict
  xfail so that fixing it will flip the test instead of silently passing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.cavitation_airfoil import CavitationFlowConfig, MixtureCavitationSolver
from tensorlbm.cavitation_free_streamline import flow_solve, panel_geometry
from tensorlbm.cavitation_numerics import (
    brinkman_eta_for_leakage,
    brinkman_leakage,
    cavity_length_isobaric,
    integrate_closure_growth,
    physical_pressure,
    pressure_anchor,
    rayleigh_plesset_time,
    schnerr_sauer_growth_time,
    stability_budget,
)


# ---------------------------------------------------------------------------
# pressure level
# ---------------------------------------------------------------------------


def test_pressure_anchor_zeroes_the_inlet_plane():
    p = torch.linspace(-0.3, 0.9, 40 * 20, dtype=torch.float64).reshape(40, 20)
    c0 = pressure_anchor(p, inlet_columns=2)
    assert abs(float((p[:, :2].mean() + c0))) < 1e-12
    # a constant field is fully removed
    assert abs(pressure_anchor(torch.full((10, 10), 0.37, dtype=torch.float64)) + 0.37) < 1e-12


def test_pressure_anchor_ignores_solid_columns():
    p = torch.zeros(8, 6, dtype=torch.float64)
    p[2:4, :2] = 5.0  # body sitting on the inlet plane
    mask = torch.zeros_like(p, dtype=torch.bool)
    mask[2:4, :] = True
    c0 = pressure_anchor(p, inlet_columns=2, airfoil_mask=mask)
    assert abs(c0) < 1e-12, "solid cells must not enter the anchor average"


def test_physical_pressure_is_a_pure_shift():
    p = torch.rand(6, 5, dtype=torch.float64)
    pp = physical_pressure(p, 0.25)
    assert torch.allclose(pp - p, torch.full_like(p, 0.25))


# ---------------------------------------------------------------------------
# Brinkman penalisation
# ---------------------------------------------------------------------------


def test_brinkman_leakage_matches_the_split_step_ratio():
    eta, dt = 1.0e-3, 0.004
    assert brinkman_leakage(eta, dt) == pytest.approx(1.0 / (1.0 + dt / eta))
    # the P41-N02 setting leaks 20 % of the free-stream speed into the body
    assert brinkman_leakage(eta, dt) == pytest.approx(0.2)
    # round trip
    target = 0.01
    e = brinkman_eta_for_leakage(dt, target)
    assert brinkman_leakage(e, dt) == pytest.approx(target)
    assert e < eta, "1 % slip needs eta far below the historical 1e-3"


def test_brinkman_rejects_invalid_input():
    with pytest.raises(ValueError):
        brinkman_leakage(0.0, 0.1)
    with pytest.raises(ValueError):
        brinkman_eta_for_leakage(0.1, 1.5)


# ---------------------------------------------------------------------------
# phase-change time scales
# ---------------------------------------------------------------------------


def test_closure_time_is_the_closed_form():
    dp = 0.4
    alpha = 1.0e-4
    r_b = (alpha / (1.0 - alpha) * 3.0 / (4.0 * math.pi * 100.0)) ** (1.0 / 3.0)
    t_ss = schnerr_sauer_growth_time(alpha, dp, r_b_floor=1.0e-6)
    assert t_ss == pytest.approx(r_b / math.sqrt(6.0 * dp), rel=1e-9)
    t_rp = rayleigh_plesset_time(dp, r_b)
    assert t_rp == pytest.approx(r_b * math.sqrt(1.0 / (3.0 * dp)), rel=1e-12)
    # the two inertia-controlled scales agree up to the classical sqrt(2)
    assert t_rp / t_ss == pytest.approx(math.sqrt(2.0), rel=1e-6)


def test_closure_growth_reproduces_the_rayleigh_plesset_radius():
    g = integrate_closure_growth(0.4, n_steps=20000, alpha0=1.0e-4, r_b_floor=1.0e-4)
    assert g["alpha_final"] > 0.5, "the closure must grow the void at fixed dp"
    # bubble radius from the void fraction vs the inertia asymptote R = t sqrt(2 dp/3 rho)
    assert g["ratio_r_rp"] == pytest.approx(1.0, abs=0.1)


def test_stability_budget_reports_the_explicit_limits():
    cfg = CavitationFlowConfig(cells_per_chord=50, cfl=0.2)
    b = stability_budget(cfg)
    assert b["advective_cfl"] == pytest.approx(0.2)
    assert b["viscous_number"] < 0.5
    assert b["sgs_number"] < 0.5
    assert b["brinkman_slip"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# isobaric cavity metric
# ---------------------------------------------------------------------------


def _synthetic_pressure(cfg: CavitationFlowConfig, span: float) -> torch.Tensor:
    """Pressure field with a low-pressure strip of chordwise extent ``span``."""
    x = (torch.arange(cfg.nx) + 0.5) * cfg.h - cfg.le_x_chords
    y = (torch.arange(cfg.ny) + 0.5) * cfg.h - cfg.le_y_chords
    X, Y = torch.meshgrid(x, y, indexing="ij")
    tongue = ((X > -0.02) & (X < span) & (Y > -0.45) & (Y < 0.45)).double()
    return -0.6 * tongue + 0.05 * (X / 6.0)


def test_isobaric_cavity_tracks_the_low_pressure_extent():
    cfg = CavitationFlowConfig(cells_per_chord=50, sigma=0.8)
    short = cavity_length_isobaric(_synthetic_pressure(cfg, 0.20), cfg, anchor_offset=0.0)
    long = cavity_length_isobaric(_synthetic_pressure(cfg, 0.40), cfg, anchor_offset=0.0)
    assert short["l_cav"] > 0.10
    assert long["l_cav"] > short["l_cav"]
    # the isobaric sampler over-reads by the sigmoid transition + station
    # spacing, so the *difference* is the physical assertion
    assert long["l_cav"] - short["l_cav"] == pytest.approx(0.20, abs=0.03)


def test_isobaric_criterion_is_sensitive_to_the_pressure_level():
    """Un-anchored pressure (the P41-N02 defect) moves the cavity contour."""
    cfg = CavitationFlowConfig(cells_per_chord=50, sigma=0.8)
    base = _synthetic_pressure(cfg, 0.40)
    anchored = cavity_length_isobaric(base, cfg, anchor_offset=0.0)
    shifted = cavity_length_isobaric(base + 0.2, cfg, anchor_offset=0.0)
    assert shifted["l_cav"] < 0.25 * anchored["l_cav"]


def test_mixture_solver_anchor_flag_sets_the_offset():
    cfg = CavitationFlowConfig(
        cells_per_chord=24,
        sigma=0.8,
        c_dest=0.0,
        c_prod=0.0,
        spinup_steps=0,
        ramp_steps=0,
        pressure_anchor=True,
        n_steps=3,
        cfl=0.2,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    for _ in range(2):
        solver.step()
    assert solver.p_offset != 0.0
    cfg_off = CavitationFlowConfig(cells_per_chord=24, sigma=0.8, n_steps=1)
    assert CavitationFlowConfig(**{**cfg_off.__dict__, "pressure_anchor": False}).pressure_anchor is False


# ---------------------------------------------------------------------------
# panel-method flow engine
# ---------------------------------------------------------------------------


def _cylinder(n: int = 120, radius: float = 0.5) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    return np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)


def test_panel_engine_reproduces_the_analytic_cylinder():
    nodes = _cylinder()
    geom = panel_geometry(nodes)
    st = flow_solve(geom, 0.0, kutta="none")
    th = np.linspace(0.0, 2.0 * math.pi, nodes.shape[0])
    th_mid = th[:-1] + np.diff(th) / 2.0
    analytic = -2.0 * np.sin(th_mid)
    assert np.abs(st.qt - analytic).max() < 5e-3
    assert st.cp.min() == pytest.approx(-3.0, abs=0.02)
    # d'Alembert: no force on a cylinder without circulation
    assert abs(st.cl) < 1e-6
    assert abs(st.cd) < 1e-6


def test_panel_engine_outward_normals():
    geom = panel_geometry(_cylinder())
    outward = np.sum(geom["n"] * (geom["mid"] - geom["mid"].mean(axis=0)), axis=1)
    assert outward.min() > 0.0


@pytest.mark.xfail(
    strict=True,
    reason="known gap: the Kutta closure of the panel engine returns ~0.5x "
    "thin-aerofoil theory, so lifting/cavity results are WITHHELD",
)
def test_lifting_branch_is_not_yet_validated():
    th = np.linspace(0.0, math.pi, 81)
    x = 0.5 * (1.0 - np.cos(th))
    yt = 5.0 * 0.15 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    up = np.stack([x, yt], axis=1)
    lo = np.stack([x[::-1], -yt[::-1]], axis=1)
    st = flow_solve(panel_geometry(np.vstack([lo, up[1:]])), 5.0)
    thin = 2.0 * math.pi * math.radians(5.0)
    assert st.cl == pytest.approx(thin, rel=0.05)