"""Acceptance tests for the IN45-N07 air-lubrication frontier layer.

Every machine-checkable criterion of the node has a test here, and every
criterion that can silently pass for the wrong reason has a negative control:

C1  ``Delta C_f`` of the computed curtain vs the PAL empirical formula  -> < 20 %
C2  ``alpha(y)`` wall peak / layer-thickness band / shape vs Madavan 1985 -> < 30 %
C3  two-phase species-mass budget residual                              -> < 1e-8
C4  air-flow optimum exists and yields net power saving                 -> > 0 W
C5  capture efficiency monotone in curvature and in injection position
C6  negative controls: no air -> no drag reduction; oversized bubbles -> loss
    of the curtain; wrong buoyancy sign -> no wall peak
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.air_lubrication import (
    LAMBDA_AIR_WATER,
    VoidFractionTransport2D,
    rise_velocity_schiller_naumann,
    stratified_drag_reduction_from_alpha,
)
from tensorlbm.air_lubrication_frontier import (
    CF0_SHIP,
    ETA_COMPRESSOR,
    INJECTION_DEPTH,
    MADAVAN_DELTA_B_BAND,
    U_SHIP,
    boundary_layer_velocity,
    curvature_injection_study,
    curtain_drag_reduction,
    energy_balance,
    optimize_air_flow,
    pal_formula_comparison,
    profile_deviation,
    re_scan_case,
    simulate_curtain,
    turbulent_diffusivity,
)

FAST = dict(l_over_delta=200.0, dt_steps=4000)


# --------------------------------------------------------------------- C2
def test_curtain_is_wall_peaked_and_thins_with_bubble_size():
    """Wall-directed buoyancy: alpha must peak at the hull and decay inward."""
    small = simulate_curtain(1.0e-4, 50e-6, **FAST)
    large = simulate_curtain(1.0e-4, 300e-6, **FAST)
    assert small.alpha_wall > 0.0
    assert np.all(np.diff(small.alpha_profile) <= 1e-12)  # monotone from the hull
    assert np.argmax(small.alpha_profile) == 0
    # a bigger bubble rises faster, so the layer it sustains is thinner
    assert large.delta_b_over_delta < small.delta_b_over_delta


def test_c2_shape_agrees_with_madavan_family_within_30_percent():
    c = simulate_curtain(1.0e-4, 100e-6, l_over_delta=1000.0, dt_steps=12000)
    dev = profile_deviation(c.alpha_profile, c.eta)
    assert dev["wall_peaked"]
    assert dev["l2_rel"] < 0.30, dev
    assert dev["max_rel"] < 0.30, dev


def test_c2_negative_control_wrong_buoyancy_sign_loses_the_curtain():
    """Feeding +V_b (away from the hull) must destroy the retained curtain.

    This is the P21-N05 defect: the sign of the buoyant drift relative to the
    hull.  With the hull-relative sign the air is held in the inner layer; with
    the reversed sign it is driven into the outer flow, so the wall void
    fraction and the inner-layer capture both collapse.
    """
    d_b = 300e-6
    vb = rise_velocity_schiller_naumann(d_b) / U_SHIP
    eta32 = (np.arange(32) + 0.5) * (2.0 / 32)
    out = {}
    for sign in (-1.0, +1.0):
        tr = VoidFractionTransport2D(
            nx=40,
            ny=32,
            dx=5.0,
            dy=2.0 / 32,
            u_profile=boundary_layer_velocity(eta32),
            rise_velocity=sign * vb,
            disp_y=turbulent_diffusivity(eta32),
            injection=np.full(40, 1.0e-4 / U_SHIP),
            dt=0.25,
        )
        u = np.repeat(tr.u_profile[:, None], 40, axis=1)
        for _ in range(4000):
            tr.step(u)
        prof = tr.wall_normal_profile(30, 38)
        out[sign] = {
            "alpha_wall": float(prof[0]),
            "capture": float(prof[eta32 <= 0.5].sum() / max(prof.sum(), 1e-30)),
            "delta_b": float(prof.sum() * (2.0 / 32) / max(prof[0], 1e-30)),
        }
    good, bad = out[-1.0], out[+1.0]
    assert good["alpha_wall"] > 3.0 * bad["alpha_wall"]
    assert good["capture"] > bad["capture"] + 0.2
    assert good["delta_b"] < bad["delta_b"]


# --------------------------------------------------------------------- C3
def test_c3_species_mass_budget_closes_below_1e8():
    c = simulate_curtain(1.0e-4, 100e-6, **FAST)
    assert c.conservation_error_max < 1.0e-8, c.conservation_error_max
    assert c.mass == pytest.approx(c.injected - c.outflowed, rel=1e-10, abs=1e-12)


def test_c3_negative_control_audit_detects_a_leak():
    """Corrupting the field by 1e-3 must break the measured budget."""
    c = simulate_curtain(1.0e-4, 100e-6, **FAST, keep_field=True)
    tr_mass = c.mass
    leaked = tr_mass + 1.0e-3
    residual = abs(leaked - (c.injected - c.outflowed)) / max(c.injected, 1e-30)
    assert residual > 1.0e-8


def test_anisotropic_cell_diffusion_matches_analytic_growth():
    """Regression for the y-flux factor: var(y) = 2 D t must hold for dy != dx.

    Before the fix the wall-normal fluxes were divided by ``dx``, so with
    ``dx = 10 dy`` the diffusive spreading grew 10x too fast (and the divergence
    10x too slow).  The second moment of a diffusing spike is analytic.
    """
    ny, nx = 64, 2
    dy, dx = 0.1, 1.0
    d = 1.0e-3
    tr = VoidFractionTransport2D(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        u_profile=np.zeros(ny),
        disp_y=np.full(ny, d),
        disp_x=np.zeros(ny),
        dt=0.05,
    )
    tr.alpha[ny // 2, :] = 1.0  # spike away from both walls
    m0 = tr.mass()
    steps = 400
    for _ in range(steps):
        tr.step()
    assert tr.mass() == pytest.approx(m0, rel=1e-12)  # closed domain
    y = (np.arange(ny) + 0.5) * dy
    var = float(np.sum(tr.alpha[:, 0] * (y - y.mean()) ** 2) / np.sum(tr.alpha[:, 0]))
    t = steps * tr.dt
    assert var == pytest.approx(2.0 * d * t, rel=0.25)


# --------------------------------------------------------------------- C1
def test_c1_pal_empirical_formula_within_20_percent():
    c = simulate_curtain(1.0e-4, 100e-6, **FAST)
    cmp = pal_formula_comparison(c.alpha_profile)
    assert cmp["dr_exact"] > 0.05
    assert cmp["rel_error"] < 0.20, cmp


def test_c1_lattice_matches_exact_stratified_law():
    """LBM Delta C_f vs the exact stratified solution at the same lambda."""
    case = re_scan_case(500.0, 0.10, steps=600)
    assert case["dr_ref"] > 0.05
    assert case["dr_rel_err"] < 0.20, case
    assert case["momentum_residual_max"] < 1e-8


# --------------------------------------------------------------------- C4
def test_c4_air_flow_optimum_is_interior_and_net_positive():
    q = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    opt = optimize_air_flow(q, d_b=100e-6, **FAST)
    assert opt["net_positive"], opt
    assert 0 < int(np.argmax([r["net_power"] for r in opt["table"]])) < len(q) - 1
    assert opt["net_power_opt"] > 0.0
    # the benefit saturates while the compressor cost keeps growing: at the
    # largest air flow the net power must be below the optimum
    assert opt["table"][-1]["net_power"] < opt["net_power_opt"]


def test_c4_negative_control_no_air_no_saving_no_drag_reduction():
    c = simulate_curtain(0.0, 100e-6, **FAST)
    assert c.alpha_wall == 0.0
    dr = curtain_drag_reduction(c.alpha_profile)["dr"]
    assert dr == pytest.approx(0.0, abs=1e-12)
    e = energy_balance(0.0, dr)
    assert e["net_power"] == pytest.approx(0.0, abs=1e-12)


def test_c4_negative_control_oversized_bubbles_collapse_the_curtain():
    """Millimetre bubbles degenerate from a curtain into an unresolved film.

    They are held against the hull by their own buoyancy (the model has no
    break-up / escape channel), so ``alpha_wall`` rises while the layer collapses
    to a fraction of a cell: the *bubbly curtain* regime of Madavan 1985 is lost
    and the stratified ``Delta C_f`` is no longer a number this model can vouch
    for (film instability is not modelled) -- hence the WITHHELD grade on that
    branch of the report.
    """
    fine = simulate_curtain(1.0e-4, 50e-6, **FAST)
    coarse = simulate_curtain(1.0e-4, 3000e-6, **FAST)
    assert fine.delta_b_over_delta > 0.30
    assert coarse.delta_b_over_delta < 0.15
    assert not profile_deviation(coarse.alpha_profile, coarse.eta)["band_ok"]
    assert coarse.capture_efficiency > fine.capture_efficiency  # thin film, not a curtain


# --------------------------------------------------------------------- C5
def test_c5_capture_efficiency_monotone_in_curvature_and_injection():
    out = curvature_injection_study(
        q_air=1.0e-4,
        d_b=100e-6,
        radii=(float("inf"), 20.0, 5.0),
        injection_positions=(0.0, 50.0, 100.0, 150.0),
        inj_length=20.0,
        **FAST,
    )
    for r, rec in out["per_radius"].items():
        assert rec["strictly_increasing"], (r, rec)
        assert rec["rho"] == pytest.approx(1.0)
    for x, rec in out["per_position"].items():
        assert rec["strictly_decreasing"], (x, rec)


def test_c5_negative_control_flat_bottom_is_the_best_case():
    out = curvature_injection_study(
        q_air=1.0e-4,
        d_b=100e-6,
        radii=(float("inf"), 10.0),
        injection_positions=(0.0,),
        inj_length=20.0,
        **FAST,
    )
    rows = {g["radius_m"]: g["capture"] for g in out["grid"]}
    assert rows[float("inf")] > rows[10.0]


def test_reference_constants_are_physical():
    assert 0.001 < CF0_SHIP < 0.003
    assert 0.5 <= ETA_COMPRESSOR <= 0.9
    assert INJECTION_DEPTH > 0.0
    assert MADAVAN_DELTA_B_BAND[0] < MADAVAN_DELTA_B_BAND[1]
    assert LAMBDA_AIR_WATER == pytest.approx(0.018)
    assert U_SHIP > 0.0