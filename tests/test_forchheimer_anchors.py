"""Contract tests for :mod:`tensorlbm.porous_weak_inertia`.

The module supplies the *self-produced* reference machinery for the
Forchheimer (weak-inertia) drag of periodic cylinder arrays.  These tests pin

* the geometry masks (solid fraction and lattice spacing),
* the analytic anchors (dilute-lattice log law slope one, Lamb-Oseen constant),
* the estimators (expansion fit, free-intercept fit, Ma -> 0 and algebraic
  tail extrapolation),
* the float64 mass conservation of the D2Q9 equilibrium (regression for the
  float32 weight-table bias, which used to inject a linear density ramp),
* one short end-to-end steady solve with the library kernels.

All tests are fast (no long campaign); the numerical campaign lives in
``benchmarks/verified/forchheimer_weak_inertia``.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tensorlbm.d2q9 import _w_for, equilibrium, macroscopic
from tensorlbm.porous_weak_inertia import (
    CS,
    _tables,
    LAMB_CONST,
    build_array,
    dilute_log_law_slope,
    expansion_fit,
    extrapolate_origin,
    free_intercept_fit,
    guo_source_term,
    lamb_oseen_cd,
    power_law_extrapolate,
    resistance_ratio,
    run_fixed_steps,
    tail_extrapolate,
)
from tensorlbm.solver import collide_trt


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """Keep torch single-threaded: these are tiny tensors and thread overhead dominates."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# ------------------------------------------------------------------ geometry
@pytest.mark.unit
def test_square_mask_solid_fraction_matches_target():
    geom = build_array("square", 80, 0.40)
    assert abs(geom.phi_actual - 0.40) < 0.01
    assert geom.mask.shape == (80, 80)
    assert geom.n_solid == int(geom.mask.sum().item())
    assert geom.diameter == pytest.approx(80 * 2 * math.sqrt(0.40 / math.pi))


@pytest.mark.unit
def test_square_mask_is_periodic_and_centred():
    geom = build_array("square", 64, 0.30)
    mask = geom.mask
    # one disc per cell: the centre of the cell is solid, its four corners are fluid
    assert bool(mask[32, 32])
    assert not bool(mask[0, 0]) and not bool(mask[0, 63]) and not bool(mask[63, 0])
    # the mask must reproduce an independent periodic-distance construction
    idx = torch.arange(64, dtype=torch.float64)
    dx = torch.remainder(idx - 32.0 + 32.0, 64.0) - 32.0
    dist2 = dx[:, None] ** 2 + dx[None, :] ** 2
    radius = 64 * math.sqrt(0.30 / math.pi)
    assert torch.equal(mask, dist2 <= radius**2)
    # solid area matches the analytic disc area to within the pixelisation error
    assert geom.n_solid / (64 * 64) == pytest.approx(0.30, abs=0.02)


@pytest.mark.unit
def test_hex_mask_two_disc_cell_and_lattice_constant():
    n = 40
    geom = build_array("hex", n, 0.55)
    ny = int(round(n * math.sqrt(3.0)))
    assert geom.mask.shape == (ny, n)
    assert abs(geom.phi_actual - 0.55) < 0.02
    # nearest-neighbour spacing of the triangular lattice equals n lattice cells:
    # the two discs of the cell sit exactly on top of their periodic images.
    r = geom.radius
    assert r < n / 2.0  # discs never touch
    assert abs(2.0 * r - geom.diameter) < 1e-12


@pytest.mark.unit
def test_geometry_meta_is_json_serialisable():
    geom = build_array("square", 20, 0.2)
    meta = geom.meta()
    assert json.loads(json.dumps(meta))["phi_target"] == 0.2
    assert meta["n_solid"] > 0


# ------------------------------------------------------------ analytic anchors
@pytest.mark.unit
def test_lamb_constant_and_oseen_drag():
    assert LAMB_CONST == pytest.approx(0.5 - 0.5772156649015329 + math.log(8.0), abs=1e-15)
    assert LAMB_CONST == pytest.approx(2.00222, abs=1e-5)
    cd_small = lamb_oseen_cd(1e-3)
    cd_large = lamb_oseen_cd(1e-1)
    assert cd_small > cd_large > 0.0
    # closed form check at Re = 0.1
    expected = 8.0 * math.pi / (0.1 * (LAMB_CONST - math.log(0.1)))
    assert lamb_oseen_cd(0.1) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError):
        lamb_oseen_cd(0.0)


@pytest.mark.unit
def test_dilute_log_law_slope_recovers_analytic_one():
    # 4 pi / f = ln(1/sqrt(phi)) + C  ->  analytic slope exactly one
    phis = [0.02, 0.04, 0.08]
    drags = [4.0 * math.pi / (math.log(1.0 / math.sqrt(p)) + 1.31) for p in phis]
    fit = dilute_log_law_slope(phis, drags)
    assert fit["slope"] == pytest.approx(1.0, abs=1e-12)
    assert fit["intercept"] == pytest.approx(1.31, abs=1e-12)
    assert fit["max_abs_resid"] < 1e-12


@pytest.mark.unit
def test_dilute_log_law_slope_detects_wrong_exponent():
    phis = [0.02, 0.04, 0.08]
    drags = [4.0 * math.pi / (0.5 * math.log(1.0 / math.sqrt(p)) + 1.0) for p in phis]
    fit = dilute_log_law_slope(phis, drags)
    assert fit["slope"] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.unit
def test_dilute_log_law_requires_two_points():
    with pytest.raises(ValueError):
        dilute_log_law_slope([0.02], [100.0])


# ---------------------------------------------------------------- estimators
@pytest.mark.unit
def test_expansion_fit_recovers_quadratic_law():
    c2 = 1.2e-4
    res = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ratios = [1.0 + c2 * r * r for r in res]
    fit = expansion_fit(res, ratios, degree=2)
    assert fit["c2"] == pytest.approx(c2, rel=1e-9)
    assert abs(fit["c1"]) < 1e-12
    assert fit["c1_share"] < 1e-8


@pytest.mark.unit
def test_expansion_fit_flags_a_linear_term():
    c1 = 1e-3
    res = [0.05, 0.1, 0.2, 0.5, 1.0]
    ratios = [1.0 + c1 * r for r in res]
    fit = expansion_fit(res, ratios, degree=2)
    assert fit["c1"] == pytest.approx(c1, rel=1e-6)
    assert fit["c1_share"] > 0.5  # linear term dominates


@pytest.mark.unit
def test_free_intercept_fit_sees_a_stokes_offset():
    res = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ratios = [1.004 + 1.2e-4 * r * r for r in res]
    fit = free_intercept_fit(res, ratios)
    assert fit["c0"] == pytest.approx(1.004, abs=1e-9)
    assert fit["c2"] == pytest.approx(1.2e-4, rel=1e-6)
    assert fit["c1_share"] < 1e-6


@pytest.mark.unit
def test_extrapolate_origin_recovers_mach_limit():
    y0, b = 1.000123, 3.0e-3
    mas = [0.01, 0.05, 0.1, 0.2]
    vals = [y0 + b * m * m for m in mas]
    ext = extrapolate_origin(mas, vals, degree=2)
    assert ext["value_at_origin"] == pytest.approx(y0, rel=1e-9)
    assert ext["relative_correction"] > 0.0
    assert ext["max_abs_resid"] < 1e-15


@pytest.mark.unit
def test_tail_and_power_law_extrapolation():
    limit, amp = 2.5e-4, 1e-5
    samples = [limit + amp * 0.7**k for k in range(6)]
    u_inf, q = tail_extrapolate(samples[-3], samples[-2], samples[-1])
    assert u_inf == pytest.approx(limit, rel=1e-9)
    assert 0.6 < q < 0.8

    times = [100.0 * k for k in range(1, 21)]
    vals = [limit + 3.0e-6 / (t**1.5) for t in times]
    ext = power_law_extrapolate(times, vals)
    assert ext["u_inf"] == pytest.approx(limit, rel=1e-5)
    assert ext["rms"] < 1e-9


@pytest.mark.unit
def test_resistance_ratio_definition():
    assert resistance_ratio(2.0, 2.0, a_ref=1.0, u_ref=1.0) == pytest.approx(1.0)
    assert resistance_ratio(1.0, 1.0, a_ref=1.0, u_ref=0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        resistance_ratio(1.0, 0.0, a_ref=1.0, u_ref=1.0)


# ------------------------------------------------------- lattice consistency
@pytest.mark.unit
def test_float64_weights_are_exact_unit_sum():
    w = _w_for(torch.float64, torch.device("cpu"))
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-16)
    w32 = _w_for(torch.float32, torch.device("cpu"))
    assert torch.equal(w32, w.float())


@pytest.mark.unit
def test_equilibrium_conserves_mass_in_float64():
    rho = torch.ones((8, 8), dtype=torch.float64)
    z = torch.zeros_like(rho)
    f = equilibrium(rho, z, z)
    assert float(f.sum()) == pytest.approx(64.0, abs=1e-15)
    # regression: the float32 weight table summed to 1 + 7.45e-9 and produced a
    # linear density ramp of 7.45e-9/tau per step in long float64 runs.
    f2 = f.clone()
    for _ in range(200):
        f2 = collide_trt(f2, 15.5)
    assert abs(float(f2.sum()) - 64.0) < 1e-12


@pytest.mark.unit
def test_split_kick_is_mass_conserving_and_sets_momentum():
    """The campaign's split force is a momentum kick with no mass source."""
    rho = torch.full((6, 6), 1.0, dtype=torch.float64)
    a = 1e-4
    cix, _, w = _tables(torch.float64, torch.device("cpu"))
    kick = 3.0 * a * w * rho.unsqueeze(0) * cix
    # opposite lattice directions cancel pairwise -> no mass source at all
    assert abs(float(kick.sum())) < 1e-30
    assert float((kick * cix).sum()) == pytest.approx(a * 36.0, rel=1e-15)


def test_guo_source_is_mass_conserving_and_has_the_right_momentum():
    rho = torch.full((6, 6), 1.0, dtype=torch.float64)
    ux = torch.full((6, 6), 0.05, dtype=torch.float64)
    uy = torch.zeros_like(ux)
    a, tau = 1e-4, 2.0
    src = guo_source_term(rho, ux, uy, a, tau)
    assert src.shape == (9, 6, 6)
    assert abs(float(src.sum())) < 1e-18  # mass conserving by construction
    cix = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float64).view(9, 1, 1)
    # Guo injects (1 - 1/(2 tau)) a per node and step; the half-force velocity
    # correction F/(2 rho) restores the exact force rho a.
    assert float((src * cix).sum()) == pytest.approx(a * (1.0 - 1.0 / (2.0 * tau)) * 36.0, rel=1e-12)


def test_lbm_step_schemes_agree_at_zero_force():
    """With no body force both forcing paths reduce to the same TRT step."""
    from tensorlbm.porous_weak_inertia import lbm_step

    geom = build_array("square", 16, 0.30)
    solid = geom.mask
    rho = torch.ones((16, 16), dtype=torch.float64)
    z = torch.zeros_like(rho)
    f0 = equilibrium(rho, z, z)
    f_split = lbm_step(f0, solid, 1.0, 0.0, scheme="split")
    f_guo = lbm_step(f0, solid, 1.0, 0.0, scheme="guo")
    assert torch.allclose(f_split, f_guo, atol=1e-13)
    with pytest.raises(ValueError):
        lbm_step(f0, solid, 1.0, 0.0, scheme="nope")


# --------------------------------------------------------------- end-to-end
@pytest.mark.integration
@pytest.mark.physics
def test_short_steady_solve_on_square_array():
    """One short forced solve: positive flow, correct Re, Stokes-like drag."""
    geom = build_array("square", 24, 0.30)
    nu, re_target, steps = 0.2, 0.05, 4000
    tau = 3.0 * nu + 0.5
    u_sup_target = re_target * nu / geom.diameter
    a = u_sup_target * nu * geom.fluid_fraction * 103.0 / geom.diameter**2  # f(0.30) ~ 103
    state = run_fixed_steps(geom, tau, a, steps)
    assert math.isfinite(state.u_superficial) and state.u_superficial > 0.0
    assert state.rho_mean == pytest.approx(1.0, abs=1e-9)
    re_ach = state.u_superficial * geom.diameter / nu
    # the body force here is a rough analytic guess (no calibration loop), so
    # only order-one agreement with the requested Reynolds number is expected
    assert 0.5 * re_target < re_ach < 2.5 * re_target
    f_drag = geom.cell_length**2 / (geom.fluid_fraction * nu * state.u_fluid / a)
    # coarse staircase grid (d ~ 15 cells): 35 % band around the Sangani table
    assert f_drag == pytest.approx(103.0, rel=0.35)
    assert abs(state.drift) < 1e-3
    assert state.monitor == "fixed_protocol"


@pytest.mark.unit
def test_interstitial_mach_definition():
    rho = torch.ones((4, 4), dtype=torch.float64)
    u = torch.full((4, 4), CS, dtype=torch.float64)
    _, ux, _ = macroscopic(equilibrium(rho, u, torch.zeros_like(u)))
    assert float(ux.mean()) / CS == pytest.approx(1.0, rel=1e-12)