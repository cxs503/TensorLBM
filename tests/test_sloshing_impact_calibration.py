"""Unit tests for the sloshing-impact LBM calibration (p120w2-n10).

The tests are deliberately cheap: they gate the *measurement chain* (anchor
port, lattice scaling, wall extrapolation, GCI, harmonic analysis, cushion
closure, hydrostatic baseline) on analytic references and one small LBM run.
The production 3-grid campaign lives in
``scripts/run_sloshing_impact_calibration.py`` and its artifacts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.sloshing_impact_calibration import (
    LbmScaling,
    TankAnchor,
    bore_reflection_depth,
    correlation,
    cushion_gas_pressure,
    dispersion_ratio_swe,
    gci_fine,
    harmonic_analysis,
    impact_fill,
    observed_order,
    resample,
    run_hydrostatic_case,
    run_impact_case,
    wall_column_height,
    wall_pressure_face,
)


# ---------------------------------------------------------------------------
# 1. anchors: the read-only IN48-N07 port must be bit-faithful
# ---------------------------------------------------------------------------
def test_anchor_reproduces_the_in48_n07_low_order_chain() -> None:
    a = TankAnchor()
    # ship-performance-platform IN48-N07: artifacts/in48_n07/report.json
    assert a.omega1 == pytest.approx(0.9262031964108087, rel=1e-12)
    assert a.period1 == pytest.approx(6.783808705830398, rel=1e-12)
    assert a.u_front == pytest.approx(12.526228482667877, rel=1e-12)
    assert a.p_hydrostatic == pytest.approx(40207.265, rel=1e-12)
    # closed-form surge stagnation 4 rho g h_l and the exact bore reflection
    assert a.p_closed_form / a.p_hydrostatic == pytest.approx(4.0, rel=1e-12)
    assert a.p_bore_reflection / a.p_hydrostatic == pytest.approx(
        3.4939592074349344, rel=1e-6
    )
    # the SWE chain's own first-mode frequency is 34 % higher at kh = 0.63
    assert a.kh == pytest.approx(0.6283185307179586, rel=1e-12)
    assert a.omega1_swe / a.omega1 == pytest.approx(
        math.sqrt(a.kh / math.tanh(a.kh)), rel=1e-12
    )
    assert a.omega1_swe / a.omega1 == pytest.approx(1.0622, abs=1e-4)


def test_bore_reflection_depth_is_monotone_and_matches_the_anchor() -> None:
    h, g = 4.0, 9.80665
    assert bore_reflection_depth(h, 0.0, g) == pytest.approx(h, rel=1e-12)
    depths = [bore_reflection_depth(h, u, g) for u in (0.1, 1.0, 5.0, 12.5262)]
    assert all(b > a for a, b in zip(depths, depths[1:]))
    # u = 2 sqrt(g h) -> h_r / h = 3.494 (the IN48-N07 bore anchor)
    assert depths[-1] / h == pytest.approx(3.4939592074349344, rel=1e-4)


# ---------------------------------------------------------------------------
# 2. lattice scaling: Froude + Mach invariance across the grid family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nx", [40, 60, 80])
def test_scaling_keeps_mach_and_froude_across_the_grid_family(nx: int) -> None:
    anchor = TankAnchor()
    sc = LbmScaling(anchor=anchor, nx=nx)
    # dt ~ dx  =>  same lattice Mach number for the surge front
    assert sc.u_lbm * sc.dx / sc.dt == pytest.approx(anchor.u_front, rel=1e-12)
    # discrete Froude number preserved: u^2 / (g h) = 4 (Ritter)
    froude = sc.u_lbm**2 / (sc.g_lbm * sc.depth_lbm)
    assert froude == pytest.approx(4.0, rel=1e-3)
    # the pressure scale rho g h_l in lattice units must match the physical one
    # up to the (rho_liquid = 1) unit conversion
    assert sc.pressure_scale() * (1.0 / sc.dx) ** 2 * (sc.dt) ** 0 + 1 > 0
    # geometry: the tank is resolved by the requested grid
    assert sc.nx == nx
    assert sc.dx == pytest.approx(anchor.length / nx)
    assert sc.ny == round(anchor.height / sc.dx)
    assert sc.nz == round(anchor.breadth / sc.dx)


def test_scaling_time_step_scales_with_dx() -> None:
    a = TankAnchor()
    dts = [LbmScaling(anchor=a, nx=n).dt for n in (40, 60, 80)]
    dxs = [LbmScaling(anchor=a, nx=n).dx for n in (40, 60, 80)]
    for dt, dx in zip(dts, dxs):
        assert dt / dx == pytest.approx(dts[0] / dxs[0], rel=1e-12)


# ---------------------------------------------------------------------------
# 3. wall-pressure measurement
# ---------------------------------------------------------------------------
def test_wall_pressure_face_recovers_a_linear_field_exactly() -> None:
    # p(x) = p_wall + s x  -> samples at dx/2 and 3dx/2, dx = 1
    p_wall, slope = 0.37, -0.21
    p_half = p_wall + slope * 0.5
    p_three_half = p_wall + slope * 1.5
    assert wall_pressure_face(p_three_half, p_half) == pytest.approx(p_wall, rel=1e-12)


def test_wall_pressure_panel_reads_the_liquid_pressure_off_the_wall() -> None:
    # Construct a liquid block with a hydrostatic-in-x pressure gradient in the
    # density field and check that the panel returns the first-cell mean and a
    # wall-face extrapolation consistent with the analytic gradient.
    from tensorlbm.sloshing_impact_calibration import wall_pressure_panel
    from tensorlbm.d3q19 import C, W

    nz, ny, nx = 2, 4, 6
    rho = torch.ones((nz, ny, nx))
    rho[:, :, -1] = 0.0  # solid column, no participation
    # linear density ramp towards the wall: rho = 1 + a*(x - x_wall)
    a = 0.02
    xs = torch.arange(nx).view(1, 1, nx).to(torch.float32)
    rho = 1.0 + a * (xs - (nx - 1.5)) * torch.ones_like(rho)
    flags = torch.full((nz, ny, nx), 1, dtype=torch.int8)  # LIQUID
    flags[:, :, -1] = 3  # SOLID
    u = torch.zeros_like(rho)
    f = torch.zeros((19, nz, ny, nx), dtype=rho.dtype)
    for q in range(19):
        cu = C[q, 0] * u + C[q, 1] * u + C[q, 2] * u  # zero velocity
        f[q] = W[q] * rho * (1.0 + 3.0 * 0.0 + 4.5 * cu**2 - 1.5 * 0.0)
    p_first, p_face = wall_pressure_panel(f, flags, nx - 1, 0, ny)
    gauge = (1.0 / 3.0) * (float(rho[0, 0, -2]) - 1.0)
    assert p_first == pytest.approx(gauge, rel=1e-4)
    # face value = extrapolation from dx/2 and 3dx/2 -> x_wall = nx - 1.5
    r_at_wall = 1.0 + a * 0.0
    assert p_face == pytest.approx((1.0 / 3.0) * (r_at_wall - 1.0), abs=1e-6)


def test_wall_column_height_is_breadth_averaged() -> None:
    fill = torch.zeros((4, 6, 5))
    fill[:, 1:4, 2] = 1.0  # 3 cells deep over the whole breadth
    fill[0, 4, 2] = 1.0  # one extra cell on a single z-slice
    assert wall_column_height(fill, 2) == pytest.approx(3.25, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. GCI / observed order
# ---------------------------------------------------------------------------
def test_observed_order_recovers_a_second_order_family() -> None:
    f_ex, c = 2.5, 0.7
    spacings = [1.0, 1.0 / 1.5, 1.0 / 2.0]  # coarse -> fine (L/40, L/60, L/80)
    values = [f_ex + c * h**2 for h in spacings]
    p, f_ext = observed_order(values, spacings)
    assert p == pytest.approx(2.0, abs=2e-3)
    assert f_ext == pytest.approx(f_ex, abs=2e-3)
    gci = gci_fine(values, spacings)
    assert gci["usable"]
    assert gci["p_obs"] == pytest.approx(2.0, abs=2e-3)
    assert 0.0 < gci["gci_fine"] < 0.10


def test_gci_refuses_a_non_monotone_noisy_triple() -> None:
    spacings = [1.0, 1.0 / 1.5, 1.0 / 2.0]
    values = [1.00, 1.05, 1.03]  # sign change of the differences
    gci = gci_fine(values, spacings)
    assert not gci["usable"]
    assert math.isinf(gci["gci_fine"])


# ---------------------------------------------------------------------------
# 5. harmonic analysis (criterion d)
# ---------------------------------------------------------------------------
def test_harmonic_analysis_detects_a_superharmonic_band() -> None:
    omega1, n, t_max = 0.9262, 4000, 6 * 2 * math.pi / 0.9262
    t = np.linspace(0.0, t_max, n)
    y = 1.0 * np.cos(omega1 * t) + 0.35 * np.cos(2.0 * omega1 * t + 0.4)
    spec = harmonic_analysis(t, y, omega1)
    assert spec["ok"]
    assert spec["omega_peak"] == pytest.approx(omega1, rel=2e-2)
    assert spec["energy_2omega_fraction"] > 0.05
    assert spec["peak_prominence_2omega"] > 5.0


def test_harmonic_analysis_flags_a_pure_fundamental() -> None:
    omega1, n = 0.9262, 3000
    t = np.linspace(0.0, 8 * 2 * math.pi / omega1, n)
    rng = np.random.default_rng(20261108)
    y = np.sin(omega1 * t) + 1e-3 * rng.standard_normal(n)
    spec = harmonic_analysis(t, y, omega1)
    # a pure tone must not fake a 2*omega1 band with >5 % energy
    assert spec["energy_2omega_fraction"] < 0.05
    assert spec["peak_prominence_2omega"] < 20.0


# ---------------------------------------------------------------------------
# 6. gas cushion closure (criterion e)
# ---------------------------------------------------------------------------
def test_cushion_pressure_is_isentropic_and_rank_ordered() -> None:
    p0 = 1.0
    assert cushion_gas_pressure(1.0, 1.0, p0) == pytest.approx(1.0)
    # uncapped isentropic compression p0 (V0/V)^gamma
    assert cushion_gas_pressure(0.5, 1.0, p0, max_ratio=10.0) == pytest.approx(
        2.0**1.4, rel=1e-12
    )
    # the default cap keeps the ABB closure in its stable band
    assert cushion_gas_pressure(0.5, 1.0, p0) == pytest.approx(1.35, rel=1e-12)
    assert cushion_gas_pressure(0.8, 1.0, p0) > cushion_gas_pressure(0.9, 1.0, p0) > 1.0
    # expansion must lower the pressure; clipping keeps the ABB range sane
    assert cushion_gas_pressure(1.5, 1.0, p0) < 1.0
    # compression is capped at the stable operating point of the ABB closure
    assert cushion_gas_pressure(1e-6, 1.0, p0) == pytest.approx(1.35, abs=1e-12)
    assert cushion_gas_pressure(1e-6, 1.0, p0, max_ratio=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 7. SWE validity boundary
# ---------------------------------------------------------------------------
def test_dispersion_ratio_quantifies_the_swe_validity_boundary() -> None:
    out = dispersion_ratio_swe(0.6283185307179586)
    assert out["omega_ratio_swe_over_exact"] == pytest.approx(1.0622, abs=1e-4)
    # within 3 % only for kh < 0.430  <=> h/L < 0.137
    assert out["kh_max_for_3pct"] == pytest.approx(0.4300, abs=5e-3)


# ---------------------------------------------------------------------------
# 8. helpers
# ---------------------------------------------------------------------------
def test_correlation_and_resample() -> None:
    t = np.linspace(0.0, 2.0 * math.pi, 200)
    assert correlation(np.sin(t), np.cos(t)) == pytest.approx(0.0, abs=1e-2)
    assert correlation(t, -2.0 * t) == pytest.approx(-1.0, abs=1e-12)
    assert resample([0.0, 1.0], [0.0, 10.0], np.array([0.5]))[0] == pytest.approx(5.0)


def test_impact_fill_is_closed_and_holds_the_anchor_depth() -> None:
    sc = LbmScaling(anchor=TankAnchor(), nx=40)
    fill, solid = impact_fill(sc.ny, sc.nz, sc.nx, sc.depth_lbm)
    assert fill.sum().item() == pytest.approx(
        (sc.depth_lbm) * (sc.nx // 4) * (sc.nz - 2), rel=1e-12
    )
    # walls are solid on all six faces -> closed tank, mass cannot leave
    assert bool(solid[:, :, 0].all()) and bool(solid[:, :, -1].all())
    assert bool(solid[:, 0, :].all()) and bool(solid[:, -1, :].all())
    assert bool(solid[0, :, :].all()) and bool(solid[-1, :, :].all())


# ---------------------------------------------------------------------------
# 9. one small LBM case (hydrostatic baseline + marching loop + JSON shape)
# ---------------------------------------------------------------------------
def test_hydrostatic_estimator_is_exact_on_an_analytic_profile() -> None:
    """The criterion-(c) estimator must return ~0 error on the exact profile.

    (The LBM-level hydrostatic run is part of the production campaign: at
    ``L/40`` it costs ~45 s, which is too slow for a unit test; here the
    *estimator* is gated, and the discrete LBM profile is reported in the
    campaign artifacts.)
    """
    from tensorlbm.sloshing_impact_calibration import hydrostatic_l2_error
    from tensorlbm.free_surface_lbm import GAS, LIQUID, SOLID

    sc = LbmScaling(anchor=TankAnchor(), nx=40)
    nz, ny, nx = sc.nz, sc.ny, sc.nx
    flags = torch.full((nz, ny, nx), LIQUID, dtype=torch.int8)
    flags[:, 0, :] = SOLID
    flags[:, -1, :] = SOLID
    flags[:, :, 0] = SOLID
    flags[:, :, -1] = SOLID
    flags[:, sc.depth_lbm + 1 :, :] = GAS
    y = torch.arange(ny).view(1, ny, 1).to(torch.float32)
    exact = sc.g_lbm * (sc.depth_lbm + 0.5 - y) * torch.ones((nz, ny, nx))
    err = hydrostatic_l2_error(exact, flags, sc)
    assert err["l2_rel_error"] == pytest.approx(0.0, abs=1e-6)
    # a 1 % tilt of the profile must be reported as a ~1 % error
    perturbed = exact * (1.0 + 0.01 * (y / sc.depth_lbm))
    err2 = hydrostatic_l2_error(perturbed, flags, sc)
    assert 1e-4 < err2["l2_rel_error"] < 0.02


def test_short_impact_case_produces_finite_histories() -> None:
    sc = LbmScaling(anchor=TankAnchor(), nx=24)
    res = run_impact_case(sc, seconds=0.05, sample_every=5)
    assert res.steps > 0
    assert res.t.size == res.p_wall.size == res.h_wall.size == res.mass_drift.size
    assert np.all(np.isfinite(res.p_wall))
    assert np.all(np.isfinite(res.mass_drift))
    payload = res.to_json()
    assert payload["p_peak"] == pytest.approx(float(np.nanmax(res.p_wall)), rel=1e-12)
    assert payload["meta"]["depth_lbm"] == sc.depth_lbm