"""Deck thermal field: solar boundary split + analytic anchors + LBM field.

Acceptance criterion (d) of P93W3-N02 is pinned here: the steady deck
temperature field from the D2Q5 lattice-Boltzmann solver must agree with an
**independent analytic solution** to better than 3 %.  Two anchors are used:

* through-thickness: nonlinear surface balance solved by bisection + linear
  interior profile (:func:`analytic_steady_field_1d`),
* shadow edge: exact cosine-series solution of the linearised 2-D problem
  (:func:`analytic_steady_2d_fourier`).

The heavy LBM runs carry the ``slow`` marker (CI deselects them on PRs), but a
representative pair is executed here so the gate is genuinely covered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.deck_thermal_field import (
    DeckMaterial,
    analytic_steady_2d_fourier,
    analytic_steady_field_1d,
    cosine_coefficients,
    solar_flux_components,
    solve_deck_field,
    surface_net_flux,
)
from tensorlbm.deck_thermal_load import (
    DeckPanel,
    SkySeaEnvironment,
    SunState,
    absorbed_solar_flux,
    solve_deck_temperature,
)

GATE = 0.03  # criterion (d): 3 %


def make_case(tilt_deg: float = 0.0) -> tuple[DeckPanel, SkySeaEnvironment]:
    panel = DeckPanel(
        tilt_deg=tilt_deg, emissivity=0.9, solar_absorptance=0.6, h_conv=12.0
    )
    sun = SunState(
        beam_irradiance=800.0, elevation_deg=60.0, azimuth_deg=180.0, diffuse_horizontal=120.0
    )
    env = SkySeaEnvironment(T_air=303.0, T_dewpoint=288.0, T_sea=293.0, sun=sun)
    return panel, env


# ---------------------------------------------------------------------------
# solar boundary: three components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.0, 30.0, 60.0, 90.0])
def test_solar_components_sum_to_established_aggregate(tilt: float) -> None:
    """The split must never drift from ``absorbed_solar_flux``."""
    panel, env = make_case(tilt)
    comp = solar_flux_components(panel, env.sun)
    ref = absorbed_solar_flux(panel, env.sun)
    assert abs(comp["total"] - ref) <= 1e-12 * max(abs(ref), 1.0)
    assert abs(
        comp["direct"] + comp["diffuse"] + comp["ground_reflected"] - ref
    ) <= 1e-12 * max(abs(ref), 1.0)


def test_solar_three_components_are_physically_distinct() -> None:
    panel, env = make_case(0.0)
    comp = solar_flux_components(panel, env.sun)
    assert comp["direct"] > 0.0 and comp["diffuse"] > 0.0
    assert comp["ground_reflected"] == pytest.approx(0.0, abs=1e-12)  # F_sea = 0 at beta=0
    panel_t, env_t = make_case(60.0)
    comp_t = solar_flux_components(panel_t, env_t.sun)
    assert comp_t["ground_reflected"] > 0.0  # tilted panel sees the sea
    night = SunState(beam_irradiance=0.0, elevation_deg=-20.0, diffuse_horizontal=0.0)
    panel_n, env_n = make_case(0.0)
    env_n.sun = night
    comp_n = solar_flux_components(panel_n, night)
    assert comp_n["total"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------


def test_adiabatic_underside_reproduces_0d_equilibrium() -> None:
    """h_in = 0 -> the field collapses onto the established 0-D solver."""
    panel, env = make_case()
    mat = DeckMaterial(thickness=0.025)
    ana = analytic_steady_field_1d(panel, env, mat, T_interior=300.0, h_interior=0.0)
    ref = solve_deck_temperature(panel, env)["T_deck"]
    assert abs(ana["T_surface"] - ref) < 1e-6
    assert ana["q"] == 0.0
    assert abs(ana["residual"]) < 1e-9


def test_analytic_1d_balance_and_linear_profile() -> None:
    panel, env = make_case()
    mat = DeckMaterial(thickness=0.025)
    res = analytic_steady_field_1d(panel, env, mat, T_interior=300.0, h_interior=100.0)
    assert abs(res["residual"]) < 1e-8
    q_surf = float(surface_net_flux(res["T_surface"], panel, env, res["q_solar_abs"])[0])
    assert abs(q_surf - res["q"]) < 1e-8
    z = np.array(res["z"])
    T = np.array(res["T_profile"])
    slope = np.diff(T) / np.diff(z)
    assert np.allclose(slope, res["q"] / mat.k, rtol=1e-9)
    assert T[-1] > T[0]  # heated from above


def test_cosine_coefficients_exact() -> None:
    """Constant profile -> only the mean mode; panel means are reproduced exactly.

    (Pointwise comparison is not meaningful at a jump: a Fourier series
    converges to the midpoint there, so the *panel mean* — the conservation
    content of the coefficients — is what gets tested.)
    """
    width = 0.2
    coef = cosine_coefficients([0.0, width], [500.0], width, 6)
    assert coef[0] == pytest.approx(500.0)
    assert np.allclose(coef[1:], 0.0, atol=1e-12)

    edges = [0.0, 0.07, 0.15, width]
    values = [72.0, 487.7, 0.0]
    coef = cosine_coefficients(edges, values, width, 600)
    assert coef[0] == pytest.approx(sum(v * (b - a) for a, b, v in zip(edges, edges[1:], values)) / width)
    xs = np.linspace(0.0, width, 200_001)
    recon = np.full_like(xs, coef[0])
    for n in range(1, len(coef)):
        recon += coef[n] * np.cos(n * math.pi * xs / width)
    for a, b, v in zip(edges[:-1], edges[1:], values):
        m = (xs >= a) & (xs <= b)
        assert abs(recon[m].mean() - v) < 1.0


def test_fourier_uniform_flux_is_x_flat_and_matches_1d() -> None:
    panel, env = make_case()
    mat = DeckMaterial(thickness=0.025)
    q_full = solar_flux_components(panel, env.sun)["total"]
    ana1 = analytic_steady_field_1d(panel, env, mat, 300.0, 100.0, q_solar_abs=q_full)
    width = 0.2
    fourier = analytic_steady_2d_fourier(
        panel, env, mat, 300.0, 100.0, [0.0, width], [q_full], width, ana1["T_surface"], n_modes=8
    )
    x = np.linspace(0.0, width, 5)
    z = np.linspace(0.0, mat.thickness, 6)
    T = fourier["evaluate"](x, z)
    assert np.max(np.abs(T - T[:, :1])) < 1e-9  # lateral uniformity
    # linearised vs nonlinear surface balance: <0.05 K for this temperature level
    assert abs(T[-1, 0] - ana1["T_surface"]) < 0.05


def test_fourier_series_truncation_converges() -> None:
    panel, env = make_case()
    mat = DeckMaterial(thickness=0.025)
    q_full = solar_flux_components(panel, env.sun)["total"]
    q_shade = solar_flux_components(panel, env.sun)["diffuse"]
    width, Tm = 0.2, 304.0
    kw = dict(solar_edges=[0.0, width / 2, width], solar_values=[q_shade, q_full], width=width)
    a = analytic_steady_2d_fourier(panel, env, mat, 300.0, 100.0, T_linearisation=Tm, n_modes=200, **kw)
    b = analytic_steady_2d_fourier(panel, env, mat, 300.0, 100.0, T_linearisation=Tm, n_modes=800, **kw)
    x = np.linspace(0.005, width - 0.005, 40)
    z = np.linspace(0.001, mat.thickness - 0.001, 8)
    assert np.max(np.abs(a["evaluate"](x, z) - b["evaluate"](x, z))) < 1e-5


def test_guard_rails() -> None:
    panel, env = make_case()
    mat = DeckMaterial()
    with pytest.raises(ValueError):
        DeckMaterial(thickness=-1.0)
    with pytest.raises(ValueError):
        analytic_steady_field_1d(panel, env, mat, T_interior=-5.0, h_interior=8.0)
    with pytest.raises(ValueError):
        analytic_steady_field_1d(panel, env, mat, T_interior=300.0, h_interior=-1.0)
    with pytest.raises(ValueError):
        cosine_coefficients([0.0, 1.0], [1.0], width=0.5, n_modes=4)
    with pytest.raises(ValueError):
        solve_deck_field(panel, env, mat, 300.0, 100.0, nx=1, ny=1)  # ny must be >= 2
    with pytest.raises(ValueError):
        solve_deck_field(panel, env, mat, 300.0, 100.0, nx=2, ny=4, solar_profile=np.zeros(3))
    with pytest.raises(ValueError):
        solve_deck_field(panel, env, mat, 300.0, 100.0, nx=2, ny=4, tau=0.4)


@pytest.mark.slow
def test_lbm_1d_field_vs_analytic() -> None:
    """Criterion (d), through-thickness: LBM vs bisection + linear profile."""
    panel, env = make_case()
    mat = DeckMaterial(thickness=0.025)
    q_full = absorbed_solar_flux(panel, env.sun)
    ana = analytic_steady_field_1d(panel, env, mat, 300.0, 100.0, q_solar_abs=q_full)
    sol = solve_deck_field(
        panel, env, mat, 300.0, 100.0, solar_profile=q_full, nx=1, ny=6,
        tau=1.0, tol=1e-7, check_every=200, max_steps=200_000,
    )
    assert sol.converged
    T_ana = np.interp(sol.z, np.array(ana["z"]), np.array(ana["T_profile"]))
    err = np.max(np.abs(sol.T[:, 0] - T_ana)) / np.max(T_ana)
    assert err < GATE, err
    assert sol.energy_residual < 1e-6
    assert abs(sol.q_top[0] - ana["q"]) / abs(ana["q"]) < GATE


@pytest.mark.slow
def test_lbm_2d_shadow_patch_vs_fourier() -> None:
    """Criterion (d), shadow edge: LBM (frozen linearisation) vs cosine series."""
    panel, env = make_case()
    mat = DeckMaterial(thickness=0.025)
    comp = solar_flux_components(panel, env.sun)
    q_full, q_shade = comp["total"], comp["diffuse"]
    ana1 = analytic_steady_field_1d(panel, env, mat, 300.0, 100.0, q_solar_abs=q_full)
    Tm, width, nx, ny = ana1["T_surface"], 0.2, 32, 4
    fourier = analytic_steady_2d_fourier(
        panel, env, mat, 300.0, 100.0, [0.0, width / 2, width], [q_shade, q_full], width, Tm,
        n_modes=400,
    )
    x = (np.arange(nx) + 0.5) * width / nx
    sol = solve_deck_field(
        panel, env, mat, 300.0, 100.0, solar_profile=np.where(x < width / 2, q_shade, q_full),
        nx=nx, ny=ny, tau=1.0, linearise_at=Tm, tol=1e-7, check_every=200, max_steps=200_000,
    )
    assert sol.converged
    T_ana = fourier["evaluate"](sol.x, sol.z)
    err = np.max(np.abs(sol.T - T_ana)) / np.max(T_ana)
    assert err < GATE, err
    assert sol.energy_residual < 1e-6
    assert sol.T.max() - sol.T.min() > 0.1  # the shadow edge really is resolved
