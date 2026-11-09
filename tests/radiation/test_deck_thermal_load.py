"""Verification of the solar/sky boundary -> deck thermal load chain.

Criterion (c) of P92W3-N02: the deck heat load must agree with the analytic
solution to better than 5 %.  Three independent routes are compared:

1. the full chain (closed-form view factors -> gray enclosure radiosity ->
   coupled surface energy balance, Brent root),
2. the analytic/reference scalar balance (black-surroundings long-wave form,
   independent fixed-count bisection),
3. the sampling estimator (cosine-weighted Monte-Carlo ray tracing) for the
   long-wave exchange.

For ``h_conv = 0`` and a black sky the balance has a closed form
``T = (T_eff^4 + alpha G / (eps sigma))^{1/4}``, used as a fourth check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.deck_thermal_load import (
    DeckPanel,
    SkySeaEnvironment,
    SunState,
    absorbed_solar_flux,
    clear_sky_irradiance,
    closed_form_equilibrium_temperature,
    cosine_incidence,
    deck_energy_balance,
    deck_longwave_analytic,
    deck_longwave_enclosure,
    deck_longwave_raytraced,
    deck_view_factors,
    effective_surroundings_temperature,
    inject_deck_boundary,
    reference_deck_temperature,
    sky_view_factor_shaded_deck,
    solar_position,
    solve_deck_temperature,
)

LOAD_TOL = 0.05  # machine criterion (c)


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def night_env(sky_emissivity: float = 1.0) -> SkySeaEnvironment:
    return SkySeaEnvironment(
        T_air=300.0, T_dewpoint=285.0, T_sea=293.0, sky_emissivity=sky_emissivity
    )


def day_env(gray_sky: bool = False) -> SkySeaEnvironment:
    return SkySeaEnvironment(
        T_air=303.0,
        T_dewpoint=288.0,
        T_sea=293.0,
        sky_emissivity=0.85 if gray_sky else 1.0,
        sun=SunState(
            beam_irradiance=800.0,
            elevation_deg=60.0,
            azimuth_deg=180.0,
            diffuse_horizontal=120.0,
        ),
    )


# ---------------------------------------------------------------------------
# geometry: view factors and solar incidence
# ---------------------------------------------------------------------------


def test_deck_view_factors_summation_and_limits() -> None:
    for tilt in (0.0, 15.0, 45.0, 90.0, 135.0, 180.0):
        F_sky, F_sea = deck_view_factors(tilt)
        assert abs(F_sky + F_sea - 1.0) < 1e-15
    assert abs(deck_view_factors(0.0)[0] - 1.0) < 1e-15  # horizontal deck: full sky
    assert abs(deck_view_factors(90.0)[0] - 0.5) < 1e-15  # vertical: half sky


def test_cosine_incidence_and_shading() -> None:
    panel = DeckPanel(tilt_deg=0.0)
    assert abs(cosine_incidence(panel, SunState(elevation_deg=90.0)) - 1.0) < 1e-12
    assert cosine_incidence(panel, SunState(elevation_deg=-10.0)) == 0.0  # sun below horizon
    # a north-facing vertical panel sees nothing of a southern sun
    panel_n = DeckPanel(tilt_deg=90.0, azimuth_deg=0.0)
    assert cosine_incidence(panel_n, SunState(elevation_deg=30.0, azimuth_deg=180.0)) == 0.0


def test_absorbed_solar_flux_normalisation() -> None:
    """Absorptance 0/1 bracket, and horizontal deck with zenith sun."""
    sun = SunState(beam_irradiance=1000.0, elevation_deg=90.0, diffuse_horizontal=0.0)
    black = DeckPanel(tilt_deg=0.0, solar_absorptance=0.0)
    white = DeckPanel(tilt_deg=0.0, solar_absorptance=1.0)
    assert absorbed_solar_flux(black, sun) == 0.0
    assert abs(absorbed_solar_flux(white, sun) - 1000.0) < 1e-9  # F_sky = 1, rho=0.06


def test_solar_position_and_clear_sky_sanity() -> None:
    elev_noon, _ = solar_position(30.0, 172, 12.0)
    elev_midnight, _ = solar_position(30.0, 172, 0.0)
    assert elev_noon > 60.0 > elev_midnight
    G_b, G_d = clear_sky_irradiance(elev_noon)
    assert 800.0 < G_b < 1100.0
    assert 0.0 < G_d < G_b
    assert clear_sky_irradiance(-5.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# long-wave exchange: three routes
# ---------------------------------------------------------------------------


def test_longwave_analytic_vs_monte_carlo() -> None:
    """Closed-form integral vs cosine-weighted ray tracing (<= 1 %)."""
    panel = DeckPanel(tilt_deg=0.0)
    env = day_env()
    T_deck = 320.0
    q_ana = deck_longwave_analytic(panel, env, T_deck)
    mc = deck_longwave_raytraced(panel, env, T_deck, n_rays=1_000_000, seed=5)
    assert rel_err(mc["q_longwave"], q_ana) < 0.01, (mc["q_longwave"], q_ana)


def test_longwave_tilted_panel_vs_monte_carlo() -> None:
    """Tilted deck: ray tracing reproduces the (1+cos beta)/2 split."""
    panel = DeckPanel(tilt_deg=45.0)
    env = day_env()
    q_ana = deck_longwave_analytic(panel, env, 320.0)
    mc = deck_longwave_raytraced(panel, env, 320.0, n_rays=1_000_000, seed=9)
    assert abs(mc["F_sky_mc"] - 0.5 * (1.0 + math.cos(math.radians(45.0)))) < 5e-3
    assert rel_err(mc["q_longwave"], q_ana) < 0.01


def test_longwave_enclosure_vs_analytic_black_sky() -> None:
    """A black sky makes the gray enclosure collapse onto the analytic form."""
    panel = DeckPanel()
    env = night_env(sky_emissivity=1.0)
    q_ana = deck_longwave_analytic(panel, env, 300.0)
    q_enc, _ = deck_longwave_enclosure(panel, env, 300.0)
    assert rel_err(q_enc, q_ana) < 1e-9


def test_longwave_gray_sky_correction_is_small() -> None:
    """Gray sky (eps_sky = 0.85) adds < 1 % to the long-wave exchange."""
    panel = DeckPanel()
    env = day_env(gray_sky=True)
    q_ana = deck_longwave_analytic(panel, env, 320.0)
    q_enc, audit = deck_longwave_enclosure(panel, env, 320.0)
    assert rel_err(q_enc, q_ana) < 0.01
    assert audit["energy_residual"] < 1e-6


# ---------------------------------------------------------------------------
# criterion (c): deck thermal load vs analytic solution
# ---------------------------------------------------------------------------


def test_night_radiative_equilibrium_matches_closed_form() -> None:
    """h = 0, no sun, black sky: T_deck = T_sky exactly."""
    panel = DeckPanel(h_conv=0.0)
    env = night_env()
    T_closed = closed_form_equilibrium_temperature(panel, env)
    T_brent = solve_deck_temperature(panel, env, longwave="analytic")["T_deck"]
    T_ref = reference_deck_temperature(panel, env)["T_deck"]
    assert T_closed == pytest.approx(env.sky_temperature(), rel=1e-12)
    assert rel_err(T_brent, T_closed) < 1e-10
    assert rel_err(T_ref, T_closed) < 1e-10


def test_night_load_matches_analytic() -> None:
    """Night, still air: deck radiates to a colder sky (negative load = cooling)."""
    panel = DeckPanel(h_conv=0.0)
    env = night_env()
    sol = solve_deck_temperature(panel, env, longwave="analytic")
    ref = reference_deck_temperature(panel, env)
    assert sol["q_net"] == pytest.approx(0.0, abs=1e-8)
    assert rel_err(sol["T_deck"], ref["T_deck"]) < LOAD_TOL
    # the deck can only cool to the sky, never below it, and the loss is real
    assert sol["T_deck"] >= env.sky_temperature() - 1e-9
    assert sol["q_required"] >= -1e-6


def test_day_solar_load_matches_reference() -> None:
    """Sun + convection: Brent on the chain vs independent bisection."""
    panel = DeckPanel(h_conv=12.0)
    env = day_env()
    sol = solve_deck_temperature(panel, env, longwave="analytic")
    ref = reference_deck_temperature(panel, env)
    assert rel_err(sol["T_deck"], ref["T_deck"]) < LOAD_TOL
    assert sol["T_deck"] > env.T_air  # solar heating pushes the deck above air
    for key in ("q_solar", "q_longwave", "q_conv"):
        assert abs(sol[key] - ref[key]) < 1e-6 * max(1.0, abs(ref[key]))


@pytest.mark.parametrize("tilt", [0.0, 20.0, 45.0, 70.0, 90.0])
def test_tilt_sweep_load_matches_reference(tilt: float) -> None:
    """Deck load vs analytic over the tilt sweep, incl. convection."""
    panel = DeckPanel(tilt_deg=tilt, h_conv=15.0)
    env = day_env()
    sol = solve_deck_temperature(panel, env, longwave="analytic")
    ref = reference_deck_temperature(panel, env)
    assert rel_err(sol["T_deck"], ref["T_deck"]) < LOAD_TOL
    assert sol["residual"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("sky_eps", [1.0, 0.85, 0.7])
def test_full_chain_gray_enclosure_vs_analytic_load(sky_eps: float) -> None:
    """Criterion (c) on the *full* chain (enclosure long-wave) vs analytic."""
    panel = DeckPanel(h_conv=12.0)
    env = SkySeaEnvironment(
        T_air=303.0, T_dewpoint=288.0, T_sea=293.0, sky_emissivity=sky_eps, sun=day_env().sun
    )
    sol = solve_deck_temperature(panel, env, longwave="enclosure")
    ref = reference_deck_temperature(panel, env)
    assert rel_err(sol["T_deck"], ref["T_deck"]) < LOAD_TOL
    # the required duty (what the HVAC / structure must remove) must match too
    assert rel_err(sol["q_required"], ref["q_required"]) < LOAD_TOL or abs(
        ref["q_required"]
    ) < 5.0


def test_load_energy_balance_closes() -> None:
    """q_solar + q_longwave + q_conv = q_net = 0 at equilibrium (all routes)."""
    panel = DeckPanel(h_conv=10.0)
    env = day_env()
    for method in ("analytic", "enclosure"):
        sol = solve_deck_temperature(panel, env, longwave=method)
        parts = sol["q_solar"] + sol["q_longwave"] + sol["q_conv"]
        assert abs(parts - sol["q_net"]) < 1e-9
        assert abs(sol["q_net"]) < 1e-6


# ---------------------------------------------------------------------------
# superstructure shading
# ---------------------------------------------------------------------------


def test_shaded_sky_view_factor_limits_and_mc() -> None:
    assert abs(sky_view_factor_shaded_deck(0.0, 2.0) - 1.0) < 1e-9
    big = sky_view_factor_shaded_deck(1e5, 1.0)
    # a narrow band near the horizon limits the quadrature to ~1e-3 here
    assert abs(big - 0.5) < 0.01
    F_quad = sky_view_factor_shaded_deck(2.0, 2.0)
    mc = deck_longwave_raytraced(DeckPanel(), day_env(), 320.0, n_rays=1_000_000, seed=2, wall_height=2.0, wall_distance=2.0)
    assert abs(mc["F_sky_mc"] - F_quad) < 5e-3, (mc["F_sky_mc"], F_quad)


# ---------------------------------------------------------------------------
# LBM / conduction boundary injection
# ---------------------------------------------------------------------------


def test_injection_conserves_injected_energy() -> None:
    T = np.full((8, 8), 300.0)
    mask = np.zeros_like(T, dtype=bool)
    mask[0, :] = True
    q, dt, rho, cp, dx, dA = 250.0, 0.01, 1000.0, 4180.0, 0.05, 0.05 * 0.05
    T_new, audit = inject_deck_boundary(T, mask, q, dt, rho, cp, dx, dA_cell=dA)
    assert audit["n_cells"] == 8
    # round-off of the 300 K offset in float64 limits the audit to ~1e-9
    assert audit["relative_error"] < 1e-8
    assert np.allclose(T_new[1:, :], 300.0)  # interior untouched
    expected_dT = q * dt / (rho * cp * dx)
    assert abs(T_new[0, 0] - (300.0 + expected_dT)) < 1e-12


def test_injection_fail_closed() -> None:
    T = np.full((4, 4), 300.0)
    with pytest.raises(ValueError):
        inject_deck_boundary(T, np.zeros_like(T, dtype=bool), 100.0, 0.1, 1000.0, 4180.0, 0.1)
    with pytest.raises(ValueError):
        inject_deck_boundary(T, np.zeros((2, 2), dtype=bool), 100.0, 0.1, 1000.0, 4180.0, 0.1)


# ---------------------------------------------------------------------------
# fail-closed behaviour of the boundary model
# ---------------------------------------------------------------------------


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        DeckPanel(area=0.0)
    with pytest.raises(ValueError):
        DeckPanel(emissivity=0.0)
    with pytest.raises(ValueError):
        DeckPanel(h_conv=-1.0)
    with pytest.raises(ValueError):
        DeckPanel(solar_absorptance=1.5)
    with pytest.raises(ValueError):
        SkySeaEnvironment(T_air=0.0)
    with pytest.raises(ValueError):
        closed_form_equilibrium_temperature(DeckPanel(h_conv=5.0), night_env())
    with pytest.raises(ValueError):
        deck_energy_balance(250.0, DeckPanel(), night_env(), longwave="nope")
    with pytest.raises(ValueError):
        solve_deck_temperature(DeckPanel(), night_env(), bracket=(400.0, 450.0))


def test_effective_surroundings_temperature_limits() -> None:
    assert abs(effective_surroundings_temperature(1.0, 280.0, 0.0, 300.0) - 280.0) < 1e-12
    assert abs(effective_surroundings_temperature(0.0, 280.0, 1.0, 300.0) - 300.0) < 1e-12

# ---------------------------------------------------------------------------
# cross-module consistency with the legacy thermal_radiation boundary
# ---------------------------------------------------------------------------


class TestLegacySkyFluxConsistency:
    """``thermal_radiation.compute_sky_radiation_flux`` vs the new closed form.

    Documents (and pins) the two defects of the legacy boundary: its
    ``projected_normal`` view factor under-estimates tilted surfaces, and it
    returns the absorbed irradiance rather than the net exchange unless a
    surface temperature is supplied.
    """

    def _normals(self):
        import torch

        return torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.70710678, 0.70710678], [0.0, 0.5, 0.86602540]],
            dtype=torch.float64,
        )

    def test_closed_form_view_factor_matches_family(self):
        import torch

        from tensorlbm.thermal_radiation import SkyRadiationSettings, compute_sky_radiation_flux

        sky = SkyRadiationSettings(enabled=True, T_ambient=300.0, T_dewpoint=285.0)
        normals = self._normals()
        q_cf = compute_sky_radiation_flux(normals, sky, emissivity=1.0, view_factor_model="closed_form")
        q_legacy = compute_sky_radiation_flux(normals, sky, emissivity=1.0)
        T_sky = sky.effective_sky_temperature()
        from tensorlbm.gray_enclosure_radiosity import STEFAN_BOLTZMANN

        for i, n_y in enumerate(normals[:, 1].tolist()):
            F_expected = 0.5 * (1.0 + n_y)
            assert abs(q_cf[i].item() - STEFAN_BOLTZMANN * T_sky**4 * F_expected) < 1e-12
        # legacy model reproduces clamp(n_y) exactly (no regression)
        for i, n_y in enumerate(normals[:, 1].tolist()):
            assert abs(q_legacy[i].item() - STEFAN_BOLTZMANN * T_sky**4 * max(0.0, n_y)) < 1e-12
        # the legacy approximation is worst at 45 deg
        rel = abs(q_legacy[1].item() - q_cf[1].item()) / q_cf[1].item()
        assert 0.15 < rel < 0.20

    def test_net_mode_matches_deck_longwave_analytic(self):
        import torch

        from tensorlbm.thermal_radiation import SkyRadiationSettings, compute_sky_radiation_flux

        sky = SkyRadiationSettings(enabled=True, T_ambient=300.0, T_dewpoint=285.0)
        normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        T_deck = 300.0
        q_net = compute_sky_radiation_flux(
            normals, sky, emissivity=0.9, surface_temperature=T_deck, view_factor_model="closed_form"
        )[0].item()
        env = SkySeaEnvironment(T_air=300.0, T_dewpoint=285.0, T_sea=300.0, sky_emissivity=1.0)
        env.T_sea = env.sky_temperature()  # remove the sea term: pure sky exchange
        q_deck = deck_longwave_analytic(DeckPanel(tilt_deg=0.0), env, T_deck)
        assert abs(q_net - q_deck) < 1e-9
