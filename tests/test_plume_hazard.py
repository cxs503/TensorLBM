"""Acceptance tests for the plume / gas-cloud hazard models (IN70-N03).

Criteria covered:

* buoyant plume self-similarity ``Q(z) ~ z^{5/3}`` (fitted exponent in
  ``[1.6, 1.75]``) **and** the analytic prefactor
  ``(3 pi^2 c_3^4 B / 4)^{1/3}`` with ``c_3 = 6 alpha/5``;
* the discriminating negative control — momentum jet (``g' = 0``): exponent in
  ``[0.95, 1.05]`` and prefactor ``2 alpha sqrt(pi M)``;
* the falsifiable prediction that the plume prefactor scales as ``alpha^{4/3}``
  while the exponent does not move;
* LFL isopleth metrics: monotone decrease over five wind speeds and grid
  convergence of the extracted hazard distance ``<= 3 %``;
* Briggs buoyant plume rise ``dh ~ F^{1/3} x^{2/3} / U``.
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.plume_hazard import (
    SPECIES,
    GasSpecies,
    IntegralPlumeConfig,
    fit_power_law,
    gaussian_plume_concentration,
    hazard_wind_sweep,
    integrate_plume,
    jet_self_similar,
    lfl_contour_on_grid,
    lfl_hazard_distance,
    plume_rise_briggs,
    plume_self_similar,
    sigma_y_briggs,
    sigma_z_briggs,
)

ALPHA = 0.05
Z_START, Z_END, N_STEPS = 1.0, 200.0, 4000


def _plume_run(alpha: float = ALPHA, buoyancy: float = 1.0) -> dict[str, list[float]]:
    cfg = IntegralPlumeConfig(
        entrainment=alpha, buoyancy_flux=buoyancy, z_start=Z_START, z_end=Z_END, n_steps=N_STEPS
    )
    ss = plume_self_similar(cfg, Z_START)
    seeded = IntegralPlumeConfig(
        entrainment=alpha,
        buoyancy_flux=buoyancy,
        z_start=Z_START,
        z_end=Z_END,
        n_steps=N_STEPS,
        volume_flux=ss["Q"],
        momentum_flux=ss["M"],
    )
    return integrate_plume(seeded)


def _jet_run(alpha: float = ALPHA) -> dict[str, list[float]]:
    cfg = IntegralPlumeConfig(
        entrainment=alpha, buoyancy_flux=0.0, z_start=Z_START, z_end=Z_END, n_steps=N_STEPS
    )
    ss = jet_self_similar(cfg, Z_START)
    seeded = IntegralPlumeConfig(
        entrainment=alpha,
        buoyancy_flux=0.0,
        z_start=Z_START,
        z_end=Z_END,
        n_steps=N_STEPS,
        volume_flux=ss["Q"],
        momentum_flux=ss["M"],
    )
    return integrate_plume(seeded)


# ---------------------------------------------------------------------------
# integral plume / jet self-similarity
# ---------------------------------------------------------------------------


def test_plume_power_law_exponent():
    out = _plume_run()
    fit = fit_power_law(out["z"], out["Q"], window=(10.0, 150.0))
    assert 1.60 <= fit["exponent"] <= 1.75, fit
    assert fit["exponent"] == pytest.approx(5.0 / 3.0, abs=1e-4)
    assert fit["rms"] < 1e-6  # the numerical solution really is a pure power law
    w_fit = fit_power_law(out["z"], out["w"], window=(10.0, 150.0))
    assert w_fit["exponent"] == pytest.approx(-1.0 / 3.0, abs=1e-4)


def test_plume_analytic_prefactor():
    """Numeric solution reproduces the analytic self-similar constant."""
    out = _plume_run()
    alpha = ALPHA
    c3 = 6.0 * alpha / 5.0
    expected = (3.0 * math.pi**2 * c3**4 * 1.0 / 4.0) ** (1.0 / 3.0)
    fit = fit_power_law(out["z"], out["Q"], window=(10.0, 150.0))
    assert fit["coefficient"] == pytest.approx(expected, rel=1e-6)
    # and the top-hat width grows with slope 6 alpha / 5
    b_fit = fit_power_law(out["z"], out["b"], window=(10.0, 150.0))
    assert b_fit["coefficient"] == pytest.approx(c3, rel=1e-6)


def test_jet_negative_control():
    """g' = 0 momentum jet: exponent 1 (settles into [0.95, 1.05])."""
    out = _jet_run()
    fit = fit_power_law(out["z"], out["Q"], window=(10.0, 150.0))
    assert 0.95 <= fit["exponent"] <= 1.05, fit
    assert fit["coefficient"] == pytest.approx(2.0 * ALPHA * math.sqrt(math.pi), rel=1e-6)
    # momentum flux is conserved in a pure jet
    assert max(out["M"]) == pytest.approx(min(out["M"]), rel=1e-12)


def test_jet_negative_control_forced_start():
    """A forced (jet-like) start still locks onto the same z^-1 decay far away."""
    cfg = IntegralPlumeConfig(
        entrainment=ALPHA, buoyancy_flux=0.0, volume_flux=1.0, momentum_flux=1.0,
        z_start=1.0, z_end=500.0, n_steps=20000,
    )
    out = integrate_plume(cfg)
    fit = fit_power_law(out["z"], out["Q"], window=(300.0, 499.0))
    assert 0.95 <= fit["exponent"] <= 1.05, fit


def test_alpha_scaling_of_prefactor():
    """Prefactor ~ alpha^{4/3}, exponent unchanged (discriminating prediction)."""
    coefs = []
    for alpha in (0.04, 0.05, 0.08, 0.12):
        out = _plume_run(alpha=alpha)
        fit = fit_power_law(out["z"], out["Q"], window=(10.0, 150.0))
        assert fit["exponent"] == pytest.approx(5.0 / 3.0, abs=1e-4)
        coefs.append((alpha, fit["coefficient"]))
    slope = fit_power_law([a for a, _ in coefs], [c for _, c in coefs], window=None)["exponent"]
    assert slope == pytest.approx(4.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# species / dispersion correlations
# ---------------------------------------------------------------------------


def test_species_lfl_conversion():
    ch4 = SPECIES["CH4"]
    assert ch4.lfl_mass_fraction == pytest.approx(0.0282, abs=5e-4)
    assert SPECIES["H2"].lfl_mass_fraction < ch4.lfl_mass_fraction
    assert SPECIES["NH3"].lfl_mass_fraction > ch4.lfl_mass_fraction
    # mass-based threshold equals volume fraction times the density ratio
    rho = 1.2
    assert ch4.lfl_mass_concentration(rho) == pytest.approx(
        ch4.lfl_volume_fraction * ch4.molar_mass / 28.97 * rho, rel=3e-2
    )


def test_briggs_sigma_monotone():
    xs = [1.0, 10.0, 100.0, 1000.0]
    for stab in ("A", "B", "C", "D", "E", "F"):
        sy = [sigma_y_briggs(x, stab) for x in xs]
        sz = [sigma_z_briggs(x, stab) for x in xs]
        assert all(sy[i] < sy[i + 1] for i in range(len(xs) - 1))
        assert all(sz[i] < sz[i + 1] for i in range(len(xs) - 1))
    # unstable spreads more than stable
    assert sigma_z_briggs(500.0, "A") > sigma_z_briggs(500.0, "F")


def test_gaussian_plume_decays_downwind():
    prev = math.inf
    for x in (5.0, 10.0, 20.0, 40.0, 80.0):
        c = gaussian_plume_concentration(x, 0.0, 0.0, 5.0, 3.0, 0.5, "D")
        assert c < prev
        prev = c


def test_briggs_plume_rise_scaling():
    f, u = 10.0, 3.0
    r1 = plume_rise_briggs(f, u, 10.0)
    r2 = plume_rise_briggs(f, u, 10.0 * 4.0)
    assert r2 / r1 == pytest.approx(4.0 ** (2.0 / 3.0), rel=1e-12)
    r3 = plume_rise_briggs(f, 2.0 * u, 10.0)
    assert r3 / r1 == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# LFL hazard envelope
# ---------------------------------------------------------------------------


def test_lfl_hazard_distance_bracket():
    gas = SPECIES["CH4"]
    res = lfl_hazard_distance(gas, source_rate=5.0, wind=3.0, stability="D", stack_height=0.5)
    assert res["distance"] > 1.0
    thr = res["threshold"]
    c0 = gaussian_plume_concentration(
        res["distance"], 0.0, 0.0, 5.0, 3.0, 0.5, "D"
    )
    assert c0 == pytest.approx(thr, rel=1e-3)
    # stable conditions spread less -> shorter ground-level hazard
    stable = lfl_hazard_distance(gas, 5.0, 3.0, stability="F", stack_height=0.5)
    assert stable["distance"] > res["distance"]


def test_hazard_wind_sweep_monotone_five_levels():
    winds = [2.0, 3.0, 5.0, 8.0, 12.0]
    for species, rate in (("CH4", 5.0), ("H2", 5.0), ("NH3", 20.0)):
        sweep = hazard_wind_sweep(
            species, rate, winds, stabilities=("D", "F"), stack_height=0.5
        )
        assert sweep["monotone_decreasing"], (species, sweep["distances"])
        assert len(sweep["distances"]) == 5
        assert all(d > 0.0 for d in sweep["distances"])
        assert sweep["distances"][0] > 2.0 * sweep["distances"][-1]  # 6x wind speed


def test_lfl_contour_grid_convergence():
    """Extracted hazard distance converges to <= 3 % under grid refinement."""
    gas = SPECIES["CH4"]
    ref = lfl_hazard_distance(gas, 5.0, 3.0, stability="D", stack_height=0.5)["distance"]
    prev = None
    for dx in (8.0, 4.0, 2.0, 1.0):
        got = lfl_contour_on_grid(gas, 5.0, 3.0, stability="D", stack_height=0.5, dx=dx)
        assert abs(got["distance"] / ref - 1.0) <= 0.03, (dx, got, ref)
        if prev is not None:
            assert abs(got["distance"] - prev) / prev <= 0.03
        prev = got["distance"]


def test_hazard_species_ordering():
    """The threshold ordering sets the hazard ordering at equal release rate."""
    res = {
        sp: lfl_hazard_distance(SPECIES[sp], 5.0, 3.0, stability="D", stack_height=0.5)[
            "distance"
        ]
        for sp in ("H2", "CH4", "NH3")
    }
    assert res["H2"] > res["CH4"] > res["NH3"] > 0.0