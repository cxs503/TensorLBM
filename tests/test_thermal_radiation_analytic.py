"""Analytic verification of the grey-diffuse radiation (radiosity) module.

Every case here has a closed-form reference, so the assertions are quantitative
rather than qualitative.  The module previously carried only monotonicity /
"hotter surface loses heat" style checks (``app/tests/test_engineering_gap_closure.py``),
which cannot fail for a wrong-but-monotone implementation.

References
----------
* Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed., §5.4 (net-radiation
  / radiosity method) and §5.6 (enclosure with partially reflecting surfaces).
* Incropera, DeWitt, Bergman & Lavine (2007). *Fundamentals of Heat and Mass
  Transfer*, 6th ed., §13.3: Eq. 13.23 (two infinite parallel plates) and
  Eq. 13.24 (concentric spheres / infinite cylinders with area ratio).

Verification gates (each printed with its measured error in the failure message)
---------------
* key flux quantities vs closed form: ``< 3 %`` relative error (actually
  ~1e-13 relative, i.e. float64 round-off);
* closed-enclosure radiative energy balance ``Σ q_i A_i = −Σ A_i q_solar,i``:
  ``< 1e-9`` relative;
* view-factor reciprocity / closure violations: must raise ``ValueError``
  (fail-closed) instead of returning a non-conservative flux.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.thermal_radiation import (
    STEFAN_BOLTZMANN,
    RadiationEnclosureConfig,
    SolarSettings,
    SurfaceRadiationProps,
    compute_net_radiation_flux,
    radiosity_matrix_solve,
    radiative_energy_residual,
    solar_flux_on_surface,
    validate_view_factors,
)

SIGMA = STEFAN_BOLTZMANN
GATE_REL = 0.03  # 3 % acceptance gate for key flux quantities


# ---------------------------------------------------------------------------
# Closed-form reference solutions
# ---------------------------------------------------------------------------


def q_two_parallel_plates(T1: float, T2: float, eps1: float, eps2: float) -> float:
    """Net flux leaving plate 1 [W/m²] — Incropera Eq. 13.23.

    ``q = σ (T1⁴ − T2⁴) / (1/ε1 + 1/ε2 − 1)``  (F12 = F21 = 1, infinite plates).
    """
    return SIGMA * (T1**4 - T2**4) / (1.0 / eps1 + 1.0 / eps2 - 1.0)


def q_concentric_spheres(
    T1: float, T2: float, eps1: float, eps2: float, A1: float, A2: float
) -> float:
    """Net flux leaving the inner surface — Incropera Eq. 13.24.

    ``q = σ (T1⁴ − T2⁴) / (1/ε1 + (A1/A2)(1/ε2 − 1))``
    with F12 = 1, F22 = 1 − A1/A2, F21 = A1/A2.
    """
    return SIGMA * (T1**4 - T2**4) / (1.0 / eps1 + (A1 / A2) * (1.0 / eps2 - 1.0))


def _surface(T: float, eps: float, area: float = 1.0, alpha_sol: float = 0.5):
    return SurfaceRadiationProps(
        temperature=T, emissivity=eps, area=area, solar_absorptance=alpha_sol
    )


def _rel_err(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


# ---------------------------------------------------------------------------
# A. Two infinite parallel plates (Incropera 13.23)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T1,T2,eps1,eps2",
    [
        (400.0, 300.0, 0.9, 0.85),
        (800.0, 300.0, 0.9, 0.9),
        (500.0, 450.0, 0.3, 0.7),
        (400.0, 300.0, 0.2, 0.2),
    ],
)
def test_two_plate_analytic_moderate_emissivity(T1, T2, eps1, eps2):
    cfg = RadiationEnclosureConfig(surfaces=[_surface(T1, eps1), _surface(T2, eps2)])
    result = compute_net_radiation_flux(cfg)
    ref = q_two_parallel_plates(T1, T2, eps1, eps2)
    err = _rel_err(result.net_flux[0], ref)
    assert err < GATE_REL, (
        f"two-plate net flux {result.net_flux[0]:.6e} W/m² vs analytic {ref:.6e} W/m² "
        f"→ {err * 100:.4g} % (gate {GATE_REL * 100:g} %)"
    )
    # Antisymmetry: q2 = −q1 for equal areas.
    assert _rel_err(result.net_flux[1], -ref) < 1e-12


@pytest.mark.parametrize("eps", [0.02, 0.05, 0.1, 0.2])
def test_two_plate_analytic_low_emissivity_regression(eps):
    """Regression gate for the under-converged fixed-point radiosity solve.

    The former ``radiosity_matrix_solve`` used an undamped fixed point with
    ``max_iter=50, tol=1e-6``.  Its spectral radius is ``(1 − ε)‖F‖ = 1 − ε``,
    so for ε = 0.05 the residual after 50 sweeps is ``0.95**50 ≈ 7.7 %``: the
    old code returned +7.31 % high on this exact case while reporting success.
    The 3 % gate below fails on that code and passes on the exact linear solve.
    """
    T1, T2 = 400.0, 300.0
    cfg = RadiationEnclosureConfig(surfaces=[_surface(T1, eps), _surface(T2, eps)])
    result = compute_net_radiation_flux(cfg)
    ref = q_two_parallel_plates(T1, T2, eps, eps)
    err = _rel_err(result.net_flux[0], ref)
    assert err < GATE_REL, (
        f"ε={eps}: net flux {result.net_flux[0]:.6e} vs analytic {ref:.6e} W/m² "
        f"→ {err * 100:.4g} % (gate {GATE_REL * 100:g} %)"
    )


def test_blackbody_limit_two_plates():
    """ε1 = ε2 = 1 must give exactly σ(T1⁴ − T2⁴), independent of F row sums."""
    T1, T2 = 600.0, 300.0
    cfg = RadiationEnclosureConfig(surfaces=[_surface(T1, 1.0), _surface(T2, 1.0)])
    result = compute_net_radiation_flux(cfg)
    ref = SIGMA * (T1**4 - T2**4)
    assert _rel_err(result.net_flux[0], ref) < 1e-12
    # A blackbody has zero irradiation from a blackbody-only enclosure pair and
    # radiates εσT⁴ = σT⁴.
    assert result.radiosity[0] == pytest.approx(SIGMA * T1**4, rel=1e-12)


# ---------------------------------------------------------------------------
# B. Concentric spheres with an area ratio (Incropera 13.24)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T1,T2,eps1,eps2,A1,A2",
    [
        (500.0, 300.0, 0.9, 0.5, 1.0, 4.0),
        (500.0, 300.0, 0.05, 0.9, 1.0, 4.0),
        (400.0, 350.0, 0.6, 0.2, 0.25, 9.0),
        (500.0, 300.0, 1.0, 1.0, 1.0, 2.5),
    ],
)
def test_concentric_spheres_area_ratio_analytic(T1, T2, eps1, eps2, A1, A2):
    f12, f21, f22 = 1.0, A1 / A2, 1.0 - A1 / A2
    view_factors = [[0.0, f12], [f21, f22]]
    cfg = RadiationEnclosureConfig(
        surfaces=[_surface(T1, eps1, A1), _surface(T2, eps2, A2)],
        view_factors=view_factors,
    )
    result = compute_net_radiation_flux(cfg)
    ref = q_concentric_spheres(T1, T2, eps1, eps2, A1, A2)
    err = _rel_err(result.net_flux[0], ref)
    assert err < GATE_REL, (
        f"inner-surface flux {result.net_flux[0]:.6e} vs analytic {ref:.6e} W/m² "
        f"→ {err * 100:.4g} % (gate {GATE_REL * 100:g} %)"
    )
    # Reciprocity of the exchange: A1 F12 = A2 F21.
    exchange = radiative_energy_residual(result.net_flux, [A1, A2])
    assert abs(exchange) < 1e-9 * abs(result.net_flux[0] * A1)


def test_self_viewing_surface_uses_supplied_matrix():
    """F22 = 1 − A1/A2 must be honoured (a concave outer surface sees itself)."""
    view_factors = [[0.0, 1.0], [0.25, 0.75]]
    diag = validate_view_factors(
        torch.tensor(view_factors, dtype=torch.float64), [1.0, 4.0], tol=1e-9
    )
    assert diag["closed"] is True
    assert diag["max_reciprocity_dev"] < 1e-12
    assert diag["max_row_sum_dev"] < 1e-12


# ---------------------------------------------------------------------------
# C. Energy conservation of closed enclosures
# ---------------------------------------------------------------------------


def test_closed_enclosure_energy_balance_no_solar():
    """Three equal-area surfaces: Σ q_i A_i must vanish (exactly, float64)."""
    cfg = RadiationEnclosureConfig(
        surfaces=[_surface(500.0, 0.8), _surface(300.0, 0.6), _surface(350.0, 0.95)]
    )
    result = compute_net_radiation_flux(cfg)
    scale = sum(abs(q) * 1.0 for q in result.net_flux)
    residual = radiative_energy_residual(result.net_flux, [1.0, 1.0, 1.0])
    assert abs(residual) < 1e-9 * scale, f"energy residual {residual:.3e} W of {scale:.3e} W"
    # Physically: the 500 K surface must lose power, the 300 K surface gain it.
    assert result.net_flux[0] > 0.0 > result.net_flux[1]


def test_closed_enclosure_energy_balance_with_solar():
    """Σ q_i A_i = −Σ A_i q_solar,i: absorbed solar power leaves radiatively."""
    solar = SolarSettings(enabled=True, irradiance=1000.0, direction=(0.0, -1.0, 0.0))
    cfg = RadiationEnclosureConfig(
        surfaces=[_surface(300.0, 0.9, 1.0, 0.6), _surface(300.0, 0.9, 1.0, 0.6)],
        solar=solar,
    )
    normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)
    result = compute_net_radiation_flux(cfg, surface_normals=normals)
    # Only the sun-facing surface is irradiated: α I cos0 = 600 W/m².
    assert result.solar_flux[0] == pytest.approx(600.0, rel=1e-12)
    assert result.solar_flux[1] == pytest.approx(0.0, abs=1e-12)
    absorbed = sum(a * q for a, q in zip([1.0, 1.0], result.solar_flux))
    residual = radiative_energy_residual(result.net_flux, [1.0, 1.0])
    assert abs(residual + absorbed) < 1e-9 * absorbed, (
        f"Σ q A = {residual:.6e} W vs −absorbed solar {absorbed:.6e} W"
    )


def test_single_surface_radiates_to_cold_void():
    """One surface, F = 0 (open enclosure): q = εσT⁴ to a 0 K background."""
    cfg = RadiationEnclosureConfig(surfaces=[_surface(400.0, 0.3, 2.0)])
    result = compute_net_radiation_flux(cfg)
    ref = 0.3 * SIGMA * 400.0**4
    assert result.net_flux[0] == pytest.approx(ref, rel=1e-12)


# ---------------------------------------------------------------------------
# D. Solar flux model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, 85.0, 90.0])
def test_solar_cosine_law(angle_deg):
    """q = α I cosθ for a sun-facing tilt θ (Lambert cosine law)."""
    solar = SolarSettings(enabled=True, irradiance=1000.0, direction=(0.0, -1.0, 0.0))
    a = math.radians(angle_deg)
    normal = torch.tensor([[math.sin(a), math.cos(a), 0.0]], dtype=torch.float64)
    q = solar_flux_on_surface(normal, solar, absorptance=0.8)
    assert q[0].item() == pytest.approx(0.8 * 1000.0 * math.cos(a), rel=1e-12, abs=1e-9)


def test_solar_back_face_and_absorptance_scaling():
    solar = SolarSettings(enabled=True, irradiance=800.0, direction=(0.0, -1.0, 0.0))
    normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)
    alpha = torch.tensor([1.0, 0.25], dtype=torch.float64)
    q = solar_flux_on_surface(normals, solar, absorptance=alpha)
    assert q[0].item() == pytest.approx(800.0, rel=1e-12)
    assert q[1].item() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# E. Fail-closed view-factor validation
# ---------------------------------------------------------------------------


def test_unequal_area_default_view_factors_fail_closed():
    """The equal-view-factor default is only reciprocity-consistent for equal areas.

    Measured pre-fix behaviour: this three-surface enclosure (areas 1/2/0.5)
    returned Σ q_i A_i = −338.6 W against 2.3 kW of exchange — a 17 % energy
    imbalance reported as a valid result.  It must now raise.
    """
    cfg = RadiationEnclosureConfig(
        surfaces=[_surface(500.0, 0.8, 1.0), _surface(300.0, 0.6, 2.0), _surface(350.0, 0.95, 0.5)]
    )
    with pytest.raises(ValueError, match="reciprocity"):
        compute_net_radiation_flux(cfg)


def test_user_view_factors_reciprocity_violation_fail_closed():
    cfg = RadiationEnclosureConfig(
        surfaces=[_surface(400.0, 0.9, 1.0), _surface(300.0, 0.9, 4.0)],
        view_factors=[[0.0, 1.0], [1.0, 0.0]],  # F21 must be A1/A2 = 0.25
    )
    with pytest.raises(ValueError, match="reciprocity"):
        compute_net_radiation_flux(cfg)


def test_row_sum_violation_is_reported_as_open_enclosure():
    """A row sum < 1 is a legitimate open enclosure: accepted, flagged not closed."""
    F = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=torch.float64)
    diag = validate_view_factors(F, [1.0, 1.0], tol=1e-9)
    assert diag["closed"] is False
    assert diag["max_row_sum_dev"] == pytest.approx(0.5, rel=1e-12)
    # The fraction (1 − Σ_j F_ij) of each radiosity escapes to the unmodelled
    # 0 K environment, so the enclosure is no longer closed:
    #   Σ_i A_i q_i = +Σ_i A_i J_i (1 − Σ_j F_ij).
    open_cfg = RadiationEnclosureConfig(
        surfaces=[_surface(400.0, 0.9), _surface(300.0, 0.9)],
        view_factors=F.tolist(),
    )
    open_res = compute_net_radiation_flux(open_cfg)
    escape = sum(
        a * J * (1.0 - float(F[i].sum()))
        for i, (a, J) in enumerate(zip([1.0, 1.0], open_res.radiosity))
    )
    residual = radiative_energy_residual(open_res.net_flux, [1.0, 1.0])
    assert residual == pytest.approx(escape, rel=1e-12), (
        f"open-enclosure balance Σ q A = {residual:.6e} W vs escape {escape:.6e} W"
    )
    assert escape > 0.0


def test_negative_view_factor_is_rejected():
    F = torch.tensor([[0.0, 1.5], [-0.5, 0.5]], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-negative"):
        validate_view_factors(F, [1.0, 1.0])


# ---------------------------------------------------------------------------
# F. Solver-consistency and gate-power (negative control) checks
# ---------------------------------------------------------------------------


def test_direct_solve_matches_converged_fixed_point():
    """The exact linear solve must equal the fixed point iterated to convergence."""
    surfaces = [_surface(400.0, 0.9), _surface(300.0, 0.85)]
    J_ax, G_ax, q_ax = radiosity_matrix_solve(surfaces, max_iter=50, tol=1e-6)
    J_conv, G_conv, q_conv = radiosity_matrix_solve(surfaces, max_iter=200000, tol=1e-14)
    assert J_ax == pytest.approx(J_conv, rel=1e-10)
    assert q_ax == pytest.approx(q_conv, rel=1e-10)


def test_gate_power_series_resistance_formula_is_rejected():
    """Negative control: the naive "series resistance" formula must fail the gate.

    A plausible-looking wrong closure ``q = σΔT⁴/(1/ε1 + 1/ε2)`` (reflections
    not counted once, i.e. the 1/ε1 + 1/ε2 − 1 denominator miscopied) is
    off by ~34 % here, so a 3 % gate genuinely discriminates.
    """
    T1, T2, eps1, eps2 = 400.0, 300.0, 0.9, 0.85
    correct = q_two_parallel_plates(T1, T2, eps1, eps2)
    wrong = SIGMA * (T1**4 - T2**4) / (1.0 / eps1 + 1.0 / eps2)
    err = _rel_err(wrong, correct)
    assert err > GATE_REL, f"negative control only {err * 100:.2f} % off; gate has no teeth"
    # And the implementation tracks the correct closure, not the wrong one.
    result = compute_net_radiation_flux(
        RadiationEnclosureConfig(surfaces=[_surface(T1, eps1), _surface(T2, eps2)])
    )
    assert _rel_err(result.net_flux[0], correct) < _rel_err(result.net_flux[0], wrong)


def test_total_emitted_power_is_blackbody_equivalent_definition():
    """Pin the *current* semantics of ``total_emitted_power``.

    The field sums ``σ T_i⁴ A_i`` (blackbody-equivalent emission), **not** the
    grey emission ``ε σ T_i⁴ A_i``; for ε = 0.05 the two differ by 20×.  The
    value is pinned here and the label discrepancy is documented in
    ``docs/thermal_radiation_radiosity_verification_r1.md`` rather than silently
    redefined (existing API consumers read this field).
    """
    cfg = RadiationEnclosureConfig(
        surfaces=[_surface(400.0, 0.05, 1.0), _surface(300.0, 0.05, 1.0)]
    )
    result = compute_net_radiation_flux(cfg)
    blackbody = SIGMA * (400.0**4 + 300.0**4)
    grey = 0.05 * blackbody
    assert result.total_emitted_power == pytest.approx(blackbody, rel=1e-12)
    assert result.total_emitted_power / grey == pytest.approx(20.0, rel=1e-12)
    # Net absorbed equals net power lost by the hot surface (no solar).
    assert result.total_absorbed_power == pytest.approx(-result.net_flux[1], rel=1e-12)
    assert result.total_absorbed_power == pytest.approx(result.net_flux[0], rel=1e-12)