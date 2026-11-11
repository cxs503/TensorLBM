"""Independent verification of the flexural-gravity wave chain.

Erratum 2026-11-11 (TensorLBM / ice-covered wave resistance)
------------------------------------------------------------
Three closed forms in ``tensorlbm.ice_wave_resistance`` were not the stationary
points / derivatives they claimed to be, and the module's own
``ice_dispersion_verification`` compared the closed form *with itself*
(``max_relative_error`` was <= machine epsilon by construction, so it could
never fail):

1. ``ice_k_star``   returned ``(ρ_w g / (4 D))^{1/4}``; the stationary point of
   ``c(k) = sqrt((g/k + D k³/ρ_w)/(1 + m k))`` is ``(ρ_w g / (3 D))^{1/4}``.
2. ``ice_c_min``    returned ``(4 D g / ρ_w)^{1/4}`` — dimensionally
   ``m^{3/2} s^{-1}`` rather than ``m/s`` — and scaled like ``h^{3/4}``
   instead of ``h^{3/8}``; for a 0.3 m ice sheet it over-predicted the
   critical speed by 2.6× (26.39 vs 10.13 m/s).
3. ``ice_group_speed`` divided the mass-loading term by ``(1+mk)²`` instead
   of ``(1+mk)``; error 0 % at ``m = 0`` (why all ``m = 0`` checks passed).

These tests bind each quantity to an *independent* reference (brute-force
minimisation on a dense grid, a finite-difference derivative, and a
root-residual check of the dispersion solver used by the resistance
integral).  ``test_erroneous_c_min_formula_would_be_caught`` is the negative
control proving the suite has discriminating power.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.ice_wave_resistance import (
    G,
    RHO,
    flexural_rigidity,
    ice_c_min,
    ice_dispersion_k,
    ice_dispersion_verification,
    ice_group_speed,
    ice_k_star,
    ice_phase_speed,
    _ice_dispersion_residual,
)

E_ICE = 5.0e9
NU_ICE = 0.3
RHO_I = 917.0
# realistic ice sheets (thin fast-ice to thick multi-year ice)
THICKNESSES = (0.006, 0.03, 0.1, 0.3, 1.0)


def _brute_force_min(D: float, m: float, n: int = 200001) -> tuple[float, float]:
    """Independent reference: minimise c(k) on a fresh dense log grid."""
    ks = np.logspace(-5.0, 3.0, n)
    cs = np.array([ice_phase_speed(float(k), D, m) for k in ks])
    i = int(np.argmin(cs))
    return float(ks[i]), float(cs[i])


@pytest.mark.parametrize("h", THICKNESSES)
def test_c_min_and_k_star_match_independent_minimisation(h: float) -> None:
    D = flexural_rigidity(E_ICE, h, NU_ICE)
    m = RHO_I * h / RHO
    k_ref, c_ref = _brute_force_min(D, m)
    assert abs(ice_c_min(D, m) - c_ref) / c_ref < 5e-3, (
        f"c_min mismatch at h={h}: closed form {ice_c_min(D, m):.5f} "
        f"vs brute force {c_ref:.5f}"
    )
    assert abs(ice_k_star(D, m) - k_ref) / k_ref < 5e-3, (
        f"k* mismatch at h={h}: {ice_k_star(D, m):.6g} vs {k_ref:.6g}"
    )


def test_stationary_condition_holds_at_k_star() -> None:
    """dc²/dk must vanish at k* (central difference, independent of the solver)."""
    for h in (0.03, 0.3):
        D = flexural_rigidity(E_ICE, h, NU_ICE)
        m = RHO_I * h / RHO
        k = ice_k_star(D, m)
        h_rel = 1e-6
        c2 = lambda x: ice_phase_speed(x, D, m) ** 2
        slope = (c2(k * (1.0 + h_rel)) - c2(k * (1.0 - h_rel))) / (2.0 * k * h_rel)
        curvature_scale = c2(k) / k
        assert abs(slope) / curvature_scale < 1e-5, (
            f"dc²/dk not zero at k* (h={h}): {slope:.3e} vs scale {curvature_scale:.3e}"
        )


def test_c_min_scaling_law_thickness() -> None:
    """m = 0: c_min ∝ D^{1/8} ∝ h^{3/8} — an integral property no constant can fake."""
    hs = np.array([0.05, 0.1, 0.2, 0.4])
    cs = np.array([
        ice_c_min(flexural_rigidity(E_ICE, float(h), NU_ICE), 0.0) for h in hs
    ])
    exponent = float(np.polyfit(np.log(hs), np.log(cs), 1)[0])
    assert abs(exponent - 3.0 / 8.0) < 0.01, (
        f"c_min exponent in h is {exponent:.4f}, expected 0.375 (h^(3/8))"
    )


def test_mass_loading_lowers_critical_speed() -> None:
    D = flexural_rigidity(E_ICE, 0.3, NU_ICE)
    assert ice_c_min(D, 0.0) > ice_c_min(D, 0.05) > ice_c_min(D, 0.3)


def test_erroneous_c_min_formula_would_be_caught() -> None:
    """Negative control: the retired (4 D g/ρ)^{1/4} is 2.6× off at h = 0.3 m."""
    D = flexural_rigidity(E_ICE, 0.3, NU_ICE)
    _, c_ref = _brute_force_min(D, RHO_I * 0.3 / RHO)
    retired = (4.0 * D * G / RHO) ** 0.25
    assert abs(retired - c_ref) / c_ref > 1.0, (
        "negative control failed: retired formula should differ from the true "
        "minimum by >100 %"
    )
    assert abs(ice_c_min(D, RHO_I * 0.3 / RHO) - c_ref) / c_ref < 5e-3


@pytest.mark.parametrize("m", [0.0, 0.0054, 0.0895, 0.2684])
def test_group_speed_matches_finite_difference(m: float) -> None:
    """c_g must equal dω/dk (the m-loading term was wrong for m > 0)."""
    D = flexural_rigidity(E_ICE, 0.3, NU_ICE)

    def omega(k: float) -> float:
        return math.sqrt((G * k + D * k**5 / RHO) / (1.0 + m * k))

    for k in (0.05, 0.13, 0.3):
        dk = 1e-6 * k
        cg_fd = (omega(k + dk) - omega(k - dk)) / (2.0 * dk)
        cg = ice_group_speed(k, D, m)
        assert abs(cg - cg_fd) / cg_fd < 1e-5, (
            f"c_g mismatch at m={m}, k={k}: {cg:.6f} vs FD {cg_fd:.6f}"
        )


@pytest.mark.parametrize("h", [0.006, 0.3])
def test_dispersion_solver_returns_genuine_root(h: float) -> None:
    """The solver used by the resistance integral must produce F(k)=0."""
    D = flexural_rigidity(E_ICE, h, NU_ICE)
    m = RHO_I * h / RHO
    k_star = ice_k_star(D, m)
    rng = np.random.default_rng(20261111)
    for _ in range(25):
        k = float(np.exp(rng.uniform(np.log(0.05 * k_star), np.log(3.0 * k_star))))
        theta = float(rng.uniform(0.0, 1.2))
        U = ice_phase_speed(k, D, m) / math.cos(theta)  # steady wave at (k, theta)
        k_solved = ice_dispersion_k(U, theta, D, m)
        if k_solved <= 0.0:
            continue  # subcritical / no steady wave
        scale = (U**2 * k_solved**2 * math.cos(theta) ** 2 * (1.0 + m * k_solved)
                 + G * k_solved + D * k_solved**5 / RHO)
        resid = abs(_ice_dispersion_residual(k_solved, U, theta, D, m))
        assert resid / scale < 1e-8, f"root residual {resid / scale:.2e} at h={h}"


def test_subcritical_speed_has_no_steady_wave() -> None:
    D = flexural_rigidity(E_ICE, 0.3, NU_ICE)
    m = RHO_I * 0.3 / RHO
    assert ice_dispersion_k(0.5 * ice_c_min(D, m), 0.0, D, m) == 0.0


def test_tangential_double_root_returns_k_star() -> None:
    """At U = c_min the two roots coalesce; the solver must not return 0."""
    D = flexural_rigidity(E_ICE, 0.3, NU_ICE)
    m = RHO_I * 0.3 / RHO
    k_star = ice_k_star(D, m)
    k = ice_dispersion_k(ice_c_min(D, m) * (1.0 + 1e-12), 0.0, D, m)
    assert k > 0.0 and abs(k - k_star) / k_star < 1e-3


def test_open_water_limit_of_solver() -> None:
    """D → 0, m → 0 recovers k = k0 sec²θ = g/(U²cos²θ)."""
    U = 6.0
    for theta in (0.0, 0.3, 0.6, 1.0):
        k = ice_dispersion_k(U, theta, 0.0, 0.0)
        k_exact = G / (U**2 * math.cos(theta) ** 2)
        assert abs(k - k_exact) / k_exact < 1e-12


@pytest.mark.parametrize("h", [0.006, 0.3, 1.0])
def test_dispersion_verification_bundle_passes(h: float) -> None:
    """The (rewritten, non-tautological) verification bundle must pass its gates."""
    D = flexural_rigidity(E_ICE, h, NU_ICE)
    m = RHO_I * h / RHO
    rec = ice_dispersion_verification(D, m, n_k=200)
    assert rec["max_relative_error"] < 1e-6, rec["max_relative_error"]
    assert rec["max_dispersion_residual"] < 1e-10, rec["max_dispersion_residual"]
    assert rec["max_relative_cg_error"] < 1e-5, rec["max_relative_cg_error"]
    assert rec["c_min_rel_error"] < 5e-3
    assert rec["pass_2pct"] and rec["pass_cg_2pct"]