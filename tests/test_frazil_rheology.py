"""Tests for the frazil-slurry rheology surface (IN11-H2).

Covers

* the Krieger-Dougherty effective-medium curve and its packed-limit guard,
* the concentration-dependent power-law index and its shear-thinning guard,
* the supercooling growth-rate window (nucleation threshold + regime flag),
* honest truncation of the concentration-viscosity curve at ``phi >= 0.6``,
* the exact equivalence of the fast shear-rate stencil with
  :mod:`tensorlbm.powerlaw`,
* 2-D channel verification of the power-law LBM against the exact
  Poiseuille / Couette solutions (the < 2% acceptance criterion),
* same-seed reproducibility and the closure hand-off for a frazil slurry,
* the committed artifact's acceptance block (when it is present).

The negative cases (rejected indices ``n <= 0`` / ``n > 1``, refused
extrapolation above the packed limit) are marked in the test names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.frazil_lbm import (  # noqa: E402
    _central_difference_fast,
    _shear_rate_fast,
    couette_analytic_profile,
    poiseuille_analytic_profile,
    run_frazil_poiseuille,
    run_powerlaw_couette,
    run_powerlaw_poiseuille,
)
from tensorlbm.frazil_rheology import (  # noqa: E402
    DEFAULT_PHI_MAX,
    FrazilRheologyError,
    concentration_viscosity_curve,
    frazil_apparent_viscosity,
    frazil_consistency_index,
    frazil_flow_index,
    growth_rate_window,
    relative_viscosity_krieger_dougherty,
    validate_flow_index,
    viscosity_jump_window,
)
from tensorlbm.powerlaw import central_difference, strain_rate_shear_rate_2d  # noqa: E402

U_REF = 0.03
NU_REF = 0.2


def _setup(kind: str, H: int, n: float) -> dict:
    h_half = 0.5 * float(H)
    if kind == "poiseuille":
        gamma_wall = (n + 1.0) * U_REF / (n * h_half)
        return {
            "K": NU_REF * gamma_wall ** (1.0 - n),
            "ax": NU_REF * gamma_wall / h_half,
            "gamma_wall": gamma_wall,
        }
    gamma_wall = U_REF / float(H)
    return {"K": NU_REF * gamma_wall ** (1.0 - n), "ax": 0.0, "gamma_wall": gamma_wall}


# ---------------------------------------------------------------------------
# 1. Krieger-Dougherty effective-medium viscosity
# ---------------------------------------------------------------------------


def test_krieger_dougherty_curve_limits_and_monotonicity():
    assert float(relative_viscosity_krieger_dougherty(0.0)) == pytest.approx(1.0)
    phi = torch.linspace(0.0, 0.58, 30, dtype=torch.float64)
    mu = relative_viscosity_krieger_dougherty(phi)
    assert torch.all(mu[1:] > mu[:-1]), "relative viscosity must increase with concentration"
    # closed form check at a single concentration: (1 - 0.3/0.6)^(-2.5*0.6)
    assert float(relative_viscosity_krieger_dougherty(0.30)) == pytest.approx(0.5**-1.5, rel=1e-12)
    # the viscosity jump window: O(10) at phi = 0.5, O(10^2) just below the packed limit
    assert 10.0 < float(relative_viscosity_krieger_dougherty(0.50)) < 20.0
    assert float(relative_viscosity_krieger_dougherty(0.595)) > 200.0
    jump = viscosity_jump_window()
    assert jump["phi_upper"] == pytest.approx(DEFAULT_PHI_MAX)
    assert jump["mu_ratio"] > 5.0
    assert jump["dln_mu_dphi_at_centre"] > 0.0


def test_krieger_dougherty_rejects_packed_limit_and_negative_phi():
    """NEGATIVE: extrapolation at/above the packed limit and phi < 0 must raise."""
    for bad in (DEFAULT_PHI_MAX, 0.65, 0.9):
        with pytest.raises(FrazilRheologyError):
            relative_viscosity_krieger_dougherty(bad)
    with pytest.raises(FrazilRheologyError):
        relative_viscosity_krieger_dougherty(-0.01)
    with pytest.raises(FrazilRheologyError):
        relative_viscosity_krieger_dougherty(torch.tensor([0.2, 0.7]))


# ---------------------------------------------------------------------------
# 2. Power-law index
# ---------------------------------------------------------------------------


def test_flow_index_decreases_with_concentration_and_stays_thinning():
    assert float(frazil_flow_index(0.0)) == pytest.approx(1.0)
    phis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.59]
    n_vals = [float(frazil_flow_index(p)) for p in phis]
    assert all(b < a for a, b in zip(n_vals, n_vals[1:])), "n must decrease with phi"
    assert all(0.0 < n <= 1.0 for n in n_vals)
    assert 0.5 < float(frazil_flow_index(0.30)) < 0.9  # Daly (1984) range for phi = 0.1-0.4
    assert float(frazil_flow_index(0.59)) == pytest.approx(0.46, abs=0.01)
    assert float(frazil_flow_index(0.59)) >= 0.45  # n_packed floor, never below


def test_flow_index_guard_rejects_non_shear_thinning_indices():
    """NEGATIVE: n <= 0 (singular) and n > 1 (not frazil-like) must be rejected."""
    for bad in (0.0, -0.5, -1.0):
        with pytest.raises(FrazilRheologyError):
            validate_flow_index(bad)
    for bad in (1.0001, 1.5, 2.0):
        with pytest.raises(FrazilRheologyError):
            validate_flow_index(bad)
    assert validate_flow_index(1.0) == 1.0
    assert validate_flow_index(0.6) == 0.6
    with pytest.raises(FrazilRheologyError):
        frazil_flow_index(0.3, n_packed=0.0)


# ---------------------------------------------------------------------------
# 3. Apparent viscosity
# ---------------------------------------------------------------------------


def test_apparent_viscosity_is_shear_thinning_and_matches_K_at_unit_shear():
    phi = 0.30
    K = float(frazil_consistency_index(phi, mu_water=0.01))
    gamma = torch.tensor([0.01, 0.1, 1.0, 10.0], dtype=torch.float64)
    mu, n = frazil_apparent_viscosity(phi, gamma, mu_water=0.01)
    assert float(mu[2]) == pytest.approx(K, rel=1e-3), "K is the viscosity at gamma_dot = 1"
    assert torch.all(mu[1:] < mu[:-1]), "shear thinning: mu_app must fall with shear rate"
    # explicit power law check
    n_scalar = float(n)
    ratio = float(mu[3] / mu[1])
    assert ratio == pytest.approx(100.0 ** (n_scalar - 1.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Growth-rate window
# ---------------------------------------------------------------------------


def test_growth_window_nucleation_threshold_and_branches():
    win = growth_rate_window([0.0, 0.01, 0.02, 0.03, 0.1, 0.5, 2.0])
    v = win["v"]
    active = win["active"]
    assert v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0, "no growth below the nucleation threshold"
    assert active[0] is False and active[2] is False
    assert active[3] is True
    assert v[3] > 0.0 and v[4] > v[3] and v[5] > v[4], "growth rate must rise with supercooling"
    # just above nucleation the kinetic branch is rate-limiting, further out the
    # diffusive branch takes over -- that two-regime structure is the 'window'
    xover = win["series_crossover_delta_T"]
    assert win["dt_nucleation"] < xover < win["dt_regime_max"]
    win_lo = growth_rate_window([0.5 * (win["dt_nucleation"] + xover)])
    win_hi = growth_rate_window([1.5 * xover])
    assert win_lo["v_kinetic"][0] < win_lo["v_diffusive"][0]
    assert win_hi["v_kinetic"][0] > win_hi["v_diffusive"][0]
    # the frazil (equiaxed) regime flag is honest above ~1 K
    assert win["in_regime"][5] is True
    assert win["in_regime"][6] is False


def test_growth_window_negative_delta_T_is_clamped_to_zero():
    """NEGATIVE input: a negative supercooling is not 'negative growth'."""
    win = growth_rate_window([-0.2, -0.01])
    assert win["v"] == [0.0, 0.0]
    assert win["active"] == [False, False]
    with pytest.raises(FrazilRheologyError):
        growth_rate_window([0.1], dt_nucleation=-0.1)
    with pytest.raises(FrazilRheologyError):
        growth_rate_window([0.1], delta_thermal=0.0)


# ---------------------------------------------------------------------------
# 5. Honest truncation of the concentration-viscosity curve
# ---------------------------------------------------------------------------


def test_concentration_curve_truncates_instead_of_extrapolating():
    """NEGATIVE: requesting phi >= 0.6 must be dropped and counted, never extrapolated."""
    out = concentration_viscosity_curve(
        [0.1, 0.3, 0.58, 0.60, 0.65, 0.75], shear_rates=(0.1, 1.0), mu_water=0.01
    )
    assert out["phi"] == [0.1, 0.3, 0.58]
    assert out["n_truncated"] == 3
    assert out["truncated_above"] == [0.60, 0.65, 0.75]
    assert max(out["phi"]) < DEFAULT_PHI_MAX
    assert all(v > 0.0 for v in out["relative_viscosity"])
    with pytest.raises(FrazilRheologyError):
        concentration_viscosity_curve([0.60], truncate=False)
    # nothing left inside the envelope -> no curve at all rather than an empty one
    with pytest.raises(FrazilRheologyError):
        concentration_viscosity_curve([0.60, 0.70])
    with pytest.raises(FrazilRheologyError):
        concentration_viscosity_curve([-0.1, 0.2])


# ---------------------------------------------------------------------------
# 6. Fast stencil equivalence
# ---------------------------------------------------------------------------


def test_fast_central_difference_is_bit_identical_to_reference():
    torch.manual_seed(7)
    a = torch.randn(13, 9, dtype=torch.float64)
    b = torch.randn(13, 9, dtype=torch.float64)
    for dim in (0, 1):
        assert torch.equal(_central_difference_fast(a, dim), central_difference(a, dim))
    assert torch.equal(_shear_rate_fast(a, b), strain_rate_shear_rate_2d(a, b))


# ---------------------------------------------------------------------------
# 7. Channel verification against the exact solutions
# ---------------------------------------------------------------------------


def test_powerlaw_poiseuille_matches_exact_solution_below_2pct():
    H, n = 24, 0.7
    setup = _setup("poiseuille", H, n)
    res = run_powerlaw_poiseuille(
        n=n,
        K=setup["K"],
        ax=setup["ax"],
        H=H,
        nx=8,
        nu_max=3.0,
        max_steps=15000,
        tol=1e-5,
        avg_steps=100,
        closure="neq",
        init="rest",
    )
    u_exact, _ = poiseuille_analytic_profile(H, n, setup["K"], setup["ax"])
    err = float(
        torch.linalg.vector_norm(res["profile"].double() - u_exact.double())
        / torch.linalg.vector_norm(u_exact.double())
    )
    assert err < 0.02, f"power-law Poiseuille L2 error {err:.3%} exceeds the 2% criterion"
    assert res["viscosity_info"]["clipped_fraction"] == 0.0


def test_powerlaw_couette_matches_exact_solution_below_2pct():
    H, n = 16, 0.7
    setup = _setup("couette", H, n)
    res = run_powerlaw_couette(
        n=n,
        K=setup["K"],
        u_wall=U_REF,
        H=H,
        nx=8,
        nu_max=3.0,
        max_steps=8000,
        tol=1e-5,
        avg_steps=100,
        closure="neq",
        init="rest",
    )
    u_exact, _ = couette_analytic_profile(H, U_REF)
    err = float(
        torch.linalg.vector_norm(res["profile"].double() - u_exact.double())
        / torch.linalg.vector_norm(u_exact.double())
    )
    assert err < 0.02, f"power-law Couette L2 error {err:.3%} exceeds the 2% criterion"
    assert res["wall_stress"]["relative_error"] < 0.05


def test_channel_rejects_non_shear_thinning_index():
    """NEGATIVE: the frazil channel entry points must reject n out of (0, 1]."""
    for bad in (0.0, -0.7, 1.4):
        with pytest.raises(FrazilRheologyError):
            run_powerlaw_poiseuille(n=bad, K=0.02, ax=1e-5, H=16, max_steps=10)
        with pytest.raises(FrazilRheologyError):
            run_powerlaw_couette(n=bad, K=0.02, u_wall=0.03, H=16, max_steps=10)


# ---------------------------------------------------------------------------
# 8. Frazil closure hand-off, reproducibility, artifact
# ---------------------------------------------------------------------------


def test_frazil_suspension_channel_matches_its_exact_rheology():
    res = run_frazil_poiseuille(
        0.30,
        mu_water=0.02,
        u_ref=U_REF,
        H=32,
        nx=8,
        nu_max=20.0,
        max_steps=12000,
        tol=1e-5,
        avg_steps=100,
        closure="neq",
        init="rest",
    )
    assert res["error"]["l2_relative"] < 0.02
    assert 0.7 < res["frazil"]["flow_index"] < 0.9
    assert res["frazil"]["viscosity_wall"] > 0.0
    with pytest.raises(FrazilRheologyError):
        run_frazil_poiseuille(DEFAULT_PHI_MAX, max_steps=10)


def test_same_seed_runs_are_bitwise_reproducible():
    def _run() -> torch.Tensor:
        torch.manual_seed(1234)
        res = run_frazil_poiseuille(
            0.30,
            mu_water=0.02,
            u_ref=U_REF,
            H=24,
            nx=8,
            nu_max=20.0,
            max_steps=1500,
            tol=1e-30,
            avg_steps=10,
            closure="neq",
            init="rest",
        )
        return res["profile"].double()

    a, b = _run(), _run()
    assert torch.equal(a, b), "same seed must reproduce the profile bit for bit"


def test_committed_artifact_acceptance_block():
    path = ROOT / "artifacts" / "frazil_rheology.json"
    if not path.exists():
        pytest.skip("artifacts/frazil_rheology.json not generated in this checkout")
    art = json.loads(path.read_text())
    acc = art["acceptance"]
    assert acc["pass"] is True
    assert acc["poiseuille_pass"] and acc["couette_pass"]
    assert acc["reproducibility_pass"]
    assert acc["max_poiseuille_l2"] < acc["threshold"]
    assert acc["max_couette_l2"] < acc["threshold"]
    assert acc["max_frazil_channel_l2"] < acc["threshold"]
    # the rheology payload the C2/A2 consumers read
    rhe = art["rheology"]
    assert max(rhe["phi"]) < rhe["phi_max"]
    assert all(0.0 < n <= 1.0 for n in rhe["flow_index"])
    assert art["jump_window"]["mu_ratio"] > 5.0
    assert art["growth_window"]["v_at"]["0.100"] > 0.0
    assert art["anchors"]["source"] == "self-built"
