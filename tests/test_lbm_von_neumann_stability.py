"""Tests for the von Neumann linear stability / equivalent-equation surface.

Covers :mod:`tensorlbm.stability.von_neumann` and
:mod:`tensorlbm.stability.campaign`:

* operator consistency with the production code (equilibrium, MRT collision),
* analytic limit checks (Chapman-Enskog viscosity, acoustic speed, Ma^2 shift),
* the measured-vs-predicted cross-audit on a periodic box (the acceptance
  logic of P155-N08, in miniature and fast),
* divergence probes and the campaign criterion helpers.

The box experiments here use ``n = 16`` and short runs to keep the suite fast;
the full campaign runs at ``n = 32`` with 4000-5000 steps
(``scripts/run_lbm_von_neumann_stability_campaign.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorlbm.stability import von_neumann as vn  # noqa: E402
from tensorlbm.stability.campaign import (  # noqa: E402
    PREDECLARED,
    box_grid_spectral_radius,
    c3_equivalent_equation,
    c4_viscosity_ratio_table,
    ma_to_u,
)

# ---------------------------------------------------------------------------
# operator consistency with the production code
# ---------------------------------------------------------------------------


def test_equilibrium_matches_production_lattice() -> None:
    """NumPy equilibrium of the analysis must equal tensorlbm.lattice.equilibrium."""
    from tensorlbm.lattice import equilibrium as torch_equilibrium

    rng = np.random.default_rng(0)
    rho = 1.0 + 0.1 * rng.random((5, 6))
    ux = 0.05 * (rng.random((5, 6)) - 0.5)
    uy = 0.05 * (rng.random((5, 6)) - 0.5)
    mine = vn.equilibrium(rho, ux, uy)
    ref = torch_equilibrium(
        torch.tensor(rho), torch.stack([torch.tensor(ux), torch.tensor(uy)], dim=-1)
    ).numpy()
    assert mine.shape == (9, 5, 6)
    assert np.allclose(np.moveaxis(mine, 0, -1), ref, atol=1e-12)


def test_equilibrium_jacobian_matches_finite_difference() -> None:
    rho0, u0 = 1.0, (0.07, -0.03)
    jac = vn.equilibrium_jacobian(rho0, u0)
    f0 = vn.equilibrium(np.array(rho0), np.array(u0[0]), np.array(u0[1]))
    eps = 1e-7
    fd = np.zeros_like(jac)
    for j in range(9):
        # f -> f + eps * e_j : rho -> rho + eps, u -> (rho u + eps c_j) / (rho + eps)
        rho1 = rho0 + eps
        ux1 = (rho0 * u0[0] + eps * vn.CX[j]) / rho1
        uy1 = (rho0 * u0[1] + eps * vn.CY[j]) / rho1
        f1 = vn.equilibrium(np.array(rho1), np.array(ux1), np.array(uy1))
        fd[:, j] = (f1 - f0) / eps
    assert np.allclose(jac, fd, rtol=1e-6, atol=1e-9)


def test_mrt_collision_operator_matches_production() -> None:
    """MPR matrix/rates must reproduce tensorlbm.solver.collide_mrt."""
    from tensorlbm.solver import collide_mrt

    rng = np.random.default_rng(1)
    f = 0.11 + rng.random((9, 6, 7))
    coll = vn.Collision("mrt", 0.62, s_e=1.64, s_eps=1.54, s_q=1.7)
    mine = vn._collide_np(f, coll)
    ref = collide_mrt(
        torch.tensor(f, dtype=torch.float64), 0.62, s_e=1.64, s_eps=1.54, s_q=1.7
    ).numpy()
    assert np.allclose(mine, ref, atol=1e-12)


def test_bgk_collision_operator_matches_production() -> None:
    from tensorlbm.solver import collide_bgk

    rng = np.random.default_rng(2)
    f = 0.11 + rng.random((9, 5, 4))
    mine = vn._collide_np(f, vn.Collision("bgk", 0.7))
    ref = collide_bgk(torch.tensor(f, dtype=torch.float64), 0.7).numpy()
    assert np.allclose(mine, ref, atol=1e-13)


# ---------------------------------------------------------------------------
# analytic limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.55, 0.8, 1.2])
def test_shear_attenuation_recovers_chapman_enskog_viscosity(tau: float) -> None:
    """nu_num(theta) -> (tau - 1/2)/3 as theta -> 0 (at Ma = 0)."""
    coll = vn.Collision("bgk", tau)
    curve = vn.shear_branch_curve([1e-3, 1e-2], coll)
    for row in curve:
        assert abs(row["nu_num"] / coll.viscosity - 1.0) < 0.01
        assert row["overlap"] > 0.999


def test_shear_growth_rate_is_minus_nu_theta_squared() -> None:
    coll = vn.Collision("bgk", 0.8)
    k = 0.2
    sp = vn.spectrum(k, 0.0, coll)
    j = sp.index_of("shear")
    assert sp.labels[j] == "shear"
    # -nu k^2 to within the leading O(k^2) *relative* correction (b/a*k^2 ~ 2e-3)
    assert sp.growth[j] == pytest.approx(-coll.viscosity * k * k, rel=5e-3)


def test_acoustic_speed_at_small_theta() -> None:
    """The acoustic branch must propagate at c_s = 1/sqrt(3) as theta -> 0."""
    coll = vn.Collision("bgk", 0.6)
    sp = vn.spectrum(0.01, 0.0, coll)
    c_plus = sp.phase_speed[sp.index_of("sound+")]
    c_minus = sp.phase_speed[sp.index_of("sound-")]
    assert c_plus == pytest.approx(vn.CS, rel=1e-3)
    assert c_minus == pytest.approx(-vn.CS, rel=1e-3)


def test_conservation_modes_are_marginal_at_zero_wave_vector() -> None:
    for kind in ("bgk", "mrt"):
        sp = vn.spectrum(0.0, 0.0, vn.Collision(kind, 0.7))
        assert np.sort(np.abs(sp.mu))[-1] == pytest.approx(1.0, abs=1e-12)


def test_ma_definition_uses_cs() -> None:
    ux, uy = ma_to_u(0.3)
    assert math.hypot(ux, uy) / vn.CS == pytest.approx(0.3, rel=1e-12)


def test_ma_squared_viscosity_shift() -> None:
    """At finite Ma the shear attenuation carries a clean -Ma^2 relative shift.

    ``nu_num(theta -> 0, Ma) = nu_phys (1 - Ma^2 + O(Ma^4))``; this is the
    C4 'physical-viscosity contamination' law and is checked directly here.
    """
    coll = vn.Collision("bgk", 0.7)
    for ma in (0.1, 0.2, 0.3):
        row = vn.numerical_viscosity(0.002, coll, ma_to_u(ma))
        shift = row["nu_num"] / coll.viscosity - 1.0
        assert shift == pytest.approx(-ma * ma, abs=2e-4)


# ---------------------------------------------------------------------------
# von Neumann stability verdicts
# ---------------------------------------------------------------------------


def test_bgk_unstable_below_half_relaxation_time() -> None:
    scan = vn.spectral_radius(vn.Collision("bgk", 0.45), (0.0, 0.0), n_theta=17)
    assert scan["lambda_max"] == pytest.approx(1.0 / 0.45 - 1.0, rel=1e-6)
    assert not vn.stability_point(vn.Collision("bgk", 0.45), (0.0, 0.0), n_theta=17)["stable"]


@pytest.mark.parametrize("tau,ma", [(0.6, 0.0), (0.6, 0.2), (0.9, 0.3), (1.2, 0.0)])
def test_bgk_stable_in_expected_region(tau: float, ma: float) -> None:
    assert vn.stability_point(vn.Collision("bgk", tau), ma_to_u(ma), n_theta=17)["stable"]


def test_zero_mode_excluded_in_box_grid_scan() -> None:
    """The uniform (0,0) mode is the base state, not a perturbation mode."""
    grid = box_grid_spectral_radius(vn.Collision("bgk", 0.9), (0.0, 0.0), n=16)
    assert grid["mode"] != (0, 0)


# ---------------------------------------------------------------------------
# measured side: cross-audit of prediction vs periodic-box experiment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau,ma", [(0.60, 0.0), (0.70, 0.2), (0.80, 0.0)])
def test_measured_growth_matches_prediction_shear(tau: float, ma: float) -> None:
    """C1 in miniature: physical shear seed + vorticity observable, error < 5%."""
    k = 2.0 * math.pi * 2.0 / 16.0
    run = vn.run_modal_box(
        k,
        0.0,
        vn.Collision("bgk", tau),
        ma_to_u(ma),
        n=16,
        n_steps=1200,
        seed="shear",
        amplitude=1e-6,
    )
    assert run.notes["target_branch"] == "shear"
    assert run.observable == "vorticity"
    assert run.growth_rel_error < PREDECLARED["c1"]["growth_rel_err_max"]
    assert run.fit_r2 > 0.999
    assert run.purity > 0.99


def test_measured_growth_is_amplitude_independent() -> None:
    """Linearity control: x10 seed amplitude must reproduce the growth rate."""
    k = 2.0 * math.pi * 2.0 / 16.0
    coll = vn.Collision("bgk", 0.7)
    a = vn.run_modal_box(k, 0.0, coll, (0.0, 0.0), n=16, n_steps=800, amplitude=1e-6)
    b = vn.run_modal_box(k, 0.0, coll, (0.0, 0.0), n=16, n_steps=800, amplitude=1e-5)
    rel = abs(a.measured_growth - b.measured_growth) / abs(a.measured_growth)
    assert rel < 1e-3


def test_measured_acoustic_branch_phase_speed() -> None:
    k = 2.0 * math.pi * 2.0 / 16.0
    coll = vn.Collision("bgk", 0.6)
    run = vn.run_modal_box(
        k, 0.0, coll, (0.0, 0.0), n=16, n_steps=800, seed="eigen", target="sound+"
    )
    assert run.phase_speed_rel_error < 0.05
    assert run.growth_rel_error < 0.05


def test_seeded_probe_detects_instability_and_not_stability() -> None:
    """Divergence probe (C2 in miniature)."""
    unstable = vn.run_modal_box(
        2.0 * math.pi / 16.0,
        0.0,
        vn.Collision("bgk", 0.45),
        (0.0, 0.0),
        n=16,
        n_steps=2000,
        seed="eigen",
        target="max",
        amplitude=1e-8,
        blowup_tol=0.05,
        max_rel_amp=0.05,
    )
    assert unstable.predicted_growth > 0.0
    assert unstable.modal_growth_factor > 100.0
    assert unstable.diverged_or_grown

    stable = vn.run_modal_box(
        2.0 * math.pi / 16.0,
        0.0,
        vn.Collision("bgk", 0.9),
        (0.0, 0.0),
        n=16,
        n_steps=2000,
        seed="eigen",
        target="max",
        amplitude=1e-8,
        blowup_tol=0.05,
        max_rel_amp=0.05,
    )
    assert stable.predicted_growth < 0.0
    assert stable.modal_growth_factor < 1.0
    assert not stable.diverged_or_grown


# ---------------------------------------------------------------------------
# equivalent equation / campaign helpers
# ---------------------------------------------------------------------------


def test_quadratic_truncation_of_numerical_viscosity() -> None:
    """C3 in miniature: nu_num(theta) is a + b theta^2 to well under 10%."""
    res = c3_equivalent_equation(taus=(0.7,), mas=(0.0,))
    curve = res["curves"][0]
    assert curve["fit"]["max_rel_dev"] < PREDECLARED["c3"]["quad_fit_rel_dev_max"]
    assert res["pass"]
    assert curve["fit"]["intercept_rel_err"] < 0.01


def test_viscosity_ratio_band_exists_at_low_mach() -> None:
    """C4 in miniature: a theta band with contamination below 1% exists at Ma=0."""
    res = c4_viscosity_ratio_table(taus=(0.7,), mas=(0.0,), thetas=np.linspace(0.02, 1.0, 20))
    row = res["rows"][0]
    assert row["theta_max_below_1pct"] > 0.0
    assert row["bands_below_1pct"][0][0] == pytest.approx(0.02, rel=1e-6)
    assert res["pass"]


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        vn.Collision("bogus", 0.6)
    with pytest.raises(ValueError):
        vn.Collision("bgk", 0.0)
    with pytest.raises(ValueError):
        vn.run_modal_box(0.0, 0.0, vn.Collision("bgk", 0.6), n=8)
