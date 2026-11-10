"""Acceptance tests for the D3Q27 collision-operator verification face.

Mission ``p246w2-n01``.  The oracles here are analytic (equilibrium moment
identities, exact transform inverses, the parabolic Poiseuille solution) or
measured-and-pinned (the Galilean defect of the advected shear wave, which is an
*equilibrium*-truncation effect: BGK and cumulant must agree with each other,
and both must scale as Ma²).

The last test in this file pins the negative result on purpose: a future change
that really removes the O(Ma²) defect would have to be an equilibrium /
streaming change, not a collision change, and would break the pin loudly rather
than silently rewriting history.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.cumulant_verification import (
    CS2,
    _decay_rate,
    collision_conservation_error,
    covariance_spectrum,
    fit_power_law_slope,
    galilean_defect,
    make_operator,
    moment_identity_errors,
    poiseuille_profile,
    shear_wave_amplitude_series,
    single_collision_stress_defect,
    stability_summary,
    taylor_green_setup,
    transform_roundtrip_error,
)
from tensorlbm.d3q27 import stream27_roll

pytestmark = pytest.mark.physics

OPS = ("bgk", "kbc", "cumulant", "cumulant_single_rate")


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------
def test_a1_equilibrium_moment_identities():
    """A1: Σf=ρ, Σf c_i=ρu_i, Σf c_ic_j=ρu_iu_j+ρc_s²δ to < 1e-12 over 20 draws."""
    r = moment_identity_errors(n_samples=20)
    assert r["mass_rel_err"] < 1e-12
    assert r["momentum_rel_err"] < 1e-12
    assert r["second_moment_rel_err"] < 1e-12


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------
def test_a2_transform_roundtrip_is_invertible():
    """A2: κ→C→κ and m→κ→m round-trips are exact to < 1e-13."""
    r = transform_roundtrip_error(n_states=4)
    assert r["cumulant_roundtrip_rel_err"] < 1e-13
    assert r["unshift_rel_err"] < 1e-13
    assert r["population_roundtrip_rel_err"] < 1e-13


@pytest.mark.parametrize("operator", OPS)
def test_a2_collision_conserves_mass_and_momentum(operator: str):
    """A2: one collision conserves mass and momentum to round-off.

    Measured on the density scale (``|Δ|/ρ``) because the moment pipelines are
    O(ρ) while the momentum is O(Ma·ρ); both normalisations are returned by the
    rig and the density-normalised one is the machine-precision statement.
    """
    r = collision_conservation_error(operator, 0.55)
    assert r["mass_abs_err_over_rho"] < 1e-14
    assert r["momentum_abs_err_over_rho"] < 1e-14


# ---------------------------------------------------------------------------
# A3 / A7 — defect definition matrix
# ---------------------------------------------------------------------------
def test_a3_single_collision_is_blind_to_advection():
    """A3/D1: one collision produces the same stress at any Ma (all operators).

    ``Π_xy`` relaxing by ``(1−ω)`` is exact for every operator that relaxes the
    shear cumulant at ``ω = 1/τ``, so the single-collision defect is at round-off
    for BGK *and* cumulant — and also for the collision-free control.  This is
    why a one-step test cannot discriminate operators.
    """
    for op in OPS + ("no_collision",):
        d = single_collision_stress_defect(op, 0.55, mach=0.1, n=16)
        assert d < 1e-10, (op, d)


@pytest.mark.slow
def test_a3_bgk_defect_scales_as_mach_squared():
    """A3: the BGK advected-shear-wave defect is measurable and grows as Ma²."""
    res = galilean_defect("bgk", tau=0.55, machs=(0.05, 0.1, 0.2), n=32, nsteps=600, window=300)
    assert res.defects[0] > 1e-3  # measurable
    assert abs(res.slope - 2.0) <= 0.3, res.slope
    assert res.defects[1] > res.defects[0] > 0.0


@pytest.mark.slow
def test_a3_cumulant_defect_equals_bgk_defect():
    """A3 pin: the defect is carried by the truncated equilibrium, not the operator.

    BGK and the Geier cumulant operator differ by 4 relaxation rates; if the
    O(Ma²) stress defect were a collision property they would differ by O(1).
    They agree to better than 1 % at Ma = 0.1 (measured: 3e-4 relative).
    """
    kw = dict(tau=0.55, machs=(0.1,), n=32, nsteps=600, window=300)
    bgk = galilean_defect("bgk", **kw)
    cum = galilean_defect("cumulant", **kw)
    rel = abs(cum.defects[0] - bgk.defects[0]) / bgk.defects[0]
    assert rel < 0.01, rel
    # and the *transient* window (window=0) does differ: that is why the window
    # must be declared, and this pins the distinction.
    t_bgk = galilean_defect("bgk", **{**kw, "window": 0})
    t_cum = galilean_defect("cumulant", **{**kw, "window": 0})
    rel_t = abs(t_cum.defects[0] - t_bgk.defects[0]) / t_bgk.defects[0]
    assert rel_t > 5.0 * rel, (rel_t, rel)


@pytest.mark.slow
def test_a7_negative_control_recovers_bgk():
    """A7: single-rate cumulant (all ω = 1/τ) reproduces the BGK defect."""
    kw = dict(tau=0.55, machs=(0.1,), n=32, nsteps=600, window=300)
    bgk = galilean_defect("bgk", **kw)
    neg = galilean_defect("cumulant_single_rate", **kw)
    rel = abs(neg.defects[0] - bgk.defects[0]) / bgk.defects[0]
    assert rel < 0.02, rel


def test_a3_defect_slope_helper():
    """The log-log fitter used for the Ma² criterion is itself checked."""
    xs = [0.05, 0.1, 0.2]
    ys = [4.0e-4, 1.6e-3, 6.4e-3]
    assert fit_power_law_slope(xs, ys) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# A4 — realizability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("operator", OPS)
def test_a4_realizability_stays_positive_definite(operator: str):
    """A4: in-domain runs keep ρ > 0 and the covariance PSD for every operator."""
    f = taylor_green_setup(8, u0=0.05)
    op = make_operator(operator, 0.52)
    for _ in range(60):
        f = stream27_roll(op(f))
        assert torch.isfinite(f).all()
    cov = covariance_spectrum(f)
    assert cov["rho_min"] > 0.0
    assert cov["lambda_min"] >= -1e-12
    # the trace of the covariance is 3ρc_s² for a near-equilibrium state
    assert cov["trace_min"] == pytest.approx(3.0 * CS2 * cov["rho_min"], rel=0.5)


def test_a4_covariance_detects_a_nonrealizable_state():
    """The realizability monitor must flag a state that is not a distribution.

    Push the shear component far beyond the positivity cone: the covariance
    eigenvalue must go negative, so the monitor cannot silently return ``ok``.
    """
    f = taylor_green_setup(4)
    cxy = torch.zeros(27, 1, 1, 1, dtype=torch.float64)
    c = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=torch.float64,
    )
    del c
    from tensorlbm.d3q27 import C as C27  # noqa: PLC0415

    for q in range(27):
        cxy[q, 0, 0, 0] = C27[q, 0].double() * C27[q, 1].double()
    bad = f + 3.0 * cxy  # large pure-shear distortion: Π_xy ≫ ½ρc_s²
    cov = covariance_spectrum(bad)
    assert cov["lambda_min"] < 0.0


# ---------------------------------------------------------------------------
# A6 — Poiseuille
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("operator", ("bgk", "cumulant"))
def test_a6_poiseuille_profile_matches_analytic(operator: str):
    """A6: the channel profile reproduces g/(2ν)·Y(H−Y) to a few percent."""
    r = poiseuille_profile(operator, 0.8, ny=16, nx=2, nz=2, nsteps=20000)
    assert r["converged"], r["steps"]
    assert r["nu_rel_err"] < 0.01, r["nu_rel_err"]
    assert r["rel_err"] < 0.01, r["rel_err"]
    assert r["u_max"] == pytest.approx(r["u_max_analytic"], rel=0.01)


# ---------------------------------------------------------------------------
# Physics-fidelity cross-check: viscosity from the shear-wave decay rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("operator", ("bgk", "kbc", "cumulant"))
def test_shear_wave_decay_rate_reproduces_nu(operator: str):
    """Every operator must decay a shear wave at λ = νk² to < 2 % (analytic)."""
    n, tau = 32, 0.55
    series = shear_wave_amplitude_series(
        operator, tau, mach=0.0, amp=2e-2, n=n, nsteps=500
    )["amplitudes"]
    lam = _decay_rate(series[250:])
    lam_theory = (1.0 / 3.0) * (tau - 0.5) * (2.0 * math.pi / n) ** 2
    assert abs(lam / lam_theory - 1.0) < 0.02, (lam, lam_theory)


# ---------------------------------------------------------------------------
# Selection criterion helper
# ---------------------------------------------------------------------------
def test_stability_summary_flags_the_wall():
    """The ``Re_max(op)`` selection metric is monotone and honest about deaths."""
    rows = [
        {"operator": "bgk", "re": 100.0, "diverged": False, "steps_completed": 100,
         "steps_requested": 100, "realizability_violated": False, "first_violation_step": -1},
        {"operator": "bgk", "re": 1000.0, "diverged": True, "steps_completed": 37,
         "steps_requested": 100, "realizability_violated": True, "first_violation_step": 30},
        {"operator": "cumulant", "re": 100.0, "diverged": False, "steps_completed": 100,
         "steps_requested": 100, "realizability_violated": False, "first_violation_step": -1},
        {"operator": "cumulant", "re": 1000.0, "diverged": False, "steps_completed": 100,
         "steps_requested": 100, "realizability_violated": False, "first_violation_step": -1},
    ]
    s = stability_summary(rows)
    assert s["bgk"]["re_max_survived"] == 100.0
    assert s["cumulant"]["re_max_survived"] == 1000.0
    assert s["cumulant"]["re_max_ratio_vs_bgk"] == pytest.approx(10.0)
    assert s["bgk"]["died_after_realizability_violation"] == 1