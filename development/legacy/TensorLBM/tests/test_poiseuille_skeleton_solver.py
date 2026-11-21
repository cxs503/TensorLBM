"""Real-solver acceptance tests for the single-phase Poiseuille baseline skeleton.

``test_poiseuille_skeleton.py`` validates the *analytical* arm of the skeleton
(unit conversion, parabola, KPIs, geometry inversion) and by construction
compares analytic functions to analytic functions: it cannot fail because of a
solver defect.  This module supplies the missing arm -- the one the OSS
``PoiseuilleFlow2D`` skeleton exists for -- by running the in-repo D2Q9 channel
solver through :func:`run_baseline_case` and diffing the measured profile
against the independent parabola.

Acceptance (feature ledger ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO``,
``< 15%``) and the tighter internal target (3%) are asserted below, together with
the structural properties the case report claims:

* the discrete steady profile equals the parabola **plus a uniform additive
  slip**, so that the slip-corrected shape error sits at round-off;
* the slip is a function of the relaxation time ``tau`` alone (the scaling
  ``slip = g f(tau)``, ``slip/u_peak = 8 nu f(tau) / H**2``);
* steady state is step-count invariant and mass conserving;
* refinement at fixed lattice viscosity is second order.

Runtime is ~45 s on one core (the D2Q9 channel is a 2-column, y-resolved box).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.verification.xval.poiseuille_skeleton import (
    run_baseline_case,
    slip_decomposition,
)

# OSS lattice pair of the res=16 baseline, used as the fixed point of the
# textbook refinement (see artifacts/xval_poiseuille_baseline_skeleton/).
NU_FIX = 0.17320508075688773
G_FIX = 8.888888888888893e-09

_CACHE: dict[tuple, object] = {}


def run(res: int, **kwargs):
    """Memoised baseline run (several tests reuse the same configuration)."""
    key = (res, tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = run_baseline_case(res, **kwargs)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Ledger acceptance: profile vs the independent parabola
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("res", [8, 16, 32])
def test_raw_profile_error_within_15_percent_gate(res):
    """Ledger gate: every profile KPI of a real run is below the 15 % standard."""
    kpis = run(res).kpis
    for name in ("eps_inf", "eps_l2_rel", "u_peak_err", "q_err"):
        assert kpis[name] < 0.15, (res, name, kpis[name])


@pytest.mark.parametrize("res", [16, 32])
def test_raw_profile_error_within_3_percent_target(res):
    """Internal target: the OSS baseline grids beat 3 % on the raw error."""
    kpis = run(res).kpis
    assert kpis["eps_inf"] < 0.03, (res, kpis["eps_inf"])
    assert kpis["eps_l2_rel"] < 0.03, (res, kpis["eps_l2_rel"])


def test_oss_convention_error_stays_bounded_at_fine_grid():
    """The OSS convention (Re, Ma fixed) keeps nu_H -> the error plateaus < 1 %.

    This is a property of the unit convention, not of the solver: ``nu`` grows
    with the resolution, and the slip scales as ``8 nu f(tau) / H**2``.
    Documented in the case report ("注意点"); the next test gives the
    textbook refinement, where the error does converge.
    """
    errors = [run(r).kpis["eps_inf"] for r in (16, 64, 128)]
    assert max(errors) < 0.01, errors


# ---------------------------------------------------------------------------
# Structure of the residual: parabola + uniform slip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("res", [8, 16, 32])
def test_profile_is_exactly_parabolic_after_slip_removal(res):
    """Removing the uniform slip leaves the parabola at round-off (not 1e-3)."""
    r = run(res)
    assert r.slip_kpis["eps_inf_shape"] < 1e-6, r.slip_kpis
    assert r.slip_kpis["eps_l2_shape"] < 1e-6, r.slip_kpis


def test_residual_is_purely_uniform():
    """The error field has (numerically) no shape component: u_num = parabola + s."""
    r = run(32)
    residual = r.u_num - r.u_exact
    peak = float(np.max(r.u_num))
    assert np.ptp(residual) / peak < 1e-8, np.ptp(residual) / peak
    # ... and the uniform part is the slip actually reported
    assert np.isclose(float(np.mean(residual)), r.slip_kpis["slip"], rtol=1e-12, atol=0.0)


def test_run_is_not_analytic_self_comparison():
    """Vacuity guard: the raw error of a real run is small but genuinely non-zero.

    Without this, a harness that silently compared the parabola to itself would
    pass every tolerance above.
    """
    r = run(16)
    assert 1e-6 < r.kpis["eps_inf"] < 3e-2, r.kpis["eps_inf"]
    assert abs(r.slip_kpis["slip_rel"]) > 1e-4, r.slip_kpis
    assert r.steps > 0


# ---------------------------------------------------------------------------
# Physical invariants of the run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("res", [16, 32])
def test_mass_is_conserved(res):
    """The Guo source term is zero-sum: no density drift in the steady state."""
    assert abs(run(res).mass_drift) < 1e-11, run(res).mass_drift


def test_steady_state_is_run_length_invariant():
    """Extra steps cannot change the answer (a genuine steady state)."""
    short = run(16)
    long = run(16, tol=1e-13, max_steps=40_000)
    assert long.steps != short.steps or long.converged
    rel = np.max(np.abs(long.u_num - short.u_num)) / np.max(short.u_num)
    assert rel < 1e-6, rel


def test_baseline_reproduces_analytic_amplitude():
    """Peak velocity matches g H^2 / (8 nu) within the slip-corrected accuracy."""
    r = run(32)
    rel = abs(np.max(r.u_num) - r.params.u_peak_lu) / r.params.u_peak_lu
    assert rel < 0.01, rel


# ---------------------------------------------------------------------------
# The slip law: slip = g f(tau), slip/u_peak = 8 nu f(tau) / H^2
# ---------------------------------------------------------------------------

def test_slip_is_a_function_of_tau_not_of_mach():
    """Same tau via (Ma, Re) -> (2 Ma, 2 Re): the slip per unit force is unchanged.

    A compressibility-driven slip would scale with Ma and break this.
    """
    a = run(32, mach_number=0.02, reynolds_number=1.0)
    b = run(32, mach_number=0.04, reynolds_number=2.0)
    tau_a = a.params.relaxation_parameter_lu
    tau_b = b.params.relaxation_parameter_lu
    assert np.isclose(tau_a, tau_b, rtol=1e-12), (tau_a, tau_b)
    fa = a.slip_kpis["slip"] / a.params.acceleration_lu
    fb = b.slip_kpis["slip"] / b.params.acceleration_lu
    assert abs(fa - fb) / abs(fa) < 5e-3, (fa, fb)


def test_slip_scales_as_one_over_height_squared():
    """slip/u_peak = 8 nu f(tau) / H^2: matched tau at H=30 and H=62 agrees."""
    a = run(32, reynolds_number=0.35)                    # tau = 3.5682
    b = run(64, reynolds_number=0.71138)                 # tau matched by construction
    assert abs(a.params.relaxation_parameter_lu - b.params.relaxation_parameter_lu) < 5e-3
    fa = a.slip_kpis["slip"] / a.params.acceleration_lu
    fb = b.slip_kpis["slip"] / b.params.acceleration_lu
    assert abs(fa - fb) / abs(fa) < 0.05, (fa, fb)
    # the relative slip of the two runs differs by (H_a / H_b)**2
    ratio = a.slip_kpis["slip_rel"] / b.slip_kpis["slip_rel"]
    expected = (b.params.height_lu / a.params.height_lu) ** 2
    assert abs(ratio / expected - 1.0) < 0.05, (ratio, expected)


# ---------------------------------------------------------------------------
# Textbook refinement: fixed lattice viscosity and force
# ---------------------------------------------------------------------------

def test_fixed_viscosity_refinement_is_second_order():
    """With nu and g held fixed, eps_inf ~ H**-2 (observed order 2.0)."""
    coarse = run(16, nu_lu=NU_FIX, force_lu=G_FIX)
    fine = run(32, nu_lu=NU_FIX, force_lu=G_FIX)
    order = np.log(coarse.kpis["eps_inf"] / fine.kpis["eps_inf"]) / np.log(
        fine.params.height_lu / coarse.params.height_lu)
    assert 1.8 <= order <= 2.2, order


# ---------------------------------------------------------------------------
# KPI helper: slip decomposition contract
# ---------------------------------------------------------------------------

def test_slip_decomposition_is_consistent_with_profile_kpis():
    """The raw KPI and the decomposition describe the same error field."""
    r = run(16)
    direct = slip_decomposition(r.y, r.u_num, r.u_exact)
    assert direct["slip"] == pytest.approx(r.slip_kpis["slip"], rel=1e-12)
    assert direct["eps_inf_shape"] <= r.kpis["eps_inf"]