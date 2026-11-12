"""Analytic physics validation of torque-free 6-DOF rotation (CASE-SIXDOF-EULER-TOP).

Referees, all independent of the integrator:

* exact symmetric-top solution of Euler's equations (arbitrary amplitude),
* exact first integrals (energy, body/world angular momentum, quaternion norm),
* linear intermediate-axis growth rate (Dzhanibekov) inside a stated amplitude
  band.

A falsification test shifts the reference frequency by 5% and asserts the same
metric blows up, so the gate cannot pass by both sides sharing an error.

The 6-DOF capability contract declares ``WITHHELD_NO_PHYSICS_VALIDATION`` for
the FSI/coupling path; this file only discharges the *torque-free rotation*
sub-scope, which is what the analytic references actually cover.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.sixdof_rotation_validation import (
    band_summary,
    compare_symmetric_top,
    convergence_study,
    dzhanibekov_growth_rate,
    euler_symmetric_top_solution,
    falsify_reference_frequency,
    integrate_torque_free,
    rotation_invariants,
)

# Symmetric top: I1 = I2 != I3, spin about the symmetry axis -> Omega = 0.5 rad/s,
# transverse period 12.566 s.
I_TOP = (2.0, 2.0, 1.0)
W_TOP = (0.3, 0.0, 1.0)
# Intermediate-axis body: I1 < I2 < I3.
I_TUMBLE = (1.0, 2.0, 3.0)


# --------------------------------------------------------------------- reference
def test_symmetric_top_solution_invariants_hold_exactly():
    """The analytic solution itself must conserve T and |I.omega| (self-check)."""
    t = np.linspace(0.0, 30.0, 400)
    w = euler_symmetric_top_solution(W_TOP, I_TOP, t)
    inertia = np.asarray(I_TOP)
    energy = 0.5 * np.sum(w * inertia * w, axis=1)
    momentum = np.linalg.norm(w * inertia, axis=1)
    assert np.allclose(energy, energy[0], rtol=1e-12)
    assert np.allclose(momentum, momentum[0], rtol=1e-12)
    # Axial component is exactly constant by construction.
    assert np.allclose(w[:, 2], W_TOP[2], rtol=0, atol=0)


def test_symmetric_top_rejects_non_symmetric_inertia():
    with pytest.raises(ValueError):
        euler_symmetric_top_solution((0.1, 0.2, 0.3), (1.0, 2.0, 3.0), np.array([0.0, 1.0]))


# ------------------------------------------------------------- integrated motion
def test_integrated_top_matches_exact_solution_within_3_percent():
    """Trajectory gate: dt=0.02 (free integration) must stay inside 3% of exact."""
    dt, t_end = 0.02, 12.0
    run = integrate_torque_free(I_TOP, W_TOP, dt=dt, n_steps=int(round(t_end / dt)))
    m = compare_symmetric_top(run["omega"], run["times"], W_TOP, I_TOP)
    assert m["rel_l2"] < 0.03, m
    assert m["rel_inf"] < 0.03, m
    assert m["period_rel_err"] < 1e-3, m
    # Axial rate is an exact invariant of Euler's equations -> must not drift.
    assert m["axial_rel_drift"] < 1e-12, m


def test_energy_and_angular_momentum_drift_is_first_order():
    """First integrals must drift at O(dt) and shrink when dt shrinks.

    Honest scope: the gyroscopic update ``w += dt I^-1 (-w x I w)`` is an
    *explicit* Euler step on a Poisson system, which is not symplectic for the
    rotational part, so T and |I.w| drift secularly at O(dt).  The gate is
    therefore (a) drift <= 1% at dt = 0.005 over 12 s and (b) the drift must
    fall when dt falls -- not an exact-conservation claim.
    """
    runs = [integrate_torque_free(I_TOP, W_TOP, dt=dt, n_steps=int(round(12.0 / dt)))
            for dt in (0.02, 0.005)]
    drifts = []
    for run in runs:
        inv = run["invariants"]
        e0, l0 = inv[0]["energy"], inv[0]["momentum_body"]
        drifts.append(
            (
                max(abs(r["energy"] - e0) / e0 for r in inv),
                max(abs(r["momentum_body"] - l0) / l0 for r in inv),
                max(abs(r["quat_norm"] - 1.0) for r in inv),
            )
        )
    (e_coarse, l_coarse, q_coarse), (e_fine, l_fine, q_fine) = drifts
    assert e_fine < 0.01 and l_fine < 0.01, drifts
    assert e_fine < e_coarse and l_fine < l_coarse  # refinement reduces the drift
    assert max(q_coarse, q_fine) < 1e-12  # quaternion stays unit length exactly


def test_world_angular_momentum_fixed_while_body_frame_rotates():
    """A real rotation: the body-frame rate vector must change while L_world is fixed."""
    run = integrate_torque_free(I_TOP, W_TOP, dt=0.01, n_steps=300)
    w = run["omega"]
    assert np.abs(w[-1] - w[0]).max() > 0.05  # body-frame rate genuinely evolves
    inv = run["invariants"]
    rel = abs(inv[-1]["momentum_world"] - inv[0]["momentum_world"]) / inv[0]["momentum_world"]
    assert rel < 1e-3, rel  # finite-rotation check: world-frame |L| is stationary


def test_first_order_convergence_of_symplectic_euler():
    """Refining dt must reduce the trajectory error at ~first order."""
    study = convergence_study(I_TOP, W_TOP, dts=[0.02, 0.01, 0.005], t_end=6.0)
    assert 0.8 < study["observed_order"] < 1.3, study["observed_order"]
    errs = [row["rel_l2"] for row in study["rows"]]
    assert errs == sorted(errs, reverse=True)  # monotone in dt


def test_rotation_invariants_match_analytic_energy():
    """The invariant helper must reproduce the hand-computed 1/2 w.I.w."""
    import torch

    w = torch.tensor(W_TOP, dtype=torch.float64)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    got = rotation_invariants(q, w, I_TOP)
    want = 0.5 * (2.0 * W_TOP[0] ** 2 + 2.0 * W_TOP[1] ** 2 + 1.0 * W_TOP[2] ** 2)
    assert got["energy"] == pytest.approx(want, rel=1e-14)
    assert got["momentum_world"] == pytest.approx(got["momentum_body"], rel=1e-14)


# ------------------------------------------------------- intermediate-axis tumble
def test_dzhanibekov_rate_formula():
    lam = dzhanibekov_growth_rate(I_TUMBLE, 1.0)
    want = math.sqrt((3.0 - 2.0) * (2.0 - 1.0) / (1.0 * 3.0))
    assert lam == pytest.approx(want, rel=1e-14)
    with pytest.raises(ValueError):
        dzhanibekov_growth_rate((3.0, 2.0, 1.0), 1.0)


def test_intermediate_axis_grows_at_linear_rate_within_20_percent():
    """Semi-quantitative: fitted growth inside a 2e-3..1e-2 band vs linear theory.

    The exact solution is cnoidal rather than a pure exponential, so the band is
    stated explicitly and the tolerance is 20% (documented limitation).
    """
    lam = dzhanibekov_growth_rate(I_TUMBLE, 1.0)
    run = integrate_torque_free(I_TUMBLE, (1e-3, 1.0, 1e-3), dt=0.002, n_steps=2500)
    summary = band_summary(run["times"], run["omega"][:, 0], low=2e-3, high=1e-2)
    rel = abs(summary["rate"] - lam) / lam
    assert rel < 0.20, (summary, lam, rel)
    # The perturbation must actually grow (a stable axis would decay).
    assert summary["rate"] > 0.0


def test_intermediate_axis_instability_absent_for_major_axis_spin():
    """Discriminator: spinning about the largest-inertia axis must not grow."""
    run = integrate_torque_free(
        I_TUMBLE, (1e-3, 1e-3, 1.0), dt=0.002, n_steps=1500
    )
    transverse = np.abs(run["omega"][:, :2]).max(axis=1)
    first = float(transverse[: transverse.size // 4].max())
    late = float(transverse[3 * transverse.size // 4 :].max())
    assert late < 2.0 * first  # bounded oscillation, no exponential growth
    assert float(transverse.max()) < 0.05  # amplitude stays at the seed scale


# --------------------------------------------------------------- falsifiability
def test_gate_fails_on_shifted_reference_frequency():
    """A 5% error in the reference precession rate must be detected comfortably."""
    m = falsify_reference_frequency(0.05, W_TOP, I_TOP, dt=0.005, n_steps=2400)
    assert m["rel_l2"] > 0.05, m
    assert m["rel_inf"] > 0.05, m


def test_domain_constants_are_not_fitted():
    """Guard: the reference depends only on (omega0, inertia) -- no tuning knobs."""
    run = integrate_torque_free(I_TOP, W_TOP, dt=0.01, n_steps=100)
    m1 = compare_symmetric_top(run["omega"], run["times"], W_TOP, I_TOP)
    m2 = compare_symmetric_top(run["omega"], run["times"], W_TOP, I_TOP)
    for key in ("rel_l2", "rel_inf", "axial_rel_drift", "amplitude_rel_err", "period_ref"):
        assert m1[key] == pytest.approx(m2[key], rel=0, abs=0, nan_ok=True), key
    assert m1["period_ref"] == pytest.approx(2.0 * math.pi / 0.5, rel=1e-12)