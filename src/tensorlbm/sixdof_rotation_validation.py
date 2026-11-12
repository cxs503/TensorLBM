"""Analytic physics validation of the torque-free 6-DOF rigid-body integrator.

Why this module exists
----------------------
``tensorlbm.sixdof_capability_contract`` certifies the 6-DOF path only as
``CONTRACT_TESTED`` and declares ``WITHHELD_NO_PHYSICS_VALIDATION``: the
existing tests check operator algebra (quaternion normalisation, DOF
constraints, Newton-Euler consistency at a single step) but never compare a
*trajectory* against a known solution.  A symplectic-Euler gyroscopic
integrator can pass every identity test and still produce the wrong
precession frequency, the wrong sign of the gyroscopic term, or a quaternion
drift, because those errors are invisible in a one-step test.

This module supplies the missing referee.  It validates the *torque-free*
(``M = 0``) rigid-body rotation, the part of the 6-DOF dynamics that has exact
closed-form solutions, against three independent analytic anchors:

1. ``euler_symmetric_top_solution`` -- exact solution of Euler's equations for
   a symmetric top (``I1 = I2 != I3``).  The transverse rate components are
   harmonic with angular rate ``Omega = (I1 - I3) * w3 / I1`` and the axial
   component is exactly constant.  No linearisation is involved, so it is a
   genuine exact reference valid for arbitrary amplitude.
2. ``dzhanibekov_growth_rate`` -- the linear-instability rate of rotation about
   the intermediate principal axis (Dzhanibekov / tennis-racket effect),
   ``lambda = |w2| sqrt((I3 - I2)(I2 - I1) / (I1 I3))`` for ``I1 < I2 < I3``.
   Used with an explicit small-amplitude window so the linearisation holds.
3. ``rotation_invariants`` -- exact first integrals of the torque-free problem:
   rotational kinetic energy ``T = 1/2 w.I.w``, body-frame angular momentum
   magnitude ``|I w|``, its world-frame image ``R(q) I w`` (a constant vector),
   and ``|q| = 1``.  These are drift instruments, not fitted quantities.

Conventions: SI units, ``omega_body`` in the body frame (rad/s), quaternion
``(w, x, y, z)``, Euler equations ``I dw/dt = M - w x (I w)``.

All comparisons go through :func:`compare_symmetric_top`, which applies the
*same* metrics to the analytic solution itself, so a reported error is always
like-for-like rather than a fitted constant.  ``falsify_reference_frequency``
shifts the reference rate by a known amount so tests can pin that the gate
actually discriminates (a gate that cannot fail is not a gate).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .sixdof import SixDOFBody, quaternion_to_rotation_matrix
from .sixdof_common import RigidBodyState, rigid_body_step

__all__ = [
    "euler_symmetric_top_solution",
    "dzhanibekov_growth_rate",
    "rotation_invariants",
    "integrate_torque_free",
    "compare_symmetric_top",
    "fit_growth_rate",
    "band_summary",
    "falsify_reference_frequency",
    "convergence_study",
]


# --------------------------------------------------------------------------- #
# Analytic references (no LBM / integrator code path used)
# --------------------------------------------------------------------------- #
def euler_symmetric_top_solution(
    omega0: np.ndarray | tuple[float, float, float],
    inertia: np.ndarray | tuple[float, float, float],
    times: np.ndarray,
    *,
    omega_shift: float = 0.0,
) -> np.ndarray:
    """Exact body-frame angular rate of a symmetric top with ``M = 0``.

    Args:
        omega0: initial body-frame rate ``(3,)``.
        inertia: principal moments ``(I1, I2, I3)`` with ``I1 == I2``.
        times: sample times (``t = 0`` must be included for the exact match).
        omega_shift: artificial frequency offset (rad/s), used only by the
            falsification helper to prove the metric can fail.

    Returns:
        Array ``(len(times), 3)`` of ``omega_body(t)``.

    Raises:
        ValueError: if ``I1 != I2`` (the closed form requires a symmetric top).
    """
    i1, i2, i3 = (float(v) for v in inertia)
    if abs(i1 - i2) > 1e-12 * max(1.0, abs(i1)):
        raise ValueError("euler_symmetric_top_solution requires I1 == I2 (symmetric top).")
    w1_0, w2_0, w3_0 = (float(v) for v in omega0)
    omega_rate = (i1 - i3) * w3_0 / i1 + float(omega_shift)

    t = np.asarray(times, dtype=float)
    cos = np.cos(omega_rate * t)
    sin = np.sin(omega_rate * t)
    # w1 = A cos + B sin, w2 = -A sin + B cos  (A = w1_0, B = w2_0)
    w1 = w1_0 * cos + w2_0 * sin
    w2 = -w1_0 * sin + w2_0 * cos
    w3 = np.full_like(t, w3_0)
    return np.stack([w1, w2, w3], axis=-1)


def dzhanibekov_growth_rate(
    inertia: np.ndarray | tuple[float, float, float],
    spin_about_intermediate: float,
) -> float:
    """Linear growth rate of the intermediate-axis (tennis-racket) instability.

    Valid for ``I1 < I2 < I3`` and a perturbation small compared with the
    intermediate-axis spin; returns ``omega`` (rad/s) of the exponential growth
    of the two off-axis rate components.
    """
    i1, i2, i3 = (float(v) for v in inertia)
    if not (i1 < i2 < i3):
        raise ValueError("Dzhanibekov instability requires I1 < I2 < I3.")
    return abs(spin_about_intermediate) * math.sqrt((i3 - i2) * (i2 - i1) / (i1 * i3))


def rotation_invariants(
    quat: torch.Tensor,
    omega_body: torch.Tensor,
    inertia_diagonal: np.ndarray | tuple[float, float, float],
) -> dict[str, float]:
    """Return the exact first integrals of torque-free rotation.

    ``energy``: ``T = 1/2 w.(I w)``; ``momentum_body``: ``|I w|``;
    ``momentum_world``: norm of ``R(q) (I w)`` (constant vector in the world
    frame); ``quat_norm``: ``|q|``.
    """
    i = torch.as_tensor(np.asarray(inertia_diagonal, dtype=float), dtype=torch.float64)
    w = torch.as_tensor(omega_body, dtype=torch.float64)
    Iw = i * w
    R = quaternion_to_rotation_matrix(torch.as_tensor(quat, dtype=torch.float64))
    L_world = R @ Iw
    return {
        "energy": float(0.5 * (w * Iw).sum().item()),
        "momentum_body": float(Iw.norm().item()),
        "momentum_world": float(L_world.norm().item()),
        "quat_norm": float(torch.as_tensor(quat, dtype=torch.float64).norm().item()),
    }


# --------------------------------------------------------------------------- #
# Library-path integration (public API only)
# --------------------------------------------------------------------------- #
def integrate_torque_free(
    inertia: np.ndarray | tuple[float, float, float],
    omega0: np.ndarray | tuple[float, float, float],
    *,
    dt: float,
    n_steps: int,
) -> dict[str, object]:
    """Integrate a torque-free body through :func:`rigid_body_step`.

    Returns dict with ``times`` (``(n_steps+1,)``), ``omega`` (``(n_steps+1, 3)``),
    ``quat`` (``(n_steps+1, 4)``) and per-step ``invariants`` (list of dicts).
    Gravity is zero so the configuration is genuinely torque-free.
    """
    i1, i2, i3 = (float(v) for v in inertia)
    body = SixDOFBody(
        mass=1.0, ixx=i1, iyy=i2, izz=i3, gravity=(0.0, 0.0, 0.0)
    )
    state = RigidBodyState(
        pos=torch.zeros(3, dtype=torch.float64),
        vel=torch.zeros(3, dtype=torch.float64),
        quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        omega_body=torch.tensor([float(v) for v in omega0], dtype=torch.float64),
    )
    zero_force = torch.zeros(6, dtype=torch.float64)

    times = [0.0]
    omegas = [state.omega_body.numpy().copy()]
    quats = [state.quat.numpy().copy()]
    invariants = [rotation_invariants(state.quat, state.omega_body, inertia)]
    for step in range(1, int(n_steps) + 1):
        state = rigid_body_step(state, zero_force, dt, body=body)
        times.append(step * dt)
        omegas.append(state.omega_body.numpy().copy())
        quats.append(state.quat.numpy().copy())
        invariants.append(rotation_invariants(state.quat, state.omega_body, inertia))
    return {
        "times": np.asarray(times),
        "omega": np.asarray(omegas),
        "quat": np.asarray(quats),
        "invariants": invariants,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compare_symmetric_top(
    omega: np.ndarray,
    times: np.ndarray,
    omega0: np.ndarray | tuple[float, float, float],
    inertia: np.ndarray | tuple[float, float, float],
    *,
    omega_shift: float = 0.0,
) -> dict[str, float]:
    """Relate an integrated rate history to the exact symmetric-top solution.

    Reported quantities:

    * ``rel_l2``        -- ``||w - w_exact||_2 / ||w_exact||_2`` over the window.
    * ``rel_inf``       -- ``max|w - w_exact| / max|w_exact|``.
    * ``axial_rel_drift`` -- max relative deviation of the axial component
      (which is *exactly* constant analytically).
    * ``amplitude_rel_err`` -- fitted transverse amplitude vs the analytic
      ``sqrt(w1_0^2 + w2_0^2)``.
    * ``period_rel_err`` -- period recovered from the sign changes of ``w1``
      vs the analytic ``2*pi/|Omega|`` (``nan`` when fewer than two crossings).
    """
    w_exact = euler_symmetric_top_solution(omega0, inertia, times, omega_shift=omega_shift)
    diff = np.asarray(omega, dtype=float) - w_exact
    if np.linalg.norm(w_exact) == 0.0:
        raise ValueError("degenerate reference: omega0 is zero")
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(w_exact))
    rel_inf = float(np.abs(diff).max() / np.abs(w_exact).max())

    w3 = np.asarray(omega, dtype=float)[:, 2]
    w3_ref = float(np.asarray(omega0, dtype=float)[2])
    axial_rel_drift = (
        float(np.abs(w3 - w3_ref).max() / abs(w3_ref)) if abs(w3_ref) > 0 else float("nan")
    )

    w1 = np.asarray(omega, dtype=float)[:, 0]
    amp_ref = float(np.hypot(float(omega0[0]), float(omega0[1])))
    amp_got = float(0.5 * (w1.max() - w1.min()))
    amplitude_rel_err = abs(amp_got - amp_ref) / amp_ref if amp_ref > 0 else float("nan")

    i1, _i2, i3 = (float(v) for v in inertia)
    period_ref = 2.0 * math.pi / abs((i1 - i3) * w3_ref / i1) if (i1 - i3) * w3_ref != 0 else float("nan")
    crossings: list[float] = []
    for k in range(1, w1.size):
        if w1[k - 1] == 0.0:
            continue
        if w1[k - 1] * w1[k] < 0.0:
            frac = abs(w1[k - 1]) / (abs(w1[k - 1]) + abs(w1[k]))
            crossings.append(float(times[k - 1] + frac * (times[k] - times[k - 1])))
    period_got = float(2.0 * (crossings[-1] - crossings[0]) / (len(crossings) - 1)) if len(crossings) >= 2 else float("nan")
    period_rel_err = (
        abs(period_got - period_ref) / period_ref
        if len(crossings) >= 2 and not math.isnan(period_ref)
        else float("nan")
    )
    return {
        "rel_l2": rel_l2,
        "rel_inf": rel_inf,
        "axial_rel_drift": axial_rel_drift,
        "amplitude_rel_err": amplitude_rel_err,
        "period_rel_err": period_rel_err,
        "period_ref": period_ref,
        "period_got": period_got,
    }


def fit_growth_rate(
    times: np.ndarray,
    signal: np.ndarray,
    *,
    low: float,
    high: float,
) -> float:
    """Least-squares slope of ``log|signal|`` inside an amplitude band.

    Used for the Dzhanibekov instability, whose off-axis component grows
    roughly exponentially with the analytic rate
    :func:`dzhanibekov_growth_rate`.  The band ``low <= |signal| <= high``
    keeps the fit inside the small-perturbation regime where the linear
    analysis is claimed to hold, and excludes the saturated tumbling phase.

    Note: the band choice matters (the exact solution is cnoidal, not a pure
    exponential), so callers must state the band and treat the rate as a
    semi-quantitative anchor, not an exact identity.
    """
    t = np.asarray(times, dtype=float)
    a = np.abs(np.asarray(signal, dtype=float))
    keep = (a >= float(low)) & (a <= float(high))
    if keep.sum() < 5:
        raise ValueError("not enough samples inside the amplitude band to fit a growth rate")
    slope, _intercept = np.polyfit(t[keep], np.log(a[keep]), 1)
    return float(slope)


def band_summary(times: np.ndarray, signal: np.ndarray, *, low: float, high: float) -> dict[str, float]:
    """Growth-rate fit plus the sample count and the covered time window."""
    t = np.asarray(times, dtype=float)
    a = np.abs(np.asarray(signal, dtype=float))
    keep = (a >= float(low)) & (a <= float(high))
    out = {
        "rate": fit_growth_rate(t, signal, low=low, high=high),
        "n_samples": float(keep.sum()),
        "t_first": float(t[keep].min()) if keep.any() else float("nan"),
        "t_last": float(t[keep].max()) if keep.any() else float("nan"),
    }
    return out


def falsify_reference_frequency(
    omega_shift: float,
    omega0: np.ndarray | tuple[float, float, float],
    inertia: np.ndarray | tuple[float, float, float],
    *,
    dt: float,
    n_steps: int,
) -> dict[str, float]:
    """Metrics for an integrated history scored against a *deliberately wrong* reference.

    A validation gate is only meaningful if it fails on a wrong reference; tests
    assert that the returned ``rel_l2`` is large.
    """
    run = integrate_torque_free(inertia, omega0, dt=dt, n_steps=n_steps)
    return compare_symmetric_top(
        run["omega"], run["times"], omega0, inertia, omega_shift=omega_shift
    )


def convergence_study(
    inertia: np.ndarray | tuple[float, float, float],
    omega0: np.ndarray | tuple[float, float, float],
    *,
    dts: list[float],
    t_end: float,
) -> dict[str, object]:
    """Refine ``dt`` at fixed ``t_end`` and return first-order convergence evidence.

    Returns ``rows`` (per-``dt`` metrics), ``observed_order`` from a log-log fit
    of ``rel_l2`` against ``dt``, and ``finest`` (metrics at the smallest dt).
    """
    rows = []
    for dt in dts:
        n_steps = int(round(t_end / dt))
        run = integrate_torque_free(inertia, omega0, dt=dt, n_steps=n_steps)
        metrics = compare_symmetric_top(run["omega"], run["times"], omega0, inertia)
        inv = run["invariants"]
        energy0 = inv[0]["energy"]
        drift = max(abs(row["energy"] - energy0) / energy0 for row in inv)
        rows.append(
            {
                "dt": dt,
                "n_steps": n_steps,
                "rel_l2": metrics["rel_l2"],
                "rel_inf": metrics["rel_inf"],
                "axial_rel_drift": metrics["axial_rel_drift"],
                "period_rel_err": metrics["period_rel_err"],
                "energy_drift": float(drift),
            }
        )
    errs = np.array([row["rel_l2"] for row in rows], dtype=float)
    steps = np.array([row["dt"] for row in rows], dtype=float)
    # d(log err)/d(log dt) = p for an O(dt^p) scheme (err decreases with dt).
    order = float(np.polyfit(np.log(steps), np.log(errs), 1)[0])
    return {"rows": rows, "observed_order": order, "finest": rows[-1]}