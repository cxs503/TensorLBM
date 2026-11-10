"""Incipient-motion (critical Shields) probe used for criterion C1.

What is measured
----------------
A weakly perturbed bed (sinusoidal, amplitude ``amp_frac * d``) is exposed to a
uniform flow whose base Shields number is ``theta_base``.  The local shear
responds to the local water depth through the standard *locally uniform flow*
scaling

    u*(x) = u*_base ( h_ref / h(x) )^{1/2},   h(x) = h_ref - (z_b - z_ref),
    theta(x) = theta_base ( h_ref / h(x) )^3,

so the crests of the perturbation feel a slightly larger Shields number and
erode first.  The bed is then evolved with the Exner/Meyer-Peter-Mueller model
(:mod:`tensorlbm.sediment.exner`, whose flux is *exactly* zero for
``theta <= theta_cr``) over a fixed observation window ``T``.  The **numerical
onset Shields number** is the value of ``theta_base`` at which the maximum bed
change over ``T`` reaches the detection level ``detect_frac * d``; it is found
by bisection on ``theta_base`` in the fixed physical bracket ``[1e-4, 1.0]``
(the bracket never uses the reference curve).

Honest scope statement
----------------------
Because the transport law contains the incipient-motion parameter
``theta_cr``, the measured onset cannot be independent of it: any model that
needs a threshold to produce (or suppress) transport returns that same
threshold.  C1 therefore verifies the **implementation chain**
``d -> d* -> theta_cr -> gate -> Exner response`` in physical units, against a
reference curve evaluated independently in
:mod:`tensorlbm.sediment_benchmarks`, in the detectability limit.  It is *not*
an independent measurement of the Soulsby-Whitehouse empiricism -- that is
stated as a limitation in the report.  The check does have discriminating
power: a wrong ``d*`` definition, a wrong regime switch or a unit slip moves
the measured onset outside the 15 % band (demonstrated numerically by
``old_d_star_rel_err`` in the verification output, which recomputes the chain
with the dimensionally inconsistent ``d* = d sqrt((s-1)g)/nu`` that this
mission removed: the deviation becomes a factor of several).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exner import ExnerConfig, exner_step, mpm_flux, stable_dt

__all__ = ["MorphoConfig", "bed_response", "measure_onset_shields"]


@dataclass(frozen=True)
class MorphoConfig:
    """Parameters of the 1-D morphodynamic onset probe."""

    d: float  # grain diameter (m)
    s: float = 1.65
    g: float = 9.81
    porosity: float = 0.4
    h_ref: float = 1.0  # reference water depth (m)
    n_waves: int = 2  # perturbation wavelengths in the domain
    domain_len: float = 20.0  # domain length (m)
    n_cells: int = 400
    amp_frac: float = 0.02  # perturbation amplitude / d
    obs_time: float = 1800.0  # observation window (s)
    n_steps: int = 240
    detect_frac: float = 0.01  # detection level / d

    def __post_init__(self) -> None:
        if self.d <= 0.0:
            raise ValueError("require d > 0")
        if self.n_cells < 16:
            raise ValueError("require n_cells >= 16")


def _bed(mc: MorphoConfig) -> tuple[np.ndarray, float]:
    dx = mc.domain_len / mc.n_cells
    x = (np.arange(mc.n_cells) + 0.5) * dx
    z = mc.amp_frac * mc.d * np.sin(2.0 * np.pi * mc.n_waves * x / mc.domain_len)
    return z, dx


def bed_response(theta_base: float, mc: MorphoConfig, theta_cr: float) -> float:
    """Max bed change (m) over the observation window for ``theta_base``.

    The bed evolves by the conservative Exner update with the MPM flux at the
    local Shields number ``theta(x) = theta_base (h_ref/h(x))^3``.  The time
    step is CFL-limited (:func:`tensorlbm.sediment.exner.stable_dt`) so the
    run cannot drive the depth negative.
    """
    z_b, dx = _bed(mc)
    z0 = z_b.copy()
    cfg = ExnerConfig(d=mc.d, s=mc.s, g=mc.g, porosity=mc.porosity, theta_cr=theta_cr)
    h_ref = mc.h_ref

    def theta_of(z: np.ndarray) -> np.ndarray:
        h = np.maximum(h_ref - z, 0.25 * h_ref)
        return theta_base * (h_ref / h) ** 3

    t = 0.0
    while t < mc.obs_time:
        dt = stable_dt(z_b, dx, cfg, theta_of, cfl=0.2)
        dt = min(dt, mc.obs_time - t)
        z_b = exner_step(z_b, mpm_flux(theta_of(z_b), cfg), dx, dt, cfg)
        t += dt
        if not np.all(np.isfinite(z_b)):  # pragma: no cover - defensive
            raise FloatingPointError("bed elevation went non-finite")
    return float(np.max(np.abs(z_b - z0)))


def measure_onset_shields(
    mc: MorphoConfig,
    theta_cr: float,
    lo: float = 1.0e-4,
    hi: float = 1.0,
    iters: int = 30,
    tol: float = 1.0e-4,
) -> dict:
    """Bisect the base Shields number for the incipient-motion detection level.

    The bracket ``[lo, hi]`` is fixed in physical units and does *not* use the
    reference curve.  Returns a dict with the measured onset, the detector
    values at the bracket ends (for auditing) and the number of iterations.
    """
    target = mc.detect_frac * mc.d
    f_lo = bed_response(lo, mc, theta_cr) - target
    f_hi = bed_response(hi, mc, theta_cr) - target
    if f_lo > 0.0:
        return {"theta_num": lo, "converged": False, "side": "lo",
                "f_lo": f_lo, "f_hi": f_hi, "iters": 0}
    if f_hi < 0.0:
        return {"theta_num": hi, "converged": False, "side": "hi",
                "f_lo": f_lo, "f_hi": f_hi, "iters": 0}
    n_it = 0
    for n_it in range(1, iters + 1):
        mid = 0.5 * (lo + hi)
        f_mid = bed_response(mid, mc, theta_cr) - target
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol * mid:
            break
    return {
        "theta_num": 0.5 * (lo + hi),
        "converged": True,
        "side": None,
        "f_lo": f_lo,
        "f_hi": f_hi,
        "iters": n_it,
    }