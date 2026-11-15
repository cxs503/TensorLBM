"""Pure analysis helpers for the *late* (nonlinear, terminal-velocity) phase of
``CASE-RT-INSTABILITY-ATWOOD``.

Why this module exists
----------------------
``REPORT.md`` of this case gates the **linear** growth rate and declares the
nonlinear front velocity *unmeasurable in this configuration* (failure mode F4:
"the interface hits the ``a/lam = 0.60`` guardrail at ``tau ~ 1.9``, while the
pre-declared terminal-velocity window ``tau in [4, 6]`` needs a much taller
box").  Re-reading the driver shows that the guardrail ``MAX_A_FRAC = 0.60`` is
3.4 wavelengths *away* from the wall of a ``layers = 8`` box (4 wavelengths per
side): the run was stopped by an amplitude threshold, not by the box.  F4 is
therefore a driver artefact, not a property of the problem.

This module holds the replacement logic in a solver-free form so it can be unit
tested:

* :func:`wall_clearance_lam` / :func:`late_stop_reason` -- the corrected stop
  criterion (stop when a front is within ``WALL_CLEARANCE_LAM`` of the wall, or
  when the Mach guard trips), replacing the amplitude guard;
* :func:`front_velocity` -- slope of a front trajectory over a ``tau`` window;
* :func:`layzer_velocity` / :func:`effective_c` -- the potential-flow
  (Layzer-Goncharov) reference in **the form that is consistent with the
  Boussinesq equations actually solved**, plus its exact inversion, so that the
  measurement can be reported as a dimensionless geometric constant ``C``
  (comparable with the ``C = 0.90`` measured on the sibling case at A = 1/2).

Unit convention: lengths in lattice units, times in lattice steps, the
dimensionless time is ``tau = t sqrt(A g / lam)`` and the dimensionless front
velocity used for the gate is
``Fr = u / sqrt(A g lam) = 1 / sqrt(pi C)``.
"""

from __future__ import annotations

import math

#: case-pool tolerance of this case (task anchor).
GATE_PCT = 15.0
#: pre-declared terminal-velocity window, identical to the one in ``run_case.py``
#: (``TAU_LO``/``TAU_HI``); inherited, not re-tuned after seeing the numbers.
TAU_LO, TAU_HI = 4.0, 6.0
#: a run stops when a front comes closer than this to the wall (in wavelengths).
WALL_CLEARANCE_LAM = 1.5
#: Mach guard (unchanged from ``run_case.py``).
MAX_MA = 0.85
#: 2-D potential-flow geometric constant (see ``ref/ref_lock.py``).
C_2D = 1.0


# ---------------------------------------------------------------------------
# corrected stop criterion
# ---------------------------------------------------------------------------
def wall_clearance_lam(layers: float) -> float:
    """Largest front displacement from the mid-plane that still leaves
    ``WALL_CLEARANCE_LAM`` wavelengths of fluid before the wall."""
    return 0.5 * float(layers) - WALL_CLEARANCE_LAM


def late_stop_reason(front_lam: float, ma: float, layers: float,
                     max_ma: float = MAX_MA) -> str | None:
    """Return why the run must stop, or ``None`` to continue.

    ``front_lam`` is ``max(h_spike, h_bubble) / lam``.  Note there is **no**
    amplitude threshold any more: a single-mode RT front at ``h/lam ~ 0.6`` is
    still 2.9 wavelengths from the wall of a ``layers = 8`` box and is exactly
    the regime this study must reach.
    """
    if not (math.isfinite(front_lam) and math.isfinite(ma)):
        return "nonfinite"
    if ma > max_ma:
        return "mach"
    if front_lam >= wall_clearance_lam(layers):
        return "wall"
    return None


# ---------------------------------------------------------------------------
# trajectory analysis
# ---------------------------------------------------------------------------
def linear_fit(xs, ys) -> tuple[float, float, float]:
    """Least-squares ``y = a + b x``; returns ``(b, a, r2)``."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return b, a, r2


def front_velocity(taus, fronts, lo: float = TAU_LO, hi: float = TAU_HI) -> dict:
    """Drift of a front over ``tau in [lo, hi]``.

    Returns the slope in *lattice lengths per unit tau* (``slope``), its
    correlation coefficient, the sample count and the window actually covered.
    """
    pts = [(t, f) for t, f in zip(taus, fronts)
           if lo <= t <= hi and math.isfinite(f)]
    if len(pts) < 3:
        return {"slope": float("nan"), "r2": float("nan"), "n": len(pts),
                "tau_range": [float("nan")] * 2, "front_range": [float("nan")] * 2}
    s, _b, r2 = linear_fit([p[0] for p in pts], [p[1] for p in pts])
    return {"slope": s, "r2": r2, "n": len(pts),
            "tau_range": [min(p[0] for p in pts), max(p[0] for p in pts)],
            "front_range": [min(p[1] for p in pts), max(p[1] for p in pts)]}


def lattice_velocity(slope_per_tau: float, atwood: float, g: float, lam: float) -> float:
    """``u = d h/dt`` in lattice units from ``slope = d h/dtau``.

    ``tau = t sqrt(A g / lam)`` so ``dt = dtau sqrt(lam / (A g))``.
    """
    if not math.isfinite(slope_per_tau):
        return float("nan")
    return slope_per_tau * math.sqrt(atwood * g / lam)


# ---------------------------------------------------------------------------
# reference velocity in the form consistent with the solved equations
# ---------------------------------------------------------------------------
def layzer_velocity(atwood: float, g: float, lam: float, c: float = C_2D) -> float:
    """Potential-flow (Layzer-Goncharov) saturated front velocity **in the
    Boussinesq limit** ``(1 -+ A) -> 1``.

    The two-fluid model gives ``u = sqrt(2 A g / (C k (1 -+ A)))`` for the
    spike/bubble, where the ``(1 -+ A)`` factors are the *variable inertia* of
    the two fluids.  The Boussinesq equations solved here have uniform inertia
    by construction, which is the ``A -> 0`` limit of that factor -- so the
    reference that is consistent with the equations being solved is

        u_LZ = sqrt(2 A g / (C k)),   Fr = u_LZ / sqrt(A g lam) = 1/sqrt(pi C).

    The *finite-A* two-fluid asymmetry is reported separately and honestly as
    NOT SUPPORTED (see :func:`two_fluid_asymmetry`).
    """
    k = 2.0 * math.pi / lam
    return math.sqrt(2.0 * atwood * g / (c * k))


def layzer_froude(atwood: float, g: float, lam: float, c: float = C_2D) -> float:
    """``u_LZ / sqrt(A g lam)``; equals ``1/sqrt(pi C)``, i.e. independent of
    ``A``, ``g`` and ``lam`` -- a clean dimensionless target."""
    return layzer_velocity(atwood, g, lam, c) / math.sqrt(atwood * g * lam)


def effective_c(atwood: float, g: float, lam: float, u: float) -> float:
    """Invert :func:`layzer_velocity` for ``C``: ``C = 2 A g / (k u^2)``."""
    if not (math.isfinite(u) and u > 0.0):
        return float("nan")
    return 2.0 * atwood * g / ((2.0 * math.pi / lam) * u * u)


def two_fluid_asymmetry(atwood: float) -> float:
    """``u_spike / u_bubble`` of the two-fluid potential-flow model,
    ``sqrt((1 + A) / (1 - A))`` -- a factor ``sqrt(3) = 1.732`` at A = 1/2."""
    return math.sqrt((1.0 + atwood) / (1.0 - atwood))


def rel_pct(value: float, ref: float) -> float:
    """Signed relative deviation in percent (``nan`` when undefined)."""
    if not (math.isfinite(value) and math.isfinite(ref)) or ref == 0.0:
        return float("nan")
    return 100.0 * (value / ref - 1.0)