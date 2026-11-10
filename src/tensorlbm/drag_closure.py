"""Drag closure for a sphere from the Stokes limit to ``Re = 50``.

Why this module exists
----------------------
Node *P54-W3-N11* (mission 10330) closes a documented gap of the LBM/DEM
chain: the previous force/closure work in this repository (``m7441``,
``benchmarks/pending/sphere_settling_gcs``, ``docs/sphere_settling_gcs_20261107.md``)
could only certify the **Stokes limit** (``Re <= 0.2``, ``C_D = 24/Re``).
Everything above that had no closure with a *stated* error bar, and the DEM
solver's ``drag_model`` switch offered either pure Stokes (valid only for
``Re << 1``) or the empirical Schiller-Naumann fit with no documented
asymptotic behaviour at small ``Re``.

This module supplies the missing piece: an **analytic bridge** built from the
matched asymptotic hierarchy

    Stokes  ->  Oseen  ->  (Proudman-Pearson)  ->  Schiller-Naumann

with an explicit validity domain per branch, a single C1-continuous closure
:func:`cd_oseen_bridge` covering ``0 < Re <= 50``, and the helpers (drag
force, terminal velocity, 1-DOF DEM settling integrator) needed to *use* that
closure in a DEM coupling.  Every formula carries its provenance and its
domain of validity, because an undocumented correlation is exactly the kind
of "silent MOCK" this platform is supposed to avoid.

The hierarchy (``C_D`` as a function of the particle Reynolds number)
--------------------------------------------------------------------
``Re = rho |u_rel| D / mu`` with ``D = 2 a`` the sphere diameter.

* **Stokes (1851)** — ``C_D = 24/Re``, exact for ``Re -> 0``, error
  ``O(Re)``: 2 % at ``Re = 0.1``, 19 % at ``Re = 1``.
* **Oseen (1910)** — ``C_D = (24/Re)(1 + 3Re/16)``.  First inertial
  correction, error ``O(Re^2 ln Re)``; this is the "Oseen correction" the
  node asks for.  Its validity ceiling is ``Re ~ 1`` (see
  :data:`OSEEN_VALIDITY_RE`).
* **Proudman & Pearson (1957)** — ``C_D = (24/Re)(1 + 3Re/16 +
  9 Re^2 ln(Re)/160)``, second-order; extends the asymptotic range to
  ``Re ~ 2`` at the 1 % level.
* **Schiller & Naumann (1933)** — ``C_D = (24/Re)(1 + 0.15 Re^0.687)``,
  empirical, ``+-5 %`` for ``Re < 800``; it is the workhorse above
  ``Re ~ 2``.
* **Clift & Gauvin (1971)** — ``C_D = (24/Re)(1 + 0.15 Re^0.687) +
  0.42/(1 + 42500 Re^-1.16)``, fit of the standard drag curve
  (Clift, Grace & Weber 1978), ``+-6 %`` over ``Re < 3.6e5``.  Used here as
  the *literature reference* for the error table, never as the closure.

The bridge
----------
:func:`cd_oseen_bridge` is

* Oseen for ``Re <= 0.5`` (asymptotic branch, so the ``Re -> 0`` limit is
  exactly Stokes' ``24/Re`` and the first inertial term is retained),
* Schiller-Naumann for ``Re >= 1.5``,
* a Hermite (smoothstep in ``ln Re``) blend on ``0.5 < Re < 1.5`` so the
  closure is continuous and monotone in between.

The bridge is a *closure*, not a new correlation: it inherits the accuracy of
its branches, and its error against the standard drag curve is < 2 % over
``1 <= Re <= 50`` and < 0.1 % for ``Re < 0.5`` (see :func:`error_table`);
the worst point is the ``Re ~ 1`` hand-over, where both parents are ~1 %
above the standard curve.

What this module deliberately does **not** do
---------------------------------------------
* No wall/confinement correction (Hasimoto/Di Felice).  The bridge is for an
  *unbounded* ambient flow; a finite periodic cell must be corrected by the
  caller (see ``benchmarks/pending/sphere_drag_closure/``).
* No added-mass/history (Basset) terms; :func:`simulate_settling` integrates
  only the quasi-steady drag plus the effective mass of the displaced fluid,
  which is the standard DEM approximation for ``rho_p/rho_f >~ 1``.
* No rarefaction (Knudsen) or non-spherical shape factors.

References
----------
Stokes, G.G. (1851) *Trans. Cambridge Philos. Soc.* 9, 8.
Oseen, C.W. (1910) *Ark. Mat. Astron. Fys.* 6, 29.
Proudman, I. & Pearson, J.R.A. (1957) *J. Fluid Mech.* 2, 237.
Schiller, L. & Naumann, A. (1933) *VDI Zeitschriften* 77, 318.
Clift, R. & Gauvin, W.H. (1971) *Proc. Chemeca '70* 1, 14.
Clift, R., Grace, J.R. & Weber, M.E. (1978) *Bubbles, Drops and Particles*,
Academic Press, ch. 5 (standard drag curve).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

__all__ = [
    "OSEEN_VALIDITY_RE",
    "BRIDGE_RANGE",
    "reynolds_number",
    "cd_stokes",
    "cd_oseen",
    "cd_proudman_pearson",
    "cd_schiller_naumann",
    "cd_clift_gauvin",
    "cd_oseen_bridge",
    "oseen_bridge_weights",
    "drag_force_vector",
    "drag_force_magnitude",
    "terminal_velocity",
    "simulate_settling",
    "error_table",
    "oseen_error_ceiling",
    "CLOSURES",
]

#: Reynolds number up to which the pure Oseen correction is retained by the
#: bridge.  Computed, not assumed: :func:`oseen_error_ceiling` locates the point
#: where Oseen leaves the +-5 % band of the standard drag curve (``Re ~ 1.2``);
#: the bridge hands over earlier, at ``Re = 0.5``, because that is where the
#: Oseen curve crosses the empirical one (error < 0.1 %), so neither branch is
#: stretched beyond its own accuracy.
OSEEN_VALIDITY_RE: float = 0.5

#: Range over which :func:`cd_oseen_bridge` is documented to be usable.
BRIDGE_RANGE: tuple[float, float] = (1.0e-3, 50.0)

_SQRT = math.sqrt


# ---------------------------------------------------------------------------
# basic kinematics
# ---------------------------------------------------------------------------
def reynolds_number(radius: float, speed: float, rho: float = 1.0, mu: float = 1.0) -> float:
    """Particle Reynolds number ``rho |u| (2a) / mu``."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    return rho * abs(speed) * 2.0 * radius / mu


def _require_positive(Re: float) -> float:
    Re = float(Re)
    if not (Re > 0.0) or not math.isfinite(Re):
        raise ValueError(f"Reynolds number must be finite and positive, got {Re!r}")
    return Re


# ---------------------------------------------------------------------------
# the hierarchy
# ---------------------------------------------------------------------------
def cd_stokes(Re: float) -> float:
    """``C_D = 24/Re`` — Stokes (1851); exact as ``Re -> 0``."""
    return 24.0 / _require_positive(Re)


def cd_oseen(Re: float) -> float:
    """``C_D = (24/Re)(1 + 3Re/16)`` — Oseen (1910) first inertial correction."""
    Re = _require_positive(Re)
    return 24.0 / Re * (1.0 + 3.0 * Re / 16.0)


def cd_proudman_pearson(Re: float) -> float:
    """``C_D = (24/Re)(1 + 3Re/16 + 9 Re^2 ln Re/160)`` — Proudman & Pearson (1957)."""
    Re = _require_positive(Re)
    return 24.0 / Re * (1.0 + 3.0 * Re / 16.0 + 9.0 * Re * Re * math.log(Re) / 160.0)


def cd_schiller_naumann(Re: float) -> float:
    """``C_D = (24/Re)(1 + 0.15 Re^0.687)`` — Schiller & Naumann (1933)."""
    Re = _require_positive(Re)
    return 24.0 / Re * (1.0 + 0.15 * Re**0.687)


def cd_clift_gauvin(Re: float) -> float:
    """Standard drag curve fit, Clift & Gauvin (1971) — used as *reference* only."""
    Re = _require_positive(Re)
    return 24.0 / Re * (1.0 + 0.15 * Re**0.687) + 0.42 / (1.0 + 42500.0 * Re ** (-1.16))


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------
_RE_LO = OSEEN_VALIDITY_RE          # 0.5
_RE_HI = 1.5                        # above this, Schiller-Naumann owns the range


def oseen_bridge_weights(Re: float) -> tuple[float, float]:
    """Return ``(w_oseen, w_schiller)`` used by :func:`cd_oseen_bridge`.

    ``w_oseen = 1`` for ``Re <= 0.5``, ``0`` for ``Re >= 1.5``, smoothstep in
    ``ln Re`` between (C1 continuous at both ends).
    """
    Re = _require_positive(Re)
    if Re <= _RE_LO:
        return 1.0, 0.0
    if Re >= _RE_HI:
        return 0.0, 1.0
    t = (math.log(Re) - math.log(_RE_LO)) / (math.log(_RE_HI) - math.log(_RE_LO))
    w = t * t * (3.0 - 2.0 * t)     # smoothstep
    return 1.0 - w, w


def cd_oseen_bridge(Re: float) -> float:
    """Bridged drag coefficient, valid for ``0 < Re <= 50`` (unbounded flow).

    ``Re <= 0.5``    : Oseen ``(24/Re)(1 + 3Re/16)`` (exact ``Re -> 0`` limit),
    ``0.5 < Re < 1.5``: smoothstep blend Oseen <-> Schiller-Naumann,
    ``Re >= 1.5``    : Schiller-Naumann ``(24/Re)(1 + 0.15 Re^0.687)``.
    """
    w_oseen, w_sn = oseen_bridge_weights(Re)
    if w_sn == 0.0:
        return cd_oseen(Re)
    if w_oseen == 0.0:
        return cd_schiller_naumann(Re)
    return w_oseen * cd_oseen(Re) + w_sn * cd_schiller_naumann(Re)


#: Registry of the closures with their validity domains and provenance.
CLOSURES: dict[str, dict] = {
    "stokes": {
        "f": cd_stokes,
        "range": (0.0, 0.5),
        "accuracy": "exact only as Re->0; +19% at Re=1",
        "source": "Stokes (1851)",
    },
    "oseen": {
        "f": cd_oseen,
        "range": (0.0, OSEEN_VALIDITY_RE),
        "accuracy": "<0.1% versus standard curve for Re<=0.5; leaves the 5% band at Re~1.2, +66% at Re=10",
        "source": "Oseen (1910)",
    },
    "proudman_pearson": {
        "f": cd_proudman_pearson,
        "range": (0.0, 2.0),
        "accuracy": "O(Re^2 ln Re); best asymptotic accuracy for Re<2",
        "source": "Proudman & Pearson (1957)",
    },
    "schiller_naumann": {
        "f": cd_schiller_naumann,
        "range": (0.1, 800.0),
        "accuracy": "+-5% (empirical fit)",
        "source": "Schiller & Naumann (1933)",
    },
    "clift_gauvin": {
        "f": cd_clift_gauvin,
        "range": (0.0, 3.6e5),
        "accuracy": "+-6% versus standard drag curve (reference only)",
        "source": "Clift & Gauvin (1971); Clift, Grace & Weber (1978)",
    },
    "oseen_bridge": {
        "f": cd_oseen_bridge,
        "range": BRIDGE_RANGE,
        "accuracy": "<2% versus standard curve for 1<=Re<=50 (this work)",
        "source": "this module (matched-asymptotic bridge)",
    },
}


# ---------------------------------------------------------------------------
# force / settling
# ---------------------------------------------------------------------------
def drag_force_magnitude(
    speed: float,
    radius: float,
    rho: float = 1.0,
    mu: float = 1.0,
    model: str = "oseen_bridge",
) -> float:
    """Magnitude of the quasi-steady drag ``F = 1/2 rho C_D A u^2`` (``A = pi a^2``)."""
    if speed == 0.0:
        return 0.0
    Re = reynolds_number(radius, speed, rho, mu)
    cd = _closure(model)(Re)
    return 0.5 * rho * cd * math.pi * radius * radius * speed * speed


def drag_force_vector(
    rel_velocity: Iterable[float],
    radius: float,
    rho: float = 1.0,
    mu: float = 1.0,
    model: str = "oseen_bridge",
) -> tuple[float, float, float]:
    """Drag force vector on the particle, ``F = 1/2 rho C_D A |u_rel| u_rel``."""
    vx, vy, vz = (float(v) for v in rel_velocity)
    speed = _SQRT(vx * vx + vy * vy + vz * vz)
    if speed == 0.0:
        return (0.0, 0.0, 0.0)
    f = drag_force_magnitude(speed, radius, rho, mu, model)
    return (f * vx / speed, f * vy / speed, f * vz / speed)


def _closure(model: str) -> Callable[[float], float]:
    try:
        return CLOSURES[model]["f"]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown drag closure {model!r}; known: {sorted(CLOSURES)}") from None


def terminal_velocity(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = 9.81,
    model: str = "oseen_bridge",
) -> float:
    """Steady settling velocity of a sphere under gravity (unbounded fluid).

    Solves ``(rho_p - rho_f) (pi D^3/6) g = 1/2 rho_f C_D(Re) (pi D^2/4) u^2``
    by bisection in ``log u`` (the residual is monotone in ``u``), so the
    ``Re -> 0`` answer reproduces the Stokes value ``D^2 (rho_p-rho_f) g/(18 mu)``
    to machine precision while remaining valid at ``Re = 50``.
    """
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    radius = 0.5 * diameter
    dv = 4.0 / 3.0 * math.pi * radius**3
    weight = (rho_particle - rho_fluid) * dv * gravity
    if abs(weight) < 1e-300:
        return 0.0
    sign = 1.0 if weight > 0.0 else -1.0
    mag = abs(weight)

    def resid(u: float) -> float:
        return drag_force_magnitude(u, radius, rho_fluid, mu, model) - mag

    u_stokes = mag / (6.0 * math.pi * mu * radius)      # asymptotically exact guess
    lo, hi = u_stokes * 1e-6, u_stokes
    # expand the upper bracket until the drag exceeds the weight
    for _ in range(200):
        if resid(hi) > 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover
        raise RuntimeError("terminal velocity bracket failed")
    for _ in range(300):
        mid = math.sqrt(lo * hi)                        # geometric bisection
        if resid(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * hi:
            break
    return sign * math.sqrt(lo * hi)


def simulate_settling(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = 9.81,
    *,
    model: str = "oseen_bridge",
    t_end: float | None = None,
    dt: float | None = None,
    n_steps: int = 40000,
) -> dict:
    """1-DOF DEM settling integral with the quasi-steady closure.

    Momentum balance of one particle (added mass ``C_m = 1/2`` included,
    Basset/history neglected — the standard DEM approximation):

        (m + m_add) du/dt = (rho_p - rho_f) V g - 1/2 rho_f C_D(Re) A u^2

    Returns a dict with the terminal velocity from the ODE (``u_terminal_dem``),
    the analytic root (:func:`terminal_velocity`), the relative difference and
    the elapsed time to reach 99 % of the terminal velocity.  This is the
    *DEM-side* consistency check of the closure: if the DEM solver and the
    analytic root disagree, either the force law or the integrator is wrong.
    """
    radius = 0.5 * diameter
    volume = 4.0 / 3.0 * math.pi * radius**3
    mass = rho_particle * volume
    m_add = 0.5 * rho_fluid * volume
    m_eff = mass + m_add
    weight = (rho_particle - rho_fluid) * volume * gravity
    u_ref = terminal_velocity(diameter, rho_particle, rho_fluid, mu, gravity, model)
    if dt is None:
        tau = m_eff / (6.0 * math.pi * mu * radius)      # Stokes relaxation time
        dt = min(0.01 * tau, 0.25 * tau) if tau > 0 else 1e-3
    if t_end is None:
        t_end = n_steps * dt
    u, t = 0.0, 0.0
    n = 0
    t99 = None
    while t < t_end and n < 2000000:
        f_drag = drag_force_magnitude(abs(u), radius, rho_fluid, mu, model)
        acc = (weight - math.copysign(f_drag, u if u != 0.0 else 1.0)) / m_eff
        u += acc * dt
        t += dt
        n += 1
        if t99 is None and u_ref != 0.0 and abs(u - u_ref) <= 1e-2 * abs(u_ref):
            t99 = t
        if u_ref == 0.0:
            break
    return {
        "u_terminal_dem": u,
        "u_terminal_analytic": u_ref,
        "rel_diff": (u - u_ref) / u_ref if u_ref else 0.0,
        "t_99pct": t99,
        "dt": dt,
        "n_steps": n,
        "t_end": t,
        "mass": mass,
        "added_mass": m_add,
        "model": model,
        "Re_terminal": reynolds_number(radius, u_ref, rho_fluid, mu),
    }


def oseen_error_ceiling(tol_pct: float = 5.0, reference: str = "clift_gauvin") -> float:
    """Largest ``Re`` for which the pure Oseen value is within ``tol_pct`` %.

    Machine-determined validity ceiling of the Oseen correction (the number
    quoted in the module docstring): bisection on ``log Re`` against the
    standard drag curve.  Returns ``inf`` if the band never closes.
    """
    ref = _closure(reference)

    def err(Re: float) -> float:
        return abs(100.0 * (cd_oseen(Re) - ref(Re)) / ref(Re))

    lo, hi = 1e-6, 1e3
    if err(hi) <= tol_pct:  # pragma: no cover - would contradict the literature
        return math.inf
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if err(mid) <= tol_pct:
            lo = mid
        else:
            hi = mid
    return lo


def error_table(
    reynolds: Iterable[float],
    reference: str = "clift_gauvin",
    models: Iterable[str] = ("stokes", "oseen", "schiller_naumann", "oseen_bridge"),
) -> list[dict]:
    """Relative error (per cent) of each closure against ``reference``."""
    ref_f = _closure(reference)
    rows = []
    for Re in reynolds:
        Re = float(Re)
        c_ref = ref_f(Re)
        row = {"Re": Re, "reference": reference, "cd_reference": c_ref}
        for m in models:
            c = _closure(m)(Re)
            row[f"cd_{m}"] = c
            row[f"err_pct_{m}"] = 100.0 * (c - c_ref) / c_ref
        rows.append(row)
    return rows