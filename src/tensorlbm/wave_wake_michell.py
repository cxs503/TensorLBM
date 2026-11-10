"""Thin-ship wave-resistance verification face: Michell integral, Kelvin wake
geometry, hump/hollow structure and demihull interference.

Why this module
---------------
Wave-making resistance is the one resistance component that no viscosity model
or empirical regression can supply: it is set by the steady free-surface wave
system the hull drags along.  The closed-form anchor is Michell's thin-ship
integral (Michell 1898), and the scale-free checks it offers are

* **wavenumber selection** ``kappa = k0 sec^2(theta)``, ``k0 = g/U^2``
  (:func:`wave_spectrum`), which fixes the transverse wavelength ``2 pi U^2/g``
  and therefore the wake geometry;
* the **caustic (Kelvin) angle** of the steady wave system
  (:func:`caustic_angle_deg`) -- ``19.4712 deg`` in deep water
  (``sin(alpha) = 1/3``) and ``arcsin(1/Fr_h)`` in the shallow-water limit;
* the **hump/hollow** oscillation of ``C_W(Fn)`` produced by the axial phase
  ``k0 L`` (:func:`michell_cw_curve`, :func:`hump_hollow`);
* the **demihull interference** of a catamaran, which in thin-ship theory is a
  pure phase factor on the hull spectrum (:func:`catamaran_interference`).

Every quantity here is verified inside the module against an *independently
computed* counterpart (analytic derivative vs finite difference, two quadrature
routes, stationary-phase caustic vs pattern envelope), so a downstream campaign
can use it as a reference face for a numerical (LBM free-surface) solver
without inheriting the thin-ship approximation twice.

Conventions
-----------
``x`` longitudinal from midship, ``x in [-L/2, L/2]``; ``z`` vertical with
``z = 0`` at the calm waterline and ``z < 0`` below it; ``y`` transverse, hull
symmetric about ``y = 0``.  Half-breadth ``f(x, z) > 0``, full beam ``b = 2f``.
All quantities SI.

Michell integral as implemented::

    R_W = (4 rho g^2 / (pi U^2)) int_0^{pi/2} sec^3(theta) (P^2 + Q^2) dtheta
    P + i Q = int int f_x exp(k z) exp(i k x cos(theta)) dx dz,  k = k0 sec^2(theta)

``C_W = R_W / (0.5 rho U^2 S)``, ``S`` = wetted surface area (both sides).

Evidence level: REAL for the integral, the quadrature and the wake geometry
(all numerically verified in ``tests/test_wave_wake_michell.py``).  The
thin-ship model itself is first order in beam/length -- see
:func:`slenderness_report` for the admission window, and never quote ``C_W``
outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "G",
    "RHO",
    "ParabolicHull",
    "WIGLEY_STANDARD",
    "WIGLEY_III",
    "michell_resistance",
    "michell_spectrum_analytic",
    "michell_resistance_analytic",
    "michell_cw_curve",
    "hump_hollow",
    "wave_spectrum",
    "caustic_angle_deg",
    "caustic_angle_analytic_deg",
    "onaxis_wavelength",
    "shallow_wake_angle_deg",
    "shallow_ray_envelope_deg",
    "wake_field",
    "measure_wake_angle_deg",
    "catamaran_interference",
    "slenderness_report",
    "wetted_surface_area",
]

G = 9.81
RHO = 1000.0


# ---------------------------------------------------------------------------
# Hull family
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParabolicHull:
    """Parabolic (Wigley-family) hull: ``f = (B/2)(1 - xi^2) g(zeta)``.

    ``xi = 2x/L`` and ``zeta = -z/T`` (``0`` at the waterline, ``1`` at the
    keel).  ``vertical`` selects the vertical distribution ``g``:

    ``"parabolic"``
        ``g = 1 - zeta^2``  ->  ``Cb = 4/9`` (classic Wigley / ITTC benchmark).
    ``"power"``
        ``g = 1 - (1 - zeta)^alpha``  ->  ``Cb = (2/3) alpha/(alpha+1)``
        (``alpha = 4.97`` reproduces the Wigley III ``Cb = 0.555`` used by the
        repository's other Wigley benchmark family).
    """

    length: float
    beam: float
    draft: float
    vertical: str = "parabolic"
    alpha: float = 4.97
    name: str = "wigley"

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.beam <= 0.0 or self.draft <= 0.0:
            raise ValueError("length, beam and draft must be positive")
        if self.vertical not in ("parabolic", "power"):
            raise ValueError("vertical must be 'parabolic' or 'power'")
        if self.vertical == "power" and self.alpha <= 0.0:
            raise ValueError("alpha must be positive for the power profile")

    # -- shape functions ----------------------------------------------------
    def profile(self, zeta):
        """Vertical distribution ``g(zeta)``; ``zeta = -z/T`` in ``[0, 1]``."""
        zeta = np.asarray(zeta, dtype=float)
        if self.vertical == "parabolic":
            return 1.0 - zeta**2
        return 1.0 - (1.0 - zeta) ** self.alpha

    def profile_integral(self) -> float:
        """``int_0^1 g(zeta) dzeta`` (closed form)."""
        if self.vertical == "parabolic":
            return 2.0 / 3.0
        return self.alpha / (self.alpha + 1.0)

    def _inside(self, xi, zeta):
        return (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)

    def half_beam(self, x, z):
        """``f(x, z)``; zero outside the hull box (broadcasting, any shape)."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        xi = 2.0 * x / self.length
        zeta = -z / self.draft
        val = 0.5 * self.beam * (1.0 - xi**2) * self.profile(zeta)
        return np.where(self._inside(xi, zeta), np.maximum(val, 0.0), 0.0)

    def half_beam_dx(self, x, z):
        """Analytic ``df/dx`` (no finite differencing)."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        xi = 2.0 * x / self.length
        zeta = -z / self.draft
        # f = (B/2)(1 - xi^2) g, xi = 2x/L  =>  df/dx = (B/2)(-8x/L^2) g
        dfdx = -0.5 * self.beam * (8.0 * x / self.length**2) * self.profile(zeta)
        return np.where(self._inside(xi, zeta), dfdx, 0.0)

    def half_beam_dz(self, x, z):
        """Analytic ``df/dz``."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        xi = 2.0 * x / self.length
        zeta = -z / self.draft
        if self.vertical == "parabolic":
            dg = -2.0 * zeta
        else:
            dg = self.alpha * (1.0 - zeta) ** (self.alpha - 1.0)
        dfdz = 0.5 * self.beam * (1.0 - xi**2) * dg * (-1.0 / self.draft)
        return np.where(self._inside(xi, zeta), dfdz, 0.0)

    def cb_closed_form(self) -> float:
        """``Cb`` from the closed-form volume integral ``(2/3) int g``."""
        return (2.0 / 3.0) * self.profile_integral()

    def section_area(self, x):
        """Sectional area ``A(x) = 2 int_{-T}^{0} f dz`` (closed form)."""
        x = np.asarray(x, dtype=float)
        xi = 2.0 * x / self.length
        val = self.beam * self.draft * (1.0 - xi**2) * self.profile_integral()
        return np.where(np.abs(xi) <= 1.0, np.maximum(val, 0.0), 0.0)

    def area_curve_numeric(self, n_x: int = 4001, n_z: int = 2001):
        """Numeric ``A(x)`` (trapezoid in ``z``) used to audit the closed form."""
        x = np.linspace(-0.5 * self.length, 0.5 * self.length, n_x)
        z = np.linspace(-self.draft, 0.0, n_z)
        f = self.half_beam(x[:, None], z[None, :])
        return x, 2.0 * np.trapezoid(f, z, axis=1)


#: Classic Wigley: ``L/B = 10``, ``B/T = 1.6``, ``Cb = 4/9`` (unit length).
WIGLEY_STANDARD = ParabolicHull(
    length=1.0, beam=0.1, draft=0.0625, vertical="parabolic", name="wigley-standard"
)
#: Wigley III (ITTC-adjacent benchmark form, ``Cb = 0.555``).
WIGLEY_III = ParabolicHull(
    length=3.048, beam=0.406, draft=0.152, vertical="power", alpha=4.97, name="wigley-III"
)


# ---------------------------------------------------------------------------
# Wetted surface and admission window
# ---------------------------------------------------------------------------
def wetted_surface_area(hull: ParabolicHull, n_x: int = 1201, n_z: int = 601) -> dict:
    """``S = 2 int int sqrt(1 + f_x^2 + f_z^2) dx dz`` (both sides)."""
    x = np.linspace(-0.5 * hull.length, 0.5 * hull.length, int(n_x))
    z = np.linspace(-hull.draft, 0.0, int(n_z))
    X, Z = np.meshgrid(x, z, indexing="ij")
    jac = np.sqrt(1.0 + hull.half_beam_dx(X, Z) ** 2 + hull.half_beam_dz(X, Z) ** 2)
    s = 2.0 * float(np.trapezoid(np.trapezoid(jac, z, axis=1), x))
    return {"s_wet_m2": s, "n_x": int(n_x), "n_z": int(n_z)}


def slenderness_report(hull: ParabolicHull) -> dict:
    """Thin-ship admission window: ``Cb <= 0.60`` and ``L/B >= 5``."""
    lb = hull.length / hull.beam
    bt = hull.beam / hull.draft
    cb = hull.cb_closed_form()
    admitted = bool(cb <= 0.60 and lb >= 5.0)
    return {
        "name": hull.name,
        "L_over_B": lb,
        "B_over_T": bt,
        "Cb": cb,
        "admitted": admitted,
        "limits": {"Cb_max": 0.60, "L_over_B_min": 5.0},
        "note": (
            "Michell is first order in beam/length; Cb<=0.60 and L/B>=5 is the "
            "conventional window (Michell 1898; Tuck 1989 survey). Outside it the "
            "coefficient is diagnostic only."
        ),
    }


# ---------------------------------------------------------------------------
# Michell integral
# ---------------------------------------------------------------------------
def _default_grid(hull: ParabolicHull, n_x: int, n_z: int):
    """Cell-centred sample grid; ``x`` nodes sit inside the hull box."""
    dx = hull.length / float(n_x)
    dz = hull.draft / float(n_z)
    x = -0.5 * hull.length + (np.arange(n_x) + 0.5) * dx
    z = -hull.draft + (np.arange(n_z) + 0.5) * dz
    return x, z, dx, dz


def _signal(hull: ParabolicHull, fn: float, n_x: int, n_z: int, g: float):
    """Sampled thin-ship source strength ``f_x dx dz`` and the geometry factors."""
    speed = fn * math.sqrt(g * hull.length)
    k0 = g / speed**2
    x, z, dx, dz = _default_grid(hull, n_x, n_z)
    X, Z = np.meshgrid(x, z, indexing="ij")
    fx = hull.half_beam_dx(X, Z)
    return {
        "speed": speed,
        "k0": k0,
        "src": (fx * dx * dz).reshape(-1),
        "xf": X.reshape(-1),
        "zf": Z.reshape(-1),
    }


def michell_resistance(
    hull: ParabolicHull,
    fn: float,
    n_x: int = 161,
    n_z: int = 61,
    n_theta: int = 600,
    u_max: float = 4.0,
    rho: float = RHO,
    g: float = G,
    wetted_area: float | None = None,
    return_spectrum: bool = False,
) -> dict:
    """Michell wave resistance at one Froude number.

    ``u = tan(theta)`` substitution: ``sec^3(theta) dtheta = sec(u) du``, so the
    quadrature weight is the secant itself -- the integrand is smooth on
    ``[0, u_max]`` and converges spectrally in ``n_theta``.

    Returns the resistance ``rw_n``, the coefficient ``cw`` (normalised by the
    wetted surface area), the axial phase ``k0*L`` and (optionally) the hull
    wave spectrum ``P(θ) + iQ(θ)``.
    """
    if not (0.0 < float(fn) < 1.0):
        raise ValueError("Froude number must be in (0, 1)")
    fn = float(fn)
    sig = _signal(hull, fn, n_x, n_z, g)
    speed, k0, src, xf, zf = sig["speed"], sig["k0"], sig["src"], sig["xf"], sig["zf"]

    u = np.linspace(0.0, float(u_max), int(n_theta))
    du = u[1] - u[0]
    sec = np.sqrt(1.0 + u**2)
    # blocked evaluation: the (n_theta, n_cells) phase matrix is chunked so that
    # memory stays bounded while the dot products run as one BLAS-like reduce.
    p_spec = np.empty_like(u)
    q_spec = np.empty_like(u)
    block = max(1, int(4_000_000 // max(src.size, 1)))
    for i0 in range(0, u.size, block):
        sl = slice(i0, min(i0 + block, u.size))
        s_blk = sec[sl][:, None]
        k = k0 * s_blk * s_blk
        phase = k0 * s_blk * xf[None, :]  # k x cos(theta) = k0 sec(theta) x
        amp = src[None, :] * np.exp(k * zf[None, :])  # z <= 0 -> depth decay
        p_spec[sl] = amp @ np.cos(phase).T if False else np.sum(amp * np.cos(phase), axis=1)
        q_spec[sl] = np.sum(amp * np.sin(phase), axis=1)
    total = float(np.sum((p_spec**2 + q_spec**2) * sec) * du)
    rw = 4.0 * rho * g**2 / (math.pi * speed**2) * total

    if wetted_area is None:
        wetted_area = wetted_surface_area(hull)["s_wet_m2"]
    cw = rw / (0.5 * rho * speed**2 * float(wetted_area))
    out = {
        "model": "michell-thin-ship",
        "hull": hull.name,
        "fn": fn,
        "speed_ms": speed,
        "k0_per_m": k0,
        "k0_L": k0 * hull.length,
        "rw_n": rw,
        "cw": cw,
        "s_wet_m2": float(wetted_area),
        "n_x": int(n_x),
        "n_z": int(n_z),
        "n_theta": int(n_theta),
        "u_max": float(u_max),
        "admission": slenderness_report(hull),
    }
    if return_spectrum:
        theta = np.arctan(u)
        out["spectrum"] = {
            "theta_rad": theta,
            "u": u,
            "sec": sec,
            "P": p_spec,
            "Q": q_spec,
            "kappa_per_m": k0 * sec**2,
            "amplitude": np.hypot(p_spec, q_spec),
        }
    return out


def michell_spectrum_analytic(theta, hull: ParabolicHull, fn: float,
                              g: float = G) -> np.ndarray:
    """Closed-form Michell spectrum ``P + iQ`` for the parabolic hull.

    The double integral collapses analytically for ``g(zeta) = 1 - zeta^2``::

        P + iQ = (B/2)(-8/L^2) * I_z(kappa) * X(m),
        I_z = T [1/a - 2/a^3 + e^{-a}(2/a^2 + 2/a^3)],      a = kappa T
        X   = -i L cos(m L/2)/m + 2 i sin(m L/2)/m^2,        m = kappa cos(theta)

    with ``kappa = k0 sec^2(theta)``, ``k0 = g/U^2``.  This is an *exact*
    reference for the sampled quadrature in :func:`michell_resistance`: the two
    share no discretisation, so their agreement certifies the source strength
    ``f_x``, the depth weighting ``e^{kappa z}`` and the axial phase together.
    """
    if hull.vertical != "parabolic":
        raise ValueError("closed-form spectrum is only available for vertical='parabolic'")
    theta = np.asarray(theta, dtype=float)
    speed = float(fn) * math.sqrt(g * hull.length)
    k0 = g / speed**2
    kappa = k0 / np.cos(theta) ** 2
    m = kappa * np.cos(theta)
    a = kappa * hull.draft
    iz = hull.draft * (1.0 / a - 2.0 / a**3
                       + np.exp(-a) * (2.0 / a**2 + 2.0 / a**3))
    half = 0.5 * m * hull.length
    x_int = (-1j * hull.length * np.cos(half) / m
             + 2j * np.sin(half) / m**2)
    return 0.5 * hull.beam * (-8.0 / hull.length**2) * iz * x_int


def michell_resistance_analytic(hull: ParabolicHull, fn: float, rho: float = RHO,
                                g: float = G, n_theta: int = 400001,
                                wetted_area: float | None = None) -> dict:
    """Michell resistance from the closed-form spectrum (quadrature only in theta).

    Independent of :func:`michell_resistance` (which samples ``x`` and ``z``):
    only the ``theta`` integral is discretised, so agreement between the two is a
    test of the sampling, not of the theory.
    """
    th = np.linspace(1e-9, math.pi / 2.0 - 1e-9, int(n_theta))
    spec = michell_spectrum_analytic(th, hull, fn, g)
    speed = float(fn) * math.sqrt(g * hull.length)
    integrand = np.abs(spec) ** 2 / np.cos(th) ** 3
    rw = 4.0 * rho * g**2 / (math.pi * speed**2) * float(np.trapezoid(integrand, th))
    if wetted_area is None:
        wetted_area = wetted_surface_area(hull)["s_wet_m2"]
    return {
        "rw_n": rw,
        "cw": rw / (0.5 * rho * speed**2 * float(wetted_area)),
        "s_wet_m2": float(wetted_area),
        "n_theta": int(n_theta),
        "fn": float(fn),
    }


def michell_cw_curve(
    hull: ParabolicHull,
    froude_numbers: Sequence[float],
    n_x: int = 161,
    n_z: int = 61,
    n_theta: int = 600,
    u_max: float = 4.0,
) -> dict:
    """``C_W(Fn)`` on a Froude ladder (shared wetted area, one geometry audit)."""
    s = wetted_surface_area(hull)["s_wet_m2"]
    fns: list[float] = []
    cws: list[float] = []
    rws: list[float] = []
    for fn in froude_numbers:
        rec = michell_resistance(
            hull, float(fn), n_x=n_x, n_z=n_z, n_theta=n_theta, u_max=u_max, wetted_area=s
        )
        fns.append(float(fn))
        cws.append(rec["cw"])
        rws.append(rec["rw_n"])
    return {
        "hull": hull.name,
        "fn": np.asarray(fns),
        "cw": np.asarray(cws),
        "rw_n": np.asarray(rws),
        "s_wet_m2": s,
        "admission": slenderness_report(hull),
    }


def hump_hollow(fn, cw, smooth_window: int = 5, min_prominence_rel: float = 2.0e-3) -> dict:
    """Locate the humps (local maxima) and hollows (local minima) of ``C_W(Fn)``.

    The oscillation of ``C_W(Fn)`` comes from the axial phase ``k0 L``: the
    integrand ``P^2 + Q^2`` is a coherent sum over the hull length, so it
    re-phases as ``k0 L = (g/U^2) L = g L / (Fn^2 g L) = 1/Fn^2`` -- i.e. the
    hump/hollow structure is a function of ``1/Fn^2`` alone for a fixed
    geometry (this is the Froude scaling law the campaign asks to verify).

    Returns refined extremum locations (parabolic vertex refinement on the
    smoothed curve) and their coefficients.
    """
    fn = np.asarray(fn, dtype=float)
    cw = np.asarray(cw, dtype=float)
    order = np.argsort(fn)
    fn, cw = fn[order], cw[order]
    k = int(max(3, smooth_window | 1))
    pad = k // 2
    padded = np.pad(cw, pad, mode="edge")
    kernel = np.ones(k) / k
    smooth = np.convolve(padded, kernel, mode="valid")

    # 1) candidate extrema = sign changes of the smoothed slope
    cand = []
    for i in range(1, len(fn) - 1):
        left = smooth[i] - smooth[i - 1]
        right = smooth[i + 1] - smooth[i]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if (left > 0.0 and right < 0.0) or (left < 0.0 and right > 0.0):
            denom = smooth[i - 1] - 2.0 * smooth[i] + smooth[i + 1]
            shift = 0.0 if denom == 0.0 else 0.5 * (smooth[i - 1] - smooth[i + 1]) / denom
            shift = float(np.clip(shift, -1.0, 1.0))
            dfn = fn[i + 1] - fn[i]
            cand.append({
                "index": int(i),
                "kind": "hump" if left > 0.0 else "hollow",
                "fn": float(fn[i] + shift * dfn),
                "cw": float(smooth[i]),
            })

    # 2) topographic prominence against the nearest opposite-type extremum on
    #    each side (the curve endpoint stands in when there is none)
    for pos, c in enumerate(cand):
        left_ref = smooth[0]
        for prev in reversed(cand[:pos]):
            if prev["kind"] != c["kind"]:
                left_ref = prev["cw"]
                break
        right_ref = smooth[-1]
        for nxt in cand[pos + 1:]:
            if nxt["kind"] != c["kind"]:
                right_ref = nxt["cw"]
                break
        if c["kind"] == "hump":
            ref = max(left_ref, right_ref)
            c["prominence"] = float(c["cw"] - ref)
        else:
            ref = min(left_ref, right_ref)
            c["prominence"] = float(ref - c["cw"])
        c["prominence_rel"] = float(c["prominence"] / max(abs(c["cw"]), 1e-30))

    humps = [{"fn": c["fn"], "fn_index": c["index"], "cw": c["cw"],
              "prominence_rel": c["prominence_rel"]}
             for c in cand if c["kind"] == "hump"
             and c["prominence_rel"] >= float(min_prominence_rel)]
    hollows = [{"fn": c["fn"], "fn_index": c["index"], "cw": c["cw"],
                "prominence_rel": c["prominence_rel"]}
               for c in cand if c["kind"] == "hollow"
               and c["prominence_rel"] >= float(min_prominence_rel)]
    return {
        "humps": humps,
        "hollows": hollows,
        "inv_fn2_humps": [1.0 / h["fn"] ** 2 for h in humps],
        "inv_fn2_hollows": [1.0 / h["fn"] ** 2 for h in hollows],
        "n_samples": int(len(fn)),
        "fn_range": [float(fn.min()), float(fn.max())],
    }


# ---------------------------------------------------------------------------
# Wave spectrum and wake geometry
# ---------------------------------------------------------------------------
def wave_spectrum(theta, k0: float, depth: float | None = None) -> np.ndarray:
    """Steady free-wave wavenumber ``kappa(theta)`` for a disturbance moving at
    speed ``U`` (``k0 = g/U^2``).

    Deep water: ``kappa = k0 sec^2(theta)`` (exact, closed form).
    Finite depth ``h = depth``: the wave must satisfy
    ``omega(kappa) = U kappa cos(theta)`` with ``omega^2 = g kappa tanh(kappa h)``,
    i.e. the implicit ``kappa cos^2(theta) = k0 tanh(kappa h)`` (solved by
    fixed-point/Newton iteration here).  The shallow limit
    ``kappa h -> 0`` returns ``cos^2(theta) = k0 h = 1/Fr_h^2``.
    """
    theta = np.asarray(theta, dtype=float)
    cos2 = np.cos(theta) ** 2
    if depth is None or not np.isfinite(depth):
        return k0 / np.maximum(cos2, 1e-12)
    depth = float(depth)
    if not (depth > 0.0):
        raise ValueError("depth must be positive")
    c = np.maximum(cos2, 1e-12)
    # Newton on g(kappa) = kappa c - k0 tanh(kappa h) = 0, starting from the
    # deep-water root (always an upper bound, since tanh <= 1).
    kappa = k0 / c
    for _ in range(60):
        kd = np.clip(kappa * depth, 0.0, 60.0)
        th = np.tanh(kd)
        g_val = kappa * c - k0 * th
        dg = c - k0 * depth * (1.0 - th**2)
        dg = np.where(np.abs(dg) < 1e-14, c, dg)
        step = g_val / dg
        kappa = np.maximum(kappa - step, 1e-12)
        if np.max(np.abs(step)) <= 1e-13 * np.max(np.abs(kappa)):
            break
    return kappa


def _phase_direction(theta, alpha, k0, depth):
    """``d/dtheta``-ready phase function ``kappa(theta) cos(theta - alpha)``.

    ``k x + l y = kappa r cos(theta - alpha)`` for a field point at polar angle
    ``alpha``; dividing by ``r`` leaves the angular factor used for the
    stationary-phase analysis.
    """
    kappa = wave_spectrum(theta, k0, depth)
    return kappa * np.cos(theta - alpha)


def caustic_angle_deg(k0: float, theta_max: float = math.radians(89.5),
                      n_theta: int = 20001, alpha_tol: float = 1e-8) -> dict:
    """Deep-water caustic (Kelvin) half-angle of the steady wave system [deg].

    Far field along the polar direction ``alpha`` receives the contributions of
    the stationary points of

        F(theta) = kappa(theta) cos(theta + alpha),   kappa = k0 sec^2(theta)

    (the steady phase is ``k x + l y = kappa r cos(theta +- alpha)``; the branch
    that reaches a field point at ``+alpha`` downstream is ``theta < 0``, written
    here as ``cos(theta + alpha)`` with ``theta > 0``).  The wake
    edge -- the caustic -- is the largest ``alpha`` for which a stationary point
    still exists; it is bracketed by bisection on the sign of ``min F'``.

    The angle is found *numerically* here (no ``arcsin`` shortcut), which is the
    point: it verifies both the wavenumber selection ``kappa = k0 sec^2 theta``
    and the classical ``sin(alpha) = 1/3`` (``19.4712 deg``) result.
    """
    th = np.linspace(1e-9, float(theta_max), int(n_theta))
    dth = th[1] - th[0]
    kappa = wave_spectrum(th, k0, None)

    def has_stationary(alpha: float) -> bool:
        f = kappa * np.cos(th + alpha)
        df = np.gradient(f, dth)
        return bool(np.any(df[:-1] * df[1:] < 0.0))

    lo, hi = 1.0e-3, math.radians(60.0)
    if not has_stationary(lo):
        return {"alpha_deg": 0.0, "found": False,
                "note": "no stationary point at small alpha (unexpected)"}
    if has_stationary(hi):
        return {"alpha_deg": float(np.degrees(hi)), "found": False,
                "note": "no caustic inside 60 deg"}
    while (hi - lo) > alpha_tol:
        mid = 0.5 * (lo + hi)
        if has_stationary(mid):
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return {
        "alpha_deg": float(np.degrees(alpha)),
        "found": True,
        "sin_alpha": float(math.sin(alpha)),
        "tan_alpha": float(math.tan(alpha)),
        "bracket_rad": [float(lo), float(hi)],
        "theta_at_caustic_deg": _caustic_wave_direction_deg(alpha, k0),
        "classical_sin_alpha": 1.0 / 3.0,
        "classical_deg": 19.471220634490691,
    }


def _caustic_wave_direction_deg(alpha: float, k0: float, n_theta: int = 20001) -> float | None:
    """Wave-vector direction ``theta`` at the caustic (deep water).

    Solves ``F'(theta) = 0`` for the double root at the caustic; the classical
    value is ``theta = arctan(1/sqrt(2)) = 35.264 deg``.
    """
    th = np.linspace(1e-9, math.radians(89.5), int(n_theta))
    f = wave_spectrum(th, k0, None) * np.cos(th + alpha)
    df = np.gradient(f, th[1] - th[0])
    idx = int(np.argmin(np.abs(df)))
    return float(np.degrees(th[idx]))


def caustic_angle_analytic_deg(n_u: int = 200001) -> dict:
    """Caustic angle by maximising ``tan(alpha) = u / (2 u^2 + 1)``.

    Independent of :func:`caustic_angle_deg` (which bisects on the existence of
    stationary points of the far-field phase).  The relation follows from the
    stationary-phase condition ``u x + y (2u^2 + 1) = 0`` with ``u = tan(theta)``
    and ``x, y = r cos(alpha), r sin(alpha)``; the maximum over ``u`` is at
    ``u = 1/sqrt(2)``, giving ``tan(alpha) = 1/(2 sqrt(2))``.
    """
    u = np.linspace(1e-6, 20.0, int(n_u))
    tan_alpha = u / (2.0 * u**2 + 1.0)
    i = int(np.argmax(tan_alpha))
    return {
        "alpha_deg": float(np.degrees(np.arctan(tan_alpha[i]))),
        "tan_alpha": float(tan_alpha[i]),
        "u_at_caustic": float(u[i]),
        "theta_at_caustic_deg": float(np.degrees(np.arctan(u[i]))),
        "closed_form_tan_alpha": 1.0 / (2.0 * math.sqrt(2.0)),
        "closed_form_deg": float(np.degrees(math.atan(1.0 / (2.0 * math.sqrt(2.0))))),
    }


def onaxis_wavelength(field: dict) -> dict:
    """Dominant transverse wavelength from the on-axis elevation trace.

    FFT of ``zeta(r, alpha=0)``; the steady wave system must select
    ``lambda = 2 pi U^2/g = 2 pi Fn^2 L`` (deep-water dispersion).  This is a
    genuine independent check of the wavenumber selection: it only uses the
    synthesised trace, not the dispersion formula.
    """
    z = np.asarray(field["zeta"], float)
    r = np.asarray(field["r"], float)
    if r.ndim == 2:
        r = r[:, 0]
    trace = z[:, 0] - np.mean(z[:, 0])
    n = trace.size
    spec = np.abs(np.fft.rfft(trace * np.hanning(n)))
    dr = float(r[1] - r[0])
    k = 2.0 * math.pi * np.fft.rfftfreq(n, d=dr)
    k_peak = float(k[1 + int(np.argmax(spec[1:]))])
    lam = 2.0 * math.pi / k_peak if k_peak > 0 else float("inf")
    theory = float(field["transverse_wavelength_m"])
    return {
        "wavelength_measured_m": lam,
        "wavelength_theory_m": theory,
        "deviation_pct": 100.0 * (lam - theory) / theory if theory else None,
        "k_peak_per_m": k_peak,
        "k0_per_m": float(field["k0_per_m"]),
    }


def shallow_wake_angle_deg(froude_depth: float) -> dict:
    """Mach-cone half-angle of a supercritical disturbance in shallow water.

    Non-dispersive waves (``c = sqrt(g h)``, independent of ``kappa``) give a
    cone at ``alpha = arcsin(1/Fr_h)``, ``Fr_h = U/sqrt(g h)``.  Derived from the
    group-velocity ray envelope ``X = U + c cos(theta)``, ``Y = c sin(theta)``,
    whose extremal ray satisfies ``cos(theta) = -c/U`` and ``sin(alpha) = c/U``
    (closed form reproduced in :func:`shallow_ray_envelope_deg`).
    """
    fr = float(froude_depth)
    if fr <= 1.0:
        return {"froude_depth": fr, "alpha_deg": None, "supercritical": False,
                "note": "Fr_h <= 1: no steady wave cone (subcritical)"}
    return {
        "froude_depth": fr,
        "alpha_deg": float(np.degrees(math.asin(1.0 / fr))),
        "supercritical": True,
        "limit_deg": 90.0 if fr <= 1.0 else float(np.degrees(math.asin(1.0 / fr))),
    }


def shallow_ray_envelope_deg(froude_depth: float, n_theta: int = 20001) -> float | None:
    """Numerical check of :func:`shallow_wake_angle_deg` (ray-envelope route).

    Rays: ``X(theta) = U + c cos(theta)``, ``Y = c sin(theta)`` with ``c =
    sqrt(g h)``; the envelope is the maximum of ``atan2(Y, X)`` over ``theta``.
    Independent of the ``arcsin`` closed form.
    """
    fr = float(froude_depth)
    if fr <= 1.0:
        return None
    th = np.linspace(0.0, math.pi, int(n_theta))
    x = fr + np.cos(th)
    y = np.sin(th)
    return float(np.degrees(np.max(np.arctan2(y, x))))


def wake_field(
    hull: ParabolicHull,
    fn: float,
    radii: Sequence[float] | None = None,
    alphas: Sequence[float] | None = None,
    n_x: int = 361,
    n_z: int = 121,
    n_theta: int = 1200,
    u_max: float = 4.0,
    g: float = G,
) -> dict:
    """Stationary-phase synthesis of the steady wake on a polar grid.

        zeta(r, alpha) = sum_j |I(theta_j)| cos(kappa_j r cos(theta_j + alpha)
                                              + arg I(theta_j))

    ``I(theta) = P + iQ`` is the hull wave spectrum from
    :func:`michell_resistance` (so the *relative* amplitude of the transverse and
    divergent systems comes from the hull, not from a fit).  Only the wave-system
    *geometry* is claimed -- absolute elevation needs the free-surface Green
    function, which the thin-ship integral does not provide.  Resistance is taken
    from :func:`michell_resistance`, never from the pattern.
    """
    fn = float(fn)
    rec = michell_resistance(
        hull, fn, n_x=n_x, n_z=n_z, n_theta=n_theta, u_max=u_max, g=g,
        return_spectrum=True,
    )
    k0 = rec["k0_per_m"]
    spec = rec["spectrum"]
    theta = spec["theta_rad"][1:-1]
    amp = spec["amplitude"][1:-1]
    phase0 = np.arctan2(spec["Q"][1:-1], spec["P"][1:-1])
    kappa = wave_spectrum(theta, k0, None)
    if radii is None:
        radii = np.linspace(0.2, 4.0, 120) * hull.length
    if alphas is None:
        alphas = np.linspace(0.0, math.radians(60.0), 241)
    radii = np.asarray(radii, float)
    alphas = np.asarray(alphas, float)
    R, A = np.meshgrid(radii, alphas, indexing="ij")
    field = np.zeros_like(R)
    for i in range(theta.size):
        if amp[i] <= 0.0:
            continue
        field += amp[i] * np.cos(kappa[i] * R * np.cos(theta[i] + A) + phase0[i])
    peak = float(np.max(np.abs(field))) if field.size else 1.0
    return {
        "r": R,
        "alpha": A,
        "zeta": field / max(peak, 1e-30),
        "amplitude_units": "normalised to unit peak (pattern geometry only)",
        "fn": fn,
        "k0_per_m": k0,
        "transverse_wavelength_m": 2.0 * math.pi * hull.length * fn**2,
        "kappa_per_m": kappa,
        "theta_rad": theta,
        "spectrum_amplitude": amp,
    }


def measure_wake_angle_deg(field: dict, thresholds: Sequence[float] = (0.5, 0.3, 0.1)) -> dict:
    """Wake envelope half-angle measured from a synthesised polar field.

    For each azimuth the radial mean of ``|zeta|`` is formed (the caustic is
    where the stationary points vanish and the amplitude collapses); the angle is
    the first azimuth at which that mean falls below ``threshold`` times the
    on-axis value.  Reporting three thresholds makes the measurement's
    sensitivity explicit instead of hiding it behind one number.
    """
    z = np.abs(np.asarray(field["zeta"], float))
    alpha = np.asarray(field["alpha"], float)
    if alpha.ndim == 2:
        alpha = alpha[0]
    amps = z.reshape(z.shape[0], -1).mean(axis=0)
    ref = float(amps[0]) if amps[0] > 0.0 else float(np.max(amps))
    out = {"reference_amplitude": ref, "thresholds": {}}
    for thr in thresholds:
        below = np.where(amps < float(thr) * ref)[0]
        out["thresholds"][str(thr)] = (
            float(np.degrees(alpha[below[0]])) if below.size else None
        )
    vals = [v for v in out["thresholds"].values() if v is not None]
    out["alpha_deg"] = float(np.median(vals)) if vals else None
    return out


# ---------------------------------------------------------------------------
# Demihull interference
# ---------------------------------------------------------------------------
def catamaran_interference(
    hull: ParabolicHull,
    fn: float,
    separations_over_l: Sequence[float],
    mode: str = "transverse",
    n_x: int = 481,
    n_z: int = 161,
    n_theta: int = 1600,
    u_max: float = 4.0,
    g: float = G,
) -> dict:
    """Interference factor ``C_W(cat) / (2 C_W(single))`` for two demihulls.

    Thin-ship superposition: two identical hulls offset by
    ``(dx, dy) = (0, S)`` (``mode="transverse"``, side by side) or
    ``(dx, dy) = (S, 0)`` (``mode="longitudinal"``, tandem) have
    ``I_cat(theta) = I(theta) (1 + exp(i phi(theta)))`` with

        phi = k0 sec(theta) dx + k0 sec(theta) tan(theta) dy,     sec = sec(theta)

    because ``k = kappa cos(theta) = k0 sec(theta)`` and
    ``l = kappa sin(theta) = k0 sec(theta) tan(theta)``.  Hence
    ``|I_cat|^2 = 2 |I|^2 (1 + cos phi)`` and

        f(S) = 1 + <cos phi>_W,   <...>_W = spectrum-weighted mean
                                  with weight sec^3(theta) |I(theta)|^2.

    The ``analytic`` column is the single-wave (textbook) approximation
    ``f_simple = 1 + cos(k0 S)`` (longitudinal) / ``1 + cos(k0 S tan(theta_*))``
    with ``theta_* = 0`` (transverse, i.e. the pure transverse-wave picture).
    Reporting both makes explicit how much of the textbook number is spectrum
    weighting rather than physics.
    """
    if mode not in ("transverse", "longitudinal"):
        raise ValueError("mode must be 'transverse' or 'longitudinal'")
    s_area = wetted_surface_area(hull)["s_wet_m2"]
    rec = michell_resistance(
        hull, fn, n_x=n_x, n_z=n_z, n_theta=n_theta, u_max=u_max, g=g,
        wetted_area=s_area, return_spectrum=True,
    )
    k0 = rec["k0_per_m"]
    spec = rec["spectrum"]
    theta, sec, P, Q = spec["theta_rad"], spec["sec"], spec["P"], spec["Q"]
    weight = sec**3 * (P * P + Q * Q)
    total_w = float(np.trapezoid(weight, theta))
    cw_single = float(rec["cw"])

    rows = []
    for s_over_l in separations_over_l:
        sep = float(s_over_l) * hull.length
        kx = k0 * sec
        ly = k0 * sec * np.tan(theta)
        phi = kx * sep if mode == "longitudinal" else ly * sep
        cos_w = float(np.trapezoid(weight * np.cos(phi), theta))
        factor = 1.0 + (cos_w / total_w if total_w > 0 else 0.0)
        simple = 1.0 + (math.cos(k0 * sep) if mode == "longitudinal" else math.cos(0.0))
        rows.append(
            {
                "s_over_l": float(s_over_l),
                "separation_m": sep,
                "interference_factor": factor,
                "cw_cat": 2.0 * cw_single * factor,
                "single_wave_analytic": simple,
                "deviation_pct": 100.0 * (factor - simple) / simple if simple else None,
            }
        )
    return {
        "mode": mode,
        "fn": float(fn),
        "k0_per_m": k0,
        "transverse_wavelength_m": 2.0 * math.pi * hull.length * fn**2,
        "cw_single": cw_single,
        "rows": rows,
    }