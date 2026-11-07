"""Squeeze-film (Reynolds lubrication) analytic anchor for ice-hull / ice-ice gaps.

Physics
-------
Two rigid plates of lateral extent :math:`2a` separated by a film of viscous fluid
of thickness :math:`h \\ll a`, approaching each other at the relative rate
:math:`V = -\\mathrm{d}h/\\mathrm{d}t`.  In the lubrication limit
(:math:`h/a \\ll 1`, :math:`Re (h/a)^2 \\ll 1`) the film is quasi-steady: the
viscous shear balances the in-plane pressure gradient (plane Poiseuille),

.. math::  q(x) = -\\frac{h^3}{12\\mu}\\,\\frac{\\mathrm{d}p}{\\mathrm{d}x},

and mass conservation of the squeezed fluid gives

.. math::  \\frac{\\mathrm{d}q}{\\mathrm{d}x} = V .

With the pressure-release condition :math:`p(\\pm a) = 0` at the free edges this
integrates to the classical Reynolds squeeze-film pressure and load

plane strain (2-D, plate width ``2 a``, depth ``L``)::

    p(x) = 6 mu V (a^2 - x^2) / h^3
    F    = 8 mu V a^3 L / h^3   ( = mu w^3 L V / h^3 with w = 2 a )

axisymmetric (circular plates of radius ``a``)::

    p(r) = 3 mu V (a^2 - r^2) / h^3
    F    = 3 pi mu V a^4 / (2 h^3)

Both carry the same ``h^{-3}`` squeeze-film scaling.  The ``h^{-3}`` law is the
signature of the *lubrication* (thin-film) regime and is the quantity this module
exists to pin down: a DEM ice-load model that omits the film misses a resistance
that grows as the third power of the inverse gap, i.e. it is irrelevant for
:math:`h \\sim a` and dominant for :math:`h \\ll a`.

References
----------
* Reynolds, O. (1886) Phil. Trans. R. Soc. Lond. 177, 157-234.
* Stefan, J. (1874) Sitzber. Akad. Wiss. Wien 69, 713 (squeeze film between disks).
* Gross, W.A. (1962) *Gas Film Lubrication*, Wiley.
* Hamrock, B.J. (1994) *Fundamentals of Fluid Film Lubrication*, McGraw-Hill.
* Batchelor, G.K. (1967) *An Introduction to Fluid Dynamics*, CUP, §4.8.

Conventions
-----------
* ``V`` is the *approach* speed, ``V = -dh/dt > 0`` for closing plates.
* SI units throughout; ``mu`` is the dynamic viscosity [Pa s].
* ``length`` (plane-strain only) is the out-of-plane extent [m].

Everything in this module is closed form / deterministic quadrature: no LBM, no
random numbers, no hard-coded reference tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MU_WATER_20C",
    "RHO_WATER_20C",
    "SqueezeFilmSolution",
    "reynolds_pressure_plane",
    "reynolds_force_plane",
    "reynolds_pressure_disk",
    "reynolds_force_disk",
    "reynolds_force_numeric",
    "reynolds_velocity_plane",
    "squeeze_drainage_number",
    "fit_loglog_slope",
    "lubrication_share",
    "lubrication_share_contour",
    "lubrication_onset_gap",
]

#: dynamic viscosity of fresh water at 20 degC [Pa s] (IAPWS-95 value).
MU_WATER_20C = 1.0016e-3
#: density of fresh water at 20 degC [kg m^-3].
RHO_WATER_20C = 998.2


@dataclass(frozen=True)
class SqueezeFilmSolution:
    """Reynolds squeeze-film solution on a set of abscissae.

    Attributes:
        s: abscissa (``x`` for plane strain, ``r`` for the disk) [m].
        p: film pressure (gauge, relative to the free-edge ambient) [Pa].
        force: integrated load carried by the film [N].
        mu: dynamic viscosity [Pa s].
        a: plate half-width (or disk radius) [m].
        h: film thickness [m].
        V: approach speed [m s^-1].
        kind: ``"plane"`` or ``"disk"``.
    """

    s: np.ndarray
    p: np.ndarray
    force: float
    mu: float
    a: float
    h: float
    V: float
    kind: str

    @property
    def p_max(self) -> float:
        """Centre-line film pressure [Pa]."""
        return float(np.max(self.p))


def _check(mu: float, a: float, h: float, V: float) -> None:
    for name, value in (("mu", mu), ("a", a), ("h", h)):
        if not (value > 0.0) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and > 0 (got {value!r})")
    if not math.isfinite(V) or V < 0.0:
        raise ValueError(f"V must be finite and >= 0 (got {V!r})")


def reynolds_pressure_plane(
    x: np.ndarray | float,
    mu: float,
    a: float,
    h: float,
    V: float,
) -> np.ndarray:
    """Reynolds squeeze-film gauge pressure between two plane plates.

    ``p(x) = 6 mu V (a^2 - x^2) / h^3`` for ``|x| <= a`` (zero outside), the
    solution of ``d2p/dx2 = -12 mu V / h^3`` with ``p(+-a) = 0``.

    Args:
        x: in-plane coordinate [m], any shape.
        mu: dynamic viscosity [Pa s].
        a: plate half-width [m].
        h: film thickness [m].
        V: approach speed ``-dh/dt`` [m s^-1].

    Returns:
        Gauge pressure array with the broadcast shape of ``x`` [Pa].
    """
    _check(mu, a, h, V)
    x = np.asarray(x, dtype=float)
    p = 6.0 * mu * V * (a * a - x * x) / (h**3)
    return np.where(np.abs(x) <= a, p, 0.0) * np.ones_like(x, dtype=float)


def reynolds_force_plane(
    mu: float,
    a: float,
    h: float,
    V: float,
    length: float = 1.0,
) -> float:
    """Plane-strain Reynolds squeeze-film load ``F = 8 mu V a^3 L / h^3``.

    Derivation: ``F/L = int_{-a}^{a} p dx = 8 mu V a^3 / h^3`` with
    ``p = 6 mu V (a^2 - x^2)/h^3``.  Equivalently, for a strip of width
    ``w = 2a`` this is the textbook one-dimensional squeeze-film result
    ``F = mu w^3 L V / h^3`` (Gross 1962; Hamrock 1994).

    Args:
        mu: dynamic viscosity [Pa s].
        a: plate half-width [m].
        h: film thickness [m].
        V: approach speed ``-dh/dt`` [m s^-1].
        length: out-of-plane plate extent ``L`` [m].

    Returns:
        Film load [N].
    """
    _check(mu, a, h, V)
    if not (length > 0.0):
        raise ValueError("length must be > 0")
    return 8.0 * mu * V * a**3 * length / h**3


def reynolds_pressure_disk(
    r: np.ndarray | float,
    mu: float,
    a: float,
    h: float,
    V: float,
) -> np.ndarray:
    """Reynolds squeeze-film gauge pressure between two circular plates.

    ``p(r) = 3 mu V (a^2 - r^2) / h^3`` for ``r <= a`` (zero outside); this is
    the axisymmetric solution of ``(1/r) d/dr (r dp/dr) = -12 mu V / h^3``.
    """
    _check(mu, a, h, V)
    r = np.asarray(r, dtype=float)
    p = 3.0 * mu * V * (a * a - r * r) / (h**3)
    return np.where(r <= a, p, 0.0) * np.ones_like(r, dtype=float)


def reynolds_force_disk(mu: float, a: float, h: float, V: float) -> float:
    """Axisymmetric squeeze-film load ``F = 3 pi mu V a^4 / (2 h^3)``.

    This is the Stefan/Reynolds disk formula quoted in the campaign card
    (``F = 3 pi mu a^4 / (2 h^3) * dh/dt``; note ``dh/dt = -V``).
    """
    _check(mu, a, h, V)
    return 3.0 * math.pi * mu * V * a**4 / (2.0 * h**3)


def reynolds_force_numeric(
    kind: str,
    mu: float,
    a: float,
    h: float,
    V: float,
    length: float = 1.0,
    n: int = 4001,
) -> float:
    """Load from an *independent* numerical solution of the Reynolds equation.

    Delegates to :mod:`tensorlbm.squeeze_film_reference`, which solves the
    lubrication pressure equation with a finite-volume scheme on a fine grid and
    integrates the pressure.  This exercises a different code path (grid,
    tridiagonal solve, quadrature) than the closed forms above and is the
    reference used to certify them.

    Args:
        kind: ``"plane"`` (``d2p/dx2 = -12 mu V/h^3``) or ``"disk"``
            (``(1/r) d/dr (r dp/dr) = -12 mu V/h^3``).
        mu, a, h, V: viscosity, half-width/radius, gap, approach speed (SI).
        length: out-of-plane extent (plane strain only) [m].
        n: number of grid points on ``[0, a]``.

    Returns:
        Integrated film load [N].
    """
    from .squeeze_film_reference import reynolds_force_numeric as _numeric

    _check(mu, a, h, V)
    return _numeric(kind, mu, a, h, V, length=length, n=n)


def reynolds_velocity_plane(
    x: np.ndarray | float,
    y: np.ndarray | float,
    mu: float,
    a: float,
    h: float,
    V: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lubrication velocity field of the plane squeeze film.

    ``u(x, y) = 6 V x y (h - y) / h^3`` and ``v(x, y) = V (3 (y/h)^2 - 2 (y/h)^3)``
    (the latter from ``dv/dy = -du/dx`` with ``v(0) = 0``, ``v(h) = V``).

    Args:
        x, y: coordinates [m] (``y`` measured from the stationary plate).
        mu, a, h, V: as in :func:`reynolds_force_plane` (``mu`` unused; kept for
            signature symmetry).

    Returns:
        ``(u, v)`` arrays broadcast to ``x``/``y`` [m s^-1].
    """
    _check(mu, a, h, V)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = 6.0 * V * x * y * (h - y) / h**3
    eta = y / h
    v = V * (3.0 * eta**2 - 2.0 * eta**3)
    return u, v


def squeeze_drainage_number(
    mu: float,
    a: float,
    h: float,
    V: float,
    rho: float = RHO_WATER_20C,
) -> float:
    """Reduced squeeze Reynolds number ``Re_h = rho V h / mu`` (inertia control).

    The persistent-inertia correction to the Reynolds load scales as
    ``(h/a)^2 Re_h``; values ``<< 1`` mean the quasi-static lubrication
    approximation is legitimate.
    """
    _check(mu, a, h, V)
    return rho * V * h / mu


def fit_loglog_slope(
    h: np.ndarray,
    force: np.ndarray,
) -> dict:
    """Least-squares fit of ``log F = slope * log h + c``.

    Args:
        h: film thicknesses (> 0), any units (consistent with ``force``).
        force: film loads (> 0), any consistent units.

    Returns:
        Dict with ``slope``, ``intercept``, ``r2``, ``stderr`` (slope standard
        error), ``n`` and ``ratio`` = ``exp(intercept)`` (the pre-factor when
        ``h`` is expressed in the units supplied).
    """
    h = np.asarray(h, dtype=float)
    force = np.asarray(force, dtype=float)
    if h.shape != force.shape:
        raise ValueError("h and force must have the same shape")
    if h.size < 3:
        raise ValueError("need at least 3 points for a 2-parameter log-log fit")
    if np.any(h <= 0) or np.any(force <= 0):
        raise ValueError("h and force must be strictly positive")
    lx = np.log(h)
    ly = np.log(force)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    dof = max(h.size - 2, 1)
    stderr = math.sqrt(ss_res / dof / float(np.sum((lx - lx.mean()) ** 2)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ratio": float(math.exp(intercept)),
        "r2": float(r2),
        "stderr": float(stderr),
        "n": int(h.size),
        "max_rel_residual": float(np.max(np.abs(np.exp(pred - ly) - 1.0))),
    }


def lubrication_share(
    film_force: np.ndarray | float,
    dry_force: np.ndarray | float,
) -> np.ndarray | float:
    """Share ``R_lub / (R_dry + R_lub)`` of the total resistance."""
    film = np.asarray(film_force, dtype=float)
    dry = np.asarray(dry_force, dtype=float)
    total = dry + film
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0.0, film / total, np.nan)
    return float(share) if np.isscalar(film_force) else share


def lubrication_onset_gap(
    share: float,
    dry_force: float,
    mu: float,
    a: float,
    V: float,
    length: float = 1.0,
) -> float:
    """Gap ``h*`` at which the film carries ``share`` of the total resistance.

    Inverts ``R_lub/(R_dry + R_lub) = share`` with the plane-strain film law:
    ``h* = (8 mu V a^3 L (1 - share) / (share R_dry))^{1/3}``.

    Args:
        share: target share in ``(0, 1)``.
        dry_force: dry (contact/crushing) resistance ``R_dry`` [N].
        mu, a, V, length: as in :func:`reynolds_force_plane`.

    Returns:
        Onset gap ``h*`` [m]; ``inf`` when ``dry_force <= 0``.
    """
    if not (0.0 < share < 1.0):
        raise ValueError("share must lie in (0, 1)")
    if dry_force <= 0.0:
        return float("inf")
    num = 8.0 * mu * V * a**3 * length * (1.0 - share)
    return float((num / (share * dry_force)) ** (1.0 / 3.0))


def lubrication_share_contour(
    share_levels: tuple[float, ...] = (0.05, 0.15),
    *,
    mu: float = MU_WATER_20C,
    a: float = 0.5,
    length: float = 1.0,
    dry_force: float = 500.0,
    V_range: tuple[float, float] = (1e-3, 1.0),
    h_range: tuple[float, float] = (1e-5, 1e-1),
    n: int = 241,
) -> dict:
    """Contours of ``R_lub / R_total = 5 %, 15 % ...`` in the ``(V, h)`` plane.

    The share is evaluated from the plane-strain Reynolds film law against a
    prescribed dry resistance ``R_dry``.  Because ``R_lub ~ h^{-3}``, each
    contour is a curve ``h*(V) = (8 mu a^3 L V (1 - s) / (s R_dry))^{1/3}``:
    a straight line of slope ``1/3`` in ``log h`` vs ``log V``.

    Args:
        share_levels: contour levels of the lubricated share.
        mu, a, length: film geometry / fluid properties.
        dry_force: dry resistance ``R_dry`` [N].
        V_range: approach-speed window [m s^-1].
        h_range: gap window [m].
        n: number of grid nodes per axis for the field evaluation.

    Returns:
        Dict with ``V``, ``h`` (1-D grids), ``share`` (2-D field), and
        ``contours`` = ``{level: {"V": ..., "h": ...}}`` sampled on a common
        approach-speed vector.
    """
    V = np.logspace(math.log10(V_range[0]), math.log10(V_range[1]), n)
    h = np.logspace(math.log10(h_range[0]), math.log10(h_range[1]), n)
    VAR = V[None, :]
    H = h[:, None]
    film = 8.0 * mu * VAR * a**3 * length / H**3
    share = lubrication_share(film, dry_force)

    contours: dict[float, dict[str, np.ndarray]] = {}
    v_sample = V
    for level in share_levels:
        h_star = np.empty_like(v_sample)
        for j, v in enumerate(v_sample):
            h_star[j] = lubrication_onset_gap(float(level), dry_force, mu, a, v, length)
        contours[float(level)] = {
            "V": v_sample.copy(),
            "h": h_star,
            # closed-form cross-check: h* ~ V^{1/3}
            "slope_in_loglog": float(
                np.polyfit(np.log(v_sample), np.log(h_star), 1)[0]
            ),
        }
    return {
        "V": V,
        "h": h,
        "share": share,
        "contours": contours,
        "dry_force": dry_force,
        "mu": mu,
        "a": a,
        "length": length,
    }