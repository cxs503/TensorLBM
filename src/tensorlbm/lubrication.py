"""Sub-lattice lubrication closure for near-contact cylinder arrays (2-D).

Motivation
----------
Lattice-Boltzmann bounce-back resolves the solid boundary to O(dx) (half-way
wall).  For a fibrous bed the permeating flow is carried by the throats between
*neighbouring* cylinders.  When the throat width ``delta`` becomes comparable to
or smaller than one lattice spacing, the binary voxel mask quantises the throat
to an integer number of cells and, because the throat conductance scales as a
*power* of its width, a half-cell quantisation error in ``delta`` becomes an
O(1) error in the permeability.  That failure is a resolution artefact, not
physics; it is removed by the closure implemented here.

Near-contact two-cylinder conductance (Sangani & Acrivos 1982)
--------------------------------------------------------------
Two cylinders of radius R whose surfaces are separated by the minimum gap
``delta`` form a parabolic channel

    h(s) = delta + s^2 / R ,

with s the arc length measured from the closest point.  In the lubrication limit
the volume flux per unit pressure gradient through that channel is
``\\int ds h(s)^3 / (12 mu)``, i.e. the conductance

    G(delta) = \\int_{-inf}^{inf} ds / h(s)^3 = (3 pi / 8) sqrt(R) / delta^{5/2} .

The delta^{-5/2} law is the origin of the (delta/R)^{5/2} asymptote behind
Gebart (1992) transverse permeability, so a closure built on G(delta) is
asymptotically consistent with the reference used to judge it.

Closure used here
-----------------
A resolved slot of width delta has volumetric conductance ~ delta^3 / nu; the
lattice reproduces that law for the *mask* width delta_mask (mid-link walls, so
a one-cell throat is 1 dx wide).  Requiring the lattice throat to carry the same
flux as the true throat gives a per-node effective viscosity

    nu_eff = nu * (delta_mask / delta_true)^3 ,

i.e. the momentum-exchange (viscous) coefficient of the gap cells is scaled by
the gap ratio cubed.  The ratio is tapered to unity at
``delta_true >= 1.5 dx``, so the closure is a strict no-op on resolved
geometries (dilute limit) and in smooth wide pores: it only ever acts on
sub-lattice throats.

Gap measurement
---------------
:func:`cylinder_gap_fields` casts, for every fluid node, four axis-aligned rays
(exact analytic ray/cylinder intersections and, separately, the voxel mask) and
takes the narrowest pair of opposed surfaces that belong to *two distinct
cylinders*.  The distinct-cylinder rule is what separates a genuine throat from
the neighbourhood of one convex wall; it is exactly the near-contact pair
configuration of Sangani & Acrivos.

References
----------
Gebart, B.R. (1992) J. Composite Materials 26, 1100-1133.
Sangani, A.S. & Acrivos, A. (1982) Int. J. Multiphase Flow 8, 193-206.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "PHI_MAX_SQ",
    "PHI_MAX_HEX",
    "GEBART_C1_SQ",
    "GEBART_C1_HEX",
    "GAP_INTEGRAL_CONST",
    "gebart_sqrt_gap",
    "gebart_permeability",
    "delta_min_over_cell",
    "sangani_acrivos_gap_conductance",
    "gap_conductance_truncated",
    "equivalent_slot_viscosity",
    "cylinder_gap_fields",
    "assign_cylinder_ids",
    "make_cylinder_mask",
]

PHI_MAX_SQ = math.pi / 4.0                       # square array: 0.785398...
PHI_MAX_HEX = math.pi / (2.0 * math.sqrt(3.0))   # hexagonal:    0.906900...
GEBART_C1_SQ = 16.0 / (9.0 * math.pi * math.sqrt(2.0))
GEBART_C1_HEX = 16.0 / (9.0 * math.pi * math.sqrt(6.0))
GAP_INTEGRAL_CONST = 3.0 * math.pi / 8.0


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------
def gebart_sqrt_gap(phi: float, arrangement: str = "hex") -> float:
    """Return ``s = sqrt(phi_max/phi) - 1`` (Gebart's near-jamming parameter)."""
    if arrangement == "hex":
        phi_max = PHI_MAX_HEX
    elif arrangement in ("sq", "square"):
        phi_max = PHI_MAX_SQ
    else:
        raise ValueError(f"unknown arrangement {arrangement!r}")
    if not 0.0 < phi < phi_max:
        raise ValueError(f"phi={phi} outside (0, {phi_max})")
    return math.sqrt(phi_max / phi) - 1.0


def gebart_permeability(phi: float, radius: float, arrangement: str = "hex") -> float:
    """Gebart (1992) near-jamming transverse permeability ``k = C1 R^2 s^(5/2)``."""
    s = gebart_sqrt_gap(phi, arrangement)
    c1 = GEBART_C1_HEX if arrangement == "hex" else GEBART_C1_SQ
    return c1 * radius**2 * s**2.5


def delta_min_over_cell(phi: float, arrangement: str = "hex") -> float:
    """Grid-independent minimum surface-to-surface gap in units of the cell size.

    Square array: ``R/W = sqrt(phi/pi)`` -> ``delta/W = 1 - 2 sqrt(phi/pi)``.
    Hexagonal array with lattice constant ``a``:
    ``R/a = sqrt(sqrt(3) phi / (2 pi))`` -> ``delta/a = 1 - 2 R/a``.
    """
    if arrangement == "hex":
        ratio = math.sqrt(math.sqrt(3.0) * phi / (2.0 * math.pi))
    elif arrangement in ("sq", "square"):
        ratio = math.sqrt(phi / math.pi)
    else:
        raise ValueError(f"unknown arrangement {arrangement!r}")
    return 1.0 - 2.0 * ratio


# ---------------------------------------------------------------------------
# Sangani & Acrivos near-contact conductance
# ---------------------------------------------------------------------------
def sangani_acrivos_gap_conductance(radius: float, delta: float) -> float:
    """Parabolic-gap conductance ``(3 pi/8) sqrt(R) / delta^(5/2)``.

    Units 1/length^2; it is ``\\int ds/h^3``.  Diverges as ``delta^{-5/2}``,
    the near-contact singularity of Sangani & Acrivos (1982).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return GAP_INTEGRAL_CONST * math.sqrt(radius) / delta**2.5


def gap_conductance_truncated(radius: float, delta: float, s_max: float) -> float:
    """``\\int_{-s_max}^{s_max} ds / (delta + s^2/R)^3`` in closed form.

    The limit ``s_max -> inf`` must recover
    :func:`sangani_acrivos_gap_conductance`.
    """
    if radius <= 0.0 or delta <= 0.0 or s_max <= 0.0:
        raise ValueError("radius, delta and s_max must be positive")
    a = math.sqrt(delta)
    x = s_max / math.sqrt(radius)
    prim = (
        x * (3.0 * x * x + 5.0 * a * a) / (8.0 * a**4 * (x * x + a * a) ** 2)
        + 3.0 * math.atan(x / a) / (8.0 * a**5)
    )
    return 2.0 * prim * math.sqrt(radius)


# ---------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------
def equivalent_slot_viscosity(
    nu: float,
    delta_true: float,
    delta_mask: float,
    dx: float = 1.0,
    cutoff: float = 1.5,
    exponent: float = 3.0,
    min_factor: float = 0.05,
    max_factor: float = 1.0e3,
) -> tuple[float, float, bool]:
    """Gap-based effective viscosity for one sub-lattice throat.

    Returns ``(nu_eff, weight, clipped)`` with

        ``nu_eff = nu * (delta_mask / delta_true) ** (exponent * weight)``
        ``weight = max(0, 1 - delta_true / (cutoff * dx))``

    ``weight`` is exactly 0 (``nu_eff == nu``) for resolved throats
    ``delta_true >= cutoff*dx`` and ramps linearly to 1 as ``delta_true -> 0``,
    so the closure is a strict no-op away from sub-lattice gaps.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    if delta_true <= 0.0 or delta_mask <= 0.0:
        raise ValueError("gaps must be positive")
    weight = max(0.0, 1.0 - delta_true / (cutoff * dx))
    if weight == 0.0:
        return nu, 0.0, False
    factor = (delta_mask / delta_true) ** (exponent * weight)
    clipped = False
    if factor < min_factor:
        factor, clipped = min_factor, True
    elif factor > max_factor:
        factor, clipped = max_factor, True
    return nu * factor, weight, clipped


# ---------------------------------------------------------------------------
# geometry: masks, ids and gap fields
# ---------------------------------------------------------------------------
def make_cylinder_mask(
    nx: int,
    ny: int,
    centres: list[tuple[float, float]],
    radius: float,
) -> np.ndarray:
    """Boolean solid mask (True = solid) for periodic cylinders.

    A node is solid when its centre lies inside any periodically imaged
    cylinder, ``r <= radius``.
    """
    ys, xs = np.meshgrid(
        np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij"
    )
    d2 = np.full((ny, nx), np.inf)
    for cx, cy in centres:
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                sx = cx + ix * nx
                sy = cy + iy * ny
                d2 = np.minimum(d2, (xs - sx) ** 2 + (ys - sy) ** 2)
    return d2 <= radius**2


def assign_cylinder_ids(
    solid: np.ndarray, centres: list[tuple[float, float]]
) -> np.ndarray:
    """Nearest-cylinder index for every solid node (-1 for fluid nodes)."""
    ny, nx = solid.shape
    ys, xs = np.meshgrid(
        np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij"
    )
    best = np.full((ny, nx), np.inf)
    idx = np.full((ny, nx), -1, dtype=int)
    for k, (cx, cy) in enumerate(centres):
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                d2 = (xs - (cx + ix * nx)) ** 2 + (ys - (cy + iy * ny)) ** 2
                better = d2 < best
                best = np.where(better, d2, best)
                idx = np.where(better, k, idx)
    return np.where(solid, idx, -1)


def _ray_exact(
    xs: np.ndarray,
    ys: np.ndarray,
    direction: tuple[int, int],
    centres: list[tuple[float, float]],
    radius: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the first exact cylinder surface along ``direction``.

    Returns ``(t, body)``; ``body = -1`` marks "no surface hit".
    """
    dx, dy = direction
    t_best = np.full(xs.shape, np.inf)
    body_best = np.full(xs.shape, -1, dtype=int)
    for k, (cx, cy) in enumerate(centres):
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                px = cx + ix * nx
                py = cy + iy * ny
                ox = xs - px
                oy = ys - py
                b = ox * dx + oy * dy
                c0 = ox * ox + oy * oy - radius**2
                disc = b * b - c0
                ok = (disc >= 0.0) & (t_best == np.inf)
                if not np.any(ok):
                    continue
                sq = np.sqrt(np.where(ok, disc, 0.0))
                t = np.where((-b - sq) > 0.0, -b - sq, -b + sq)
                cand = ok & (t > 0.0) & (t < t_best)
                t_best = np.where(cand, t, t_best)
                body_best = np.where(cand, k, body_best)
    return t_best, body_best


def _ray_mask(
    x0: np.ndarray,
    y0: np.ndarray,
    direction: tuple[int, int],
    solid: np.ndarray,
    ids: np.ndarray,
    max_steps: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the first solid *node* along ``direction`` on the voxel mask.

    The bounce-back wall sits half-way between the last fluid node and the first
    solid node, so ``t = n_steps - 0.5`` in lattice units.
    """
    ny, nx = solid.shape
    dx, dy = direction
    steps = np.zeros(x0.shape, dtype=int)
    hit = np.zeros(x0.shape, dtype=bool)
    body = np.full(x0.shape, -1, dtype=int)
    curx = x0.copy()
    cury = y0.copy()
    for n in range(1, max_steps + 1):
        curx = (curx + dx) % nx
        cury = (cury + dy) % ny
        s = solid[cury, curx]
        newly = s & (~hit)
        if np.any(newly):
            body[newly] = ids[cury[newly], curx[newly]]
            steps[newly] = n
            hit |= newly
        if hit.all():
            break
    t = np.where(hit, steps - 0.5, np.inf)
    return t, body


def cylinder_gap_fields(
    nx: int,
    ny: int,
    centres: list[tuple[float, float]],
    radius: float,
    solid: np.ndarray | None = None,
) -> dict:
    """Local throat geometry for every fluid node of a periodic cylinder array.

    For each of the two opposed axis pairs (x, -x) and (y, -y) the first surface
    hit by an exact ray cast (and, separately, by a mask ray cast) is recorded;
    a pair contributes a throat width only when the two hits belong to *different
    cylinders*.  The node's throat width is the minimum over the contributing
    pairs.

    Returns a dict with ``delta_true`` / ``delta_mask`` (inf where no throat),
    ``pair_dir`` (0 = x-pair, 1 = y-pair, -1 = none), ``valid`` and
    ``delta_min`` (the smallest true throat width found, grid-dependent estimate
    of the geometric delta_min).
    """
    ids = assign_cylinder_ids(solid, centres) if solid is not None else None
    if solid is None:
        solid = make_cylinder_mask(nx, ny, centres, radius)
        ids = assign_cylinder_ids(solid, centres)

    fluid_y, fluid_x = np.nonzero(~solid)
    xs = fluid_x.astype(float)
    ys = fluid_y.astype(float)

    delta_true = np.full(xs.shape, np.inf)
    delta_mask = np.full(xs.shape, np.inf)
    pair_dir = np.full(xs.shape, -1, dtype=int)

    for pi, (dpair, mpair) in enumerate(
        (((1, 0), (-1, 0)), ((0, 1), (0, -1)))
    ):
        te_p, be_p = _ray_exact(xs, ys, dpair, centres, radius, nx, ny)
        te_m, be_m = _ray_exact(xs, ys, mpair, centres, radius, nx, ny)
        tm_p, bm_p = _ray_mask(fluid_x, fluid_y, dpair, solid, ids)
        tm_m, bm_m = _ray_mask(fluid_x, fluid_y, mpair, solid, ids)
        # opposed first-hit distances: a genuine throat is the only place where
        # *both* rays stop within a lattice spacing, so the sum is the throat
        # width.  Next to one convex wall one of the two rays crosses the pore and
        # the sum is large -- that is what excludes convex-wall neighbourhoods.
        hit_e = (be_p >= 0) & (be_m >= 0)
        hit_m = (bm_p >= 0) & (bm_m >= 0)
        w_true = np.where(hit_e, te_p + te_m, np.inf)
        w_mask = np.where(hit_m, tm_p + tm_m, np.inf)
        better = w_true < delta_true
        delta_true = np.where(better, w_true, delta_true)
        delta_mask = np.where(better, w_mask, delta_mask)
        pair_dir = np.where(better, pi, pair_dir)

    valid = np.isfinite(delta_true)
    return {
        "delta_true": delta_true,
        "delta_mask": delta_mask,
        "pair_dir": pair_dir,
        "valid": valid,
        "fluid_x": fluid_x,
        "fluid_y": fluid_y,
        "delta_min": float(np.min(delta_true[valid])) if np.any(valid) else float("nan"),
    }


def closure_nu_field(
    nu: float,
    gap: dict,
    shape: tuple[int, int],
    dx: float = 1.0,
    cutoff: float = 1.5,
    exponent: float = 3.0,
) -> dict:
    """Build the per-node effective-viscosity field implied by a gap field.

    Returns ``nu_field`` (ny, nx), ``corrected`` (bool mask), ``min_factor``,
    ``max_factor``, ``n_corrected``, ``n_valid``.
    """
    ny, nx = shape
    nu_field = np.full((ny, nx), nu)
    corrected = np.zeros((ny, nx), dtype=bool)
    fx, fy = gap["fluid_x"], gap["fluid_y"]
    d_true, d_mask = gap["delta_true"], gap["delta_mask"]
    lo = np.inf
    hi = 0.0
    n_corr = 0
    for k in range(fx.size):
        dt = d_true[k]
        dm = d_mask[k]
        if not np.isfinite(dt) or not np.isfinite(dm):
            continue
        if dt >= cutoff * dx:
            continue
        nu_eff, _w, _c = equivalent_slot_viscosity(
            nu, dt, dm, dx=dx, cutoff=cutoff, exponent=exponent
        )
        nu_field[fy[k], fx[k]] = nu_eff
        corrected[fy[k], fx[k]] = True
        lo = min(lo, nu_eff / nu)
        hi = max(hi, nu_eff / nu)
        n_corr += 1
    return {
        "nu_field": nu_field,
        "corrected": corrected,
        "min_factor": float(lo) if n_corr else 1.0,
        "max_factor": float(hi) if n_corr else 1.0,
        "n_corrected": n_corr,
        "n_valid": int(np.sum(gap["valid"])),
    }