"""Model-scale ship total-resistance closure with per-leg evidence grades.

Motivation
----------
The repository's hull-resistance cases are stuck between two calibers: a
verified laminar ``Re=O(10^3)`` friction frame (``suboff_re1000`` family) that
cannot see wave-making, and a model-scale ``Re=O(10^7)``, ``Fn``-dependent
experimental frame that the lattice solver cannot reach (``benchmarks/STATUS.md``:
"ship resistance needs wave-making").  This module supplies the missing
*bookkeeping*: it closes the model-scale resistance identity

    C_T(Fn) = (1 + k) C_F(Re) + C_W(Fn)

with each leg carrying its own, explicitly graded, evidence, and it refuses to
emit a comparison number when a leg is outside its validity domain.

Evidence grades (never mixed, never promoted)
---------------------------------------------
``GEOMETRY``
    Half-breadth grid ``b(x, z)`` -> ``nabla``, ``Cb``, ``L``, ``B``, ``T``,
    ``S_wet``.  Fail-closed against locked principal particulars: a grid that
    does not reproduce ``Cb`` within the requested tolerance returns
    ``status='geometry_rejected'`` and no resistance number.
``CORRELATION``
    ITTC-1957 friction line (reused from :mod:`tensorlbm.hydrodynamics`) plus
    the Holtrop-Mennen (1984) form-factor regression.  Not a simulation.
``THEORY``
    Michell thin-ship integral evaluated on the hull's own ``db/dx`` (real
    quadrature), gated by an explicit slenderness admission test.  Outside the
    gate the wave leg returns ``outside_validity`` and no ``C_W``.

Numerical verification of the wave leg rests on identities the thin-ship
integral must satisfy exactly -- zero for ``db/dx = 0``, invariance under
fore-aft mirroring, exact linearity in half-breadth, and geometric convergence
of the quadrature -- so the quadrature is verified independently of any
experiment.  What is *not* verified by those identities is the thin-ship
approximation itself; that is a documented model limitation.

Lattice-Boltzmann legs are deliberately absent: at the grids reachable in this
repository the resolution Reynolds number delivers the friction coefficient at
``Re_resolved`` (``Ct_fric/ITTC ~ 2.5`` in
``benchmarks/pending/suboff_highre_resistance``), so an LBM total-resistance
claim at model scale would be an unearned promotion.  See
``benchmarks/pending/ship_resistance_closure/README.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .hydrodynamics import ittc57_friction_coefficient

__all__ = [
    "EVALUATION_GRADES",
    "MAX_PHASE_PER_CELL",
    "MICHELL_LEVELS",
    "SLENDERNESS_GATE",
    "HullSectionGrid",
    "ResistanceClosure",
    "analytic_wetted_area",
    "band_verdict",
    "evaluate_case",
    "gci",
    "holtrop_form_factor",
    "hull_section_grid",
    "michell_wave_resistance",
    "quadrature_adequacy",
    "monotone_increasing",
    "resistance_closure",
    "series60_fullness_grid",
    "slenderness_admission",
    "stl_hull_section_grid",
]

G = 9.81
RHO = 1000.0
NU = 1.139e-6  # m^2/s, fresh water at ~15.5 C (Todd model-test convention)

EVALUATION_GRADES = ("GEOMETRY", "CORRELATION", "THEORY")

#: Thin-ship admission limits (Michell 1898; Tuck 1989 survey).  ``Cb <= 0.60``
#: is the classical slenderness statement; the beam and draft ratios bound the
#: neglected ``O((B/L)^2)`` terms.
SLENDERNESS_GATE = {
    "min_length_over_beam": 6.0,
    "max_block_coefficient": 0.60,
    "max_beam_over_length": 0.16,
    "max_draft_over_length": 0.10,
}


# ---------------------------------------------------------------------------
# Geometry leg
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HullSectionGrid:
    """Half-breadth grid ``b(x, z)`` in metres plus locked principal particulars.

    ``x`` runs fore->aft over ``[0, length_m]``; ``z`` runs keel->waterline over
    ``[-draft_m, 0]`` (waterline at ``z = 0``).  ``half_breadth_m[i, j]`` is the
    half-breadth (m) at ``x[i], z[j]``; zero outside the hull envelope.
    """

    name: str
    length_m: float
    beam_m: float
    draft_m: float
    x_m: np.ndarray
    z_m: np.ndarray
    half_breadth_m: np.ndarray
    cb_locked: float | None = None
    cb_source: str = ""
    geometry_source: str = ""
    wetted_area_override_m2: float | None = None
    wetted_area_source: str = ""

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=float)
        z = np.asarray(self.z_m, dtype=float)
        b = np.asarray(self.half_breadth_m, dtype=float)
        if b.shape != (x.size, z.size):
            raise ValueError("half_breadth must have shape (len(x), len(z))")
        if not (np.all(np.diff(x) > 0.0) and np.all(np.diff(z) > 0.0)):
            raise ValueError("x and z grids must be strictly increasing")
        if not (self.length_m > 0 and self.beam_m > 0 and self.draft_m > 0):
            raise ValueError("principal particulars must be positive")
        if np.any(b < 0.0):
            raise ValueError("half-breadth must be non-negative")
        if abs(z[0] + self.draft_m) > 1e-9 * max(1.0, self.draft_m) or abs(z[-1]) > 1e-9:
            raise ValueError("z grid must span exactly [-draft_m, 0]")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "half_breadth_m", b)

    # -- integrals -------------------------------------------------------
    @property
    def dx_m(self) -> float:
        return float(self.x_m[1] - self.x_m[0])

    @property
    def dz_m(self) -> float:
        return float(self.z_m[1] - self.z_m[0])

    @property
    def volume_m3(self) -> float:
        """Submerged volume ``2 * int int b dx dz`` (trapezoid in both axes)."""
        b = self.half_breadth_m
        return float(2.0 * np.trapezoid(np.trapezoid(b, self.z_m, axis=1), self.x_m))

    @property
    def block_coefficient(self) -> float:
        return self.volume_m3 / (self.length_m * self.beam_m * self.draft_m)

    @property
    def mid_ship_section_coefficient(self) -> float:
        """``Cm`` from the section at the maximum-breadth station."""
        b = self.half_breadth_m
        i = int(np.argmax(b.max(axis=1)))
        area = 2.0 * float(np.trapezoid(b[i], self.z_m))
        return area / (self.beam_m * self.draft_m)

    @property
    def prismatic_coefficient(self) -> float:
        return self.block_coefficient / self.mid_ship_section_coefficient

    @property
    def waterplane_coefficient(self) -> float:
        """``Cwp`` measured 5 % of the draft below the design waterline.

        The waterline itself is not used because a sampled surface (STL) has an
        open cut there, which under-populates the topmost bin; 5 % of draft is
        inside the wall-sided band of every hull treated here and reproduces the
        analytic waterplane to better than 1 % for the parametric forms.
        """
        z_probe = -0.05 * self.draft_m
        b_row = np.array(
            [np.interp(z_probe, self.z_m, self.half_breadth_m[i]) for i in range(self.x_m.size)]
        )
        return 2.0 * float(np.trapezoid(b_row, self.x_m)) / (self.length_m * self.beam_m)

    @property
    def wetted_area_m2(self) -> float:
        """Wetted hull surface (waterplane excluded).

        ``dA = sqrt(1 + b_x^2 + b_z^2) dx dz`` for a surface ``y = b(x, z)``.
        The override exists for sampled surfaces (STL): there the first-hand
        triangle area is authoritative and the finite-difference area is
        inflated by sampling jitter.
        """
        if self.wetted_area_override_m2 is not None:
            return float(self.wetted_area_override_m2)
        bx, bz = np.gradient(self.half_breadth_m, self.dx_m, self.dz_m)
        return float(2.0 * np.sqrt(1.0 + bx**2 + bz**2).sum() * self.dx_m * self.dz_m)

    # -- audit -----------------------------------------------------------
    def audit(self, cb_tolerance: float = 0.02) -> dict[str, object]:
        """Fail-closed geometry audit against the locked block coefficient."""
        cb = self.block_coefficient
        record: dict[str, object] = {
            "grade": "GEOMETRY",
            "name": self.name,
            "length_m": self.length_m,
            "beam_m": self.beam_m,
            "draft_m": self.draft_m,
            "cb_grid": cb,
            "cb_locked": self.cb_locked,
            "cb_source": self.cb_source,
            "cm_grid": self.mid_ship_section_coefficient,
            "cp_grid": self.prismatic_coefficient,
            "cwp_grid": self.waterplane_coefficient,
            "volume_m3": self.volume_m3,
            "wetted_area_m2": self.wetted_area_m2,
            "wetted_area_source": self.wetted_area_source
            or ("override" if self.wetted_area_override_m2 is not None else "half_breadth_integral"),
            "geometry_source": self.geometry_source,
            "grid": (int(self.x_m.size), int(self.z_m.size)),
            "cb_tolerance": cb_tolerance,
        }
        if self.cb_locked is None:
            record["status"] = "unlocked"
            record["cb_error_pct"] = None
            return record
        err = (cb - self.cb_locked) / self.cb_locked
        record["cb_error_pct"] = 100.0 * err
        record["status"] = "pass" if abs(err) <= cb_tolerance else "geometry_rejected"
        return record


def analytic_wetted_area(
    half_beam_norm: Callable[[np.ndarray, np.ndarray], np.ndarray],
    length_m: float,
    beam_m: float,
    draft_m: float,
    nx: int = 801,
    nz: int = 641,
) -> float:
    """Wetted surface of an analytic half-breadth function, integrated finely.

    The keel form ``zeta^q`` has an unbounded ``db/dz`` at ``zeta -> 0`` for the
    fuller members, so the area integral is evaluated on a ladder four times
    finer than any map used downstream (converged to <0.1 % between ``nz=641``
    and ``nz=1281``; see the pytest ladder test).  The stored map is then free
    to be coarse without biasing ``C_W`` through its normalising area.
    """
    xi = np.linspace(-1.0, 1.0, int(nx))
    zeta = np.linspace(0.0, 1.0, int(nz))
    XI, ZE = np.meshgrid(xi, zeta, indexing="ij")
    hb = np.asarray(half_beam_norm(XI, ZE), dtype=float) * (beam_m / 2.0)
    x = np.linspace(0.0, length_m, int(nx))
    z = draft_m * (zeta - 1.0)
    bx, bz = np.gradient(hb, x, z)
    return float(2.0 * np.trapezoid(np.trapezoid(np.sqrt(1.0 + bx**2 + bz**2), z, axis=1), x))


def _grid_from_half_beam(
    name: str,
    half_beam_norm: Callable[[np.ndarray, np.ndarray], np.ndarray],
    length_m: float,
    beam_m: float,
    draft_m: float,
    nx: int,
    nz: int,
    cb_locked: float | None = None,
    cb_source: str = "",
    geometry_source: str = "",
) -> HullSectionGrid:
    """Sample a normalised half-breadth function onto the section grid.

    ``half_beam_norm(xi, zeta)`` returns ``b / (B/2)`` for ``xi in [-1, 1]`` and
    ``zeta in [0, 1]`` -- the convention of :mod:`tensorlbm.ship_cad`.
    """
    if nx < 3 or nz < 3:
        raise ValueError("need at least 3 grid points per axis")
    xi = np.linspace(-1.0, 1.0, nx)
    zeta = np.linspace(0.0, 1.0, nz)
    XI, ZE = np.meshgrid(xi, zeta, indexing="ij")
    hb = np.asarray(half_beam_norm(XI, ZE), dtype=float) * (beam_m / 2.0)
    return HullSectionGrid(
        name=name,
        length_m=length_m,
        beam_m=beam_m,
        draft_m=draft_m,
        x_m=np.linspace(0.0, length_m, nx),
        z_m=draft_m * (zeta - 1.0),
        half_breadth_m=np.clip(hb, 0.0, None),
        cb_locked=cb_locked,
        cb_source=cb_source,
        geometry_source=geometry_source or "analytic_half_beam",
        wetted_area_override_m2=analytic_wetted_area(half_beam_norm, length_m, beam_m, draft_m),
        wetted_area_source="analytic_half_beam_fine_trapezoid(nx=801,nz=641)",
    )


def series60_fullness_grid(
    cb: float,
    length_m: float = 6.096,
    beam_m: float | None = None,
    draft_m: float | None = None,
    nx: int = 401,
    nz: int = 81,
    section_coefficient: float = 0.977,
    vertical_exponent: float | None = None,
    cb_locked: bool = True,
) -> HullSectionGrid:
    """Series 60 *fullness family* calibrated to an exact block coefficient.

    Section family ``b(x, z) = (B/2) (1 - xi^2)^p zeta^q`` -- the shape used by
    :func:`tensorlbm.ship_cad._series60_half_beam`, whose ``p = 0.51``,
    ``q = 0.30`` member is the ``Cb = 0.60`` form.  Two exponents, two targets:
    ``q = 1/Cm - 1`` is fixed by the declared midship section coefficient
    (``section_coefficient``, default 0.977 = the Series 60 ``Cb = 0.60``
    member) and ``p`` is then solved by bisection so that the *grid* block
    coefficient equals ``cb`` to ``1e-12``.  Passing ``vertical_exponent``
    overrides ``q`` (useful to reproduce one specific published member; the
    reachable block coefficient is then capped at ``1/(1 + q)``).

    The default model is Todd's 20 ft Series 60 model (``L = 6.096 m``,
    ``L/B = 6.5``, ``B/T = 2.74``).  Only the ``cb = 0.60`` member has a locked
    model-test reference in this repository; the fuller members are *declared
    parametric derivatives*, not the Todd lines.
    """
    if not (0.30 <= cb <= 0.90):
        raise ValueError("cb must be within [0.30, 0.90]")
    length_m = float(length_m)
    beam_m = float(beam_m) if beam_m is not None else length_m / 6.5
    draft_m = float(draft_m) if draft_m is not None else beam_m / 2.74
    if vertical_exponent is None:
        if not (0.5 < section_coefficient < 1.0):
            raise ValueError("section_coefficient must be within (0.5, 1.0)")
        q = 1.0 / float(section_coefficient) - 1.0
    else:
        q = float(vertical_exponent)
    if cb >= 1.0 / (1.0 + q):
        raise ValueError(
            f"cb={cb} is unreachable with vertical_exponent={q}: the family caps at "
            f"1/(1+q)={1.0 / (1.0 + q):.3f}"
        )

    def builder(p: float) -> HullSectionGrid:
        def hb(xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
            inside = (np.abs(xi) <= 1.0) & (zeta >= 0.0) & (zeta <= 1.0)
            xi_c = np.clip(xi, -1.0, 1.0)
            zeta_c = np.clip(zeta, 0.0, 1.0)
            return np.where(inside, np.clip((1.0 - xi_c**2) ** p * zeta_c**q, 0.0, 1.0), 0.0)

        return _grid_from_half_beam(
            name=f"series60_cb{cb:.3f}",
            half_beam_norm=hb,
            length_m=length_m,
            beam_m=beam_m,
            draft_m=draft_m,
            nx=nx,
            nz=nz,
            cb_locked=cb if cb_locked else None,
            cb_source="Todd (1963) Series 60 fullness target" if cb_locked else "",
            geometry_source=f"series60_fullness_family(p={p:.6g}, q={q})",
        )

    lo, hi = 0.02, 3.0
    best = builder(0.5 * (lo + hi))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        best = builder(mid)
        got = best.block_coefficient
        if abs(got - cb) < 1e-12 or (hi - lo) < 1e-7:
            break
        if got > cb:
            lo = mid
        else:
            hi = mid
    return best


def _binomial_smooth(b: np.ndarray) -> np.ndarray:
    """One pass of a separable ``[1,2,1]/4`` filter with edge replication."""
    out = b
    for axis in (0, 1):
        pad = [(1, 1) if a == axis else (0, 0) for a in (0, 1)]
        padded = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        out = 0.25 * (
            np.take(padded, range(0, n), axis=axis)
            + 2.0 * np.take(padded, range(1, n + 1), axis=axis)
            + np.take(padded, range(2, n + 2), axis=axis)
        )
    return np.clip(out, 0.0, None)


def stl_hull_section_grid(
    stl_path: str,
    length_m: float,
    waterline_z_m: float,
    beam_m: float,
    draft_m: float,
    nx: int = 145,
    nz: int = 97,
    subdivision: int = 7,
    smooth_passes: int = 2,
    cb_locked: float | None = None,
    cb_source: str = "",
    name: str = "stl_hull",
) -> HullSectionGrid:
    """Sample a half-breadth map from an STL hull surface.

    Every triangle is sampled on a barycentric lattice and each ``(x, z)`` bin
    keeps the largest ``|y|`` seen.  Bins are then lightly smoothed (binomial,
    ``smooth_passes`` times) so ``db/dx`` is not dominated by sampling jitter.
    Bins with no surface sample stay zero: the map never invents breadth the
    surface does not carry.

    The wetted area is taken from the triangle areas below ``waterline_z_m``
    (first-hand geometry), not from the finite-difference map.
    """
    import trimesh  # local import: STL geometry is an optional extra

    mesh = trimesh.load(stl_path, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        raise ValueError("STL has no triangles")
    tri = verts[faces]

    k = max(1, int(subdivision))
    bary = np.asarray(
        [(i / k, j / k, (k - i - j) / k) for i in range(k + 1) for j in range(k + 1 - i)]
    )
    samples = np.einsum("pb,fbd->fpd", bary, tri).reshape(-1, 3)

    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    scale = float(length_m) / (x_max - x_min)
    xs = np.linspace(x_min, x_max, int(nx))
    zs = np.linspace(waterline_z_m - float(draft_m) / scale, waterline_z_m, int(nz))
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    ix = np.clip(((samples[:, 0] - x_min) / dx).astype(int), 0, int(nx) - 1)
    iz = np.clip(((samples[:, 2] - zs[0]) / dz).astype(int), 0, int(nz) - 1)
    keep = (samples[:, 2] >= zs[0]) & (samples[:, 2] <= waterline_z_m)
    b = np.zeros((int(nx), int(nz)))
    np.maximum.at(b, (ix[keep], iz[keep]), np.abs(samples[keep, 1]))
    for _ in range(int(smooth_passes)):
        b = _binomial_smooth(b)

    below = tri[:, :, 2].max(axis=1) <= waterline_z_m
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    wetted = float(areas[below].sum()) * scale**2

    return HullSectionGrid(
        name=name,
        length_m=float(length_m),
        beam_m=float(beam_m),
        draft_m=float(draft_m),
        x_m=(xs - x_min) * scale,
        z_m=(zs - waterline_z_m) * scale,
        half_breadth_m=b * scale,
        cb_locked=cb_locked,
        cb_source=cb_source,
        geometry_source=(
            f"stl_surface_sampling({str(stl_path).rsplit('/', 1)[-1]}, k={k}, "
            f"smooth={smooth_passes})"
        ),
        wetted_area_override_m2=wetted,
        wetted_area_source="stl_triangle_area_below_waterline",
    )


def hull_section_grid(name: str, length_m: float | None = None, **kwargs) -> HullSectionGrid:
    """Dispatcher for the hulls used by the resistance campaign."""
    from .ship_cad import _kcs_half_beam, _series60_half_beam, _wigley_half_beam

    key = str(name).lower()
    if key == "kcs":
        length = 7.28 if length_m is None else float(length_m)
        return _grid_from_half_beam(
            name="kcs",
            half_beam_norm=_kcs_half_beam,
            length_m=length,
            beam_m=length * 32.2 / 230.0,
            draft_m=length * 10.8 / 230.0,
            nx=int(kwargs.pop("nx", 401)),
            nz=int(kwargs.pop("nz", 81)),
            cb_locked=float(kwargs.pop("cb_locked", 0.6505)),
            cb_source="KCS principal particulars L/B=7.143, B/T=2.981 (Tokyo-2015)",
            geometry_source="ship_cad._kcs_half_beam (parametric approximation)",
        )
    if key in {"series60", "series_60"}:
        return series60_fullness_grid(
            cb=float(kwargs.pop("cb", 0.60)),
            length_m=6.096 if length_m is None else float(length_m),
            **kwargs,
        )
    if key == "wigley":
        length = 6.096 if length_m is None else float(length_m)
        return _grid_from_half_beam(
            name="wigley",
            half_beam_norm=_wigley_half_beam,
            length_m=length,
            beam_m=length / 10.0,
            draft_m=length / 16.0,
            nx=int(kwargs.pop("nx", 401)),
            nz=int(kwargs.pop("nz", 81)),
            cb_locked=4.0 / 9.0,
            cb_source="Wigley parabolic closed form Cb = 4/9",
            geometry_source="ship_cad._wigley_half_beam",
        )
    if key in {"dtmb5415", "5415"}:
        stl = kwargs.pop("stl_path", None)
        if stl is None:
            raise ValueError("dtmb5415 requires an explicit stl_path")
        length = 5.46 if length_m is None else float(length_m)
        return stl_hull_section_grid(
            stl_path=stl,
            length_m=length,
            waterline_z_m=float(kwargs.pop("waterline_z_m", 6.15)),
            beam_m=float(kwargs.pop("beam_m", 19.1034 * length / 142.03)),
            draft_m=float(kwargs.pop("draft_m", 6.15 * length / 142.03)),
            cb_locked=float(kwargs.pop("cb_locked", 0.507)),
            cb_source="DTMB 5415 bare-hull Cb = 0.507 (IIHR model-test convention)",
            name="dtmb5415",
            **kwargs,
        )
    raise ValueError(f"unknown hull {name!r}")


# ---------------------------------------------------------------------------
# Viscous leg (correlation)
# ---------------------------------------------------------------------------


def holtrop_form_factor(
    length_m: float,
    beam_m: float,
    draft_m: float,
    volume_m3: float,
    prismatic_coefficient: float,
    lcb_fraction: float = 0.0,
    cstern: float = 0.0,
) -> float:
    """Holtrop-Mennen (1984) form factor ``1 + k``.

    ``1 + k = 0.93 + 0.487118 c14 (B/L)^1.06806 (T/L)^0.46106 (L/L_R)^0.121563
    (L^3/nabla)^0.36486 (1 - Cp)^-0.604247`` with ``L_R`` the length of the run
    and ``c14 = 1 + 0.011 cstern``.  ``lcb_fraction`` and ``cstern`` are
    *explicit assumptions* (0 % and 0 for a conventional transom-less hull)
    because these hull forms carry no locked LCB/Cstern; the sensitivity of
    ``C_T`` to those inputs is reported by :func:`resistance_closure`.
    """
    if volume_m3 <= 0 or not (0.0 < prismatic_coefficient < 1.0):
        raise ValueError("volume and prismatic coefficient must be physical")
    if length_m <= 0 or beam_m <= 0 or draft_m <= 0:
        raise ValueError("principal particulars must be positive")
    c14 = 1.0 + 0.011 * cstern
    lr = length_m * (
        1.0
        - prismatic_coefficient
        + 0.06 * prismatic_coefficient * lcb_fraction / max(4.0 * prismatic_coefficient - 1.0, 1e-9)
    )
    return float(
        0.93
        + 0.487118
        * c14
        * (beam_m / length_m) ** 1.06806
        * (draft_m / length_m) ** 0.46106
        * (length_m / lr) ** 0.121563
        * (length_m**3 / volume_m3) ** 0.36486
        * (1.0 - prismatic_coefficient) ** (-0.604247)
    )


# ---------------------------------------------------------------------------
# Wave leg (theory)
# ---------------------------------------------------------------------------


def slenderness_admission(grid: HullSectionGrid) -> dict[str, object]:
    """Thin-ship admission test evaluated on the *measured* geometry."""
    b_over_l = grid.beam_m / grid.length_m
    checks = {
        "length_over_beam": {
            "value": grid.length_m / grid.beam_m,
            "limit": SLENDERNESS_GATE["min_length_over_beam"],
            "ok": grid.length_m / grid.beam_m >= SLENDERNESS_GATE["min_length_over_beam"],
        },
        "block_coefficient": {
            "value": grid.block_coefficient,
            "limit": SLENDERNESS_GATE["max_block_coefficient"],
            "ok": grid.block_coefficient <= SLENDERNESS_GATE["max_block_coefficient"],
        },
        "beam_over_length": {
            "value": b_over_l,
            "limit": SLENDERNESS_GATE["max_beam_over_length"],
            "ok": b_over_l <= SLENDERNESS_GATE["max_beam_over_length"],
        },
        "draft_over_length": {
            "value": grid.draft_m / grid.length_m,
            "limit": SLENDERNESS_GATE["max_draft_over_length"],
            "ok": grid.draft_m / grid.length_m <= SLENDERNESS_GATE["max_draft_over_length"],
        },
    }
    admitted = all(bool(c["ok"]) for c in checks.values())
    return {
        "status": "applicable" if admitted else "outside_validity",
        "admitted": admitted,
        "checks": checks,
        "gate": dict(SLENDERNESS_GATE),
    }


def _bilinear(grid: HullSectionGrid, XI: np.ndarray, ZE: np.ndarray) -> np.ndarray:
    x, z, b = grid.x_m, grid.z_m, grid.half_breadth_m
    xi = np.interp(XI.ravel(), x, np.arange(x.size))
    zi = np.interp(ZE.ravel(), z, np.arange(z.size))
    i0 = np.clip(np.floor(xi).astype(int), 0, x.size - 2)
    j0 = np.clip(np.floor(zi).astype(int), 0, z.size - 2)
    tx = np.clip(xi - i0, 0.0, 1.0)
    tz = np.clip(zi - j0, 0.0, 1.0)
    out = (
        b[i0, j0] * (1 - tx) * (1 - tz)
        + b[i0 + 1, j0] * tx * (1 - tz)
        + b[i0, j0 + 1] * (1 - tx) * tz
        + b[i0 + 1, j0 + 1] * tx * tz
    )
    return out.reshape(XI.shape)


def _michell_single(
    grid: HullSectionGrid, fn: float, n_x: int, n_z: int, n_theta: int, u_max: float
) -> tuple[float, float, float]:
    """One quadrature level of the Michell integral -> (R_W, C_W, S_wet)."""
    x = np.linspace(0.0, grid.length_m, int(n_x))
    z = np.linspace(-grid.draft_m, 0.0, int(n_z))
    XI, ZE = np.meshgrid(x, z, indexing="ij")
    b = _bilinear(grid, XI, ZE)
    dx = x[1] - x[0]
    dz = z[1] - z[0]

    speed = float(fn) * math.sqrt(G * grid.length_m)
    k0 = G / speed**2
    dbdx = np.zeros_like(b)
    dbdx[1:-1] = (b[2:] - b[:-2]) / (2.0 * dx)
    src = dbdx * dx * dz

    xf = np.repeat(x, b.shape[1])
    zf = np.tile(z, b.shape[0])
    sf = src.reshape(-1)
    u = np.linspace(0.0, u_max, int(n_theta))
    sec = np.sqrt(1.0 + u**2)
    total = 0.0
    for s in sec:
        k = k0 * s * s
        amp = sf * np.exp(k * zf)  # z <= 0 -> depth decay
        phase = k0 * s * xf
        re = float(np.dot(amp, np.cos(phase)))
        im = float(np.dot(amp, np.sin(phase)))
        total += (re * re + im * im) * s
    total *= u[1] - u[0]
    rw = 4.0 * RHO * G**2 / (math.pi * speed**2) * total

    bx, bz = np.gradient(b, dx, dz)
    s_wet = float(2.0 * np.sqrt(1.0 + bx**2 + bz**2).sum() * dx * dz)
    cw = rw / (0.5 * RHO * speed**2 * grid.wetted_area_m2)
    return rw, cw, s_wet


MICHELL_LEVELS = ((91, 31, 300), (181, 61, 600), (361, 121, 1200))

#: Axial phase advance per x-cell of the finest quadrature level above which the
#: oscillatory integrand is aliased (Nyquist = pi rad/cell; a factor 8 margin is
#: demanded, i.e. >= ~8 cells per axial wave).
MAX_PHASE_PER_CELL = math.pi / 8.0


def quadrature_adequacy(grid: HullSectionGrid, fn: float, n_x: int, u_max: float = 4.0) -> dict:
    """Phase resolution of the Michell integrand on the finest quadrature level.

    ``k0 = g/U^2`` and the secant factor raise the axial wavenumber to
    ``k0 sec(theta)``; the finest level must resolve the shortest resulting
    wavelength.  Below the resolution limit the integral aliases and the module
    refuses to report a coefficient (fail-closed), which is what makes a
    ``Fn -> 0`` evaluation honest instead of merely fast.
    """
    speed = float(fn) * math.sqrt(G * grid.length_m)
    k0 = G / speed**2
    dx = grid.length_m / max(int(n_x) - 1, 1)
    # Only the secant range that still reaches into the hull contributes: the
    # vertical factor exp(k0 sec^2 z) is dead once k0 sec^2 T exceeds
    # ~ln(1/eps) (eps = 1e-6 relative amplitude).  Using u_max unconditionally
    # would over-restrict the low-Froude end.
    decay_budget = math.log(1.0e6)
    kt = k0 * grid.draft_m
    sec_effective = math.sqrt(max(1.0, decay_budget / kt)) if kt > 0 else float(u_max)
    sec_max = min(math.sqrt(1.0 + float(u_max) ** 2), sec_effective)
    phase_per_cell = k0 * sec_max * dx
    return {
        "k0_per_m": k0,
        "dx_m": dx,
        "sec_max": sec_max,
        "u_max_supplied": float(u_max),
        "phase_per_cell_rad": phase_per_cell,
        "cells_per_axial_wave": 2.0 * math.pi / phase_per_cell if phase_per_cell else float("inf"),
        "limit_rad_per_cell": MAX_PHASE_PER_CELL,
        "resolved": phase_per_cell <= MAX_PHASE_PER_CELL,
    }


def michell_wave_resistance(
    grid: HullSectionGrid,
    fn: float,
    levels: Sequence[Sequence[int]] = MICHELL_LEVELS,
    u_max: float = 4.0,
) -> dict[str, object]:
    """Thin-ship wave-resistance coefficient with a quadrature refinement study.

    ``C_W = R_W / (0.5 rho U^2 S_wet)`` with

        R_W = 4 rho g^2 / (pi U^2) int_0^{pi/2} (I^2 + J^2) sec^3(theta) dtheta

    and ``I + i J = int int (db/dx) exp(k z) exp(i k x sec(theta)) dx dz``,
    ``k = k0 sec^2(theta)``, ``k0 = g / U^2`` (Michell 1898; Havelock form as in
    :func:`tensorlbm.icebow_cad.michell_wave_resistance`, re-derived here for a
    generic half-breadth grid).  Only admitted geometries receive a ``C_W``.
    """
    if not (0.0 < float(fn) < 1.0):
        raise ValueError("Froude number must be in (0, 1)")
    fn = float(fn)
    admission = slenderness_admission(grid)
    speed = fn * math.sqrt(G * grid.length_m)
    points: list[dict[str, object]] = []
    for n_x, n_z, n_theta in [tuple(int(v) for v in level) for level in levels]:
        rw, cw, s_wet = _michell_single(grid, fn, n_x, n_z, n_theta, u_max)
        points.append(
            {
                "grid": (n_x, n_z, n_theta),
                "rw_n": rw,
                "cw": cw,
                "s_wet_m2": s_wet,
            }
        )
    record: dict[str, object] = {
        "grade": "THEORY",
        "model": "michell-thin-ship",
        "fn": fn,
        "speed_ms": speed,
        "re": speed * grid.length_m / NU,
        "k0_per_m": G / speed**2,
        "s_wet_m2": grid.wetted_area_m2,
        "s_wet_source": grid.wetted_area_source or "half_breadth_integral",
        "levels": points,
        "admission": admission,
    }
    if not admission["admitted"]:
        record.update(
            {
                "status": "outside_validity",
                "cw": None,
                "gci_pct": None,
                "convergence": None,
                "diagnostic_cw": points[-1]["cw"],
                "note": (
                    "thin-ship premise violated by the measured geometry: the diagnostic "
                    "C_W is for attribution only and must not be compared with experiment"
                ),
            }
        )
        return record

    adequacy = quadrature_adequacy(grid, fn, levels[-1][0], u_max)
    record["quadrature"] = adequacy
    if not adequacy["resolved"]:
        record.update(
            {
                "status": "quadrature_unresolved",
                "cw": None,
                "gci_pct": None,
                "convergence": None,
                "diagnostic_cw": points[-1]["cw"],
                "note": (
                    "axial phase advance exceeds %.3f rad/cell: the oscillatory "
                    "integrand is aliased, so no coefficient is reported"
                    % MAX_PHASE_PER_CELL
                ),
            }
        )
        return record

    cw_levels = [float(p["cw"]) for p in points]
    if len(cw_levels) >= 3:
        conv = gci(cw_levels[-1], cw_levels[-2], cw_levels[-3], ratio=2.0)
    else:  # pragma: no cover - defensive
        conv = {"gci_fine_pct": None, "status": "insufficient_levels"}
    record.update(
        {
            "status": "applicable" if conv.get("gci_fine_pct") is not None else "not_converged",
            "cw": cw_levels[-1],
            "cw_levels": cw_levels,
            "gci_pct": conv.get("gci_fine_pct"),
            "convergence": conv,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Convergence + verdict helpers
# ---------------------------------------------------------------------------


def gci(fine: float, medium: float, coarse: float, ratio: float = 2.0) -> dict[str, object]:
    """Roache grid-convergence index with the observed order of accuracy.

    A non-monotone or degenerate ladder returns ``gci_fine_pct = None`` with an
    explanatory ``status`` instead of a fabricated index.
    """
    f, m, c = float(fine), float(medium), float(coarse)
    record: dict[str, object] = {"fine": f, "medium": m, "coarse": c, "ratio": float(ratio)}
    if not all(math.isfinite(v) for v in (f, m, c)) or ratio <= 1.0:
        record.update({"status": "invalid_input", "p_observed": None, "gci_fine_pct": None})
        return record
    d1, d2 = m - c, f - m
    if d1 == 0.0 and d2 == 0.0:
        record.update(
            {"status": "exact", "p_observed": None, "extrapolated": f, "gci_fine_pct": 0.0}
        )
        return record
    if d1 == 0.0 or d2 / d1 <= 0.0:
        record.update({"status": "non_monotone", "p_observed": None, "gci_fine_pct": None})
        return record
    p_observed = abs(math.log(abs(d2 / d1)) / math.log(ratio))
    if p_observed < 1e-6:
        record.update({"status": "degenerate", "p_observed": p_observed, "gci_fine_pct": None})
        return record
    extrapolated = f + (f - m) / (ratio**p_observed - 1.0)
    if f == 0.0:
        record.update({"status": "degenerate", "p_observed": p_observed, "gci_fine_pct": None})
        return record
    record.update(
        {
            "status": "converged" if p_observed > 0.5 else "low_order",
            "p_observed": p_observed,
            "extrapolated": extrapolated,
            "gci_fine_pct": 100.0 * 1.25 * abs((f - m) / f) / (ratio**p_observed - 1.0),
            "relative_change_pct": 100.0 * abs((f - m) / f),
        }
    )
    return record


def band_verdict(
    predicted: float | None, reference: float | None, band_pct: float
) -> dict[str, object]:
    """Fail-closed comparison band; missing or non-finite evidence never passes."""
    out: dict[str, object] = {
        "predicted": predicted,
        "reference": reference,
        "band_pct": band_pct,
    }
    if (
        predicted is None
        or reference is None
        or not math.isfinite(float(predicted))
        or not math.isfinite(float(reference))
        or float(reference) <= 0.0
    ):
        out.update({"status": "withheld", "error_pct": None, "pass": False})
        return out
    err = 100.0 * (float(predicted) - float(reference)) / float(reference)
    out.update(
        {
            "error_pct": err,
            "abs_error_pct": abs(err),
            "status": "pass" if abs(err) <= band_pct else "fail",
            "pass": bool(abs(err) <= band_pct),
        }
    )
    return out


def monotone_increasing(values: Sequence[float]) -> bool:
    """Strict monotonicity of a ladder (equal neighbours rejected)."""
    v = [float(x) for x in values]
    if len(v) < 2:
        return False
    return all(b > a for a, b in zip(v[:-1], v[1:]))


# ---------------------------------------------------------------------------
# Closure record
# ---------------------------------------------------------------------------


@dataclass
class ResistanceClosure:
    """One ``(hull, Fn)`` closure record: three graded legs and one identity."""

    geometry: dict[str, object]
    viscous: dict[str, object]
    wave: dict[str, object]
    total: dict[str, object]
    decomposition: dict[str, object]
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "geometry": self.geometry,
            "viscous": self.viscous,
            "wave": self.wave,
            "total": self.total,
            "decomposition": self.decomposition,
            "limitations": list(self.limitations),
        }


_LIMITATIONS = [
    "the viscous leg is the ITTC-1957 correlation plus a Holtrop-Mennen form-factor "
    "regression: it is not a simulation and cannot be credited to the lattice solver",
    "the wave leg is first-order thin-ship theory; it is gated by a slenderness "
    "admission test and is out of scope wherever the measured hull is fuller than Cb=0.60",
    "no free-surface/VOSET lattice run stands behind either leg at model Reynolds "
    "number; see benchmarks/pending/ship_resistance_closure/README.md for the blocker",
    "form factor uses lcb=0 % and Cstern=0 (no locked LCB/Cstern for these forms); "
    "the resulting C_T sensitivity is reported per case",
]


def resistance_closure(
    grid: HullSectionGrid,
    fn: float,
    cb_tolerance: float = 0.02,
    reference_ct: float | None = None,
    reference_source: str = "",
    band_pct: float | None = None,
    lcb_fraction: float = 0.0,
    cstern: float = 0.0,
) -> ResistanceClosure:
    """Close ``C_T = (1 + k) C_F + C_W`` for one hull at one Froude number.

    ``total['ct']`` is emitted only when the geometry audit passes *and* the
    wave leg is admitted and converged; otherwise it is ``None`` with an
    explanatory ``status`` (fail-closed).
    """
    geometry = grid.audit(cb_tolerance=cb_tolerance)
    fn = float(fn)
    speed = fn * math.sqrt(G * grid.length_m)
    re = speed * grid.length_m / NU
    cf = ittc57_friction_coefficient(re)
    one_plus_k = holtrop_form_factor(
        length_m=grid.length_m,
        beam_m=grid.beam_m,
        draft_m=grid.draft_m,
        volume_m3=grid.volume_m3,
        prismatic_coefficient=grid.prismatic_coefficient,
        lcb_fraction=lcb_fraction,
        cstern=cstern,
    )
    # form-factor assumption sensitivity: lcb +-2 % L, Cstern = -10..+10
    k_lo = min(
        holtrop_form_factor(
            grid.length_m, grid.beam_m, grid.draft_m, grid.volume_m3,
            grid.prismatic_coefficient, -0.02, -10.0,
        ),
        holtrop_form_factor(
            grid.length_m, grid.beam_m, grid.draft_m, grid.volume_m3,
            grid.prismatic_coefficient, 0.02, 10.0,
        ),
    )
    k_hi = max(
        holtrop_form_factor(
            grid.length_m, grid.beam_m, grid.draft_m, grid.volume_m3,
            grid.prismatic_coefficient, -0.02, -10.0,
        ),
        holtrop_form_factor(
            grid.length_m, grid.beam_m, grid.draft_m, grid.volume_m3,
            grid.prismatic_coefficient, 0.02, 10.0,
        ),
    )
    viscous = {
        "grade": "CORRELATION",
        "model": "ITTC-1957 friction line + Holtrop-Mennen (1984) form factor",
        "re": re,
        "cf": cf,
        "form_factor_1k": one_plus_k,
        "form_factor_1k_sensitivity": (k_lo, k_hi),
        "lcb_fraction": lcb_fraction,
        "cstern": cstern,
        "cv": one_plus_k * cf,
    }
    wave = michell_wave_resistance(grid, fn)

    ct: float | None = None
    status = "ok"
    if geometry["status"] == "geometry_rejected":
        status = "geometry_rejected"
    elif wave.get("status") != "applicable":
        status = f"wave_{wave.get('status')}"
    else:
        ct = one_plus_k * cf + float(wave["cw"])

    total: dict[str, object] = {
        "fn": fn,
        "speed_ms": speed,
        "re": re,
        "ct": ct,
        "status": status,
        "expression": "C_T = (1 + k) C_F + C_W",
    }
    if reference_ct is not None or band_pct is not None:
        total["comparison"] = band_verdict(ct, reference_ct, float(band_pct or 0.0))
        total["reference_source"] = reference_source

    cf_share = one_plus_k * cf
    cw = wave.get("cw")
    if ct is None or cw is None:
        decomposition = {
            "cf_share": cf_share,
            "cw_share": cw,
            "residual": None,
            "residual_relative": None,
            "pass": False,
            "status": "incomplete",
        }
    else:
        residual = float(ct) - (cf_share + float(cw))
        scale = max(abs(float(ct)), 1e-30)
        decomposition = {
            "cf_share": cf_share,
            "cw_share": float(cw),
            "residual": residual,
            "residual_relative": abs(residual) / scale,
            "pass": bool(abs(residual) / scale < 1e-6),
            "status": "self_consistent" if abs(residual) / scale < 1e-6 else "inconsistent",
            "cf_share_fraction": cf_share / float(ct),
            "cw_share_fraction": float(cw) / float(ct),
        }
    return ResistanceClosure(
        geometry=geometry,
        viscous=viscous,
        wave=wave,
        total=total,
        decomposition=decomposition,
        limitations=list(_LIMITATIONS),
    )


def evaluate_case(
    grid: HullSectionGrid,
    froude_numbers: Sequence[float],
    reference_ct: float | None = None,
    reference_fn: float | None = None,
    reference_source: str = "",
    band_pct: float | None = None,
    cb_tolerance: float = 0.02,
) -> dict[str, object]:
    """Evaluate a ``Fn`` ladder and judge the whole case.

    The case verdict carries:
    ``ct_monotone``  -- is ``C_T(Fn)`` strictly increasing over the ladder,
    ``band``         -- the band verdict at ``reference_fn``,
    ``gci_pct``      -- quadrature GCI of the wave leg at ``reference_fn``,
    ``decomposition``-- identity residual at ``reference_fn``.
    """
    rows: list[dict[str, object]] = []
    closures: list[ResistanceClosure] = []
    for fn in froude_numbers:
        closure = resistance_closure(
            grid,
            fn,
            cb_tolerance=cb_tolerance,
            reference_ct=reference_ct if reference_fn is not None and abs(fn - reference_fn) < 1e-12 else None,
            reference_source=reference_source,
            band_pct=band_pct if reference_fn is not None and abs(fn - reference_fn) < 1e-12 else None,
        )
        closures.append(closure)
        row = closure.as_dict()
        row["case"] = grid.name
        rows.append(row)

    ct_values = [c.total.get("ct") for c in closures]
    complete = [c for c in ct_values if isinstance(c, (int, float))]
    monotone = monotone_increasing(complete) if len(complete) == len(ct_values) and len(complete) > 1 else False
    ref_index = None
    if reference_fn is not None:
        for i, fn in enumerate(froude_numbers):
            if abs(float(fn) - float(reference_fn)) < 1e-12:
                ref_index = i
    ref = closures[ref_index] if ref_index is not None else closures[0]
    comparison = ref.total.get("comparison") or band_verdict(ref.total.get("ct"), reference_ct, float(band_pct or 0.0))
    return {
        "case": grid.name,
        "geometry": ref.geometry,
        "froude_numbers": [float(f) for f in froude_numbers],
        "ct_ladder": ct_values,
        "ct_monotone": monotone,
        "reference_fn": reference_fn,
        "reference_ct": reference_ct,
        "reference_source": reference_source,
        "band_pct": band_pct,
        "band": comparison,
        "gci_pct": ref.wave.get("gci_pct"),
        "decomposition": ref.decomposition,
        "wave_status": ref.wave.get("status"),
        "rows": rows,
        "limitations": list(_LIMITATIONS),
    }