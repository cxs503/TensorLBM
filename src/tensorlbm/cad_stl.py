"""CAD STL round trip for the parametric SUBOFF hull family.

The product story is *CAD gives us geometry, we give real-time
performance*: :mod:`tensorlbm.voxelize` reads arbitrary STL surface
meshes and rasterises them onto the B4 canonical grid, while
:mod:`tensorlbm.suboff_cad` builds the parametric DARPA SUBOFF corpus
masks analytically.  This module closes the loop that the voxelize
docs promised as the post-merge gate: tessellate the SAME analytic
description ``suboff_cad`` voxelises, export it as a binary STL the
way a CAD tool would, re-read it through ``voxelize`` and measure how
faithfully the STL path reproduces the corpus masks -- i.e. whether
geometry authored in real CAD software lands where the training data
lives.

Geometry conventions (full map in docs/cad_stl_roundtrip_20260825.md)
---------------------------------------------------------------------
* ``suboff_cad`` authored frame: hull axis = mesh x with the bow tip
  at ``x=0`` and the stern tip at ``x=length``, the axis at ``y=z=0``
  and the sail on +z.  All tessellators here work in that frame, in
  lattice units, reusing the ``suboff_cad`` helpers (radius profile,
  sail half-thickness polynomials, NACA thickness, hull-form
  piecewise-linear axial maps) instead of re-deriving them.
* ``voxelize`` grid layout is ``(nz, ny, nx)`` with mesh x on the last
  array axis.  ``place_on_grid`` maps the authored frame onto the
  canonical B4 placement (hull centred at ``cx = 0.35 * nx``, length
  ``0.6 * nx``) which is exactly the placement of
  ``tensorlbm.cases.suboff`` and of the corpus masks.
* Sampling alignment: ``build_suboff_mask`` evaluates its predicates
  at integer lattice nodes; ``mask_from_stl`` samples cell centres
  (``origin + (i + 0.5) * spacing``).  The round trip passes
  ``origin=(-0.5, -0.5, -0.5)`` so both paths evaluate the *same*
  sample points -- the documented sampling knob, not a fudge.
* Components overlap on purpose: the DARPA sail predicate grows from
  the hull axis (``z > 0``) and the fin roots are buried in the stern
  taper, exactly as in ``suboff_cad``.  Ray parity on a single soup of
  overlapping closed shells yields the XOR (not the union) of their
  interiors, so the round trip voxelises each closed component
  separately and ORs them -- mirroring ``build_suboff_mask``'s
  ``mask | sail | fins`` composition one level up.

Every component is tessellated as a closed, consistently oriented
2-manifold (verified by :func:`tensorlbm.voxelize.is_watertight`):

* hull: surface of revolution with adaptive axial stations (chord-
  deviation refinement, see :func:`_hull_stations`) and tip fans;
* sail: closed cross-section rings (``suboff_cad`` 3-segment profile
  + semi-elliptical cap + bottom face at the axis plane) lofted along
  the axis with planar end caps;
* fins: closed NACA airfoil rings (sharp leading/trailing edge as a
  single shared edge) lofted along the span with planar root/tip caps.

Only numpy + stdlib are imported at module level;
:mod:`tensorlbm.suboff_cad` (which pulls torch) is imported lazily
inside the functions that need the analytic description.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from .voxelize import is_watertight, load_stl, mask_from_stl, place_on_grid

if TYPE_CHECKING:
    from .suboff_cad import SuboffConfig

__all__ = [
    "DEFAULT_GATE_CASES",
    "PRODUCTION_SHAPE",
    "RoundtripReport",
    "StlExportReport",
    "roundtrip_mask",
    "run_roundtrip_gate",
    "suboff_to_stl",
    "tessellate_suboff",
]

#: B4 production grid (nz, ny, nx) the gate runs at.
PRODUCTION_SHAPE = (40, 40, 128)

#: Acceptance targets from the physics of the thing.
IOU_TARGET = 0.98
BOUNDARY_TARGET = 0.05

#: Default tessellation resolutions.  ``chord_tol`` is the axial
#: station refinement tolerance (max radius chord deviation, lattice
#: units); the sail/fin grids are uniform (their profiles are short
#: and smooth).  Justified by the resolution study in
#: docs/cad_stl_roundtrip_20260825.md.
DEFAULT_CHORD_TOL = 0.02
DEFAULT_N_CIRC = 64
DEFAULT_SAIL_STATIONS = 48
DEFAULT_SAIL_ARC = 16
DEFAULT_FIN_CHORD = 24
DEFAULT_FIN_SPAN = 12

_STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])

_RING_EPS = 1.0e-9


def _cad() -> ModuleType:
    """Import :mod:`tensorlbm.suboff_cad` lazily (it imports torch)."""
    from . import suboff_cad

    return suboff_cad


# ---------------------------------------------------------------------------
# 1. Tessellation building blocks
# ---------------------------------------------------------------------------


def _surface_of_revolution(x: np.ndarray, r: np.ndarray, n_circ: int) -> np.ndarray:
    """Closed surface of revolution of a profile ``r(x)`` as triangles.

    Stations with ``r > eps`` become rings of ``n_circ`` vertices; the
    two ends are capped by fans from the axis point at the *original*
    end stations, so profiles tapering to ``r == 0`` (SUBOFF tips) get
    cone caps and flat-ended profiles (cylinder, frustum) get disc
    caps -- the same closed manifold either way.

    Args:
        x: Increasing axial station coordinates (any consistent unit).
        r: Radius at each station, same length as ``x``.
        n_circ: Circumferential resolution (>= 3, divisible by 4 so the
            ring hits the exact axis extremes).

    Returns:
        ``(T, 3, 3)`` float64 triangle table, mesh axes (x, y, z),
        consistently outward oriented.
    """
    if n_circ < 3:
        msg = f"n_circ must be >= 3, got {n_circ!r}"
        raise ValueError(msg)
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if x.shape != r.shape or x.ndim != 1 or x.size < 2:
        msg = f"x and r must be 1-D arrays of equal length >= 2, got {x.shape}, {r.shape}"
        raise ValueError(msg)
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("axial stations must be strictly increasing")
    if np.any(r < 0.0):
        raise ValueError("radii must be non-negative")
    tol = _RING_EPS * max(float(r.max()), 1.0)
    keep = r > tol
    if not keep.any():
        raise ValueError("profile has no positive-radius station")
    xs, rs = x[keep], r[keep]
    theta = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)
    ring = np.empty((xs.size, n_circ, 3), dtype=np.float64)
    ring[..., 0] = xs[:, None]
    ring[..., 1] = rs[:, None] * np.cos(theta)[None, :]
    ring[..., 2] = rs[:, None] * np.sin(theta)[None, :]
    nxt = np.roll(ring, -1, axis=1)
    # tube quads (same winding as suboff_cad._build_suboff_triangles):
    # (i, j) -> (i+1, j) -> (i+1, j+1) and (i, j) -> (i+1, j+1) -> (i, j+1)
    t1 = np.stack([ring[:-1], ring[1:], nxt[1:]], axis=2)
    t2 = np.stack([ring[:-1], nxt[1:], nxt[:-1]], axis=2)
    bow_tip = np.array([x[0], 0.0, 0.0])
    stern_tip = np.array([x[-1], 0.0, 0.0])
    cap_bow = np.stack([np.broadcast_to(bow_tip, ring[0].shape), ring[0], nxt[0]], axis=1)
    cap_stern = np.stack([np.broadcast_to(stern_tip, ring[-1].shape), nxt[-1], ring[-1]], axis=1)
    tris = np.concatenate([t1.reshape(-1, 3, 3), t2.reshape(-1, 3, 3), cap_bow, cap_stern])
    # suboff_cad's historical ring winding yields inward normals; flip to
    # outward so signed volumes come out positive (divergence theorem).
    return tris[:, [0, 2, 1]]


def _loft_closed_rings(rings: np.ndarray) -> np.ndarray:
    """Loft closed cross-section rings into a capped manifold.

    ``rings`` is ``(S, M, 3)``: ``S`` stations each carrying the same
    number ``M`` of ring vertices (cyclically ordered, consistently
    wound).  The two ends are capped by fans from the ring centroids.
    Every quad is split along the same diagonal as the hull tube so
    the whole module shares one winding convention.
    """
    if rings.ndim != 3 or rings.shape[1] < 3 or rings.shape[0] < 2:
        msg = f"rings must be (S >= 2, M >= 3, 3), got {rings.shape}"
        raise ValueError(msg)
    nxt = np.roll(rings, -1, axis=1)
    t1 = np.stack([rings[:-1], rings[1:], nxt[1:]], axis=2)
    t2 = np.stack([rings[:-1], nxt[1:], nxt[:-1]], axis=2)
    root_c = rings[0].mean(axis=0)
    tip_c = rings[-1].mean(axis=0)
    cap_root = np.stack([np.broadcast_to(root_c, rings[0].shape), rings[0], nxt[0]], axis=1)
    cap_tip = np.stack([np.broadcast_to(tip_c, rings[-1].shape), nxt[-1], rings[-1]], axis=1)
    return np.concatenate([t1.reshape(-1, 3, 3), t2.reshape(-1, 3, 3), cap_root, cap_tip])


def _mesh_volume(tris: np.ndarray) -> float:
    """Enclosed volume via the signed tetrahedron sum (divergence theorem)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))


# ---------------------------------------------------------------------------
# 2. SUBOFF component tessellators (mirror suboff_cad conventions exactly)
# ---------------------------------------------------------------------------


def _hull_stations(
    params: SuboffConfig,
    *,
    length: float,
    n_stations: int | None,
    chord_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Axial stations and radii of the hull profile in lattice units.

    Stations live on the mother-frame normalised axis ``xi`` (0 = bow
    tip, 1 = stern tip) and always include the four profile knots.  With
    ``n_stations`` given they are uniform (plus the knots); otherwise
    they are refined adaptively: any interval whose profile midpoint
    deviates from the chord by more than ``chord_tol`` lattice units of
    radius is split, so stations concentrate where the curvature is
    (bow entrance, stern taper) and skip the flat midbody.  Hull-form
    multipliers are applied by mapping the mother stations through the
    same piecewise-linear node map ``suboff_cad`` uses, with the
    diameter-fixed-in-feet radius scaling ``r / l_over_d_mult``.
    """
    cad = _cad()
    radius_eff = params.r_over_l * length / params.l_over_d_mult
    knots = np.unique(
        np.asarray(
            [
                0.0,
                cad._BOW_END_FT / cad._SUBOFF_L_FT,
                cad._MID_END_FT / cad._SUBOFF_L_FT,
                cad._STERN_END_FT / cad._SUBOFF_L_FT,
                1.0,
            ]
        )
    )
    if n_stations is not None:
        if n_stations < 8:
            msg = f"n_stations must be >= 8, got {n_stations!r}"
            raise ValueError(msg)
        xi = np.unique(np.concatenate([np.linspace(0.0, 1.0, int(n_stations)), knots]))
    else:
        if not chord_tol > 0.0:
            msg = f"chord_tol must be positive, got {chord_tol!r}"
            raise ValueError(msg)
        xi = knots
        for _ in range(24):
            if xi.size > 4096:
                break
            r = cad.suboff_radius_profile(xi, params) * radius_eff
            mid = 0.5 * (xi[:-1] + xi[1:])
            r_mid = cad.suboff_radius_profile(mid, params) * radius_eff
            err = np.abs(r_mid - 0.5 * (r[:-1] + r[1:]))
            split = err > chord_tol
            if not split.any():
                break
            xi = np.unique(np.concatenate([xi, mid[split]]))
    r = cad.suboff_radius_profile(xi, params) * radius_eff
    if cad._hull_lines_is_mother(params):
        x_lu = xi * length
    else:
        nodes_v = cad._variant_nodes_ft(params)
        x_ft_var = cad._pw_map_np(xi * cad._SUBOFF_L_FT, cad._HULL_NODES_FT, nodes_v)
        x_lu = x_ft_var * (length / nodes_v[-1])
    if not np.all(np.diff(x_lu) > 0.0):
        raise RuntimeError("hull-form axial map produced non-monotonic stations")
    return x_lu, r


def _sail_tris(
    params: SuboffConfig,
    *,
    length: float,
    sail_stations: int,
    sail_arc: int,
) -> np.ndarray:
    """Closed sail manifold in the authored (bow-at-0) lattice frame.

    Rings are the ``suboff_cad`` sail cross-sections (3-segment DARPA
    half-thickness polynomial, semi-elliptical cap) closed by the
    bottom face on the hull axis plane ``z = 0`` -- the sail solid the
    analytic predicate evaluates, buried base included.  The first and
    last stations sit half a spacing inside the DARPA footprint ends
    (where the profile degenerates to zero width) and the ends are
    capped by planar centroid fans; the excluded tip slivers are orders
    of magnitude below one voxel.  ``sail_scale`` and the hull-form
    axes are applied with the exact expressions of
    ``suboff_cad._real_sail_triangles``.
    """
    cad = _cad()
    inv = length / cad._SUBOFF_L_FT  # mother ft -> lattice units
    span = cad._SAIL_X3_END - cad._SAIL_X1_START
    step = span / sail_stations
    x_ft = np.linspace(cad._SAIL_X1_START + step, cad._SAIL_X3_END - step, int(sail_stations))
    half = cad._sail_half_thickness_np(x_ft)
    outlines = [
        np.asarray(cad._sail_cross_section_pts(float(h), int(sail_arc)), dtype=np.float64)
        for h in half
    ]
    if any(o.shape != outlines[0].shape for o in outlines):
        raise RuntimeError("sail stations produced mismatched ring outlines")
    ring_ft = np.asarray(outlines, dtype=np.float64)  # (S, M, 2) = (y, z) in ft
    if params.sail_scale != 1.0:
        s = float(params.sail_scale)
        x_ft = cad._SAIL_X_CENTER + (x_ft - cad._SAIL_X_CENTER) * s
        ring_ft = np.stack(
            [ring_ft[..., 0] * s, cad._SAIL_Z_DECK + (ring_ft[..., 1] - cad._SAIL_Z_DECK) * s],
            axis=-1,
        )
    if not cad._hull_lines_is_mother(params) or params.sail_x_mult != 1.0:
        x_ft = x_ft + cad._sail_x_shift_ft(params)
        x_ft = cad._pw_map_np(x_ft, cad._HULL_NODES_FT, cad._variant_nodes_ft(params)) / (
            params.l_over_d_mult
        )
    rings = np.empty((x_ft.size, ring_ft.shape[1], 3), dtype=np.float64)
    rings[..., 0] = x_ft[:, None] * inv
    rings[..., 1] = ring_ft[..., 0] * inv
    rings[..., 2] = ring_ft[..., 1] * inv
    return _loft_closed_rings(rings)


def _fin_tris(
    params: SuboffConfig,
    *,
    length: float,
    fin_chord: int,
    fin_span: int,
) -> np.ndarray:
    """Closed manifolds of the four cruciform swept-NACA stern fins.

    Each span station carries a closed airfoil ring: sharp leading and
    trailing edges (the SUBOFF NACA coefficients close the trailing
    edge to exactly zero thickness, so each is a single shared edge)
    and ``fin_chord - 2`` points per side.  ``fin_scale`` (chord about
    the common trailing edge, span about the root radius, thickness
    together) and the hull-form axial map mirror
    ``suboff_cad._real_fin_triangles``.  Returns the concatenated
    triangles of all four fins (y-span port/starboard + z-span
    top/bottom).
    """
    cad = _cad()
    inv = length / cad._SUBOFF_L_FT
    s_arr = np.linspace(0.0, 1.0, int(fin_chord))
    r_arr = np.linspace(cad._FIN_R_INNER, cad._FIN_R_OUTER, int(fin_span))
    t_arr = cad._naca4_thickness_np(s_arr)
    t_arr[0] = 0.0  # exact sharp edges: shared single vertices, watertight
    t_arr[-1] = 0.0
    cy_arr = cad._FIN_SWEEP_K * r_arr + cad._FIN_SWEEP_C
    x_ft = cad._FIN_H + (s_arr[None, :] - 1.0) * cy_arr[:, None]
    scale = float(params.fin_scale)
    if scale != 1.0:
        x_ft = cad._FIN_H + (x_ft - cad._FIN_H) * scale
        r_arr = cad._FIN_R_INNER + (r_arr - cad._FIN_R_INNER) * scale
        t_arr = t_arr * scale
    if not cad._hull_lines_is_mother(params):
        x_ft = cad._pw_map_np(x_ft, cad._HULL_NODES_FT, cad._variant_nodes_ft(params)) / (
            params.l_over_d_mult
        )
    # closed ring in the (x, thickness) profile plane: LE, upper side,
    # TE, lower side (matching vertex counts across span stations).
    ring_x = np.concatenate([x_ft[:, :1], x_ft[:, 1:-1], x_ft[:, -1:], x_ft[:, -2:0:-1]], axis=1)
    zeros = np.zeros((x_ft.shape[0], 1))
    upper = np.broadcast_to(t_arr[1:-1], (x_ft.shape[0], fin_chord - 2))
    ring_t = np.concatenate([zeros, upper, zeros, -upper[:, ::-1]], axis=1)
    x_lu = ring_x * inv
    t_lu = ring_t * inv
    r_lu = r_arr * inv
    fins = []
    for span_axis, sign in (("y", 1.0), ("y", -1.0), ("z", 1.0), ("z", -1.0)):
        rings = np.empty((r_lu.size, ring_x.shape[1], 3), dtype=np.float64)
        rings[..., 0] = x_lu
        if span_axis == "y":
            rings[..., 1] = sign * r_lu[:, None]
            rings[..., 2] = t_lu
        else:
            rings[..., 1] = t_lu
            rings[..., 2] = sign * r_lu[:, None]
        mirrored = sign < 0.0
        axis_swap = span_axis == "y"
        if mirrored != axis_swap:  # xor: net reflection -> inward winding
            # a reflection (mirror about the centreplane, or the y<->z swap
            # of the profile plane) flips the shell orientation; restore
            # the outward winding so volumes add up across the four fins.
            rings = rings[:, ::-1]
        fins.append(_loft_closed_rings(rings))
    return np.concatenate(fins)


def tessellate_suboff(
    params: SuboffConfig,
    *,
    hull_type: str = "full",
    length: float = 0.6 * PRODUCTION_SHAPE[2],
    n_stations: int | None = None,
    chord_tol: float = DEFAULT_CHORD_TOL,
    n_circ: int = DEFAULT_N_CIRC,
    sail_stations: int = DEFAULT_SAIL_STATIONS,
    sail_arc: int = DEFAULT_SAIL_ARC,
    fin_chord: int = DEFAULT_FIN_CHORD,
    fin_span: int = DEFAULT_FIN_SPAN,
) -> dict[str, np.ndarray]:
    """Tessellate the analytic SUBOFF description into closed components.

    Args:
        params: :class:`~tensorlbm.suboff_cad.SuboffConfig` (the 4
            hull-form axes, ``r_over_l`` and the appendage scales all
            apply).
        hull_type: ``"bare_hull"``, ``"with_sail"`` or ``"full"``.
        length: Hull length in lattice units (authored frame: bow tip
            at x=0, stern tip at x=length).
        n_stations: Uniform hull axial station count (plus profile
            knots); ``None`` selects adaptive refinement.
        chord_tol: Adaptive station tolerance (lattice units of radius
            chord deviation).
        n_circ: Hull circumferential resolution.
        sail_stations: Sail axial stations; ``sail_arc``: cap
            semi-ellipse resolution.
        fin_chord: Chordwise points per fin side; ``fin_span``: span
            stations.

    Returns:
        Dict ``{"hull": (T, 3, 3), ...}`` with ``"sail"`` added for
        ``with_sail``/``full`` and ``"fins"`` for ``full`` -- each a
        closed, consistently oriented manifold in mesh (x, y, z) order.
    """
    cad = _cad()
    hull = (
        cad.SuboffHullType(hull_type)
        if not isinstance(hull_type, cad.SuboffHullType)
        else hull_type
    )
    comps: dict[str, np.ndarray] = {}
    x_lu, r = _hull_stations(params, length=length, n_stations=n_stations, chord_tol=chord_tol)
    comps["hull"] = _surface_of_revolution(x_lu, r, n_circ)
    if hull in (cad.SuboffHullType.WITH_SAIL, cad.SuboffHullType.FULL):
        comps["sail"] = _sail_tris(
            params, length=length, sail_stations=sail_stations, sail_arc=sail_arc
        )
    if hull is cad.SuboffHullType.FULL:
        comps["fins"] = _fin_tris(params, length=length, fin_chord=fin_chord, fin_span=fin_span)
    for key, tris in comps.items():
        if not is_watertight(tris):
            msg = f"component {key!r} tessellated to a non-watertight manifold"
            raise RuntimeError(msg)
    return comps


# ---------------------------------------------------------------------------
# 3. Binary STL writer + export report
# ---------------------------------------------------------------------------


def _write_binary_stl(path: Path, tris: np.ndarray) -> None:
    """Write a binary STL (same record layout ``voxelize.load_stl`` reads)."""
    rec = np.zeros(tris.shape[0], dtype=_STL_RECORD)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    rec["normal"] = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0.0)
    rec["verts"] = tris
    header = b"tensorlbm cad_stl SUBOFF surface export"
    path.write_bytes(header.ljust(80, b"\0") + struct.pack("<I", tris.shape[0]) + rec.tobytes())


def _params_echo(params: SuboffConfig) -> dict[str, float]:
    """Echo the geometry-bearing SuboffConfig fields."""
    return {
        "r_over_l": float(params.r_over_l),
        "sail_scale": float(params.sail_scale),
        "fin_scale": float(params.fin_scale),
        "l_over_d_mult": float(params.l_over_d_mult),
        "nose_len_mult": float(params.nose_len_mult),
        "stern_len_mult": float(params.stern_len_mult),
        "sail_x_mult": float(params.sail_x_mult),
    }


@dataclass(frozen=True)
class StlExportReport:
    """Result of :func:`suboff_to_stl`.

    Attributes:
        path: Absolute path of the written binary STL.
        hull_type: Model variant string.
        params: Echo of the SuboffConfig geometry fields.
        length: Hull length in lattice units (authored frame).
        radius: Effective maximum hull radius in lattice units
            (after the diameter-fixed ``l_over_d_mult`` scaling).
        n_triangles: Total triangle count in the file.
        n_stations: Hull axial stations used (including the tip caps).
        n_circumferential: Hull circumferential resolution.
        watertight: :func:`voxelize.is_watertight` of the reloaded file.
        bbox_min: Mesh lower corner (x, y, z), lattice units.
        bbox_max: Mesh upper corner (x, y, z), lattice units.
        volume_lu3: Enclosed volume (signed tetrahedron sum) of the
            reloaded file, lattice units cubed.
        components: Triangle count per component.
    """

    path: str
    hull_type: str
    params: dict[str, float]
    length: float
    radius: float
    n_triangles: int
    n_stations: int
    n_circumferential: int
    watertight: bool
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    volume_lu3: float
    components: dict[str, int] = field(default_factory=dict)


def suboff_to_stl(
    params: SuboffConfig,
    path: str | Path,
    *,
    hull_type: str = "full",
    length: float = 0.6 * PRODUCTION_SHAPE[2],
    n_stations: int | None = None,
    chord_tol: float = DEFAULT_CHORD_TOL,
    n_circ: int = DEFAULT_N_CIRC,
    sail_stations: int = DEFAULT_SAIL_STATIONS,
    sail_arc: int = DEFAULT_SAIL_ARC,
    fin_chord: int = DEFAULT_FIN_CHORD,
    fin_span: int = DEFAULT_FIN_SPAN,
) -> StlExportReport:
    """Export the parametric SUBOFF as a watertight binary STL.

    The mesh is the tessellation of the SAME analytic description
    ``suboff_cad`` voxelises (see :func:`tessellate_suboff`); the file
    is written with the binary record layout ``voxelize.load_stl``
    reads and validated by reloading it.

    Args:
        params: :class:`~tensorlbm.suboff_cad.SuboffConfig`.
        path: Destination STL path (parents created as needed).
        hull_type: ``"bare_hull"``, ``"with_sail"`` or ``"full"``.
        length: Hull length in lattice units.
        n_stations: Uniform hull stations, or ``None`` for adaptive.
        chord_tol: Adaptive station refinement tolerance.
        n_circ: Circumferential resolution.
        sail_stations: Sail axial stations; ``sail_arc``: cap resolution.
        fin_chord: Fin chordwise points per side; ``fin_span``: span
            stations.

    Returns:
        The :class:`StlExportReport` (triangle count, watertightness of
        the reloaded file, bounding box, signed volume, params echo).
    """
    comps = tessellate_suboff(
        params,
        hull_type=hull_type,
        length=length,
        n_stations=n_stations,
        chord_tol=chord_tol,
        n_circ=n_circ,
        sail_stations=sail_stations,
        sail_arc=sail_arc,
        fin_chord=fin_chord,
        fin_span=fin_span,
    )
    tris = np.concatenate(list(comps.values()))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_stl(p, tris)
    mesh = load_stl(p)
    flat = mesh.vertices.reshape(-1, 3)
    cad = _cad()
    radius_eff = params.r_over_l * length / params.l_over_d_mult
    return StlExportReport(
        path=str(p.resolve()),
        hull_type=cad.SuboffHullType(hull_type).value,
        params=_params_echo(params),
        length=float(length),
        radius=float(radius_eff),
        n_triangles=int(mesh.vertices.shape[0]),
        n_stations=int(comps["hull"].shape[0] // (2 * n_circ) + 2),
        n_circumferential=int(n_circ),
        watertight=bool(is_watertight(mesh.vertices)),
        bbox_min=(float(flat[:, 0].min()), float(flat[:, 1].min()), float(flat[:, 2].min())),
        bbox_max=(float(flat[:, 0].max()), float(flat[:, 1].max()), float(flat[:, 2].max())),
        volume_lu3=_mesh_volume(mesh.vertices),
        components={k: int(v.shape[0]) for k, v in comps.items()},
    )


# ---------------------------------------------------------------------------
# 4. Round trip: params -> STL -> voxelize -> mask vs analytic mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundtripReport:
    """Result of :func:`roundtrip_mask`.

    Attributes:
        name: Case label.
        hull_type: Model variant string.
        params: Echo of the SuboffConfig geometry fields.
        shape: Grid shape (nz, ny, nx).
        n_triangles: Triangles in the (in-memory) component meshes.
        solid_cells_stl / solid_cells_analytic: Solid cell counts of the
            STL path mask and of ``build_suboff_mask``.
        volume_ratio: ``solid_cells_stl / solid_cells_analytic``.
        iou: Global IoU of the two masks.
        boundary_cells: Analytic solid cells with a fluid 6-neighbour.
        disagreement_cells: Cells where the two masks differ (XOR).
        boundary_disagreement_frac: XOR size over boundary cells -- the
            honest surface-fidelity metric.
        localized_frac: Fraction of the XOR inside the 1-voxel boundary
            band; 1.0 means every disagreement sits on the tessellated
            surface.
        interior_exact: True when no disagreement lies outside the
            boundary band (the interior agrees exactly).
        components: Per-component analytic/stl cell counts and IoU.
        mesh_volume_lu3: Enclosed volume of the merged mesh.
        analytic_displacement_lu3: ``suboff_statistics`` displacement of
            the bare hull at the same placement.
        elapsed_s: Wall time of the full loop.
        iou_pass / boundary_pass: Acceptance flags.
    """

    name: str
    hull_type: str
    params: dict[str, float]
    shape: tuple[int, int, int]
    n_triangles: int
    solid_cells_stl: int
    solid_cells_analytic: int
    volume_ratio: float
    iou: float
    boundary_cells: int
    disagreement_cells: int
    boundary_disagreement_frac: float
    localized_frac: float
    interior_exact: bool
    components: dict[str, dict[str, float]] = field(default_factory=dict)
    mesh_volume_lu3: float = 0.0
    analytic_displacement_lu3: float = 0.0
    elapsed_s: float = 0.0
    iou_pass: bool = False
    boundary_pass: bool = False


def _boundary_mask(solid: np.ndarray) -> np.ndarray:
    """Solid cells with at least one fluid 6-neighbour (border = fluid)."""
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    flips = np.zeros(solid.shape, dtype=bool)
    for d in range(3):
        for step in (-1, 1):
            src = np.roll(padded, step, axis=d)[tuple(slice(1, s + 1) for s in solid.shape)]
            flips |= src != solid
    return solid & flips


def _dilate26(mask: np.ndarray) -> np.ndarray:
    """One Chebyshev step (26-neighbourhood) dilation of a bool array."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = mask.copy()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                shifted = np.roll(padded, (dz, dy, dx), (0, 1, 2))[
                    tuple(slice(1, s + 1) for s in mask.shape)
                ]
                out |= shifted
    return out


def _node_origin() -> tuple[float, float, float]:
    """Sample origin that puts ``mask_from_stl`` samples on integer nodes.

    ``build_suboff_mask`` evaluates its predicates at integer lattice
    nodes; ``mask_from_stl`` samples ``origin + (i + 0.5) * spacing``,
    so ``origin = -0.5`` per axis evaluates the identical points.
    """
    return (-0.5, -0.5, -0.5)


def roundtrip_mask(
    params: SuboffConfig,
    shape: tuple[int, int, int] = PRODUCTION_SHAPE,
    *,
    name: str = "case",
    hull_type: str = "full",
    center_frac: tuple[float, float, float] = (0.35, 0.5, 0.5),
    n_stations: int | None = None,
    chord_tol: float = DEFAULT_CHORD_TOL,
    n_circ: int = DEFAULT_N_CIRC,
    sail_stations: int = DEFAULT_SAIL_STATIONS,
    sail_arc: int = DEFAULT_SAIL_ARC,
    fin_chord: int = DEFAULT_FIN_CHORD,
    fin_span: int = DEFAULT_FIN_SPAN,
) -> RoundtripReport:
    """Full CAD loop: params -> tessellate -> place -> voxelize vs analytic.

    The STL path is :func:`tessellate_suboff` -> ``place_on_grid``
    (canonical B4 placement, uniform scale 1; the transform is derived
    from the hull component, whose authored bounding box is exactly the
    analytic frame, and applied unchanged to the appendages so the
    whole assembly lands where ``build_suboff_mask`` puts it) ->
    ``mask_from_stl`` per closed component (overlapping closed shells
    parity-XOR, so the union is composed by OR, mirroring the analytic
    builder) sampled on integer nodes.  The analytic reference is
    ``build_suboff_mask`` with the same params, grid and placement.

    Args:
        params: :class:`~tensorlbm.suboff_cad.SuboffConfig`.
        shape: Grid shape (nz, ny, nx).
        name: Case label for the report.
        hull_type: ``"bare_hull"``, ``"with_sail"`` or ``"full"``.
        center_frac: Canonical placement target (matches
            ``place_on_grid`` defaults = the ``cases/suboff`` frame).
        n_stations, chord_tol, n_circ, sail_stations, sail_arc,
            fin_chord, fin_span: Tessellation resolutions forwarded to
            :func:`tessellate_suboff`.

    Returns:
        The :class:`RoundtripReport` with IoU, boundary-band
        localization, volume ratio and per-component counts.
    """
    cad = _cad()
    nz, ny, nx = (int(v) for v in shape)
    length = 0.6 * nx
    t0 = time.perf_counter()
    comps = tessellate_suboff(
        params,
        hull_type=hull_type,
        length=length,
        n_stations=n_stations,
        chord_tol=chord_tol,
        n_circ=n_circ,
        sail_stations=sail_stations,
        sail_arc=sail_arc,
        fin_chord=fin_chord,
        fin_span=fin_span,
    )
    # Canonical placement from the hull (authored bbox == analytic
    # frame); the identical translation carries the appendages.
    placement = place_on_grid(comps["hull"], (nz, ny, nx), scale=1.0, center_frac=center_frac)
    offset = placement.tris[0, 0] - comps["hull"][0, 0]
    origin = _node_origin()
    stl_masks: dict[str, np.ndarray] = {
        key: mask_from_stl(tris + offset, (nz, ny, nx), origin=origin, spacing=1.0)
        for key, tris in comps.items()
    }
    mask_stl = np.zeros((nz, ny, nx), dtype=bool)
    for m in stl_masks.values():
        mask_stl |= m

    cx, cy, cz = center_frac[0] * nx, center_frac[1] * ny, center_frac[2] * nz
    analytic, _stats = cad.build_suboff_mask(
        hull_type=hull_type,
        nx=nx,
        ny=ny,
        nz=nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=length,
        config=params,
        device="cpu",
    )
    mask_an = analytic.numpy()

    # Analytic per-component masks via the public point predicates
    # (bit-identical to the builder internals by the hullform tests).
    import torch

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    comp_masks: dict[str, np.ndarray] = {
        "hull": cad.suboff_hull_mask(
            nx, ny, nz, cx, cy, cz, length, -1.0, torch.device("cpu"), params
        ).numpy()
    }
    if "sail" in comps:
        comp_masks["sail"] = cad.suboff_sail_contains_points(
            xx, yy, zz, center=(cx, cy, cz), length=length, scale=params.sail_scale, config=params
        ).numpy()
    if "fins" in comps:
        comp_masks["fins"] = cad.suboff_fins_contain_points(
            xx, yy, zz, center=(cx, cy, cz), length=length, scale=params.fin_scale, config=params
        ).numpy()

    inter = int(np.logical_and(mask_stl, mask_an).sum())
    union = int(np.logical_or(mask_stl, mask_an).sum())
    iou = inter / union if union else 0.0
    xor = np.logical_xor(mask_stl, mask_an)
    n_xor = int(xor.sum())
    boundary = _boundary_mask(mask_an)
    n_boundary = int(boundary.sum())
    band = _dilate26(boundary)
    in_band = int(np.logical_and(xor, band).sum())
    localized = in_band / n_xor if n_xor else 1.0
    interior_exact = not bool(np.logical_and(xor, ~band).any())
    n_stl = int(mask_stl.sum())
    n_an = int(mask_an.sum())

    components: dict[str, dict[str, float]] = {}
    for key, an in comp_masks.items():
        stl = stl_masks[key]
        c_inter = int(np.logical_and(stl, an).sum())
        c_union = int(np.logical_or(stl, an).sum())
        components[key] = {
            "analytic_cells": int(an.sum()),
            "stl_cells": int(stl.sum()),
            "iou": c_inter / c_union if c_union else 1.0,
        }
    stats = cad.suboff_statistics(
        cad.SuboffHullType(hull_type), length, params.r_over_l * length, params
    )
    merged = np.concatenate(list(comps.values()))
    elapsed = time.perf_counter() - t0
    return RoundtripReport(
        name=name,
        hull_type=cad.SuboffHullType(hull_type).value,
        params=_params_echo(params),
        shape=(nz, ny, nx),
        n_triangles=int(merged.shape[0]),
        solid_cells_stl=n_stl,
        solid_cells_analytic=n_an,
        volume_ratio=n_stl / n_an if n_an else 0.0,
        iou=iou,
        boundary_cells=n_boundary,
        disagreement_cells=n_xor,
        boundary_disagreement_frac=n_xor / n_boundary if n_boundary else 0.0,
        localized_frac=localized,
        interior_exact=interior_exact,
        components=components,
        mesh_volume_lu3=_mesh_volume(merged + offset),
        analytic_displacement_lu3=float(stats["displacement_lu3"]),
        elapsed_s=round(elapsed, 3),
        iou_pass=bool(iou > IOU_TARGET),
        boundary_pass=bool(n_xor / n_boundary < BOUNDARY_TARGET if n_boundary else False),
    )


# ---------------------------------------------------------------------------
# 5. The gate
# ---------------------------------------------------------------------------

#: Default gate cases: the mother hull plus family variants and the
#: appendage variants, all at the production grid.
DEFAULT_GATE_CASES: list[dict[str, Any]] = [
    {"name": "mother", "hull_type": "full", "params": {}},
    {"name": "bare", "hull_type": "bare_hull", "params": {}},
    {"name": "with_sail", "hull_type": "with_sail", "params": {}},
    {"name": "slender", "hull_type": "full", "params": {"l_over_d_mult": 1.25}},
    {"name": "blunt", "hull_type": "full", "params": {"l_over_d_mult": 0.85}},
    {"name": "long_nose", "hull_type": "full", "params": {"nose_len_mult": 1.4}},
    {"name": "aft_sail", "hull_type": "full", "params": {"sail_x_mult": 1.12}},
]


def run_roundtrip_gate(
    cases: list[dict[str, Any]] | None = None,
    out_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the CAD STL round-trip gate over a case set.

    On-demand (not in CI: it needs the full CAD machinery); runtime for
    the default 7-case set at the production grid is well under two
    minutes.  Each case dict carries ``name``, ``hull_type``, a
    ``params`` dict of :class:`~tensorlbm.suboff_cad.SuboffConfig`
    kwargs and optionally ``shape`` and ``tessellation`` (a dict of
    :func:`tessellate_suboff` resolution kwargs).

    Args:
        cases: Case list; ``None`` uses :data:`DEFAULT_GATE_CASES`.
        out_json: Where to write the machine-readable results (JSON);
            ``None`` skips the write.

    Returns:
        ``{"cases": {name: report-dict}, "all_pass": bool}`` where
        ``all_pass`` requires every case to clear both the IoU and the
        boundary-band targets.
    """
    cad = _cad()
    results: dict[str, Any] = {}
    for case in cases if cases is not None else DEFAULT_GATE_CASES:
        params = cad.SuboffConfig(**case.get("params", {}))
        report = roundtrip_mask(
            params,
            tuple(case.get("shape", PRODUCTION_SHAPE)),
            name=str(case.get("name", "case")),
            hull_type=str(case.get("hull_type", "full")),
            **(case.get("tessellation", {})),
        )
        results[report.name] = asdict(report)
        print(
            f"{report.name:>10s} {report.hull_type:>9s} "
            f"tris={report.n_triangles:6d} "
            f"IoU={report.iou:.4f} "
            f"boundary%={100.0 * report.boundary_disagreement_frac:5.2f} "
            f"localized={report.localized_frac:.3f} "
            f"interior_exact={report.interior_exact} "
            f"vol_ratio={report.volume_ratio:.4f} "
            f"pass={report.iou_pass and report.boundary_pass} "
            f"[{report.elapsed_s:.1f}s]",
            flush=True,
        )
    summary = {
        "cases": results,
        "targets": {"iou": IOU_TARGET, "boundary_frac": BOUNDARY_TARGET},
        "all_pass": all(r["iou_pass"] and r["boundary_pass"] for r in results.values()),
    }
    if out_json is not None:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["written"] = str(p.resolve())
    return summary
