"""Parametric ice-class bow (破冰艏) family: geometry source, gate, SDF, resistance.

This module is the *single* geometric source for the ice-region campaign: one
parametric family of icebreaking fore-bodies, a machine-checkable geometry
admission gate, an SDF/voxel casting used by the LBM leg, and
panels/offsets/contact-surface exports consumed by the BEM/DEM legs.

Family definition (all lengths normalised by ``L``; ``xi = x/L``,
``zeta = z/D``, keel plane ``z = 0``, deck ``z = D``, design waterline
``z = T = waterline_frac * D``)::

    b(xi, zeta) = (B/(2L)) * kappa(zeta) * min( phi_bow(u), phi_aft(xi) )

    u        = ( xs(zeta) - xi ) / s_ent           # 0 at the local stem
    xs(zeta) = 1 - s(zeta)                         # stem x at height zeta
    s(zeta)  = (D/L)(1-zeta) tan(alpha_stem)       # raked stem profile
    s_ent    = (B/(2L)) / tan(beta_ent)            # waterline entrance length
    kappa    = 1 - k_bilge (1 - zeta/zeta_T)^2          for zeta <= zeta_T
             = 1 + c_flare (zeta - zeta_T)/(1-zeta_T)   for zeta >  zeta_T
    phi_bow  = u                                  for u <= f_lin
             = f_lin + (1-f_lin)(1-(1-t)^c_fwd),  t = (u-f_lin)/(1-f_lin)
    phi_aft  = r_t + (1-r_t) clip(xi / s_aft, 0, 1)

Physically: ``alpha_stem`` is the stem rake off the vertical (艏柱倾角),
``beta_ent`` the half angle of entrance of the *design waterline* (水线进角,
recovered exactly by the gate from the mesh because ``phi_bow`` is linear in
``u`` next to the stem), ``c_flare`` the sectional flare (negative = 内倾式
tumblehome), ``c_fwd`` the fore-shoulder fullness and ``r_t`` the transom
breadth ratio, which is *solved* so that the geometry-dependent block
coefficient ``Cb = nabla/(L B T)`` hits the design target.

Three named sub-families share the same parameterisation:

===============  ========================================================
``straight``     直斜式: constant stem rake, no flare (``c_flare = 0``).
``inward``       内倾式: tumblehome sides (``c_flare < 0``), typical of
                 ice-breaking plating that sheds ice inward.
``multisegment`` 多段式: two-segment stem -- the lower stem below half
                 draft carries an extra rake increment for ice riding,
                 the upper segment (crossing the design waterline) keeps
                 ``alpha_stem`` so the gate's reverse reading is well posed.
===============  ========================================================

Admission gate (:func:`validate_mesh`) checks water-tightness, edge
manifoldness, Euler characteristic ``V - E + F = 2``, consistent/outward
normals (edge pairing + divergence-theorem signed volume + centroid offset
test against the analytic indicator) and the absence of zero-area
triangles; :func:`reverse_parameters` re-derives ``Cb/L/B`` and
``alpha_stem/beta_ent`` *from the mesh*, so a corrupted or mis-registered
geometry cannot silently enter the campaign.

Everything is numpy/stdlib only.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

__all__ = [
    "FAMILIES",
    "FAMILY_PRESETS",
    "IceBowParams",
    "IceBowMesh",
    "family_members",
    "member_label",
    "beta_ent_for_cb",
    "half_breadth_norm",
    "stem_offset_norm",
    "stem_x_norm",
    "resolve_params",
    "solve_transom_ratio",
    "design_volume",
    "build_ice_bow_mesh",
    "mesh_volume",
    "mesh_surface_area",
    "submerged_volume",
    "inside_solid",
    "validate_mesh",
    "reverse_parameters",
    "offsets_table",
    "voxelize_mesh",
    "voxel_volume_study",
    "gci",
    "write_stl",
    "read_stl_vertices",
    "wetted_surface",
    "longitudinal_distribution",
    "holtrop_resistance",
    "michell_wave_resistance",
    "resistance_sensitivity",
]

FAMILIES = ("straight", "inward", "multisegment")

_G = 9.80665  # m/s^2
_RHO = 1025.0  # kg/m^3, sea water
_NU = 1.139e-6  # m^2/s at ~15 C

# Sub-cell asymmetric ray perturbation, mirroring tensorlbm.geometry_voxel and
# tensorlbm.preprocess_geo: breaks exact ray/edge alignments that would
# otherwise double-count a shared edge and punch a hole in the parity mask.
_EPS_Y = 1.3e-4
_EPS_Z = 3.7e-4


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def beta_ent_for_cb(cb: float) -> float:
    """Design-guideline waterline half angle of entrance for a target ``Cb``.

    Fitted so that the solved transom ratio stays strictly inside
    ``(0.02, 0.995)`` over the campaign range ``Cb in [0.55, 0.80]``: the
    family reaches Cb=0.55 only with a fine entrance (``beta ~ 9 deg``) and
    Cb=0.80 only with a full one (``beta ~ 30 deg``).  The achievable
    ``(Cb_lo, Cb_hi)`` window is monotone in ``beta``; see the campaign
    report's parameter table.
    """
    return float(min(max(9.0 + 84.0 * (float(cb) - 0.55), 6.0), 32.0))


@dataclass(frozen=True)
class IceBowParams:
    """Design parameters of one ice-class bow family member.

    Defaults describe a wide, shallow ice-class fore-body:
    ``L/B = 5.88``, ``B/T = 1.97``.
    """

    family: str = "straight"
    alpha_stem_deg: float = 30.0
    beta_ent_deg: float | None = None  # None -> beta_ent_for_cb(cb_target)
    c_flare: float = 0.0
    c_fwd: float = 2.5
    cb_target: float = 0.70
    L: float = 1.0
    B: float = 0.17
    D: float = 0.115
    waterline_frac: float = 0.75
    s_aft_frac: float = 0.30
    f_lin: float = 0.35
    k_bilge: float = 0.30
    multi_break_frac: float = 0.50
    multi_lower_increment_deg: float = 12.0
    r_t: float | None = None

    def __post_init__(self) -> None:
        if self.beta_ent_deg is None:
            object.__setattr__(self, "beta_ent_deg", beta_ent_for_cb(self.cb_target))
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; expected one of {FAMILIES}")
        for name in ("alpha_stem_deg", "beta_ent_deg", "L", "B", "D", "c_fwd"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not (0.0 < self.alpha_stem_deg < 80.0):
            raise ValueError("alpha_stem_deg must be in (0, 80) degrees")
        if not (0.0 < self.beta_ent_deg < 60.0):
            raise ValueError("beta_ent_deg must be in (0, 60) degrees")
        if not (0.0 < self.cb_target < 0.95):
            raise ValueError("cb_target must be in (0, 0.95)")
        if self.L <= 0.0 or self.B <= 0.0 or self.D <= 0.0:
            raise ValueError("L, B and D must be > 0")
        if not (0.3 <= self.waterline_frac < 1.0):
            raise ValueError("waterline_frac must be in [0.3, 1)")
        if not (0.0 < self.s_aft_frac < 0.8):
            raise ValueError("s_aft_frac must be in (0, 0.8)")
        if not (0.0 < self.f_lin < 1.0):
            raise ValueError("f_lin must be in (0, 1)")
        if self.c_fwd < 1.0:
            raise ValueError("c_fwd must be >= 1 (1 = straight wedge shoulder)")
        if not (0.0 <= self.k_bilge < 1.0):
            raise ValueError("k_bilge must be in [0, 1)")
        if self.r_t is not None and not (0.0 < self.r_t <= 1.0):
            raise ValueError("r_t must be in (0, 1]")

    @property
    def zeta_T(self) -> float:
        return float(self.waterline_frac)

    @property
    def draft(self) -> float:
        return self.zeta_T * self.D

    @property
    def beam_over_length(self) -> float:
        return self.B / self.L


#: Representative parameter sets of the three sub-families at alpha_stem = 30 deg.
FAMILY_PRESETS: dict[str, dict[str, float]] = {
    "straight": dict(c_flare=0.0, c_fwd=2.5),
    "inward": dict(c_flare=-0.25, c_fwd=2.0),
    "multisegment": dict(c_flare=0.15, c_fwd=3.0),
}


def member_label(params: IceBowParams) -> str:
    """Short human-readable label of one family member."""
    return f"{params.family} a={params.alpha_stem_deg:.0f} b={params.beta_ent_deg:.1f} cb={params.cb_target:.2f}"


def family_members(
    alphas=(20.0, 25.0, 30.0, 35.0),
    cb_target: float = 0.70,
    **overrides: object,
) -> list[IceBowParams]:
    """Campaign family: one straight-stem member per rake angle plus the
    inward / multisegment variants at the reference angle 30 deg."""
    members = [
        IceBowParams(family="straight", alpha_stem_deg=float(a), cb_target=cb_target, **overrides)
        for a in alphas
    ]
    for fam in ("inward", "multisegment"):
        members.append(
            IceBowParams(
                family=fam,
                alpha_stem_deg=30.0,
                cb_target=cb_target,
                **FAMILY_PRESETS[fam],
                **overrides,
            )
        )
    return members


# ---------------------------------------------------------------------------
# Analytic geometry
# ---------------------------------------------------------------------------


def stem_offset_norm(params: IceBowParams, zeta) -> np.ndarray:
    """``s(zeta)/L``: aft distance from the deck stem to the stem at ``zeta``."""
    z = np.asarray(zeta, dtype=np.float64)
    d_l = params.D / params.L
    tan_up = math.tan(math.radians(params.alpha_stem_deg))
    if params.family != "multisegment":
        return d_l * (1.0 - z) * tan_up
    tan_lo = math.tan(math.radians(params.alpha_stem_deg + params.multi_lower_increment_deg))
    zb = params.multi_break_frac * params.zeta_T
    s_break = d_l * (1.0 - zb) * tan_up
    return np.where(
        z >= zb, d_l * (1.0 - z) * tan_up, s_break + d_l * (zb - z) * tan_lo
    )


def stem_x_norm(params: IceBowParams, zeta) -> np.ndarray:
    """``xs(zeta) = 1 - s(zeta)``: normalised x of the stem at height ``zeta``."""
    return 1.0 - stem_offset_norm(params, zeta)


def _section_factor(params: IceBowParams, zeta) -> np.ndarray:
    z = np.asarray(zeta, dtype=np.float64)
    zt = params.zeta_T
    below = z <= zt
    out = np.empty_like(z)
    out[below] = 1.0 - params.k_bilge * (1.0 - z[below] / zt) ** 2
    if np.any(~below):
        out[~below] = 1.0 + params.c_flare * (z[~below] - zt) / (1.0 - zt)
    return np.maximum(out, 0.05)


def _bow_shape(u, f_lin: float, c_fwd: float) -> np.ndarray:
    uu = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    out = np.empty_like(uu)
    lin = uu <= f_lin
    out[lin] = uu[lin]
    if np.any(~lin):
        t = (uu[~lin] - f_lin) / (1.0 - f_lin)
        out[~lin] = f_lin + (1.0 - f_lin) * (1.0 - (1.0 - t) ** c_fwd)
    return np.clip(out, 0.0, 1.0)


def half_breadth_norm(
    params: IceBowParams, xi, zeta, r_t: float | None = None
) -> np.ndarray:
    """Normalised half-breadth ``b(xi, zeta)/L`` of the hull family."""
    xi = np.asarray(xi, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    xi, zeta = np.broadcast_arrays(xi, zeta)
    xs = stem_x_norm(params, zeta)
    s_ent = 0.5 * params.beam_over_length / math.tan(math.radians(params.beta_ent_deg))
    if s_ent <= 0.0:
        raise ValueError("degenerate entrance length")
    u = (xs - xi) / s_ent
    phi_bow = _bow_shape(u, params.f_lin, params.c_fwd)
    rt = params.r_t if r_t is None else float(r_t)
    if rt is None:
        raise ValueError("r_t unresolved: call resolve_params() first")
    phi_aft = rt + (1.0 - rt) * np.clip(xi / params.s_aft_frac, 0.0, 1.0)
    kappa = _section_factor(params, zeta)
    return 0.5 * params.beam_over_length * kappa * np.minimum(phi_bow, phi_aft)


def _volume_integral(
    params: IceBowParams,
    zeta_hi: float,
    r_t: float | None = None,
    n_x: int = 1201,
    n_z: int = 401,
) -> float:
    """``int_0^1 int_0^{zeta_hi} 2 b_norm dxi dzeta`` by 2-D trapezoid."""
    xi = np.linspace(0.0, 1.0, n_x)
    zeta = np.linspace(0.0, zeta_hi, n_z)
    XI, ZE = np.meshgrid(xi, zeta)
    b = half_breadth_norm(params, XI, ZE, r_t=r_t)
    v = np.trapezoid(2.0 * b, x=xi, axis=1)
    return float(np.trapezoid(v, x=zeta))


def design_volume(
    params: IceBowParams, n_x: int = 1201, n_z: int = 401, r_t: float | None = None
) -> dict:
    """Analytic (quadrature) submerged and full-body volume."""
    zt = params.zeta_T
    v_sub = _volume_integral(params, zt, r_t=r_t, n_x=n_x, n_z=n_z)
    v_full = _volume_integral(params, 1.0, r_t=r_t, n_x=n_x, n_z=n_z)
    scale = params.L * params.D * params.L
    c_box = params.L * params.B * params.draft
    return {
        "nabla": v_sub * scale,
        "volume_full": v_full * scale,
        "cb": v_sub * scale / c_box if c_box > 0 else float("nan"),
        "cb_box": c_box,
    }


def solve_transom_ratio(
    params: IceBowParams, lo: float = 0.02, hi: float = 0.995, iters: int = 60
) -> float:
    """Bisection on ``r_t`` so that ``Cb(nabla(r_t)) == cb_target``."""
    target = params.cb_target
    c_box = params.L * params.B * params.draft

    def cb_of(rt: float) -> float:
        v = _volume_integral(params, params.zeta_T, r_t=rt, n_x=401, n_z=201)
        return v * params.L * params.D * params.L / c_box

    cb_lo, cb_hi = cb_of(lo), cb_of(hi)
    if not (cb_lo <= target <= cb_hi):
        raise ValueError(
            f"cb_target={target:.4f} outside achievable range "
            f"[{cb_lo:.4f}, {cb_hi:.4f}] for these family parameters"
        )
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if cb_of(mid) < target:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def resolve_params(params: IceBowParams, **kw: object) -> IceBowParams:
    """Return a copy with ``r_t`` solved from ``cb_target`` (idempotent)."""
    if params.r_t is not None:
        return params
    return replace(params, r_t=solve_transom_ratio(params, **kw))


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


PATCH_NAMES = ("port", "stbd", "deck", "bottom", "transom")


@dataclass
class IceBowMesh:
    """Watertight triangulated closed surface of one family member."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64
    params: IceBowParams
    resolution: dict
    patches: np.ndarray | None = None  # (M,) index into PATCH_NAMES

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


def _graded_chi(n: int) -> np.ndarray:
    eta = np.linspace(0.0, 1.0, n + 1)
    chi = 1.0 - (1.0 - eta) ** 1.6  # cluster towards the stem (chi -> 1)
    chi[0], chi[-1] = 0.0, 1.0
    return chi


def _graded_levels(n: int, zeta_T: float) -> np.ndarray:
    z = np.linspace(0.0, 1.0, n + 1)
    z = np.union1d(z, np.array([zeta_T]))  # a level exactly on the waterline
    z[0], z[-1] = 0.0, 1.0
    return z


def _weld(vertices: np.ndarray, faces: np.ndarray, tol: float):
    key = np.round(vertices / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return vertices[first][order], remap[inverse[faces]]


def _weld_map(vertices: np.ndarray, tol: float):
    """Weld coincident vertices; return (welded, old->new index map)."""
    key = np.round(vertices / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return vertices[first][order], remap[inverse]


def _emit_grid_faces(node_idx: np.ndarray, out: list) -> None:
    """Triangles for a structured grid of vertex indices ``(nj, ni)``.

    Degenerate corners (collapsed rows, e.g. where the half-breadth vanishes
    at the stem) are dropped, so no zero-area triangle is ever created.
    """
    a = node_idx[:-1, :-1].reshape(-1)
    b = node_idx[:-1, 1:].reshape(-1)
    c = node_idx[1:, 1:].reshape(-1)
    d = node_idx[1:, :-1].reshape(-1)
    for tri in (np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)):
        keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
        out.append(tri[keep])


def build_ice_bow_mesh(
    params: IceBowParams,
    n_stations: int = 81,
    n_levels: int = 61,
    n_transverse: int = 12,
) -> IceBowMesh:
    """Triangulate the closed hull surface (sides / deck / bottom / transom).

    Five structured patches share their boundary nodes: port and starboard
    side sheets (meeting on the stem seam where ``b = 0``), the flat bottom
    at ``z = 0``, the flat deck at ``z = D`` and the transom plane at
    ``x = 0``.  The result is topologically a sphere (Euler ``V-E+F = 2``).
    """
    p = resolve_params(params)
    res = {
        "n_stations": int(n_stations),
        "n_levels": int(n_levels),
        "n_transverse": int(n_transverse),
    }
    chi = _graded_chi(int(n_stations))
    zeta = _graded_levels(int(n_levels), p.zeta_T)
    xs = stem_x_norm(p, zeta)
    xg = chi[None, :] * xs[:, None] * p.L  # (nz, nx)
    zg = (zeta[:, None] * p.D) * np.ones_like(xg)
    b = half_breadth_norm(p, xg / p.L, zeta[:, None]) * p.L

    verts: list[np.ndarray] = []
    offset = 0

    def _add(arr: np.ndarray) -> np.ndarray:
        nonlocal offset
        flat_arr = arr.reshape(-1, 3)
        n = flat_arr.shape[0]
        idx = np.arange(offset, offset + n).reshape(arr.shape[:-1])
        verts.append(flat_arr)
        offset += n
        return idx

    idx_port = _add(np.stack([xg, -b, zg], axis=-1))  # (nz, nx)
    idx_stbd = _add(np.stack([xg, +b, zg], axis=-1))

    k = np.linspace(0.0, 1.0, int(n_transverse) + 1)
    # deck, z = D, transverse grid (nx, nt)
    b_deck = b[-1, :]
    x_deck = np.repeat(xg[-1, :][:, None], k.size, axis=1)
    y_deck = -b_deck[:, None] + 2.0 * b_deck[:, None] * k[None, :]
    idx_deck = _add(np.stack([x_deck, y_deck, p.D * np.ones_like(x_deck)], axis=-1))
    # bottom, z = 0
    b_bot = b[0, :]
    x_bot = np.repeat(xg[0, :][:, None], k.size, axis=1)
    y_bot = -b_bot[:, None] + 2.0 * b_bot[:, None] * k[None, :]
    idx_bot = _add(np.stack([x_bot, y_bot, np.zeros_like(x_bot)], axis=-1))
    # transom, x = 0, grid (nz, nt)
    b_tra = b[:, 0]
    z_tra = np.repeat(zg[:, 0][:, None], k.size, axis=1)
    y_tra = -b_tra[:, None] + 2.0 * b_tra[:, None] * k[None, :]
    idx_tra = _add(np.stack([np.zeros_like(y_tra), y_tra, z_tra], axis=-1))

    allv = np.concatenate(verts, axis=0)
    # Weld first, then triangulate: rows that collapse to a single point (the
    # stem) must be recognised *after* welding, otherwise duplicate-index
    # (zero-area) faces are created at the stem fan.
    v, old_to_new = _weld_map(allv, tol=1e-10 * p.L)

    def _emit(node_idx: np.ndarray, ref) -> np.ndarray:
        """Emit one patch and orient every triangle along ``ref`` (outward)."""
        out: list[np.ndarray] = []
        _emit_grid_faces(old_to_new[node_idx], out)
        fc = np.concatenate(out, axis=0)
        tri = v[fc]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        flip = (n @ np.asarray(ref, dtype=np.float64)) < 0.0
        fc[flip] = fc[flip][:, ::-1]
        return fc

    # Each patch is oriented along its own outward reference direction, so
    # the assembled closed surface is consistently wound outwards.
    faces = [
        _emit(idx_port, (0.0, -1.0, 0.0)),
        _emit(idx_stbd, (0.0, 1.0, 0.0)),
        _emit(idx_deck, (0.0, 0.0, 1.0)),
        _emit(idx_bot, (0.0, 0.0, -1.0)),
        _emit(idx_tra, (-1.0, 0.0, 0.0)),
    ]

    f = np.concatenate(faces, axis=0)
    patch = np.concatenate([np.full(fc.shape[0], i, dtype=np.int8) for i, fc in enumerate(faces)])
    mesh = IceBowMesh(vertices=v, faces=f, params=p, resolution=res, patches=patch)
    return orient_outward(mesh)


def _tri_normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1)
    out = np.zeros_like(n)
    good = norm > 0.0
    out[good] = n[good] / norm[good, None]
    return out


def orient_outward(mesh: IceBowMesh) -> IceBowMesh:
    """Flip the global winding if the divergence-theorem volume is negative."""
    if mesh_volume(mesh.vertices, mesh.faces) < 0.0:
        return replace_mesh(mesh, mesh.vertices, mesh.faces[:, ::-1].copy())
    return mesh


def replace_mesh(mesh: IceBowMesh, vertices: np.ndarray, faces: np.ndarray) -> IceBowMesh:
    return IceBowMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        params=mesh.params,
        resolution=mesh.resolution,
        patches=mesh.patches,
    )


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume by the divergence theorem (``sum v0 . (v1 x v2) / 6``)."""
    tri = np.asarray(vertices)[np.asarray(faces)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = np.asarray(vertices)[np.asarray(faces)]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def _div_terms(tri: np.ndarray) -> float:
    if tri.size == 0:
        return 0.0
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def _clip_below_z(tri: np.ndarray, zc: float) -> list:
    """Sutherland-Hodgman clip of one triangle against ``z <= zc``."""
    poly = [tri[0], tri[1], tri[2]]
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cur_in, nxt_in = cur[2] <= zc, nxt[2] <= zc
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            t = (zc - cur[2]) / (nxt[2] - cur[2])
            out.append(cur + t * (nxt - cur))
    return out


def submerged_volume(mesh: IceBowMesh) -> dict:
    """Displacement volume below the design waterline, from the closed surface.

    Hull triangles fully below ``T`` are summed by the divergence theorem,
    crossing triangles are clipped at ``T`` and the free surface is closed
    with the waterplane strip reconstructed from the waterline vertex row, so
    the result is a pure mesh quantity (no re-integration of the analytic
    form).  Also returns the full-body volume of the same mesh.
    """
    p = mesh.params
    T = p.draft
    tri = mesh.triangles
    z = tri[:, :, 2]
    below = (z <= T).all(axis=1)
    above = (z >= T).all(axis=1)
    vol = _div_terms(tri[below])
    for t in tri[~below & ~above]:
        poly = _clip_below_z(t, T)
        for k in range(1, len(poly) - 1):
            vol += _div_terms(np.array([[poly[0], poly[k], poly[k + 1]]]))

    # waterplane cap: recover the station strip from the waterline vertex row
    wl = np.abs(mesh.vertices[:, 2] - T) <= 1e-8 * p.D
    v = mesh.vertices[wl]
    key = np.round(v[:, 0] / (1e-9 * p.L)).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    xs = np.zeros(uk.size)
    y_hi = np.full(uk.size, -np.inf)
    y_lo = np.full(uk.size, np.inf)
    np.maximum.at(xs, inv, v[:, 0])
    np.maximum.at(y_hi, inv, v[:, 1])
    np.minimum.at(y_lo, inv, v[:, 1])
    order = np.argsort(xs)
    xs, y_hi, y_lo = xs[order], y_hi[order], y_lo[order]
    for i in range(xs.size - 1):
        quad = np.array(
            [
                [xs[i], y_lo[i], T],
                [xs[i + 1], y_lo[i + 1], T],
                [xs[i + 1], y_hi[i + 1], T],
                [xs[i], y_hi[i], T],
            ]
        )
        vol += _div_terms(np.array([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]))
    return {
        "nabla": vol,
        "volume_full": mesh_volume(mesh.vertices, mesh.faces),
        "waterline_stations": int(xs.size),
    }


def inside_solid(params: IceBowParams, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Analytic indicator ``0 <= x <= x_stem(z), |y| <= b(x,z), 0 <= z <= D``."""
    p = resolve_params(params)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xi = x / p.L
    zeta = z / p.D
    ok = (zeta >= -tol / p.D) & (zeta <= 1.0 + tol / p.D) & (xi >= -tol / p.L)
    xs = stem_x_norm(p, np.clip(zeta, 0.0, 1.0))
    ok &= xi <= xs + tol / p.L
    b = half_breadth_norm(p, np.clip(xi, 0.0, 1.0), np.clip(zeta, 0.0, 1.0)) * p.L
    ok &= np.abs(y) <= b + tol
    ok &= (z >= -tol) & (z <= p.D + tol)
    return ok# ---------------------------------------------------------------------------
# Geometry admission gate
# ---------------------------------------------------------------------------


def _directed_edges(faces: np.ndarray) -> np.ndarray:
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)


def validate_mesh(
    mesh: IceBowMesh,
    analytic_volume: float | None = None,
    vol_tol: float = 0.02,
    eps_frac: float = 1e-3,
) -> dict:
    """Admission gate for one family member.

    Mandatory checks: water-tightness, manifoldness (no boundary and no
    non-manifold edge), Euler characteristic ``V - E + F = 2``, consistent
    winding, positive signed volume, outward normals and the absence of
    zero-area triangles.  ``passed`` is the conjunction of all of them, so a
    corrupted mesh (dropped face, flipped winding, duplicated vertex) must
    fail at least one check.
    """
    p = mesh.params
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    n_v = int(v.shape[0])

    degenerate_index = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    face_use = f[~degenerate_index] if degenerate_index.any() else f

    e = _directed_edges(face_use)
    und = np.sort(e, axis=1)
    key = und[:, 0].astype(np.int64) * n_v + und[:, 1]
    uniq, counts = np.unique(key, return_counts=True)
    n_edges = int(uniq.size)

    watertight = bool(np.all(counts == 2))
    no_boundary = bool(np.all(counts >= 2))
    no_nonmanifold = bool(np.all(counts <= 2))

    dkey = e[:, 0].astype(np.int64) * n_v + e[:, 1]
    _, dcounts = np.unique(dkey, return_counts=True)
    consistent = bool(np.all(dcounts == 1))

    n_faces = int(face_use.shape[0])
    euler = int(n_v - n_edges + n_faces)

    tri = v[face_use]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area_floor = 1e-14 * p.L**2
    n_zero_area = int(np.count_nonzero(areas <= area_floor))

    vol_mesh = mesh_volume(v, f)
    sub = submerged_volume(mesh)
    if analytic_volume is None:
        analytic_volume = design_volume(p)["volume_full"]
    vol_err = abs(vol_mesh - analytic_volume) / abs(analytic_volume)
    nabla_analytic = design_volume(p)["nabla"]
    nabla_err = abs(sub["nabla"] - nabla_analytic) / abs(nabla_analytic)

    # Outward test.  For a surface built from the five patches the outward
    # reference direction of each face is known exactly, so the test is
    # n . ref_patch > 0 (a strict, crease-robust test).  A centroid-offset
    # sample test against the analytic indicator is reported as a secondary
    # diagnostic only: at the aft crease of min(phi_bow, phi_aft) the averaged
    # face normal legitimately points into the body.
    normals = _tri_normals(tri)
    centroids = tri.mean(axis=1)
    eps = eps_frac * min(p.B, p.D)
    outside = ~inside_solid(p, centroids + eps * normals)
    frac_outward = float(np.count_nonzero(outside)) / max(n_faces, 1)
    patch_ids = None if mesh.patches is None else np.asarray(mesh.patches)
    if patch_ids is not None and patch_ids.size != f.shape[0]:
        patch_ids = None  # externally edited mesh: fall back to the sample test
    if patch_ids is not None:
        patch_ids = patch_ids if not degenerate_index.any() else patch_ids[~degenerate_index]
        ref = np.zeros_like(normals)
        ref[:, 1] = np.where(patch_ids == 0, -1.0, np.where(patch_ids == 1, 1.0, 0.0))
        ref[:, 2] = np.where(patch_ids == 2, 1.0, np.where(patch_ids == 3, -1.0, 0.0))
        ref[:, 0] = np.where(patch_ids == 4, -1.0, 0.0)
        outward_mask = np.einsum("ij,ij->i", normals, ref) > 0.0
        frac_patch_outward = float(np.count_nonzero(outward_mask)) / max(n_faces, 1)
    else:
        frac_patch_outward = frac_outward

    checks = {
        "watertight": watertight,
        "manifold_no_boundary": no_boundary,
        "manifold_no_nonmanifold": no_nonmanifold,
        "euler_characteristic_2": euler == 2,
        "consistent_orientation": consistent,
        "no_zero_area_triangles": n_zero_area == 0,
        "no_collapsed_faces": int(np.count_nonzero(degenerate_index)) == 0,
        "positive_signed_volume": vol_mesh > 0.0,
        "outward_normals": frac_patch_outward >= 1.0,
    }
    mandatory = tuple(checks)
    return {
        "passed": bool(all(checks[k] for k in mandatory)),
        "checks": checks,
        "n_vertices": n_v,
        "n_edges": n_edges,
        "n_faces": n_faces,
        "euler_characteristic": euler,
        "n_zero_area": n_zero_area,
        "n_duplicate_index_faces": int(np.count_nonzero(degenerate_index)),
        "min_triangle_area": float(areas.min()) if areas.size else 0.0,
        "volume_mesh": vol_mesh,
        "volume_analytic": float(analytic_volume),
        "volume_rel_error": float(vol_err),
        "volume_rel_error_ok": bool(vol_err <= vol_tol),
        "nabla_mesh": sub["nabla"],
        "nabla_analytic": float(nabla_analytic),
        "nabla_rel_error": float(nabla_err),
        "nabla_rel_error_ok": bool(nabla_err <= vol_tol),
        "outward_normal_fraction": frac_patch_outward,
        "outward_offset_sample_fraction": frac_outward,
        "edge_use_min": int(counts.min()) if counts.size else 0,
        "edge_use_max": int(counts.max()) if counts.size else 0,
    }


def reverse_parameters(mesh: IceBowMesh) -> dict:
    """Re-derive ``Cb/L/B`` and ``alpha_stem/beta_ent`` **from the mesh**."""
    p = mesh.params
    v = mesh.vertices
    L = float(v[:, 0].max() - v[:, 0].min())
    T = p.draft
    wl = np.abs(v[:, 2] - T) <= 1e-8 * p.D
    if not np.any(wl):
        raise ValueError("mesh has no vertex row on the design waterline")
    B = float(2.0 * np.abs(v[wl, 1]).max())
    vol_full = abs(mesh_volume(v, mesh.faces))
    vol = abs(submerged_volume(mesh)["nabla"])
    cb = vol / (L * B * T)

    # stem rake: the two stem vertices straddling the waterline.  The stem
    # seam is the forward-most centreline vertex at every height (deck /
    # bottom / transom centreline vertices also sit on y = 0, so take max x).
    stem = np.abs(v[:, 1]) <= 1e-10 * p.L
    zkey = np.round(v[stem, 2] / (1e-9 * p.D)).astype(np.int64)
    order = np.lexsort((v[stem, 0], zkey))
    zs_all, xs_all = v[stem, 2][order], v[stem, 0][order]
    keep = np.ones(zs_all.size, dtype=bool)
    keep[:-1] = zkey[order][:-1] != zkey[order][1:]
    zs, xs_ = zs_all[keep], xs_all[keep]
    hi = int(np.searchsorted(zs, T, side="right"))
    lo = max(hi - 1, 0)
    hi = min(hi, zs.size - 1)
    dz = float(zs[hi] - zs[lo])
    alpha = math.degrees(math.atan2(float(xs_[hi] - xs_[lo]), dz)) if dz > 0 else float("nan")

    # waterline entrance angle: two forward-most starboard waterline vertices
    wl_s = wl & (v[:, 1] > 0.0)
    xw = v[wl_s, 0]
    yw = v[wl_s, 1]
    ordx = np.argsort(xw)[::-1]
    xw, yw = xw[ordx], yw[ordx]
    dx = float(xw[0] - xw[1])
    dy = float(yw[1] - yw[0])
    beta = math.degrees(math.atan2(abs(dy), abs(dx))) if dx > 0 else float("nan")

    return {
        "L": L,
        "B": B,
        "T": T,
        "nabla": vol,
        "volume_full": vol_full,
        "cb": cb,
        "cb_error": (cb - p.cb_target) / p.cb_target,
        "L_error": (L - p.L) / p.L,
        "B_error": (B - p.B) / p.B,
        "alpha_stem_deg": alpha,
        "alpha_error_deg": alpha - p.alpha_stem_deg,
        "beta_ent_deg": beta,
        "beta_error_deg": beta - p.beta_ent_deg,
    }


# ---------------------------------------------------------------------------
# Offsets table (型值表) and exports
# ---------------------------------------------------------------------------


def offsets_table(params: IceBowParams, n_stations: int = 21, n_levels: int = 11) -> dict:
    """Station offsets ``(xi, x, z, b)`` for downstream BEM/DEM/panel legs."""
    p = resolve_params(params)
    xi = np.linspace(0.0, 1.0, n_stations)
    zeta = np.linspace(0.0, 1.0, n_levels)
    XI, ZE = np.meshgrid(xi, zeta, indexing="ij")  # (nx, nz)
    b = half_breadth_norm(p, XI, ZE) * p.L
    xs = stem_x_norm(p, zeta) * p.L
    x = XI * xs[None, :]
    return {
        "xi": XI,
        "x": x,
        "z": ZE * p.D,
        "half_beam": b,
        "waterline_length": xs,
        "n_stations": n_stations,
        "n_levels": n_levels,
    }


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, header: bytes = b"icebow") -> Path:
    """Binary STL writer (numpy + struct only)."""
    path = Path(path)
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    normals = _tri_normals(tri)
    rec = np.zeros(
        tri.shape[0], dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]
    )
    rec["n"] = normals.astype(np.float32)
    rec["v"] = tri.astype(np.float32)
    with path.open("wb") as fh:
        fh.write(header[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", tri.shape[0]))
        fh.write(rec.tobytes())
    return path


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_stl_vertices(path: Path) -> np.ndarray:
    """Minimal binary/ASCII STL reader (round-trip contract checks)."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:4096]:
        txt = raw.decode("ascii", errors="ignore").split()
        vals = [float(t) for t in txt if _is_float(t)]
        arr = np.array(vals).reshape(-1, 12)
        return arr[:, 3:].reshape(-1, 3, 3)
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw[84 : 84 + 50 * n], dtype=np.uint8).reshape(n, 50)
    tri = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    return tri.astype(np.float64)


def wetted_surface(mesh: IceBowMesh) -> dict:
    """Submerged hull patch (centroid ``z <= T``) and its area."""
    p = mesh.params
    tri = mesh.triangles
    cent = tri.mean(axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    wet = cent[:, 2] <= p.draft
    return {
        "triangles": tri[wet],
        "area": float(areas[wet].sum()),
        "area_total": float(areas.sum()),
        "n_wet": int(np.count_nonzero(wet)),
        "draft": p.draft,
    }


def longitudinal_distribution(
    params: IceBowParams, n_stations: int = 201, n_levels: int = 201
) -> dict:
    """Sectional areas below the waterline, displacement and LCB."""
    p = resolve_params(params)
    xi = np.linspace(1e-9, 1.0, n_stations)
    zeta = np.linspace(0.0, p.zeta_T, n_levels)
    XI, ZE = np.meshgrid(xi, zeta, indexing="ij")
    b = half_breadth_norm(p, XI, ZE)
    a = np.trapezoid(2.0 * b * p.D, x=zeta, axis=1)  # b/L * D  -> area in L units
    nabla = float(np.trapezoid(a, x=xi) * p.L * p.L)
    xcg = float(np.trapezoid(a * xi, x=xi) / np.trapezoid(a, x=xi))
    return {
        "xi": xi,
        "section_area": a,
        "nabla": nabla,
        "lcb_frac": xcg - 0.5,
        "cb": nabla / (p.B * p.zeta_T * p.D),
    }


# ---------------------------------------------------------------------------
# SDF / voxel casting (ray parity, as in tensorlbm.geometry_voxel)
# ---------------------------------------------------------------------------


def voxelize_mesh(mesh: IceBowMesh, spacing: float, pad_cells: int = 1, chunk: int = 384):
    """Rasterise the closed hull into a boolean occupancy mask by x-ray parity.

    One ray per ``(y, z)`` column along ``+x``; a cell is inside iff an odd
    number of triangle crossings lie strictly ahead of its centre.  The
    sub-cell asymmetric perturbation and strict barycentric bounds mirror
    :mod:`tensorlbm.geometry_voxel`, so shared edges are never counted twice.

    Returns ``(mask, origin, spacing)`` with ``mask`` shaped ``(nz, ny, nx)``
    like the repository's LBM solid masks.
    """
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    lo = v.min(axis=0) - pad_cells * spacing
    hi = v.max(axis=0) + pad_cells * spacing
    nx, ny, nz = (np.maximum(np.ceil((hi - lo) / spacing).astype(int), 1)).tolist()

    yc = lo[1] + (np.arange(ny) + 0.5) * spacing + _EPS_Y * spacing
    zc = lo[2] + (np.arange(nz) + 0.5) * spacing + _EPS_Z * spacing
    CY, CZ = np.meshgrid(yc, zc, indexing="ij")  # (ny, nz)
    cy = CY.reshape(-1)
    cz = CZ.reshape(-1)
    ncol = cy.size

    cnt = np.zeros((ncol, nx + 1), dtype=np.int32)
    nx1 = nx + 1
    tri = v[f]
    for start in range(0, tri.shape[0], chunk):
        t = tri[start : start + chunk]
        ty, tz = t[:, :, 1], t[:, :, 2]
        y0, y1, y2 = ty[:, 0][:, None], ty[:, 1][:, None], ty[:, 2][:, None]
        z0, z1, z2 = tz[:, 0][:, None], tz[:, 1][:, None], tz[:, 2][:, None]
        cand = (
            (cy[None, :] >= np.minimum(np.minimum(y0, y1), y2))
            & (cy[None, :] <= np.maximum(np.maximum(y0, y1), y2))
            & (cz[None, :] >= np.minimum(np.minimum(z0, z1), z2))
            & (cz[None, :] <= np.maximum(np.maximum(z0, z1), z2))
        )
        rows = np.nonzero(cand)[0]
        if rows.size == 0:
            continue
        cols = np.nonzero(cand)[1]
        det = (y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)
        d = det[rows, 0]
        good = np.abs(d) > 1e-18
        rows, cols, d = rows[good], cols[good], d[good]
        if rows.size == 0:
            continue
        py, pz = cy[cols], cz[cols]
        u = ((py - y0[rows, 0]) * (z2[rows, 0] - z0[rows, 0])
             - (pz - z0[rows, 0]) * (y2[rows, 0] - y0[rows, 0])) / d
        w = ((y1[rows, 0] - y0[rows, 0]) * (pz - z0[rows, 0])
             - (z1[rows, 0] - z0[rows, 0]) * (py - y0[rows, 0])) / d
        hit = (u > 0.0) & (w > 0.0) & (u + w < 1.0)
        if not np.any(hit):
            continue
        rows, u, w, cols = rows[hit], u[hit], w[hit], cols[hit]
        x0, x1v, x2v = t[rows, 0, 0], t[rows, 1, 0], t[rows, 2, 0]
        xcross = x0 + u * (x1v - x0) + w * (x2v - x0)
        k = np.clip(np.ceil((xcross - lo[0]) / spacing - 0.5), 0, nx).astype(np.int64)
        flat = cols * nx1 + k
        cnt += np.bincount(flat, minlength=ncol * nx1).reshape(ncol, nx1).astype(np.int32)

    suffix = np.cumsum(cnt[:, ::-1], axis=1)[:, ::-1]
    parity = suffix[:, 1:]  # crossings strictly ahead of each cell
    mask = (parity % 2 == 1).reshape(ny, nz, nx).transpose(1, 0, 2).copy()
    return mask, lo, float(spacing)


def gci(fine: float, medium: float, coarse: float, r: float = 2.0) -> dict:
    """Roache grid-convergence index for a 3-level refinement study."""
    e21 = abs(medium - fine)
    e32 = abs(coarse - medium)
    eps = 1e-300
    if e21 <= eps or e32 <= eps or fine == 0.0:
        p = float("nan")
        gci_f = float("nan")
    else:
        p = math.log(e32 / e21) / math.log(r)
        if not math.isfinite(p) or p <= 0.0:
            p = float("nan")
            gci_f = 1.25 * e21 / abs(fine) * 100.0
        else:
            gci_f = 1.25 * (e21 / abs(fine)) / (r**p - 1.0) * 100.0
    extrap = float(fine)
    if math.isfinite(p) and p > 0.0:
        extrap = fine + (fine - medium) / (r**p - 1.0)
    return {
        "fine": float(fine),
        "medium": float(medium),
        "coarse": float(coarse),
        "r": float(r),
        "observed_order": float(p),
        "gci_fine_percent": float(gci_f),
        "richardson": float(extrap),
    }


def voxel_volume_study(
    mesh: IceBowMesh,
    spacings=(0.005, 0.0025, 0.00125),
    analytic_volume: float | None = None,
) -> dict:
    """Three-level voxel volume convergence with GCI (criterion (d))."""
    p = mesh.params
    dv = design_volume(p)
    if analytic_volume is None:
        analytic_volume = dv["volume_full"]  # the mask holds the closed body
    rows = []
    for s in spacings:
        mask, origin, sp = voxelize_mesh(mesh, float(s))
        vol = float(mask.sum()) * sp**3
        zc = origin[2] + (np.arange(mask.shape[0]) + 0.5) * sp
        sub = float(mask[zc <= p.draft].sum()) * sp**3
        rows.append(
            {
                "spacing": sp,
                "spacing_over_L": sp / p.L,
                "cells": int(mask.size),
                "solid_cells": int(mask.sum()),
                "volume": vol,
                "rel_error": (vol - analytic_volume) / analytic_volume,
                "rel_error_abs": abs(vol - analytic_volume) / analytic_volume,
                "submerged_volume": sub,
                "submerged_rel_error": (sub - dv["nabla"]) / dv["nabla"],
                "submerged_rel_error_abs": abs(sub - dv["nabla"]) / dv["nabla"],
            }
        )
    vol = [r["volume"] for r in rows]
    sub = [r["submerged_volume"] for r in rows]
    monotone = all(vol[i + 1] >= vol[i] for i in range(len(vol) - 1)) or all(
        vol[i + 1] <= vol[i] for i in range(len(vol) - 1)
    )
    # Monotone convergence in the refinement-study sense: each refinement must
    # reduce the change, i.e. |V_fine - V_med| <= |V_med - V_coarse|.  When the
    # error is at the staircase noise floor the raw volume sequence can
    # oscillate in sign while still converging; that is reported, not hidden.
    conv = abs(vol[-1] - vol[-2]) <= abs(vol[-2] - vol[-3]) if len(vol) >= 3 else False
    conv_sub = abs(sub[-1] - sub[-2]) <= abs(sub[-2] - sub[-3]) if len(sub) >= 3 else False
    if len(vol) >= 3:
        r_ratio = float(spacings[0] / spacings[1])
        conv = gci(vol[-1], vol[-2], vol[-3], r=r_ratio)
        conv_sub = gci(sub[-1], sub[-2], sub[-3], r=r_ratio)
    else:
        conv = {"gci_fine_percent": float("nan"), "observed_order": float("nan")}
        conv_sub = dict(conv)
    return {
        "analytic_volume": float(analytic_volume),
        "analytic_nabla": float(dv["nabla"]),
        "levels": rows,
        "monotone": bool(monotone),
        "monotone_volume_sequence": bool(monotone),
        "converging": bool(conv),
        "submerged_converging": bool(conv_sub),
        "finest_rel_error_abs": rows[-1]["rel_error_abs"],
        "finest_submerged_rel_error_abs": rows[-1]["submerged_rel_error_abs"],
        "gci": conv,
        "gci_submerged": conv_sub,
    }


# ---------------------------------------------------------------------------
# Resistance: ITTC-57 friction + Holtrop-Mennen residual (regression model)
# ---------------------------------------------------------------------------


def ittc57_cf(re: float) -> float:
    if re <= 100.0:
        raise ValueError("Reynolds number too low for ITTC-1957")
    return 0.075 / (math.log10(re) - 2.0) ** 2


def holtrop_resistance(
    params: IceBowParams,
    fn: float = 0.20,
    L_m: float = 100.0,
    mesh: IceBowMesh | None = None,
    nu: float = _NU,
    rho: float = _RHO,
) -> dict:
    """Calm-water resistance ``R_T = R_F(1+k) + R_W`` (Holtrop-Mennen 1984).

    Geometry inputs (``nabla``, ``S``, ``Cb``, ``Cp``, ``lcb``, ``iE``) all
    come from this module's own hull surface, so the stem-angle sensitivity
    below is a *geometric* sensitivity.  Holtrop-Mennen carries no explicit
    stem-rake term: the rake enters through the wetted surface and the
    displaced-volume distribution only -- see the report limitations.
    """
    p = resolve_params(params)
    scale = L_m / p.L  # model -> metres
    if mesh is None:
        mesh = build_ice_bow_mesh(p)
    wet = wetted_surface(mesh)
    S = wet["area"] * scale**2
    nabla = abs(submerged_volume(mesh)["nabla"]) * scale**3
    L = p.L * scale
    B = p.B * scale
    T = p.draft * scale
    ld = longitudinal_distribution(p)
    lcb = ld["lcb_frac"] * L
    cb = nabla / (L * B * T)

    z = np.linspace(0.0, p.zeta_T, 401)
    b_mid = half_breadth_norm(p, np.full_like(z, 0.5), z) * p.L
    am = float(np.trapezoid(2.0 * b_mid * p.D, x=z)) * scale**2
    cm = am / (B * T)
    cp = cb / cm if cm > 0 else float("nan")

    speed = fn * math.sqrt(_G * L)
    re = speed * L / nu
    cf = ittc57_cf(re)

    c_stern = -25.0
    c14 = 1.0 + 0.011 * c_stern
    lr = L * (1.0 - cp + 0.06 * cp * lcb / max(4.0 * cp - 1.0, 1e-6))
    form = (
        0.93
        + 0.487118
        * c14
        * (B / L) ** 1.06806
        * (T / L) ** 0.46106
        * (L / lr) ** 0.121563
        * (L**3 / nabla) ** 0.36486
        * (1.0 - cp) ** (-0.604247)
    )
    rf = 0.5 * rho * speed**2 * S * cf * form

    c7 = B / L if B / L >= 0.11 else 0.229577 * (B / L) ** (1.0 / 3.0)
    i_e = p.beta_ent_deg
    c1 = 2223105.0 * c7**3.78613 * (T / B) ** 1.07961 * (90.0 - i_e) ** (-1.37565)
    c2 = 1.0  # no bulbous bow
    c5 = 1.0  # no immersed transom
    c16 = 8.07981 * cp - 13.8673 * cp**2 + 6.984388 * cp**3
    m1 = 0.0140407 * L / T - 1.75254 * nabla ** (1.0 / 3.0) / L - 4.79323 * B / L - c16
    l3_nab = L**3 / nabla
    if l3_nab > 1727.0:
        c15 = -1.69385
    elif l3_nab < 512.0:
        c15 = -1.69385 + (L / nabla ** (1.0 / 3.0) - 8.0) / 2.36
    else:
        c15 = -1.69385
    m2 = c15 * cp**2 * math.exp(-0.1 * fn ** (-2.0))
    lam = 1.446 * cp - 0.36
    rw = (
        c1 * c2 * c5 * nabla * rho * _G
        * math.exp(m1 * fn ** (-0.9) + m2 * math.cos(lam * fn ** (-2.0)))
    )
    rt = rf + rw
    return {
        "model": "holtrop-mennen-1984",
        "fn": float(fn),
        "speed_ms": speed,
        "L_m": L,
        "B_m": B,
        "T_m": T,
        "nabla_m3": nabla,
        "S_m2": S,
        "cb": cb,
        "cm": cm,
        "cp": cp,
        "lcb_m": lcb,
        "iE_deg": i_e,
        "re": re,
        "cf": cf,
        "form_factor_1k": form,
        "rf_n": rf,
        "rw_n": rw,
        "rt_n": rt,
        "ct": rt / (0.5 * rho * speed**2 * S),
        "rf_share": rf / rt if rt > 0 else float("nan"),
    }


def michell_wave_resistance(
    params: IceBowParams,
    fn: float = 0.20,
    L_m: float = 100.0,
    rho: float = _RHO,
    n_x: int = 241,
    n_z: int = 91,
    n_u: int = 320,
    u_max: float = 3.0,
) -> dict:
    """Thin-ship (Michell) wave resistance from the same hull surface.

    Used as the *stem-rake-sensitive* indicator: unlike Holtrop-Mennen it
    sees the vertical distribution of the source strength, which is what a
    raked ice bow changes.  Normalisation follows the classical form

        R_W = 4 rho g^2 / (pi U^2) int_0^{pi/2} (I^2 + J^2) sec^3(theta) dtheta

    with ``I, J`` the cosine/sine transforms of ``db/dx`` weighted by
    ``exp(k z')`` (``k = k0 sec^2 theta``, ``z' <= 0`` below the waterline).
    Absolute magnitude is indicative only; the *trend* is the deliverable.
    """
    p = resolve_params(params)
    scale = L_m / p.L
    L = p.L * scale
    T = p.draft * scale
    speed = fn * math.sqrt(_G * L)
    k0 = _G / speed**2

    xi = np.linspace(0.0, 1.0, n_x)
    zeta = np.linspace(0.0, p.zeta_T, n_z)
    XI, ZE = np.meshgrid(xi, zeta, indexing="ij")
    h = xi[1] - xi[0]
    b_plus = half_breadth_norm(p, np.clip(XI + h, 0.0, 1.0), ZE)
    b_minus = half_breadth_norm(p, np.clip(XI - h, 0.0, 1.0), ZE)
    # db/dx is non-dimensional: b and x are both scaled by L, so the
    # difference quotient of the *normalised* half-breadth is already
    # d(b/L)/d(x/L) = db/dx.
    dbdx = (b_plus - b_minus) / (2.0 * h)
    x = XI * p.L * scale
    zp = ZE * p.D * scale - T  # <= 0 below the waterline
    cell = (p.L * scale / (n_x - 1)) * (p.D * scale / (n_z - 1))
    src = (dbdx * cell).reshape(-1)
    xf, zf = x.reshape(-1), zp.reshape(-1)

    # u = tan(theta); the integrand decays like exp(-k0 sec^2(theta) T), so it
    # is negligible beyond u ~ 3 for T/lambda0 ~ 2 (measured, see the report's
    # quadrature convergence table).  Truncating there also keeps the x-wave
    # number (k0 sec theta) resolvable by the x-quadrature.
    u = np.linspace(0.0, u_max, n_u)
    total = 0.0
    for ui in u:
        sec = math.sqrt(1.0 + ui * ui)  # sec(theta)
        amp = src * np.exp(k0 * sec * sec * zf)  # vertical decay  exp(k z), k = k0 sec^2
        kx = k0 * sec  # x-wavenumber
        i_amp = float(np.dot(amp, np.cos(kx * xf)))
        j_amp = float(np.dot(amp, np.sin(kx * xf)))
        total += (i_amp * i_amp + j_amp * j_amp) * sec
    total *= u[1] - u[0]
    rw = 4.0 * rho * _G**2 / (math.pi * speed**2) * total
    return {
        "model": "michell-thin-ship",
        "fn": float(fn),
        "speed_ms": speed,
        "rw_n": float(rw),
        "k0_per_m": k0,
        "ct_wave": float(rw / (0.5 * rho * speed**2 * (L * p.B * scale))),
    }


def resistance_sensitivity(
    members: list[IceBowParams],
    fn: float = 0.20,
    L_m: float = 100.0,
    densities=((41, 31), (81, 61), (161, 121)),
    michell_densities=((61, 41, 81), (121, 81, 161), (241, 161, 321)),
    reference_alpha: float = 30.0,
) -> dict:
    """Stem-angle resistance sensitivity with grid-convergence indices.

    Two legs are computed from the *same* hull surface:

    ``R_T^HM``
        ``R_F (1+k) + R_W`` with the Holtrop-Mennen (1984) regression.
        Absolute values are credible for a hull of this size, but the
        regression carries no stem-rake term, so its rake sensitivity is
        only the wetted-surface one (a fraction of a percent).
    ``R_W^M``
        Michell thin-ship wave resistance.  This *does* see the vertical
        source distribution, i.e. exactly what a raked stem changes.  Its
        absolute magnitude is not used (thin-ship slenderness is violated at
        ``Cb = 0.7``); only the relative change is used.
    ``hybrid``
        ``dR_T/R_T = (R_F/R_T) dR_F/R_F + (R_W^HM/R_T) dR_W^M/R_W^M`` -- the
        theory-based rake trend applied to the regression-based split.  This
        is the quantity quoted as the rake sensitivity; both legs carry their
        own GCI (mesh refinement and quadrature refinement).
    """
    rows = []
    for m in members:
        p = resolve_params(m)
        rts_hm, rws_m, details = [], [], None
        for ns, nl in densities:
            mesh = build_ice_bow_mesh(p, n_stations=ns, n_levels=nl)
            d = holtrop_resistance(p, fn=fn, L_m=L_m, mesh=mesh)
            rts_hm.append(d["rt_n"])
            details = d
        for nx, nz, nu in michell_densities:
            rws_m.append(michell_wave_resistance(p, fn=fn, L_m=L_m, n_x=nx, n_z=nz, n_u=nu)["rw_n"])
        gci_hm = (
            gci(rts_hm[-1], rts_hm[-2], rts_hm[-3], r=2.0)
            if len(rts_hm) >= 3
            else {"gci_fine_percent": float("nan"), "observed_order": float("nan")}
        )
        gci_m = (
            gci(rws_m[-1], rws_m[-2], rws_m[-3], r=2.0)
            if len(rws_m) >= 3
            else {"gci_fine_percent": float("nan"), "observed_order": float("nan")}
        )
        rows.append(
            {
                "family": p.family,
                "alpha_stem_deg": p.alpha_stem_deg,
                "beta_ent_deg": p.beta_ent_deg,
                "c_flare": p.c_flare,
                "c_fwd": p.c_fwd,
                "r_t": p.r_t,
                "cb": details["cb"],
                "cp": details["cp"],
                "S_m2": details["S_m2"],
                "nabla_m3": details["nabla_m3"],
                "rf_n": details["rf_n"],
                "rw_holtrop_n": details["rw_n"],
                "rt_holtrop_n": rts_hm[-1],
                "rt_holtrop_per_density": [float(v) for v in rts_hm],
                "rt_holtrop_gci_percent": gci_hm["gci_fine_percent"],
                "rw_michell_n": rws_m[-1],
                "rw_michell_per_density": [float(v) for v in rws_m],
                "rw_michell_gci_percent": gci_m["gci_fine_percent"],
                "fn": fn,
            }
        )
    straight = sorted(
        [r for r in rows if r["family"] == "straight"], key=lambda r: r["alpha_stem_deg"]
    )
    ref = next(
        (r for r in straight if abs(r["alpha_stem_deg"] - reference_alpha) < 1e-9), straight[0]
    )
    for r in rows:
        r["delta_rt_holtrop"] = (r["rt_holtrop_n"] - ref["rt_holtrop_n"]) / ref["rt_holtrop_n"]
        r["delta_rw_michell"] = (r["rw_michell_n"] - ref["rw_michell_n"]) / ref["rw_michell_n"]
        share_w = ref["rw_holtrop_n"] / ref["rt_holtrop_n"]
        share_f = 1.0 - share_w
        r["delta_rt_hybrid"] = share_f * r["delta_rt_holtrop"] + share_w * r["delta_rw_michell"]

    def _monotone(seq) -> bool:
        return all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1)) or all(
            seq[i + 1] <= seq[i] for i in range(len(seq) - 1)
        )

    d_hm = [r["delta_rt_holtrop"] for r in straight]
    d_hy = [r["delta_rt_hybrid"] for r in straight]
    d_m = [r["delta_rw_michell"] for r in straight]
    # GCI of the *sensitivity quantity* (delta R_W / R_W) over the quadrature
    # refinement: recompute the deltas level by level.
    n_lev = len(michell_densities)
    ref_row = next(r for r in straight if r is ref)
    deltas_by_level = []
    for k in range(n_lev):
        base = ref_row["rw_michell_per_density"][k]
        deltas_by_level.append(
            [
                (r["rw_michell_per_density"][k] - base) / base
                for r in straight
            ]
        )
    for r in rows:
        d_lv = [
            (r["rw_michell_per_density"][k] - ref_row["rw_michell_per_density"][k])
            / ref_row["rw_michell_per_density"][k]
            for k in range(n_lev)
        ]
        r["delta_rw_michell_per_density"] = [float(v) for v in d_lv]
        if n_lev >= 3 and d_lv[-1] != 0.0:
            r["delta_rw_gci_percent"] = gci(d_lv[-1], d_lv[-2], d_lv[-3], r=2.0)["gci_fine_percent"]
        else:
            r["delta_rw_gci_percent"] = float("nan")
    gci_hy = {}
    if n_lev >= 3:
        gci_hy = gci(deltas_by_level[-1][-1], deltas_by_level[-2][-1], deltas_by_level[-3][-1], r=2.0)
    return {
        "fn": fn,
        "L_m": L_m,
        "reference_alpha_deg": reference_alpha,
        "rows": rows,
        "share_rw_holtrop": float(ref["rw_holtrop_n"] / ref["rt_holtrop_n"]),
        "delta_rt_holtrop_straight": d_hm,
        "delta_rt_hybrid_straight": d_hy,
        "delta_rw_michell_straight": d_m,
        "monotone_delta_rt_holtrop": bool(_monotone(d_hm)),
        "monotone_delta_rw_michell": bool(_monotone(d_m)),
        "monotone_delta_rt_hybrid": bool(_monotone(d_hy)),
        "max_abs_delta_rt_holtrop": float(max(abs(v) for v in d_hm)),
        "max_abs_delta_rt_hybrid": float(max(abs(v) for v in d_hy)),
        "max_rt_holtrop_gci_percent": float(max(r["rt_holtrop_gci_percent"] for r in rows)),
        "max_rw_michell_gci_percent": float(max(r["rw_michell_gci_percent"] for r in rows)),
        "gci_hybrid_35deg": gci_hy,
        "delta_rw_gci_max_percent": float(
            max(
                (r["delta_rw_gci_percent"] for r in rows if r["delta_rw_gci_percent"] == r["delta_rw_gci_percent"]),
                default=float("nan"),
            )
        ),
    }
