"""Differentiable SUBOFF voxelization: parameters -> SDF -> masks -> dC_D/dshape.

This module adds the *adjoint of the voxelizer* to the B4 drag-surrogate
stack: given SuboffConfig design parameters (``sail_scale``, ``fin_scale``,
``l_over_d_mult``, ...) it computes end-to-end gradients of the ensemble
drag prediction ``d(log10 C_D)/d(theta)`` by making every stage of the
conditioning chain differentiable:

    params -> analytic SDF -> soft/STE occupancy masks
           -> geometry channels (log_aproj_ratio, sail_frac, fin_frac,
              solid_frac; semantics identical to
              :func:`tensorlbm.ai.drag_cond.suboff_geometry_features`)
           -> condition_v3 vector -> CondFNODrag ensemble -> log10 C_D

Geometry model
--------------
Each SUBOFF component is written as a *signed distance-like field* in feet
(negative inside), reproducing the exact DARPA predicates of
:mod:`tensorlbm.suboff_cad` (hull 4-segment radius profile, sail 3-segment
polynomial cross-section with semi-elliptical cap, swept NACA cruciform
fins):

- ``hull``  -- smooth-max of the radial distance ``r - R(xi)`` and the
  axial-interval distance of the surface of revolution (radial distance,
  not exact normal distance);
- ``sail``  -- 2-D cross-section SDF (exact box + approximated ellipse
  cap) swept along the footprint half-width ``h(x)``, intersected with the
  axial footprint interval;
- ``fins``  -- smooth-max of thickness / span / chord-end constraint
  distances for each of the cruciform pairs, smooth-min'ed between pairs.

Unions/intersections use the polynomial smooth-min/-max
(:func:`smooth_min`/:func:`smooth_max`) with blending radius ``smooth_k``
in feet.  ``smooth_k`` is the explicit fidelity-vs-differentiability knob:

- ``smooth_k -> 0`` recovers the exact boolean predicates (hard mask IoU
  against :func:`tensorlbm.suboff_cad.build_suboff_mask` rises, pinned by
  tests), but gradients vanish away from surfaces and kinks appear;
- larger ``smooth_k`` blends component corners smoothly (gradients flow
  across joins) at the cost of an O(``smooth_k``) displacement of the zero
  level set near surface intersections.

Masking
-------
``soft_mask`` is sigmoid occupancy with temperature ``tau`` (feet; boundary
band width).  ``straight_through_mask`` evaluates the *hard* mask
(``sdf <= 0``) in the forward pass and back-propagates through the sigmoid,
so the differentiable channels are numerically identical (up to CAD
boundary cells) to the hard-voxel training features in the forward pass
while remaining differentiable in the backward pass.

Honest approximations (see docs/diff_voxelize_20260825.md for numbers)
----------------------------------------------------------------------
1. **STE bias** -- the gradient is that of the soft occupancy, not of the
   integer voxel counts; finite differences of the *hard* counts are
   piecewise constant and only agree in sign/direction on average.  The
   STE magnitude also depends on the occupancy temperature ``tau`` (it
   grows roughly linearly with ``tau`` through the component-junction
   band); the default ``tau`` is calibrated so STE, soft and hard-FD
   directions agree on the appendage axes.
2. **Smooth-union/-intersection error** -- the zero level set is displaced
   by O(``smooth_k``) near component joins (hull-sail, hull-fin, fin-pair
   corners); away from joins the SDFs reproduce the predicates exactly.
3. **Ellipse cap and swept fin distances are approximations** -- the sail
   cap uses the scaled-ellipse bound; fin chord/span distances use local
   frame distances (sweep obliquity ignored in magnitude, not in sign).
4. **Bypassed channels** -- the FNO field input channels 0-3 (normalised
   ux/uy/uz, rho) are frozen at a corpus reference row: the flow response
   to shape changes is NOT differentiated; only the solid-mask channel 4
   and the condition vector carry shape gradient.
5. **Field channel 4 uses the CAD mask slice**, whereas training used the
   simulation solid mask (they differ on a few boundary cells -- cache meta
   ``mask_bit_eq`` is false for most rows).
6. ``l_over_d_mult`` / ``nose_len_mult`` / ``stern_len_mult`` /
   ``sail_x_mult`` were **outside the training support** of the B4 v4
   ensemble (corpus sweeps ``sail_scale``/``fin_scale`` only); gradients
   along them flow through the geometry channels as an extrapolation of
   the same channel recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .ai.drag_cond import (
    PRODUCTION_GRID,
    CondFNODrag,
    SuboffGrid,
    condition_v3,
    suboff_geometry_features,
)
from .ai.inference_service import load_checkpoint
from .suboff_cad import (
    _BOW_END_FT,
    _FIN_H,
    _FIN_R_INNER,
    _FIN_R_OUTER,
    _FIN_SWEEP_C,
    _FIN_SWEEP_K,
    _HULL_NODES_FT,
    _MID_END_FT,
    _NACA_COEFFS,
    _SAIL_X1_END,
    _SAIL_X1_START,
    _SAIL_X2_END,
    _SAIL_X3_END,
    _SAIL_X_CENTER,
    _SAIL_YTMP,
    _SAIL_Z_DECK,
    _SAIL_ZMAX,
    _STERN_END_FT,
    _SUBOFF_L_FT,
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
)

__all__ = [
    "DEFAULT_SMOOTH_K_FT",
    "DEFAULT_TAU_FT",
    "DIFF_PARAM_NAMES",
    "DiffDragEnsemble",
    "DiffParams",
    "drag_finite_difference",
    "drag_forward",
    "drag_gradients",
    "mask_channels",
    "reference_drag_forward",
    "smooth_max",
    "smooth_min",
    "soft_mask",
    "straight_through_mask",
    "suboff_component_sdfs",
    "suboff_radius_profile_torch",
]

#: Default smooth-min/-max blending radius (ft).  About a twentieth of a
#: production-grid cell (0.186 ft) -- small enough that the hard mask
#: matches the CAD predicates cell-for-cell except at component joins.
DEFAULT_SMOOTH_K_FT = 0.01

#: Default sigmoid occupancy temperature (ft): the half-width of the
#: boundary band that carries gradient.  Must stay below the smallest
#: solid feature (fin half-thickness ~0.10 ft) or the soft mask saturates.
#: 0.02 ft (~a ninth of a production cell, band under half a cell per side)
#: is the measured sweet spot: the STE and soft estimators converge and the
#: appendage sensitivities agree in direction with the hard finite
#: differences, while larger tau inflates the STE magnitude roughly
#: linearly through the component-junction boundary band (see
#: docs/diff_voxelize_20260825.md and report section 3.1 for the sweep).
DEFAULT_TAU_FT = 0.02

#: SuboffConfig fields the differentiable path accepts (leaf tensors).
DIFF_PARAM_NAMES = (
    "r_over_l",
    "sail_scale",
    "fin_scale",
    "l_over_d_mult",
    "nose_len_mult",
    "stern_len_mult",
    "sail_x_mult",
)

# Numerical softening epsilons: the DARPA profile uses fractional powers
# (x^(1/2.1), sqrt) whose derivative is unbounded at 0; adding eps inside
# the root bounds the gradient at ~1e6 and biases the radius by <1e-5 ft.
_EPS_ROOT: float = 1e-9
_EPS_SQRT: float = 1e-12


# --------------------------------------------------------------------------- #
# Smooth boolean primitives
# --------------------------------------------------------------------------- #
def smooth_min(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """Polynomial smooth minimum (Quilez); exact min where ``|a-b| > k``."""
    if k <= 0.0:
        raise ValueError(f"smooth_k must be positive, got {k}")
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)


def smooth_max(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """Polynomial smooth maximum (negated :func:`smooth_min`)."""
    return -smooth_min(-a, -b, k)


def _smax_all(terms: Sequence[torch.Tensor], k: float) -> torch.Tensor:
    out = terms[0]
    for t in terms[1:]:
        out = smooth_max(out, t, k)
    return out


# --------------------------------------------------------------------------- #
# Differentiable parameter bundle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiffParams:
    """Zero-dim leaf tensors for the differentiable SuboffConfig axes."""

    r_over_l: torch.Tensor
    sail_scale: torch.Tensor
    fin_scale: torch.Tensor
    l_over_d_mult: torch.Tensor
    nose_len_mult: torch.Tensor
    stern_len_mult: torch.Tensor
    sail_x_mult: torch.Tensor

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, float | torch.Tensor] | None = None,
        *,
        requires_grad: bool = False,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> DiffParams:
        """Build the bundle from plain floats (defaults = mother geometry).

        Values already carrying a graph (0-dim tensors) are used verbatim,
        so callers can chain custom parameterisations.
        """
        device = torch.device(device)
        mother = SuboffConfig()
        defaults = {name: float(getattr(mother, name)) for name in DIFF_PARAM_NAMES}
        fields: dict[str, torch.Tensor] = {}
        for name in DIFF_PARAM_NAMES:
            raw = defaults[name] if values is None else values.get(name, defaults[name])
            if isinstance(raw, torch.Tensor):
                fields[name] = raw.to(device=device, dtype=dtype)
            else:
                fields[name] = torch.tensor(
                    float(raw), device=device, dtype=dtype, requires_grad=requires_grad
                )
        return cls(**fields)

    def values(self) -> dict[str, float]:
        """Detached plain-float view (a valid ``SuboffConfig`` payload)."""
        return {name: float(getattr(self, name).detach()) for name in DIFF_PARAM_NAMES}

    def to_config(self) -> SuboffConfig:
        """Frozen-float SuboffConfig of the current values (forward parity)."""
        return SuboffConfig(**self.values())


# --------------------------------------------------------------------------- #
# Torch mother radius profile (port of suboff_cad.suboff_radius_profile)
# --------------------------------------------------------------------------- #
def suboff_radius_profile_torch(xi: torch.Tensor) -> torch.Tensor:
    """Normalised hull radius ``r(xi)/R_max`` (mother frame), torch port.

    Mirrors :func:`tensorlbm.suboff_cad.suboff_radius_profile` segment by
    segment (bow polynomial ^1/2.1, parallel midbody, 6th-order stern
    taper, elliptic stern cap).  The only deviation is the root softening
    epsilons (forward bias < 1e-5) that bound gradients at the tips.
    """
    length_ft = float(_SUBOFF_L_FT)
    bow_end = _BOW_END_FT / length_ft
    mid_end = _MID_END_FT / length_ft
    stern_end = _STERN_END_FT / length_ft
    x_ft = xi * length_ft

    r = torch.zeros_like(xi)

    bow = (xi >= 0.0) & (xi < bow_end)
    tmp = 0.3 * x_ft - 1.0
    tmp2 = tmp * tmp
    tmp4 = tmp2 * tmp2
    a1 = (
        1.126395101 * x_ft * tmp4
        + 0.442874707 * x_ft * x_ft * (tmp2 * tmp)
        + 1.0
        - tmp4 * (1.2 * x_ft + 1.0)
    )
    r_bow = (torch.clamp(a1, min=0.0) + _EPS_ROOT).pow(1.0 / 2.1)
    r = torch.where(bow, r_bow, r)

    mid = (xi >= bow_end) & (xi <= mid_end)
    r = torch.where(mid, torch.ones_like(xi), r)

    taper = (xi > mid_end) & (xi <= stern_end)
    r1c = 0.1175
    k0 = 10.0
    k1 = 44.6244
    ksi = (_STERN_END_FT - x_ft) / 3.333333
    ksi2 = ksi * ksi
    ksi3 = ksi2 * ksi
    ksi4 = ksi3 * ksi
    ksi5 = ksi4 * ksi
    ksi6 = ksi5 * ksi
    a3 = (
        r1c * r1c
        + r1c * k0 * ksi2
        + (20.0 - 20.0 * r1c * r1c - 4.0 * r1c * k0 - k1 / 3.0) * ksi3
        + (-45.0 + 45.0 * r1c * r1c + 6.0 * r1c * k0 + k1) * ksi4
        + (36.0 - 36.0 * r1c * r1c - 4.0 * r1c * k0 - k1) * ksi5
        + (-10.0 + 10.0 * r1c * r1c + r1c * k0 + k1 / 3.0) * ksi6
    )
    r_taper = (torch.clamp(a3, min=0.0) + _EPS_SQRT).sqrt()
    r = torch.where(taper, r_taper, r)

    tail = (xi > stern_end) & (xi <= 1.0)
    val = 1.0 - (3.2 * x_ft - 44.733333) ** 2
    r_tail = (torch.clamp(val, min=0.0) + _EPS_SQRT).sqrt() * 0.1175
    r = torch.where(tail, r_tail, r)

    return torch.clamp(r, 0.0, 1.0)


def _pw_map_t(
    x: torch.Tensor,
    src: Sequence[torch.Tensor],
    dst: Sequence[float],
) -> torch.Tensor:
    """Piecewise-linear map with *tensor* source nodes (grad-capable).

    Semantics mirror ``suboff_cad._pw_map_torch``: values below/above the
    node range clamp to ``dst[0]``/``dst[-1]``.
    """
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("node tuples must have equal length >= 2")
    out = torch.where(x < src[0], torch.full_like(x, dst[0]), torch.full_like(x, dst[-1]))
    for a, b, c, d in zip(src[:-1], src[1:], dst[:-1], dst[1:]):
        seg = (x >= a) & (x <= b)
        out = torch.where(seg, c + (x - a) * ((d - c) / (b - a)), out)
    return out


def _variant_nodes_t(p: DiffParams) -> tuple[torch.Tensor, ...]:
    """Tensor version of ``suboff_cad._variant_nodes_ft``."""
    total = _SUBOFF_L_FT * p.l_over_d_mult
    return (
        torch.zeros_like(total),
        _BOW_END_FT * p.nose_len_mult,
        total - (_SUBOFF_L_FT - _MID_END_FT) * p.stern_len_mult,
        total - (_SUBOFF_L_FT - _STERN_END_FT) * p.stern_len_mult,
        total,
    )


def _interval_dist(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Signed distance from ``x`` to the closed interval [lo, hi].

    Negative inside (distance to the nearest end), positive outside.  The
    interior must be genuinely negative: a clamped-at-zero interior would
    make ``smooth_max`` lift the whole O(k) surface shell of the partner
    constraint above zero and shrink the solid by one shell of cells.
    """
    outside = torch.maximum(lo - x, x - hi)
    inside = -torch.minimum(x - lo, hi - x)
    return torch.where(outside > 0.0, outside, inside)


def _naca_thickness_t(s: torch.Tensor) -> torch.Tensor:
    """SUBOFF NACA 4-digit half-thickness (ft units, unnormalised)."""
    a, b, c, d, e = _NACA_COEFFS
    root = (torch.clamp(s, min=0.0) + _EPS_SQRT).sqrt()
    return a * root - b * s - c * s.square() + d * s.pow(3) - e * s.pow(4)


def _sail_halfwidth(x_u: torch.Tensor) -> torch.Tensor:
    """Sail cross-section half-width ``h(x)`` (ft, unscaled DARPA frame).

    Piecewise by axial segment; 0 outside the footprint ``[
    _SAIL_X1_START, _SAIL_X3_END ]``.  Mirrors ``_add_sail_mask``.
    """
    h = torch.zeros_like(x_u)
    m1 = (x_u > _SAIL_X1_START) & (x_u < _SAIL_X1_END)
    d1 = 3.0720 * (x_u - _SAIL_X1_START)
    c1 = 1.0 - (d1 - 1.0).pow(4) * (4.0 * d1 + 1.0)
    b1 = (1.0 / 3.0) * d1.square() * (d1 - 1.0).pow(3)
    a1 = 2.0 * d1 * (d1 - 1.0).pow(4)
    poly1 = torch.clamp(2.094759 * a1 + 0.2071781 * b1 + c1, min=0.0)
    h1 = _SAIL_ZMAX * (poly1 + _EPS_SQRT).sqrt()
    h = torch.where(m1, h1, h)

    m2 = (x_u > _SAIL_X1_END) & (x_u <= _SAIL_X2_END)
    h = torch.where(m2, torch.full_like(x_u, _SAIL_ZMAX), h)

    m3 = (x_u <= _SAIL_X3_END) & (x_u > _SAIL_X2_END)
    e3 = (_SAIL_X3_END - x_u) / 0.6822917
    f3 = e3 - 1.0
    h3 = _SAIL_ZMAX * (
        2.238361 * e3 * f3.pow(4)
        + 3.106529 * e3.square() * f3.pow(3)
        + 1.0
        - f3.pow(4) * (4.0 * e3 + 1.0)
    )
    h = torch.where(m3, torch.clamp(h3, min=0.0), h)
    return h


def _sail_cross_section_sdf(
    y: torch.Tensor, z: torch.Tensor, h: torch.Tensor, k: float
) -> torch.Tensor:
    """2-D cross-section SDF: box ``|y|<=h, 0<=z<=YTMP`` union ellipse cap.

    The box part is the exact slab-intersection SDF; the semi-elliptical
    cap (semi-axes ``h`` and ``h/2`` about ``z = YTMP``) uses the standard
    scaled-ellipse approximation (no closed form exists).
    """
    q1 = y.abs() - h
    q2 = -z
    q3 = z - _SAIL_YTMP
    q = torch.stack((q1, q2, q3), dim=0)
    outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=0)
    inside = torch.clamp(torch.amax(q, dim=0), max=0.0)
    d_box = outside + inside

    h_safe = torch.clamp(h, min=1e-9)
    ratio = (y / h_safe).square() + ((z - _SAIL_YTMP) / (h_safe / 2.0)).square()
    d_ell = (torch.sqrt(ratio) - 1.0) * (h_safe / 2.0)
    return smooth_min(d_box, d_ell, k)


def _fin_pair_sdf(
    x_c: torch.Tensor,
    span: torch.Tensor,
    thick: torch.Tensor,
    scale: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """One cruciform fin pair: swept NACA plate SDF.

    ``x_c`` is the chord-frame axial coordinate, ``span`` the scaled radial
    coordinate of this pair, ``thick`` the coordinate normal to the plate
    (absolute value).  Constraint distances (thickness, span in/out, chord
    ends) are combined with smooth-max; magnitudes are local frame
    distances (sweep obliquity ignored -- sign/zero set unaffected).
    """
    chord = _FIN_SWEEP_K * span + _FIN_SWEEP_C
    chord_safe = torch.where(chord.abs() < 1e-6, chord + 1e-6, chord)
    s = (x_c - _FIN_H) / chord_safe + 1.0
    t = _naca_thickness_t(s)
    d_thick = thick - t * scale
    d_span_out = (span - _FIN_R_OUTER) * scale
    d_span_in = (_FIN_R_INNER - span) * scale
    d_lead = -s * chord_safe * scale
    d_trail = (s - 1.0) * chord_safe * scale
    return _smax_all((d_thick, d_span_out, d_span_in, d_lead, d_trail), k)


# --------------------------------------------------------------------------- #
# Component SDFs on the grid
# --------------------------------------------------------------------------- #
def suboff_component_sdfs(
    grid: SuboffGrid | None = None,
    params: DiffParams | None = None,
    *,
    smooth_k: float = DEFAULT_SMOOTH_K_FT,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Component SDFs (ft, negative inside) on the voxel grid.

    Returns a dict with keys ``"hull"``, ``"sail"``, ``"fin"`` (per-
    component predicates, used for the disjoint channel decomposition) and
    ``"solid"`` (smooth-min union).  Shapes are ``(nz, ny, nx)``.
    """
    g = PRODUCTION_GRID if grid is None else grid
    p = DiffParams.from_values(None, device=device, dtype=dtype) if params is None else params
    dev = torch.device(device)
    zz, yy, xx = torch.meshgrid(
        torch.arange(g.nz, device=dev, dtype=dtype),
        torch.arange(g.ny, device=dev, dtype=dtype),
        torch.arange(g.nx, device=dev, dtype=dtype),
        indexing="ij",
    )
    ftlu = float(_SUBOFF_L_FT) / float(g.length)
    x_bow = g.cx - g.length / 2.0
    # Variant-frame axial ft coordinate: the lattice hull spans exactly
    # [0, L * l_over_d_mult] in ft (diameter fixed in ft, length in lattice
    # units unchanged) -- mirrors suboff_hull_mask / the appendage builders.
    x_var_ft = (xx - x_bow) * ftlu * p.l_over_d_mult
    y_ft = (yy - g.cy) * ftlu
    z_ft = (zz - g.cz) * ftlu

    nodes = _variant_nodes_t(p)
    x_mother_ft = _pw_map_t(x_var_ft, nodes, _HULL_NODES_FT)

    # --- hull: surface of revolution capped by the axial interval --------
    r_norm = suboff_radius_profile_torch(x_mother_ft / float(_SUBOFF_L_FT))
    r_max_ft = p.r_over_l * float(_SUBOFF_L_FT) / p.l_over_d_mult
    r_grid = torch.sqrt(y_ft.square() + z_ft.square())
    d_radial = r_grid - r_norm * r_max_ft
    d_axial = _interval_dist(x_var_ft, nodes[0], nodes[4])
    sdf_hull = smooth_max(d_radial, d_axial, smooth_k)

    # --- sail -------------------------------------------------------------
    # Appendage axial frame = mother frame, translated by sail_x_mult.
    x_sail_ft = x_mother_ft - (p.sail_x_mult - 1.0) * _SAIL_X_CENTER
    inv_s = 1.0 / p.sail_scale
    x_u = _SAIL_X_CENTER + (x_sail_ft - _SAIL_X_CENTER) * inv_s
    y_u = y_ft * inv_s
    z_u = _SAIL_Z_DECK + (z_ft - _SAIL_Z_DECK) * inv_s
    h = _sail_halfwidth(x_u)
    d_cross = _sail_cross_section_sdf(y_u, z_u, h, smooth_k)
    lo = torch.full_like(x_u, _SAIL_X1_START)
    hi = torch.full_like(x_u, _SAIL_X3_END)
    d_axial_sail = _interval_dist(x_u, lo, hi)
    sdf_sail = smooth_max(d_cross, d_axial_sail, smooth_k) * p.sail_scale

    # --- fins -------------------------------------------------------------
    inv_f = 1.0 / p.fin_scale
    x_c = _FIN_H + (x_mother_ft - _FIN_H) * inv_f
    y_span = _FIN_R_INNER + (y_ft.abs() - _FIN_R_INNER) * inv_f
    z_span = _FIN_R_INNER + (z_ft.abs() - _FIN_R_INNER) * inv_f
    d_fin_y = _fin_pair_sdf(x_c, y_span, z_ft.abs(), p.fin_scale, smooth_k)
    d_fin_z = _fin_pair_sdf(x_c, z_span, y_ft.abs(), p.fin_scale, smooth_k)
    sdf_fin = smooth_min(d_fin_y, d_fin_z, smooth_k)

    sdf_solid = smooth_min(smooth_min(sdf_hull, sdf_sail, smooth_k), sdf_fin, smooth_k)
    return {"hull": sdf_hull, "sail": sdf_sail, "fin": sdf_fin, "solid": sdf_solid}


# --------------------------------------------------------------------------- #
# Occupancy masks
# --------------------------------------------------------------------------- #
def soft_mask(sdf: torch.Tensor, tau: float = DEFAULT_TAU_FT) -> torch.Tensor:
    """Sigmoid occupancy ``sigmoid(-sdf / tau)`` (tau in ft)."""
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    return torch.sigmoid(-sdf / tau)


def straight_through_mask(sdf: torch.Tensor, tau: float = DEFAULT_TAU_FT) -> torch.Tensor:
    """STE occupancy: hard forward (``sdf <= 0``), soft backward."""
    hard = (sdf <= 0.0).to(dtype=sdf.dtype)
    soft = soft_mask(sdf, tau)
    return hard + soft - soft.detach()


def _occupancy(sdf: torch.Tensor, tau: float, ste: bool) -> torch.Tensor:
    """STE occupancy (default) or pure soft occupancy (surrogate studies)."""
    return straight_through_mask(sdf, tau) if ste else soft_mask(sdf, tau)


# --------------------------------------------------------------------------- #
# Differentiable geometry channels
# --------------------------------------------------------------------------- #
@dataclass
class DiffMaskChannels:
    """Masks, differentiable counts and geometry channels of one design."""

    ste: dict[str, torch.Tensor]
    """STE component masks ``{"hull", "sail", "fin"}`` (nz, ny, nx)."""

    cover: torch.Tensor
    """Solid occupancy (STE OR of the active components), (nz, ny, nx)."""

    counts: dict[str, torch.Tensor]
    """Differentiable voxel counts: ``v_bare``/``v_sail``/``v_fin``/
    ``v_solid``/``aproj``/``aproj_bare`` (float scalars; forward values are
    exact integers)."""

    channels: dict[str, torch.Tensor]
    """The four geometry channels (differentiable float scalars)."""

    channel_vector: torch.Tensor
    """The channels as a ``(4,)`` tensor ordered like
    :data:`tensorlbm.ai.drag_cond.GEOMETRY_CHANNEL_NAMES`."""

    midplane: torch.Tensor
    """STE solid occupancy of the ``z = nz // 2`` slice, (ny, nx)."""


def mask_channels(
    grid: SuboffGrid | None = None,
    params: DiffParams | None = None,
    *,
    hull_type: str = "full",
    smooth_k: float = DEFAULT_SMOOTH_K_FT,
    tau: float = DEFAULT_TAU_FT,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    sdfs: Mapping[str, torch.Tensor] | None = None,
    ste: bool = True,
) -> DiffMaskChannels:
    """Differentiable mask-derived geometry features of one design point.

    Forward semantics mirror ``tensorlbm.ai.drag_cond._component_counts``
    exactly (STE masks are hard in the forward pass): ``v_sail`` /
    ``v_fin`` are *net* disjoint contributions and ``aproj`` counts
    (y, z) columns with any solid voxel.  Backward semantics flow through
    the sigmoid occupancy of every contributing voxel.  ``ste=False``
    switches the forward itself to soft occupancy (a smooth surrogate of
    the counts whose finite differences must match autograd exactly).
    """
    g = PRODUCTION_GRID if grid is None else grid
    variant = SuboffHullType(hull_type)
    if sdfs is None:
        sdfs = suboff_component_sdfs(g, params, smooth_k=smooth_k, device=device, dtype=dtype)
    m_hull = _occupancy(sdfs["hull"], tau, ste)
    m_sail = (
        _occupancy(sdfs["sail"], tau, ste)
        if variant in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL)
        else torch.zeros_like(m_hull)
    )
    m_fin = (
        _occupancy(sdfs["fin"], tau, ste)
        if variant == SuboffHullType.FULL
        else torch.zeros_like(m_hull)
    )

    not_hull = 1.0 - m_hull
    v_bare = m_hull.sum()
    v_sail = (m_sail * not_hull).sum()
    v_fin = (m_fin * not_hull * (1.0 - m_sail)).sum()
    v_solid = v_bare + v_sail + v_fin

    def _projected(m: torch.Tensor) -> torch.Tensor:
        return (1.0 - torch.prod(1.0 - m, dim=2)).sum()

    aproj_bare = _projected(m_hull)
    cover = 1.0 - (1.0 - m_hull) * (1.0 - m_sail) * (1.0 - m_fin)
    aproj = _projected(cover)

    channels = {
        "log_aproj_ratio": torch.log10(aproj / aproj_bare),
        "sail_frac": v_sail / v_bare,
        "fin_frac": v_fin / v_bare,
        "solid_frac": v_solid / v_bare,
    }
    channel_vector = torch.stack(
        (
            channels["log_aproj_ratio"],
            channels["sail_frac"],
            channels["fin_frac"],
            channels["solid_frac"],
        )
    )
    return DiffMaskChannels(
        ste={"hull": m_hull, "sail": m_sail, "fin": m_fin},
        cover=cover,
        counts={
            "v_bare": v_bare,
            "v_sail": v_sail,
            "v_fin": v_fin,
            "v_solid": v_solid,
            "aproj": aproj,
            "aproj_bare": aproj_bare,
        },
        channels=channels,
        channel_vector=channel_vector,
        midplane=cover[g.nz // 2],
    )


def condition_vector_diff(
    params: DiffParams,
    re: float,
    u_in: float,
    channel_vector: torch.Tensor,
) -> torch.Tensor:
    """Differentiable ``condition_v3`` row (8,) -- logs use param tensors."""
    dev = channel_vector.device
    logs = torch.stack(
        (
            torch.log10(torch.as_tensor(re, dtype=channel_vector.dtype, device=dev)),
            torch.log10(torch.as_tensor(u_in, dtype=channel_vector.dtype, device=dev)),
            torch.log10(params.sail_scale),
            torch.log10(params.fin_scale),
        )
    )
    return torch.cat((logs, channel_vector))


# --------------------------------------------------------------------------- #
# Ensemble wrapper with grad-capable forward
# --------------------------------------------------------------------------- #
@dataclass
class DiffDragEnsemble:
    """Loaded checkpoints + fit normalisations, gradient-capable forward.

    Same arithmetic as ``ModelEnsembleBackend.predict`` (per-member channel
    and condition z-scoring, de-z-scored log10 C_D) but without
    ``torch.no_grad`` so shape parameters can be back-propagated.
    """

    models: list[CondFNODrag]
    norms: list[dict[str, np.ndarray]]
    member_labels: list[str]
    device: torch.device

    @classmethod
    def from_checkpoints(
        cls, paths: Sequence[str | Path], device: str | torch.device = "cpu"
    ) -> DiffDragEnsemble:
        if not paths:
            raise ValueError("ensemble needs at least one checkpoint path")
        ckpts = [load_checkpoint(pth) for pth in paths]
        for c in ckpts:
            missing = [
                key
                for key in ("ch_mean", "ch_std", "p_mean", "p_std", "y_mean", "y_std")
                if key not in c.norm
            ]
            if missing:
                raise ValueError(f"checkpoint norm missing keys: {missing}")
        dev = torch.device(device)
        return cls(
            models=[c.to_model(dev) for c in ckpts],
            norms=[c.norm for c in ckpts],
            member_labels=[str(c.meta.get("member", f"m{i}")) for i, c in enumerate(ckpts)],
            device=dev,
        )

    def member_log10_cd(self, field: torch.Tensor, cond: torch.Tensor) -> list[torch.Tensor]:
        """Per-member log10 C_D of one design (field (5, ny, nx), cond (8,)).

        ``field``/``cond`` may carry autograd graphs (typically only the
        solid-mask channel and the condition row do).
        """
        if field.ndim != 3 or field.shape[0] != 5:
            raise ValueError(f"field must be (5, ny, nx), got {tuple(field.shape)}")
        if cond.ndim != 1 or cond.shape[0] != 8:
            raise ValueError(f"cond must be (8,), got {tuple(cond.shape)}")
        x = field.unsqueeze(0).float().to(self.device)
        p = cond.unsqueeze(0).float().to(self.device)
        outs: list[torch.Tensor] = []
        for model, norm in zip(self.models, self.norms):
            ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=self.device)
            ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=self.device)
            p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=self.device)
            p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=self.device)
            y_s = float(norm["y_std"])
            y_m = float(norm["y_mean"])
            x_norm = (x - ch_m.view(1, -1, 1, 1)) / ch_s.view(1, -1, 1, 1)
            p_norm = (p - p_m) / p_s
            z = model(x_norm, p_norm)
            outs.append(z.double().squeeze(0) * y_s + y_m)
        return outs


def _field_with_mask(
    field_row: np.ndarray,
    midplane: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Reference field with channel 4 replaced by the STE mask slice."""
    arr = np.asarray(field_row, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 5:
        raise ValueError(f"field_row must be (5, ny, nx), got {arr.shape}")
    if tuple(arr.shape[1:]) != tuple(midplane.shape):
        raise ValueError(
            f"field plane {arr.shape[1:]} does not match mask plane {tuple(midplane.shape)}"
        )
    frozen = torch.from_numpy(arr).to(device=device, dtype=midplane.dtype)
    out = frozen.clone()
    out[4] = midplane
    return out


def _ensemble_log10_cd(member_log10: Sequence[torch.Tensor]) -> torch.Tensor:
    """log10 of the linear-space ensemble mean C_D (service convention)."""
    member_cd = torch.stack(tuple(10.0**m for m in member_log10))
    return torch.log10(member_cd.mean(dim=0))


# --------------------------------------------------------------------------- #
# End-to-end differentiable forward + gradients
# --------------------------------------------------------------------------- #
def drag_forward(
    design: Mapping[str, float] | None = None,
    ensemble: DiffDragEnsemble | None = None,
    field_row: np.ndarray | None = None,
    *,
    re: float = 200.0,
    u_in: float = 0.1,
    grid: SuboffGrid | None = None,
    hull_type: str = "full",
    smooth_k: float = DEFAULT_SMOOTH_K_FT,
    tau: float = DEFAULT_TAU_FT,
    requires_grad: bool = True,
    ste: bool = True,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """One differentiable ensemble evaluation of a design point.

    Returns a dict with tensors ``log10_cd`` (ensemble),
    ``member_log10_cd`` (list), ``params`` (:class:`DiffParams`),
    ``channels`` (the :class:`DiffMaskChannels` result) and plain-float
    copies under ``values``.  Use under ``torch.no_grad()`` for the
    hard-forward finite-difference oracle (STE masks are hard in the
    forward pass); ``ste=False`` gives the smooth soft-occupancy surrogate
    whose finite differences must reproduce autograd exactly.
    """
    if ensemble is None:
        raise ValueError("ensemble is required (DiffDragEnsemble.from_checkpoints)")
    if field_row is None:
        raise ValueError("field_row is required (corpus reference mid-plane field)")
    params = DiffParams.from_values(design, requires_grad=requires_grad, device=device, dtype=dtype)
    ch = mask_channels(
        grid,
        params,
        hull_type=hull_type,
        smooth_k=smooth_k,
        tau=tau,
        device=device,
        dtype=dtype,
        ste=ste,
    )
    cond = condition_vector_diff(params, re, u_in, ch.channel_vector)
    field = _field_with_mask(field_row, ch.midplane, device)
    member = ensemble.member_log10_cd(field, cond)
    log10_cd = _ensemble_log10_cd(member)
    return {
        "log10_cd": log10_cd,
        "member_log10_cd": member,
        "params": params,
        "channels": ch,
        "values": {
            "log10_cd": float(log10_cd.detach()),
            "member_log10_cd": [float(m.detach()) for m in member],
            "channels": {k: float(v.detach()) for k, v in ch.channels.items()},
            "counts": {k: int(round(float(v.detach()))) for k, v in ch.counts.items()},
        },
    }


def drag_gradients(
    design: Mapping[str, float] | None,
    ckpt_paths: Sequence[str | Path],
    field_row: np.ndarray,
    *,
    re: float = 200.0,
    u_in: float = 0.1,
    grid: SuboffGrid | None = None,
    hull_type: str = "full",
    smooth_k: float = DEFAULT_SMOOTH_K_FT,
    tau: float = DEFAULT_TAU_FT,
    ste: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """End-to-end ``d(log10 C_D)/d(theta)`` for every differentiable axis.

    Builds leaf tensors for all seven :data:`DIFF_PARAM_NAMES` axes (gradients
    are reported for the keys of *design*), runs the ensemble forward and
    back-propagates the log10 of the linear-space ensemble mean C_D.

    Returns a dict with ``grads`` (ensemble), ``member_grads`` (per
    checkpoint member), forward diagnostics and the channel values.
    ``ste=False`` differentiates the smooth soft-occupancy surrogate (use
    for self-consistency checks against soft finite differences).
    """
    ensemble = DiffDragEnsemble.from_checkpoints(ckpt_paths, device=device)
    out = drag_forward(
        design,
        ensemble,
        field_row,
        re=re,
        u_in=u_in,
        grid=grid,
        hull_type=hull_type,
        smooth_k=smooth_k,
        tau=tau,
        requires_grad=True,
        ste=ste,
        device=device,
    )
    axes = sorted(design.keys()) if design is not None else list(DIFF_PARAM_NAMES)
    leaves = [getattr(out["params"], name) for name in axes]
    out["log10_cd"].backward(retain_graph=True)

    def _grad_of(tensor: torch.Tensor) -> float:
        return float(tensor.grad.detach()) if tensor.grad is not None else 0.0

    grads = {name: _grad_of(leaf) for name, leaf in zip(axes, leaves)}
    member_grads: dict[str, list[float]] = {name: [] for name in axes}
    for m_log in out["member_log10_cd"]:
        gm = torch.autograd.grad(m_log, leaves, retain_graph=True, allow_unused=True)
        for name, g in zip(axes, gm):
            member_grads[name].append(float(g) if g is not None else 0.0)
    return {
        "grads": grads,
        "member_grads": member_grads,
        "member_labels": list(ensemble.member_labels),
        **out["values"],
        "design": {name: float(value) for name, value in (design or {}).items()},
        "re": float(re),
        "u_in": float(u_in),
        "hull_type": hull_type,
        "smooth_k": float(smooth_k),
        "tau": float(tau),
    }


def drag_finite_difference(
    design: Mapping[str, float],
    ckpt_paths: Sequence[str | Path],
    field_row: np.ndarray,
    h: Mapping[str, float] | float = 1e-3,
    *,
    re: float = 200.0,
    u_in: float = 0.1,
    grid: SuboffGrid | None = None,
    hull_type: str = "full",
    smooth_k: float = DEFAULT_SMOOTH_K_FT,
    tau: float = DEFAULT_TAU_FT,
    ste: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Central finite differences of the forward (hard by default).

    The default oracle is :func:`drag_forward` under ``torch.no_grad()``
    whose STE masks are the hard CAD masks, so the FD reference and the
    autograd path share one forward definition (this isolates the STE and
    smooth-union approximations from any CAD parity error).  With
    ``ste=False`` the oracle is the smooth soft-occupancy surrogate whose
    FD must reproduce the soft autograd to FD truncation error.  Steps are
    *relative* to each parameter value (all axes are multiplicative
    scales).
    """
    ensemble = DiffDragEnsemble.from_checkpoints(ckpt_paths, device=device)
    axes = sorted(design.keys())

    def _f(values: Mapping[str, float]) -> float:
        with torch.no_grad():
            out = drag_forward(
                values,
                ensemble,
                field_row,
                re=re,
                u_in=u_in,
                grid=grid,
                hull_type=hull_type,
                smooth_k=smooth_k,
                tau=tau,
                requires_grad=False,
                ste=ste,
                device=device,
            )
        return float(out["log10_cd"])

    center = _f(design)
    grads: dict[str, float] = {}
    evals: dict[str, tuple[float, float]] = {}
    for name in axes:
        step = float(h[name] if isinstance(h, Mapping) else h)
        if step <= 0.0:
            raise ValueError(f"FD step for {name} must be positive, got {step}")
        plus = dict(design)
        plus[name] = float(design[name]) * (1.0 + step)
        minus = dict(design)
        minus[name] = float(design[name]) * (1.0 - step)
        f_plus = _f(plus)
        f_minus = _f(minus)
        grads[name] = (f_plus - f_minus) / (2.0 * float(design[name]) * step)
        evals[name] = (f_plus, f_minus)
    return {
        "grads": grads,
        "log10_cd_center": center,
        "evals": evals,
        "h": dict(h) if isinstance(h, Mapping) else float(h),
    }


# --------------------------------------------------------------------------- #
# Reference (numpy training-pipeline) forward for parity checks
# --------------------------------------------------------------------------- #
def reference_drag_forward(
    design: Mapping[str, float],
    ensemble: DiffDragEnsemble,
    field_row: np.ndarray,
    *,
    re: float = 200.0,
    u_in: float = 0.1,
    grid: SuboffGrid | None = None,
    hull_type: str = "full",
) -> dict[str, Any]:
    """Gold-standard forward with the *training-pipeline* semantics.

    Geometry channels come from
    :func:`tensorlbm.ai.drag_cond.suboff_geometry_features` (numpy CAD
    predicates, ``sail_scale``/``fin_scale`` axes only) and the field mask
    channel from :func:`tensorlbm.suboff_cad.build_suboff_mask`; the
    condition row is assembled by :func:`condition_v3`.  Hull-form
    multipliers only enter through the mask slice -- the reference channel
    recipe has no hull-form axis by construction.
    """
    g = PRODUCTION_GRID if grid is None else grid
    cfg_kwargs = {k: float(v) for k, v in design.items() if k in DIFF_PARAM_NAMES}
    cfg = SuboffConfig(**cfg_kwargs)
    sail = float(design.get("sail_scale", 1.0))
    fin = float(design.get("fin_scale", 1.0))
    features = suboff_geometry_features(hull_type, sail, fin, grid=g)
    cond_np = condition_v3(
        np.array([float(re)]),
        np.array([u_in]),
        np.array([sail]),
        np.array([fin]),
        np.stack(
            [
                np.asarray(
                    [
                        features.log_aproj_ratio,
                        features.sail_frac,
                        features.fin_frac,
                        features.solid_frac,
                    ]
                )
            ]
        ),
    )
    mask, _ = build_suboff_mask(
        hull_type=hull_type,
        nx=g.nx,
        ny=g.ny,
        nz=g.nz,
        cx=g.cx,
        cy=g.cy,
        cz=g.cz,
        length=g.length,
        config=cfg,
    )
    arr = np.asarray(field_row, dtype=np.float32).copy()
    arr[4] = np.asarray(mask[g.nz // 2], dtype=np.float32)
    cond_t = torch.from_numpy(cond_np[0].astype(np.float64))
    field_t = torch.from_numpy(arr).to(dtype=torch.float64)
    member = ensemble.member_log10_cd(field_t, cond_t)
    log10_cd = _ensemble_log10_cd(member)
    return {
        "log10_cd": float(log10_cd.detach()),
        "member_log10_cd": [float(m.detach()) for m in member],
        "channels": {
            "log_aproj_ratio": float(features.log_aproj_ratio),
            "sail_frac": float(features.sail_frac),
            "fin_frac": float(features.fin_frac),
            "solid_frac": float(features.solid_frac),
        },
        "counts": {
            "v_bare": int(features.v_bare),
            "v_sail": int(features.v_sail),
            "v_fin": int(features.v_fin),
            "v_solid": int(features.v_solid),
            "aproj": int(features.aproj),
            "aproj_bare": int(features.aproj_bare),
        },
    }
