"""Inamuro wet-node / anti-bounce-back boundary closures for D2Q9 (TensorLBM).

Why this module exists
----------------------
TensorLBM had two families of wall treatment: *half-way bounce-back* on a
staircase solid set (:func:`tensorlbm.boundaries.bounce_back_cells`) and a
sub-grid *interpolated* bounce-back (:mod:`tensorlbm.boundary.interpolated_bounce_back`).
Both are **dry-node** schemes: the boundary node is solid, and the link that
crosses the wall is killed/reflected.  The third textbook family — the
**wet-node** scheme, where the boundary node stays an *active fluid node* and
the populations arriving from outside the domain are reconstructed — was
missing entirely, so there was no way to impose a Dirichlet velocity (or a
Dirichlet scalar) *on* a lattice plane without a half-cell offset.

This module adds it, in the formulation used by the open-source reference
implementation ``lbm-principles-practice/code`` (the code accompanying
Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva & Viggen, *The Lattice Boltzmann
Method: Principles and Practice*, Springer 2017; MIT licensed, repo pinned at
commit ``6e2c592fdc3592c14dfd52f860fc1ceea930bcb0``):

* ``chapter5/couette_wetnode.m`` / ``chapter5/poiseuille_wetnode.m`` —
  velocity Dirichlet by the **non-equilibrium bounce-back** rule (the
  ``wetnode=3`` branch; Inamuro, Yoshino & Ogino, *Phys. Fluids* 11 (1999)
  2924): the non-equilibrium part of the outgoing population is reflected
  with a sign flip while the equilibrium part is evaluated at the *wall*
  state,
  ``f_ī = f_ī^eq(ρ_w, u_w) + f_i^eq(ρ_w, u_w) − f_i``, which is the
  second-order-accurate ("anti-bounce-back") form of the wall closure.
* ``chapter8/film_inamuro.cpp`` — scalar (concentration) Dirichlet by the
  **residual-weight closure**: the unknown incoming populations are replaced
  by weight-scaled equilibrium at the target value, scaled so that the sum of
  *all* populations at the wet node equals the target exactly (the upstream
  *executed* block ``update_bounce_back_after_stream()`` subtracts the rest
  population ``f_0`` as part of the known set — see ``known="all_known"``;
  the older, never-called ``update_bounce_back()`` omits ``f_0``, which is
  exposed here as ``known="all_but_rest"`` and is measurably wrong: it leaves
  the node value at ``c_target + f_0``).  The zero-normal-gradient (outflow /
  free-surface) closure is the same scaling without the target, i.e.
  ``f_i = w_i · (Σ_mirror f) / (Σ_unknown w)``, where ``Σ_mirror`` runs over
  the opposite trio ``opp(unknown)``.

Both closures are re-implemented here as face-based, lattice-agnostic torch
operators (``(9, ny, nx)`` population layout, matching
:mod:`tensorlbm.boundary.solver2d`).  No geometry-specific code lives here:
the caller supplies the face and the wall velocity/target field, so the same
operator serves a channel, a lid-driven cavity, a falling film or an
arbitrary masked boundary.

Formulae (D2Q9, ``c_s² = 1/3``, ``U`` = directions arriving from outside)

Velocity Dirichlet (anti-bounce-back, quadratic equilibrium)::

    f_ī(x_b) = f_ī^eq(ρ_w, u_w) + f_i^eq(ρ_w, u_w) − f_i(x_b),   ī ∈ U, i = opp(ī)

with ``f^eq = w_i ρ (1 + 3 c_i·u + 4.5 (c_i·u)² − 1.5 u²)``.  ``ρ_w`` is
either fixed (``rho_w=1.0``, as in the OSS MATLAB scripts) or extrapolated
from the first fluid node inward (``rho_w=None``, Inamuro's original advice).
``equilibrium="linear"`` reproduces the OSS scripts bit-for-bit in the sense
that it uses their linear incompressible equilibrium
``w_i ρ (1 + 3 c_i·u)`` instead of the quadratic one.

Scalar Dirichlet (residual-weight, OSS ``film_inamuro.cpp``)::

    f_i(x_b) = w_i · (c_target − Σ_{j∉U} f_j) / (Σ_{j∈U} w_j)      ∀ i ∈ U

Zero-normal-gradient (same file, outlet / free surface)::

    f_i(x_b) = w_i · (Σ_{j∉U} f_j) / (Σ_{j∈U} w_j)                ∀ i ∈ U

Corner average (same file, the four domain corners)::

    f(x_corner) = ½ [ f(x_corner + e_a) + f(x_corner + e_b) ]

with ``e_a, e_b`` the two inward face normals.

Deviations from the OSS reference (see ``docs/XVAL_INAMURO_WETNODE_M17603.md``
for the full table and the measured effect of each)
------------------------------------------------------
1. The OSS C++ film code hard-codes a 40 × 400 grid, ``omega``, the parabolic
   advection velocity, the closure order and the output cadence; here the
   closures are pure functions of the population array (grid/parameter
   agnostic) and the *problem* lives in the benchmark driver.
2. The OSS velocity closures exist only for the two channel walls in a MATLAB
   script with a fixed direction table.  This module derives the unknown
   direction set from the face normal through the lattice constants, so all
   four faces of a 2D box (and moving walls with arbitrary ``u_w``) are
   supported by one code path.
3. ``ρ_w`` is an explicit argument (fixed or extrapolated); the OSS MATLAB
   velocity closures hard-wire ``ρ_w = 1``.
4. The OSS film code applies the closures *after* streaming in-place on the
   streamed array and averages corners afterwards.  The operators here are
   side-effect free (return a new tensor unless ``inplace=True``) and expose
   the same ordering via :func:`inamuro_corner_average`.
5. No curved/sub-grid wall support: this is the lattice-aligned wet-node
   formulation.  For sub-grid walls use
   :func:`tensorlbm.boundary.interpolated_bounce_back`.

Licence/attribution: the closures follow the MIT-licensed reference above;
the implementations here are independent PyTorch code written against the
published formulae and cross-validated numerically against the reference
binary (see the XVAL report).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .d2q9 import C2Q9, OPPOSITE_2Q9, W2Q9

__all__ = [
    "InamuroWetNodeReport",
    "FACE_NORMALS",
    "inamuro_corner_average",
    "inamuro_wetnode_scalar",
    "inamuro_wetnode_velocity",
    "inamuro_zero_gradient",
    "unknown_directions",
]

Face = Literal["left", "right", "bottom", "top"]
Corner = Literal["bl", "br", "tl", "tr"]
Equilibrium = Literal["quadratic", "linear"]

#: Outward normal (as a lattice-coordinate pair, x first) of each supported face.
FACE_NORMALS: dict[str, Sequence[int]] = {
    "left": (-1, 0),
    "right": (1, 0),
    "bottom": (0, -1),
    "top": (0, 1),
}

_W = W2Q9
_C = C2Q9
_OPP = OPPOSITE_2Q9


@dataclass(frozen=True)
class InamuroWetNodeReport:
    """Bookkeeping for one closure application (used in evidence tables)."""

    face: str
    n_cells: int
    directions: tuple[int, ...]
    value_before: float
    value_after: float
    closure: str

    def as_dict(self) -> dict[str, object]:
        return {
            "face": self.face,
            "n_cells": self.n_cells,
            "directions": list(self.directions),
            "value_before": self.value_before,
            "value_after": self.value_after,
            "closure": self.closure,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _check_field(f: torch.Tensor) -> None:
    if f.ndim != 3 or f.shape[0] != 9:
        raise ValueError(f"populations must have shape (9, ny, nx); got {tuple(f.shape)}")
    if not f.is_floating_point():
        raise TypeError("populations must be a floating point tensor")


def unknown_directions(face: Face) -> torch.Tensor:
    """Indices of the D2Q9 directions arriving at *face* from outside the domain.

    A population is "arriving from outside" when its lattice velocity points
    *into* the domain, i.e. when ``c_i · n_out < 0`` for the outward normal
    ``n_out`` of the face.
    """
    if face not in FACE_NORMALS:
        raise ValueError(f"face must be one of {sorted(FACE_NORMALS)}; got {face!r}")
    n = torch.tensor(FACE_NORMALS[face], dtype=_C.dtype)
    # c·n > 0 means the direction points outward; those directions' *populations
    # at the boundary node* would have come from inside, so they are known.
    # The unknown set is the mirror image: c·n < 0.
    dot = _C[:, 0] * n[0] + _C[:, 1] * n[1]
    return torch.nonzero(dot < 0, as_tuple=False).flatten()


def _face_slice(face: Face, shape: tuple[int, int]) -> tuple[slice, slice]:
    ny, nx = shape
    if face == "left":
        return slice(None), slice(0, 1)
    if face == "right":
        return slice(None), slice(nx - 1, nx)
    if face == "bottom":
        return slice(0, 1), slice(None)
    if face == "top":
        return slice(ny - 1, ny), slice(None)
    raise ValueError(f"unknown face {face!r}")


def _inward_normal(face: Face) -> tuple[int, int]:
    """Inward unit normal of *face* in lattice coordinates ``(dx, dy)``."""
    n = FACE_NORMALS[face]
    return (-int(n[0]), -int(n[1]))


def _shifted(a: torch.Tensor, shift: tuple[int, int], fill: float = 0.0) -> torch.Tensor:
    """Shift a ``(ny, nx)`` tensor by lattice offsets (non-periodic, zero fill).

    ``out[y, x] = a[y + sy, x + sx]`` — i.e. the value *pulled from* the
    neighbour in direction ``s``.
    """
    sy, sx = shift
    out = torch.full_like(a, fill)
    ny, nx = a.shape
    src_y = slice(max(0, sy), ny + min(0, sy))
    dst_y = slice(max(0, -sy), ny + min(0, -sy))
    src_x = slice(max(0, sx), nx + min(0, sx))
    dst_x = slice(max(0, -sx), nx + min(0, -sx))
    out[dst_y, dst_x] = a[src_y, src_x]
    return out


def _as_face_field(value, face: Face, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar / (ny, nx) tensor to the shape of the *face* slice."""
    ny, nx = like.shape[1], like.shape[2]
    if torch.is_tensor(value):
        v = value.to(dtype=like.dtype, device=like.device)
        if v.ndim == 1:
            v = v.view(-1, 1)
        if v.shape != (ny, nx):
            raise ValueError(f"face field must have shape ({ny}, {nx}); got {tuple(v.shape)}")
        return v[_face_slice(face, (ny, nx))]
    return torch.full((1, like.shape[2] if face in ("bottom", "top") else 1),
                      float(value), dtype=like.dtype, device=like.device)


def _wall_velocity_face(
    u_wall, face: Face, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ux, uy)`` on the face slice, broadcast over the face's extent."""
    ny, nx = like.shape[1], like.shape[2]
    if isinstance(u_wall, (int, float)):
        ux = _as_face_field(float(u_wall), face, like)
        uy = _as_face_field(0.0, face, like)
        return ux, uy
    if len(u_wall) != 2:
        raise ValueError("u_wall must be a scalar or a pair (ux, uy)")
    ux = _as_face_field(u_wall[0], face, like)
    uy = _as_face_field(u_wall[1], face, like)
    if ux.shape != uy.shape:  # broadcast the tangential extent
        ux, uy = torch.broadcast_tensors(ux, uy)
    return ux, uy


def _equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    equilibrium: Equilibrium,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """D2Q9 equilibrium on the face slice, shape ``(9, *face)``."""
    c = _C.to(dtype=dtype, device=device)
    w = _W.to(dtype=dtype, device=device)
    cu = c[:, 0].view(-1, *([1] * ux.ndim)) * ux.unsqueeze(0) + c[
        :, 1
    ].view(-1, *([1] * ux.ndim)) * uy.unsqueeze(0)
    if equilibrium == "linear":
        return w.view(-1, *([1] * ux.ndim)) * rho.unsqueeze(0) * (1.0 + 3.0 * cu)
    if equilibrium != "quadratic":
        raise ValueError(f"equilibrium must be 'quadratic' or 'linear'; got {equilibrium!r}")
    u2 = ux**2 + uy**2
    return w.view(-1, *([1] * ux.ndim)) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0)
    )


def _known_sum(
    f_face: torch.Tensor, dirs: torch.Tensor, *, exclude_rest: bool = False
) -> torch.Tensor:
    """Sum of the populations that are *not* reconstructed by the closure.

    ``exclude_rest`` drops the rest population ``f_0`` from the sum without
    touching it — the convention used by the OSS ``film_inamuro.cpp`` Dirichlet
    block (whose residual is built from the five non-rest known directions
    only).  With ``exclude_rest=False`` the closure is *mass consistent*: the
    node value ``Σ_i f_i`` becomes exactly the target.
    """
    mask = torch.ones(f_face.shape[0], dtype=torch.bool, device=f_face.device)
    mask[dirs] = False
    if exclude_rest:
        mask[0] = False
    return f_face[mask].sum(dim=0)


# ---------------------------------------------------------------------------
# scalar (concentration / temperature) closures — OSS chapter8/film_inamuro.cpp
# ---------------------------------------------------------------------------


def inamuro_wetnode_scalar(
    f: torch.Tensor,
    face: Face,
    value: float,
    *,
    known: Literal["all_known", "all_but_rest"] = "all_known",
    inplace: bool = False,
    return_report: bool = False,
):
    """Impose a Dirichlet scalar value on a face with the residual-weight closure.

    ``f_i = w_i (c_target − Σ_known f) / Σ_unknown w`` for every direction
    ``i`` arriving from outside the face.

    ``known="all_known"`` (default) sums every population that is not being
    reconstructed, so the node value ``Σ_i f_i`` becomes exactly ``value`` —
    the closure is *mass consistent*.  This is what the executed upstream block
    ``update_bounce_back_after_stream()`` does and it reproduces the upstream
    binary to ~1e-11 relative L2 (see the XVAL report).  ``known="all_but_rest"``
    leaves the rest population ``f_0`` out of the residual (and untouched) —
    the upstream *unused* ``update_bounce_back()`` variant, which leaves the
    node value at ``value + f_0`` (79 % relative deviation on the film problem).
    """
    _check_field(f)
    out = f if inplace else f.clone()
    ny, nx = f.shape[1], f.shape[2]
    sl = _face_slice(face, (ny, nx))
    dirs = unknown_directions(face).to(f.device)
    w = _W.to(dtype=f.dtype, device=f.device)

    f_face = out[:, sl[0], sl[1]]
    before = f_face.sum(dim=0)
    if known not in ("all_known", "all_but_rest"):
        raise ValueError(f"known must be 'all_known' or 'all_but_rest'; got {known!r}")
    residual = float(value) - _known_sum(f_face, dirs, exclude_rest=known == "all_but_rest")
    denom = w[dirs].sum()
    new = w[dirs].view(-1, 1, 1) * (residual / denom).unsqueeze(0)
    out[:, sl[0], sl[1]] = _scatter_dirs(f_face, dirs, new)

    if not return_report:
        return out
    after = out[:, sl[0], sl[1]].sum(dim=0)
    report = InamuroWetNodeReport(
        face=face,
        n_cells=int(before.numel()),
        directions=tuple(int(d) for d in dirs.tolist()),
        value_before=float(before.mean()),
        value_after=float(after.mean()),
        closure=f"scalar-dirichlet-residual-weight[{known}]",
    )
    return out, report


def inamuro_zero_gradient(
    f: torch.Tensor,
    face: Face,
    *,
    known: Literal["mirror_only", "all_known"] = "mirror_only",
    inplace: bool = False,
    return_report: bool = False,
):
    """Zero-normal-gradient scalar closure (outflow / free surface).

    ``f_i = w_i (Σ_known f) / Σ_unknown w`` — the incoming populations are
    scaled to the mirror image of the outgoing ones.

    ``known="mirror_only"`` (default) sums only the *opposite* trio
    ``{opp(i) : i ∈ unknown}``, which is what the OSS ``film_inamuro.cpp``
    outlet / free-surface blocks do and the usual outflow rule;
    ``known="all_known"`` sums every non-reconstructed population instead.
    """
    _check_field(f)
    out = f if inplace else f.clone()
    ny, nx = f.shape[1], f.shape[2]
    sl = _face_slice(face, (ny, nx))
    dirs = unknown_directions(face).to(f.device)
    w = _W.to(dtype=f.dtype, device=f.device)

    f_face = out[:, sl[0], sl[1]]
    before = f_face.sum(dim=0)
    if known == "mirror_only":
        mirror = f_face[_OPP.to(f.device)[dirs]].sum(dim=0)
    elif known == "all_known":
        mirror = _known_sum(f_face, dirs)
    else:
        raise ValueError(f"known must be 'mirror_only' or 'all_known'; got {known!r}")
    denom = w[dirs].sum()
    new = w[dirs].view(-1, 1, 1) * (mirror / denom).unsqueeze(0)
    out[:, sl[0], sl[1]] = _scatter_dirs(f_face, dirs, new)

    if not return_report:
        return out
    after = out[:, sl[0], sl[1]].sum(dim=0)
    report = InamuroWetNodeReport(
        face=face,
        n_cells=int(before.numel()),
        directions=tuple(int(d) for d in dirs.tolist()),
        value_before=float(before.mean()),
        value_after=float(after.mean()),
        closure=f"scalar-zero-gradient[{known}]",
    )
    return out, report


def _scatter_dirs(f_face: torch.Tensor, dirs: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """Return ``f_face`` with rows ``dirs`` replaced by ``new`` (no aliasing)."""
    out = f_face.clone()
    out[dirs] = new
    return out


# ---------------------------------------------------------------------------
# velocity closures — anti-bounce-back / non-equilibrium bounce-back (Inamuro)
# ---------------------------------------------------------------------------


def inamuro_wetnode_velocity(
    f: torch.Tensor,
    face: Face,
    u_wall=0.0,
    *,
    rho_w: float | torch.Tensor | None = None,
    equilibrium: Equilibrium = "quadratic",
    inplace: bool = False,
    return_report: bool = False,
):
    """Impose a Dirichlet velocity on a face (wet node, anti-bounce-back).

    ``f_ī = f_ī^eq(ρ_w, u_w) + f_i^eq(ρ_w, u_w) − f_i`` for every direction
    ``ī`` arriving from outside the face, with ``i = opp(ī)``.

    Parameters
    ----------
    f : ``(9, ny, nx)`` populations at the wet (boundary) node.
    face : which face of the box the boundary node sits on.
    u_wall : scalar (x-component) or ``(ux, uy)``; each entry may be a scalar
        or a full ``(ny, nx)`` field (moving / rotating walls).
    rho_w : wall density.  ``None`` (default) extrapolates it from the first
        fluid node inward (Inamuro's original advice); a float fixes it
        (``rho_w=1.0`` reproduces the OSS MATLAB scripts).
    equilibrium : ``"quadratic"`` (default, TensorLBM convention) or
        ``"linear"`` (the linear incompressible equilibrium used by the OSS
        MATLAB scripts; the difference is O(u³)).
    """
    _check_field(f)
    out = f if inplace else f.clone()
    ny, nx = f.shape[1], f.shape[2]
    sl = _face_slice(face, (ny, nx))
    dirs = unknown_directions(face).to(f.device)
    opp = _OPP.to(f.device)[dirs]

    f_face = out[:, sl[0], sl[1]]

    if rho_w is None:
        # Inamuro's extrapolation: density of the first fluid node inward.
        d_dx, d_dy = _inward_normal(face)
        rho = _shifted(out.sum(dim=0), (d_dy, d_dx))[sl[0], sl[1]]
    else:
        rho = _as_face_field(rho_w, face, out)

    ux, uy = _wall_velocity_face(u_wall, face, out)
    feq = _equilibrium(rho, ux, uy, equilibrium, out.dtype, out.device)

    new = feq[dirs] + feq[opp] - f_face[opp]
    out[:, sl[0], sl[1]] = _scatter_dirs(f_face, dirs, new)

    if not return_report:
        return out
    momentum_before = (f_face * _C.to(dtype=f.dtype, device=f.device)[:, 0].view(-1, 1, 1)).sum(dim=0)
    f_new = out[:, sl[0], sl[1]]
    momentum_after = (f_new * _C.to(dtype=f.dtype, device=f.device)[:, 0].view(-1, 1, 1)).sum(dim=0)
    report = InamuroWetNodeReport(
        face=face,
        n_cells=int(f_face.shape[1] * f_face.shape[2]),
        directions=tuple(int(d) for d in dirs.tolist()),
        value_before=float(momentum_before.abs().mean()),
        value_after=float(momentum_after.abs().mean()),
        closure=f"velocity-abb-{equilibrium}",
    )
    return out, report


def inamuro_corner_average(
    f: torch.Tensor,
    corner: Corner,
    *,
    inplace: bool = False,
    return_report: bool = False,
):
    """Average the four domain corners from their two adjacent face nodes.

    Reproduces the OSS film code's ``//Corners (averaging procedure)`` block:
    ``f(x_c) = ½ [f(x_c + e_a) + f(x_c + e_b)]`` with ``e_a, e_b`` the two
    inward face normals of the corner.
    """
    _check_field(f)
    out = f if inplace else f.clone()
    ny, nx = f.shape[1], f.shape[2]
    corners: dict[str, tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]] = {
        "bl": ((0, 0), ((0, 1), (1, 0))),
        "br": ((0, nx - 1), ((0, -1), (1, 0))),
        "tl": ((ny - 1, 0), ((0, 1), (-1, 0))),
        "tr": ((ny - 1, nx - 1), ((0, -1), (-1, 0))),
    }
    if corner not in corners:
        raise ValueError(f"corner must be one of {sorted(corners)}; got {corner!r}")
    (cy, cx), (ea, eb) = corners[corner]
    a = out[:, cy + ea[0], cx + ea[1]]
    b = out[:, cy + eb[0], cx + eb[1]]
    avg = 0.5 * (a + b)
    out[:, cy, cx] = avg
    if not return_report:
        return out
    return out, InamuroWetNodeReport(
        face=f"corner-{corner}",
        n_cells=1,
        directions=(0,),
        value_before=float(out[:, cy, cx].sum()),
        value_after=float(avg.sum()),
        closure="corner-average",
    )