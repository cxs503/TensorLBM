"""Vectorised immersed-boundary (IBM) force-transfer operators.

This module is the *numerical kernel* used by the ice–propeller IBM validation
campaign (node ``IN54W1-N02``).  It does **not** replace
:mod:`tensorlbm.ibm` (which keeps the per-marker reference implementation and
the Guo body-force application); it provides the vectorised, moving/rotating
rigid-body operators that are required for quantitative force-transfer
validation:

* :func:`build_stencil` – sparse (marker × node) interpolation/spreading
  stencil for the Peskin 2-point (``hat``) and 4-point (``4pt``) delta
  kernels.  The stencil is built once per marker set; every subsequent
  interpolation/spreading is a pair of gather / scatter-add operations.
* :class:`IBMStencil` – ``interpolate`` (Eulerian → Lagrangian) and
  ``spread`` (Lagrangian → Eulerian).  The two operators are discrete
  adjoints of each other, which is what makes the reaction force
  ``F_body = −Σ_k F_k`` an exact statement of momentum conservation rather
  than an approximation (see :func:`momentum_audit`).
* :class:`RigidBody` – minimal 6-DOF rigid body (position, velocity, angular
  velocity) used to prescribe the target velocity ``U + ω × (x − x_c)`` of a
  moving or rotating immersed boundary.
* :func:`direct_forcing` – two-pass direct-forcing (Uhlmann 2005) with the
  Guo half-force velocity correction; returns the Lagrangian marker forces,
  the Eulerian body-force field and the body force/torque.
* :func:`interior_mask_force` – Eulerian solid-mask forcing (used only to
  fill the interior of a solid body when it is not represented by markers).

Everything is expressed in *lattice units* (``dt = 1``), so a Lagrangian
marker force ``F_k`` is a momentum change per lattice node per time step.

References
----------
Peskin, C. S. (1972) J. Comput. Phys. 10, 252–271.
Uhlmann, M. (2005)   J. Comput. Phys. 209, 448–476.
Guo, Z. et al. (2002) Phys. Rev. E 65, 046308.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .ibm import ibm_delta_4pt, ibm_delta_hat

__all__ = [
    "IBMStencil",
    "body_source_2d",
    "body_source_3d",
    "RigidBody",
    "build_stencil",
    "build_cylinder_markers_2d",
    "build_cylinder_markers_3d",
    "sphere_markers",
    "direct_forcing",
    "direct_forcing_3d",
    "guo_force_2d",
    "guo_force_3d",
    "interior_mask_force",
    "momentum_audit",
    "kernel_profile",
]

_DELTA = {"hat": (ibm_delta_hat, 2), "4pt": (ibm_delta_4pt, 4)}


def kernel_profile(name: str):
    """Return ``(delta_fn, support)`` for a supported kernel name."""
    try:
        return _DELTA[name]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown IBM kernel {name!r}; use 'hat' or '4pt'") from None


# --------------------------------------------------------------------------- #
# Stencil construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IBMStencil:
    """Sparse marker↔node interpolation / spreading stencil (periodic grid).

    Attributes:
        idx: ``(M*S,)`` int64 linear indices into the flattened grid.
        w: ``(M*S,)`` float64 delta-kernel weights.
        shape: grid shape, ``(ny, nx)`` or ``(nz, ny, nx)``.
        kernel: ``"hat"`` or ``"4pt"``.
        points: ``(M, d)`` marker coordinates (fractional lattice units).
    """

    idx: torch.Tensor
    w: torch.Tensor
    shape: tuple[int, ...]
    kernel: str
    points: torch.Tensor
    _per_marker: int = field(default=0, repr=False)

    @property
    def n_markers(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(math.prod(self.shape))

    def interpolate(self, q: torch.Tensor) -> torch.Tensor:
        """Interpolate a field from the Eulerian grid to the markers.

        Args:
            q: field of shape ``(*shape,)`` or ``(C, *shape)``.

        Returns:
            ``(M,)`` for a scalar field, ``(C, M)`` for a vector field.
        """
        scalar = q.dim() == len(self.shape)
        flat = q.reshape(-1, self.n_nodes) if not scalar else q.reshape(1, self.n_nodes)
        # (C, M*S) -> sum over support -> (C, M)
        vals = flat[:, self.idx] * self.w.unsqueeze(0)
        out = vals.view(vals.shape[0], -1, self._per_marker).sum(dim=-1)
        return out[0] if scalar else out

    def spread(self, f_marker: torch.Tensor) -> torch.Tensor:
        """Spread marker forces onto the Eulerian grid.

        Args:
            f_marker: ``(M,)`` or ``(C, M)`` marker forces.

        Returns:
            Grid force field ``(*shape,)`` or ``(C, *shape)``.
        """
        scalar = f_marker.dim() == 1
        fm = f_marker.reshape(1, -1) if scalar else f_marker
        w2 = self.w.view(self.n_markers, self._per_marker)
        n_comp = fm.shape[1]
        contrib = (fm.t().unsqueeze(-1) * w2.unsqueeze(0)).reshape(n_comp, -1)
        flat = torch.zeros((n_comp, self.n_nodes), dtype=self.w.dtype, device=self.w.device)
        flat.index_add_(1, self.idx, contrib)
        out = flat.reshape((n_comp,) + tuple(self.shape))
        return out[0] if scalar else out

    def peak_normalization(self) -> float:
        """Sum of kernel weights per marker (partition of unity check)."""
        return float(
            self.w.view(self.n_markers, self._per_marker).sum(dim=-1).mean().item()
        )


def build_stencil(
    points: torch.Tensor,
    shape: Sequence[int],
    kernel: str = "4pt",
    *,
    axis_order: str = "xyz",
) -> IBMStencil:
    """Build a periodic-grid IBM stencil.

    Args:
        points: ``(M, d)`` marker coordinates in lattice units.
        shape: Eulerian grid shape in ``(ny, nx)`` / ``(nz, ny, nx)`` order to
            match the solver field layout (numpy/torch row-major, last axis
            fastest).
        kernel: ``"hat"`` (2-point, support 2) or ``"4pt"`` (support 4).
        axis_order: ``"xyz"`` (default) means ``points`` are physical
            ``(x, y)`` / ``(x, y, z)`` coordinates and are mapped onto the
            array axes ``(-1, -2, ...)``.  ``"array"`` means the point
            components already follow the array-axis order.

    Notes:
        The field layout used by the solvers is ``(ny, nx)`` / ``(nz, ny, nx)``
        (row-major, last axis fastest), so physical ``x`` is the *last* array
        axis.  Getting this mapping wrong silently swaps the axes and makes
        every IBM force look like a radial source term — the regression test
        ``test_stencil_axis_convention`` pins it down.
    """
    pts = torch.as_tensor(points, dtype=torch.float64)
    if pts.dim() != 2:
        raise ValueError("points must be a (M, d) tensor")
    if axis_order == "xyz":
        pts_arr = pts.flip(dims=(1,))
    elif axis_order == "array":
        pts_arr = pts
    else:
        raise ValueError("axis_order must be 'xyz' or 'array'")
    shape = tuple(int(s) for s in shape)
    if pts_arr.shape[1] != len(shape):
        raise ValueError(
            f"points have {pts_arr.shape[1]} components but shape has {len(shape)} axes"
        )
    delta, support = kernel_profile(kernel)
    half = support // 2
    d = len(shape)

    # Per-dimension node indices and weights: (M, K) each.
    dim_idx: list[torch.Tensor] = []
    dim_w: list[torch.Tensor] = []
    for ax in range(d):
        p = pts_arr[:, ax]
        base = torch.floor(p).to(torch.int64) - half + 1
        offs = torch.arange(support, dtype=torch.int64, device=p.device)
        nodes = base.unsqueeze(1) + offs.unsqueeze(0)  # (M, K)
        r = nodes.to(torch.float64) - p.unsqueeze(1)
        dim_idx.append(nodes % shape[ax])
        dim_w.append(delta(r))

    # Cartesian product across dimensions, keeps (M, S) with S = support**d.
    idx_parts: list[torch.Tensor] = []
    w_parts: list[torch.Tensor] = []
    strides = [1] * d
    for ax in range(d - 2, -1, -1):
        strides[ax] = strides[ax + 1] * shape[ax + 1]

    for combo in itertools.product(range(support), repeat=d):
        lin = torch.zeros(dim_idx[0].shape[0], dtype=torch.int64, device=dim_idx[0].device)
        wt = torch.ones(dim_w[0].shape[0], dtype=torch.float64, device=dim_w[0].device)
        for ax, k in enumerate(combo):
            lin = lin + dim_idx[ax][:, k] * strides[ax]
            wt = wt * dim_w[ax][:, k]
        idx_parts.append(lin)
        w_parts.append(wt)

    idx = torch.stack(idx_parts, dim=1).reshape(-1)
    w = torch.stack(w_parts, dim=1).reshape(-1)
    return IBMStencil(
        idx=idx,
        w=w,
        shape=shape,
        kernel=kernel,
        points=pts,
        _per_marker=support**d,
    )


# --------------------------------------------------------------------------- #
# Marker geometry generators
# --------------------------------------------------------------------------- #


def build_cylinder_markers_2d(
    center: tuple[float, float], radius: float, spacing: float = 0.9
) -> torch.Tensor:
    """Markers on a circle resampled to ~``spacing`` lattice cells."""
    cx, cy = center
    n = max(8, int(round(2.0 * math.pi * radius / spacing)))
    t = torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)
    return torch.stack([cx + radius * torch.cos(t), cy + radius * torch.sin(t)], dim=1)


def build_cylinder_markers_3d(
    center: tuple[float, float, float],
    radius: float,
    height: float,
    spacing: float = 1.0,
) -> torch.Tensor:
    """Markers on the lateral surface of a finite circular cylinder (axis z)."""
    cx, cy, cz = center
    n_theta = max(10, int(round(2.0 * math.pi * radius / spacing)))
    n_z = max(2, int(round(height / spacing)))
    th = torch.arange(n_theta, dtype=torch.float64) * (2.0 * math.pi / n_theta)
    zs = cz - height / 2.0 + torch.arange(n_z, dtype=torch.float64) * (height / (n_z - 1))
    grid_th, grid_z = torch.meshgrid(th, zs, indexing="ij")
    return torch.stack(
        [
            cx + radius * torch.cos(grid_th.reshape(-1)),
            cy + radius * torch.sin(grid_th.reshape(-1)),
            grid_z.reshape(-1),
        ],
        dim=1,
    )


def sphere_markers(
    radius: float, spacing: float = 1.0, center: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> torch.Tensor:
    """Quasi-uniform (Fibonacci) markers on a sphere."""
    n = max(12, int(round(4.0 * math.pi * radius**2 / spacing**2)))
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    pts = torch.stack([r_xy * torch.cos(phi), r_xy * torch.sin(phi), z], dim=1) * radius
    return pts + torch.tensor(center, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# Rigid body motion
# --------------------------------------------------------------------------- #


@dataclass
class RigidBody:
    """Minimal rigid body: translation + rotation about its centre of mass.

    ``omega`` is a scalar in 2-D and a 3-vector in 3-D.
    """

    center: torch.Tensor
    velocity: torch.Tensor
    omega: torch.Tensor | float = 0.0
    mass: float = 1.0
    inertia: float = 1.0
    dim: int = 3

    def surface_velocity(self, points: torch.Tensor) -> torch.Tensor:
        """Rigid-body velocity ``U + ω × (x − x_c)`` at arbitrary points."""
        rel = points - self.center
        if self.dim == 2:
            w = float(self.omega)
            return torch.stack(
                [self.velocity[0] - w * rel[:, 1], self.velocity[1] + w * rel[:, 0]], dim=1
            )
        w = torch.as_tensor(self.omega, dtype=points.dtype)
        return self.velocity.unsqueeze(0) + torch.cross(
            w.expand(rel.shape[0], 3), rel, dim=1
        )


# --------------------------------------------------------------------------- #
# Direct forcing (two-pass, Guo half-force correction)
# --------------------------------------------------------------------------- #


def direct_forcing(
    stencil: IBMStencil,
    u_grid: torch.Tensor,
    u_target: torch.Tensor,
    rho_grid: torch.Tensor | None = None,
    n_correctors: int = 2,
):
    """Two-pass direct forcing for an immersed boundary.

    The physical (Guo-corrected) velocity at a node is
    ``u* = u_moment + F/(2 ρ)``; a single-pass direct forcing therefore
    under-predicts the marker force by ``F/(2ρ)``.  ``n_correctors`` extra
    passes remove that bias, which matters for moving walls where the force
    is not a small correction.

    Args:
        stencil: marker stencil (dims match ``u_grid``).
        u_grid: ``(d, *shape)`` Eulerian velocity.
        u_target: ``(M, d)`` desired marker velocity.
        rho_grid: optional ``(*shape,)`` density (defaults to 1).
        n_correctors: number of additional interpolation passes.

    Returns:
        ``(f_marker, f_grid, body_force, body_torque)`` where ``f_marker`` is
        ``(M, d)``, ``f_grid`` is ``(d, *shape)``, ``body_force`` is ``(d,)``
        (equal to ``−Σ_k f_marker``) and ``body_torque`` is the moment about
        the marker-set centroid.
    """
    if rho_grid is None:
        rho_m = torch.ones(stencil.n_markers, dtype=torch.float64)
    else:
        rho_m = stencil.interpolate(rho_grid)

    u_int = stencil.interpolate(u_grid)  # (d, M)
    f_m = u_target.T - u_int  # (d, M)  dt = 1
    for _ in range(max(0, n_correctors)):
        u_phys = u_int + 0.5 * f_m / rho_m.unsqueeze(0)
        f_m = f_m + (u_target.T - u_phys)

    f_m = f_m.T.contiguous()  # (M, d)
    f_grid = stencil.spread(f_m)
    body_force = -f_m.sum(dim=0)
    centroid = stencil.points.mean(dim=0)
    rel = stencil.points - centroid
    if stencil.points.shape[1] == 2:
        torque = -(rel[:, 0] * f_m[:, 1] - rel[:, 1] * f_m[:, 0]).sum()
    else:
        torque = -torch.cross(rel, f_m, dim=1).sum(dim=0)
    return f_m, f_grid, body_force, torque


# Historic alias used by the 3-D coupling module.
direct_forcing_3d = direct_forcing


def guo_force_2d(
    f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, tau: float
) -> torch.Tensor:
    """Guo body force applied to a D2Q9 post-collision population."""
    from .d2q9 import C, W

    w = W.to(f.dtype).view(9, 1, 1)
    cx = C[:, 0].to(f.dtype).view(9, 1, 1)
    cy = C[:, 1].to(f.dtype).view(9, 1, 1)
    g = 1.0 - 1.0 / (2.0 * tau)
    return f + w * 3.0 * g * (cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0))


def guo_force_3d(
    f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor, tau: float
) -> torch.Tensor:
    """Guo body force applied to a D3Q19 post-collision population."""
    from .d3q19 import _c_on, _w_on

    c = _c_on(f.device)
    w = _w_on(f.device, f.dtype).view(19, 1, 1, 1)
    g = 1.0 - 1.0 / (2.0 * tau)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * fx.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * fy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * fz.unsqueeze(0)
    )
    return f + w * 3.0 * g * cu


def body_source_2d(f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """Exact momentum source for D2Q9 (no Guo ``1 − 1/(2τ)`` factor).

    The injected momentum is ``Σ_i c_i w_i 3 (c_i · F) = F`` per node, so the
    marker force and the momentum actually added to the fluid are identical.
    This is the calibration used for *force accounting*: the drag/torque on the
    immersed body is ``−Σ_k F_k`` (markers) and that same ``Σ_k F_k`` is what
    the fluid receives.  With the Guo factor the injected momentum is only
    ``(1 − 1/(2τ)) F``, which biases a direct-forcing torque estimate by
    ``1/(1 − 1/(2τ))`` (2.67× at τ = 0.8).
    """
    from .d2q9 import C, W

    w = W.to(f.dtype).view(9, 1, 1)
    cx = C[:, 0].to(f.dtype).view(9, 1, 1)
    cy = C[:, 1].to(f.dtype).view(9, 1, 1)
    return f + w * 3.0 * (cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0))


def body_source_3d(
    f: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor
) -> torch.Tensor:
    """Exact momentum source for D3Q19 (no Guo factor); see :func:`body_source_2d`."""
    from .d3q19 import _c_on, _w_on

    c = _c_on(f.device)
    w = _w_on(f.device, f.dtype).view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * fx.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * fy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * fz.unsqueeze(0)
    )
    return f + w * 3.0 * cu


def interior_mask_force(
    u_grid: torch.Tensor,
    mask: torch.Tensor,
    u_target: torch.Tensor,
) -> torch.Tensor:
    """Eulerian direct forcing on a boolean solid mask.

    Args:
        u_grid: ``(d, *shape)`` velocity field.
        mask: ``(*shape,)`` boolean solid mask.
        u_target: ``(d, *shape)`` target velocity inside the solid.

    Returns:
        ``(d, *shape)`` body-force field (zero outside the mask).
    """
    force = torch.where(mask.unsqueeze(0), u_target - u_grid, torch.zeros_like(u_grid))
    return force


# --------------------------------------------------------------------------- #
# Audits
# --------------------------------------------------------------------------- #


def momentum_audit(stencil: IBMStencil, f_marker: torch.Tensor) -> dict:
    """Check partition of unity and adjointness (interpolation ↔ spreading).

    Returns:
        dict with ``sum_marker``, ``sum_grid``, ``rel_residual`` and
        ``partition_error``.
    """
    f_grid = stencil.spread(f_marker)
    s_marker = f_marker.sum(dim=0)
    s_grid = f_grid.reshape(f_marker.shape[1], -1).sum(dim=1)
    scale = float(s_marker.abs().max().item()) or 1.0
    rel = float((s_grid - s_marker).abs().max().item()) / scale
    return {
        "sum_marker": [float(v) for v in s_marker.tolist()],
        "sum_grid": [float(v) for v in s_grid.tolist()],
        "rel_residual": rel,
        "partition_error": abs(stencil.peak_normalization() - 1.0),
    }