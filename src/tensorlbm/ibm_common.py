"""Common Immersed Boundary Method (IBM) module — solver-agnostic direct forcing.

This module provides the IBM kernels integrated with the 9 common modules
workflow (solid → near → mesh → lbm_step → drag → St).  It supports:

* **Stationary bodies** — IBM replaces bounce-back for boundary enforcement.
* **Moving bodies** — marker positions update each step; no mask rebuild.
* **Rotating bodies** — rigid-body rotation of markers; Magnus effect.

IBM advantages over bounce-back (BB):
  - Handles moving bodies without rebuilding the solid mask each step.
  - Handles thin (zero-thickness) surfaces.
  - Works with any geometry (STL / analytical).

Public contract
----------------
``ibm_direct_forcing_3d_common(f, mask, u_target, *, lattice, kernel, markers)``
    * ``f``        – distribution tensor, shape ``(Q, nz, ny, nx)``.
    * ``mask``     – solid mask, shape ``(nz, ny, nx)``; ``True`` inside the body.
    * ``u_target`` – desired marker velocity.  Accepted shapes:
        - ``(3,)``            uniform target for every marker,
        - ``(3, nz, ny, nx)`` Eulerian field sampled at the markers,
        - ``(N, 3)``          per-marker target.
    * ``markers``  – optional explicit marker positions, shape ``(N, 3)`` in
      lattice coordinates ``(x, y, z)``.  When omitted, surface markers are
      derived from ``mask`` (solid cells with at least one fluid neighbour).
    Returns ``(force, f_corrected)`` where:
        - ``force``       – Eulerian IBM body-force field, shape ``(3, nz, ny, nx)``.
        - ``f_corrected`` – distribution with the Guo body-force correction
          applied, shape ``(Q, nz, ny, nx)``.

``ibm_step_correct(f, collide_fn, tau, solid, u_in, far_field_bc_fn, markers, u_target_fn, ...)``
    One LBM step with IBM forcing instead of bounce-back:
      1. Collision (all cells)
      2. IBM: interpolate velocity → compute force → spread → Guo correction
      3. Streaming
      4. Far-field BC
      5. Mass correction (optional)

``generate_cylinder_markers(cx, cy, cz, R, nz, axis='z', n_theta=64)``
    Analytical Lagrangian markers on a cylinder surface.

``generate_sphere_markers(cx, cy, cz, R, n_theta=32, n_phi=16)``
    Analytical Lagrangian markers on a sphere surface.

``compute_ibm_drag_from_markers(marker_fx, marker_fy, marker_fz, dpS)``
    Drag/lift coefficients from IBM marker forces (momentum exchange).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .ibm import (
    ibm_direct_forcing_3d,
)

__all__ = [
    "IBMLatticeName",
    "IBMKernelName",
    "IBMCapabilityWithheldError",
    "ibm_direct_forcing_3d_common",
    "ibm_apply_body_force_3d_common",
    "ibm_step_correct",
    "derive_surface_markers_3d",
    "generate_cylinder_markers",
    "generate_sphere_markers",
    "update_moving_markers",
    "update_rotating_markers",
    "compute_ibm_drag_from_markers",
    "macroscopic_velocity_3d",
]

IBMLatticeName = Literal["D3Q19", "D3Q27"]
IBMKernelName = Literal["hat", "4pt"]


class IBMCapabilityWithheldError(NotImplementedError):
    """Raised when an IBM capability request lacks a validated kernel."""


# --------------------------------------------------------------------------- #
# Lattice registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LatticeSpec:
    q: int
    c: torch.Tensor
    w: torch.Tensor

    def on(self, device: torch.device) -> "_LatticeSpec":
        return _LatticeSpec(self.q, self.c.to(device), self.w.to(device))


def _lattice_spec(lattice: str, device: torch.device) -> _LatticeSpec:
    lattice_u = lattice.upper()
    if lattice_u == "D3Q19":
        from .d3q19 import C as C19
        from .d3q19 import W as W19

        return _LatticeSpec(19, C19.to(device), W19.to(device))
    if lattice_u == "D3Q27":
        from .d3q27 import C as C27
        from .d3q27 import W as W27

        return _LatticeSpec(27, C27.to(device), W27.to(device))
    raise IBMCapabilityWithheldError(
        f"WITHHELD_UNKNOWN_LATTICE: {lattice!r} is not an audited IBM lattice "
        f"(expected 'D3Q19' or 'D3Q27')."
    )


def _normalise_lattice(lattice: str) -> IBMLatticeName:
    value = lattice.upper()
    if value not in {"D3Q19", "D3Q27"}:
        raise IBMCapabilityWithheldError(
            f"WITHHELD_UNKNOWN_LATTICE: {lattice!r} is not an audited IBM lattice."
        )
    return value  # type: ignore[return-value]


def _normalise_kernel(kernel: str) -> str:
    value = kernel.lower()
    if value not in {"hat", "4pt"}:
        raise IBMCapabilityWithheldError(
            f"WITHHELD_UNKNOWN_KERNEL: {kernel!r} is not an audited IBM delta kernel."
        )
    return value


# --------------------------------------------------------------------------- #
# Macroscopic velocity extraction (lattice-neutral)
# --------------------------------------------------------------------------- #


def macroscopic_velocity_3d(
    f: torch.Tensor,
    lattice: IBMLatticeName = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(rho, ux, uy, uz)`` from a 3-D distribution.

    Args:
        f:       Distribution tensor of shape ``(Q, nz, ny, nx)``.
        lattice:  Lattice name, ``"D3Q19"`` or ``"D3Q27"``.

    Returns:
        Tuple ``(rho, ux, uy, uz)`` each of shape ``(nz, ny, nx)``.
    """
    spec = _lattice_spec(_normalise_lattice(lattice), f.device)
    q = spec.q
    if f.ndim != 4 or f.shape[0] != q:
        raise ValueError(
            f"{lattice} distribution must have shape ({q}, nz, ny, nx); got {tuple(f.shape)}."
        )
    rho = f.sum(dim=0)  # (nz, ny, nx)
    c = spec.c.float()  # (Q, 3)
    # momentum = sum_q c_q * f_q  -> (nz, ny, nx, 3)
    momentum = (f.unsqueeze(-1) * c.view(q, 1, 1, 1, 3)).sum(dim=0)
    inv_rho = torch.where(rho > 1e-12, 1.0 / rho, torch.zeros_like(rho))
    u = momentum * inv_rho.unsqueeze(-1)
    return rho, u[..., 0], u[..., 1], u[..., 2]


# --------------------------------------------------------------------------- #
# Surface marker derivation from a solid mask
# --------------------------------------------------------------------------- #


def derive_surface_markers_3d(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive Lagrangian marker coordinates from a solid mask's surface.

    A *surface* cell is a solid cell (``mask == True``) that has at least one
    fluid neighbour (6-connectivity).  Each surface cell centre becomes a
    marker at its integer lattice coordinate.

    Args:
        mask: Solid mask of shape ``(nz, ny, nx)``; ``True`` inside the body.

    Returns:
        Tuple ``(marker_x, marker_y, marker_z)`` each of shape ``(N,)`` in
        lattice coordinates.  Returns three empty tensors when the mask has no
        surface (e.g. fully solid or fully fluid domain).
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D (nz, ny, nx); got {mask.ndim}-D.")
    nz, ny, nx = mask.shape
    m = mask.bool()
    # Pad with False (fluid) so border solid cells still count as surface.
    pad = torch.nn.functional.pad(m.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1, 1, 1))
    fluid_neighbours = (
        (
            (1 - pad[:, :, 1:-1, 1:-1, 2:])  # +x
            + (1 - pad[:, :, 1:-1, 1:-1, :-2])  # -x
            + (1 - pad[:, :, 1:-1, 2:, 1:-1])  # +y
            + (1 - pad[:, :, 1:-1, :-2, 1:-1])  # -y
            + (1 - pad[:, :, 2:, 1:-1, 1:-1])  # +z
            + (1 - pad[:, :, :-2, 1:-1, 1:-1])  # -z
        ).squeeze()
    )  # (nz, ny, nx)
    surface = m & (fluid_neighbours > 0)
    if not surface.any():
        empty = torch.zeros(0, dtype=torch.float32, device=mask.device)
        return empty, empty.clone(), empty.clone()
    iz, iy, ix = torch.where(surface)
    return (
        ix.float(),
        iy.float(),
        iz.float(),
    )


# --------------------------------------------------------------------------- #
# Guo body-force application (lattice-neutral)
# --------------------------------------------------------------------------- #


def ibm_apply_body_force_3d_common(
    f: torch.Tensor,
    fx_grid: torch.Tensor,
    fy_grid: torch.Tensor,
    fz_grid: torch.Tensor,
    lattice: IBMLatticeName = "D3Q19",
    tau: float | None = None,
) -> torch.Tensor:
    """Apply a 3-D Guo body-force correction to a D3Q19 or D3Q27 distribution.

    Uses the Guo (2002) forcing scheme::

        f_i ← f_i + (1 − 1/(2τ)) · w_i · 3 · (c_ix F_x + c_iy F_y + c_iz F_z)

    where the factor ``3 = 1/c_s²`` is identical for D3Q19 and D3Q27, and
    ``(1 − 1/(2τ))`` is the **Guo forcing factor** that accounts for the
    discrete lattice effect of body forces applied during/after collision.

    **Critical for stability**: without the ``(1 − 1/(2τ))`` factor the force
    is applied at full strength, which overshoots by ~8× for typical τ≈0.57
    and causes immediate divergence for moving bodies.

    Args:
        f:       Distribution tensor of shape ``(Q, nz, ny, nx)``.
        fx_grid: Eulerian x-force field of shape ``(nz, ny, nx)``.
        fy_grid: Eulerian y-force field of shape ``(nz, ny, nx)``.
        fz_grid: Eulerian z-force field of shape ``(nz, ny, nx)``.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        tau:     Relaxation time τ.  When provided, the Guo factor
                 ``(1 − 1/(2τ))`` is applied.  When ``None`` (legacy),
                 the factor defaults to 1.0 (NOT recommended — causes
                 instability for moving bodies).

    Returns:
        Updated distribution tensor of the same shape as ``f``.
    """
    spec = _lattice_spec(_normalise_lattice(lattice), f.device)
    q = spec.q
    if f.ndim != 4 or f.shape[0] != q:
        raise ValueError(
            f"{lattice} distribution must have shape ({q}, nz, ny, nx); got {tuple(f.shape)}."
        )
    c = spec.c.float()
    w = spec.w.float()
    cx = c[:, 0].view(q, 1, 1, 1)
    cy = c[:, 1].view(q, 1, 1, 1)
    cz = c[:, 2].view(q, 1, 1, 1)
    w_view = w.view(q, 1, 1, 1)
    # Guo forcing factor: (1 − 1/(2τ)).  Essential for stability.
    guo_factor = 1.0
    if tau is not None:
        guo_factor = 1.0 - 1.0 / (2.0 * tau)
    forcing = (
        w_view
        * 3.0
        * guo_factor
        * (cx * fx_grid.unsqueeze(0) + cy * fy_grid.unsqueeze(0) + cz * fz_grid.unsqueeze(0))
    )
    return f + forcing


# --------------------------------------------------------------------------- #
# Public direct-forcing interface
# --------------------------------------------------------------------------- #


def _resolve_target_velocity(
    u_target: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    marker_z: torch.Tensor,
    field_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast ``u_target`` to per-marker ``(N,)`` triplets."""
    n = marker_x.shape[0]
    nz, ny, nx = field_shape
    if n == 0:
        z = torch.zeros(0, dtype=marker_x.dtype, device=marker_x.device)
        return z, z.clone(), z.clone()
    t = u_target
    if t.ndim == 1 and t.shape[0] == 3:
        # Uniform target for every marker.
        return (
            t[0].expand(n).to(marker_x.dtype),
            t[1].expand(n).to(marker_x.dtype),
            t[2].expand(n).to(marker_x.dtype),
        )
    if t.ndim == 4 and t.shape[0] == 3:
        # Eulerian field (3, nz, ny, nx): sample at marker integer cells.
        ix = marker_x.long().clamp(0, nx - 1)
        iy = marker_y.long().clamp(0, ny - 1)
        iz = marker_z.long().clamp(0, nz - 1)
        return t[0][iz, iy, ix], t[1][iz, iy, ix], t[2][iz, iy, ix]
    if t.ndim == 2 and t.shape[1] == 3 and t.shape[0] == n:
        return t[:, 0], t[:, 1], t[:, 2]
    raise ValueError(
        f"u_target has unsupported shape {tuple(t.shape)} for {n} markers; "
        f"expected (3,), (3, nz, ny, nx), or (N, 3)."
    )


def ibm_direct_forcing_3d_common(
    f: torch.Tensor,
    mask: torch.Tensor,
    u_target: torch.Tensor,
    *,
    lattice: IBMLatticeName = "D3Q19",
    kernel: IBMKernelName = "4pt",
    markers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    tau: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the direct-forcing IBM body force and apply it to ``f``.

    This is the solver-agnostic public interface.  It:

    1. Extracts the macroscopic velocity from ``f`` using the lattice weights.
    2. Resolves marker positions (from ``markers`` or derived from ``mask``).
    3. Broadcasts ``u_target`` to per-marker target velocities.
    4. Calls the validated :func:`ibm_direct_forcing_3d` kernel.
    5. Applies the Guo body-force correction to ``f`` (D3Q19 or D3Q27 aware),
       including the ``(1 − 1/(2τ))`` Guo factor when ``tau`` is provided.

    Args:
        f:        Distribution tensor, shape ``(Q, nz, ny, nx)``.
        mask:     Solid mask, shape ``(nz, ny, nx)``; ``True`` inside the body.
        u_target: Target marker velocity — ``(3,)``, ``(3, nz, ny, nx)``, or
                  ``(N, 3)``.
        lattice:  ``"D3Q19"`` or ``"D3Q27"``.
        kernel:   Delta kernel: ``"hat"`` (2-point) or ``"4pt"`` (4-point).
                  Default is ``"4pt"`` for stability.
        markers:  Optional explicit ``(marker_x, marker_y, marker_z)`` triple,
                  each of shape ``(N,)``.  When ``None``, surface markers are
                  derived from ``mask``.
        tau:      Relaxation time τ.  When provided, the Guo forcing factor
                  ``(1 − 1/(2τ))`` is applied.  Essential for stability.

    Returns:
        Tuple ``(force, f_corrected)``:
        - ``force`` of shape ``(3, nz, ny, nx)`` — the Eulerian IBM body force.
        - ``f_corrected`` of shape ``(Q, nz, ny, nx)`` — ``f`` with the Guo
          body-force correction applied.
    """
    lattice_name = _normalise_lattice(lattice)
    kernel_name = _normalise_kernel(kernel)
    spec = _lattice_spec(lattice_name, f.device)
    q = spec.q
    if f.ndim != 4 or f.shape[0] != q:
        raise ValueError(
            f"{lattice_name} distribution must have shape ({q}, nz, ny, nx); got {tuple(f.shape)}."
        )
    if mask.shape != f.shape[1:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match f spatial shape {tuple(f.shape[1:])}."
        )

    # 1. Macroscopic velocity from f.
    _, ux, uy, uz = macroscopic_velocity_3d(f, lattice=lattice_name)

    # 2. Marker positions.
    if markers is not None:
        marker_x, marker_y, marker_z = markers
    else:
        marker_x, marker_y, marker_z = derive_surface_markers_3d(mask)

    nz, ny, nx = f.shape[1:]

    # 3. Zero markers → zero force, f unchanged.
    if marker_x.shape[0] == 0:
        zero_force = torch.zeros((3, nz, ny, nx), dtype=f.dtype, device=f.device)
        return zero_force, f.clone()

    # 4. Resolve per-marker target velocity.
    ut_x, ut_y, ut_z = _resolve_target_velocity(
        u_target, marker_x, marker_y, marker_z, (nz, ny, nx)
    )

    # 5. Direct forcing (validated kernel).
    fx_grid, fy_grid, fz_grid = ibm_direct_forcing_3d(
        ux, uy, uz, marker_x, marker_y, marker_z, ut_x, ut_y, ut_z, kernel=kernel_name
    )

    # 6. Apply Guo body-force correction (lattice-aware, with Guo factor).
    f_corrected = ibm_apply_body_force_3d_common(
        f, fx_grid, fy_grid, fz_grid, lattice=lattice_name, tau=tau
    )

    force = torch.stack([fx_grid, fy_grid, fz_grid], dim=0)
    return force, f_corrected


# --------------------------------------------------------------------------- #
# Analytical marker generation (cylinder / sphere)
# --------------------------------------------------------------------------- #


def generate_cylinder_markers(
    cx: float,
    cy: float,
    cz: float,
    R: float,
    nz: int,
    axis: str = "z",
    n_theta: int = 64,
    n_axial: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Lagrangian markers on a cylinder surface.

    For a cylinder extruded along *axis* with centre (cx, cy, cz) and radius
    R, markers are placed at uniform angular and axial spacing on the surface.

    Args:
        cx, cy, cz: Cylinder centre in lattice coordinates.
        R: Cylinder radius in lattice units.
        nz: Number of cells along the extrusion axis (for axial marker span).
        axis: Extrusion axis: ``'z'`` (default), ``'y'``, or ``'x'``.
        n_theta: Number of markers around the circumference.
        n_axial: Number of markers along the axis.  If ``None``, uses
            ``max(2, nz // 4)`` to give ~4-cell spacing.
        device: Torch device for the output tensors.

    Returns:
        Tuple ``(marker_x, marker_y, marker_z)`` each of shape ``(N,)``.
    """
    if n_axial is None:
        n_axial = max(2, nz // 4)

    theta = torch.linspace(0, 2 * math.pi, n_theta, device=device, dtype=torch.float32)[:-1]
    if axis == "z":
        axial = torch.linspace(0.5, nz - 0.5, n_axial, device=device, dtype=torch.float32)
        # Broadcast: (n_axial, n_theta)
        ax, th = torch.meshgrid(axial, theta, indexing="ij")
        mx = (cx + R * torch.cos(th)).reshape(-1)
        my = (cy + R * torch.sin(th)).reshape(-1)
        mz = (cz + ax).reshape(-1)
    elif axis == "y":
        axial = torch.linspace(0.5, nz - 0.5, n_axial, device=device, dtype=torch.float32)
        ax, th = torch.meshgrid(axial, theta, indexing="ij")
        mx = (cx + R * torch.cos(th)).reshape(-1)
        my = (cy + ax).reshape(-1)
        mz = (cz + R * torch.sin(th)).reshape(-1)
    elif axis == "x":
        axial = torch.linspace(0.5, nz - 0.5, n_axial, device=device, dtype=torch.float32)
        ax, th = torch.meshgrid(axial, theta, indexing="ij")
        mx = (cx + ax).reshape(-1)
        my = (cy + R * torch.cos(th)).reshape(-1)
        mz = (cz + R * torch.sin(th)).reshape(-1)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    return mx, my, mz


def generate_sphere_markers(
    cx: float,
    cy: float,
    cz: float,
    R: float,
    n_theta: int = 32,
    n_phi: int = 16,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Lagrangian markers on a sphere surface.

    Markers are placed at uniform angular spacing (theta = azimuth, phi =
    polar) on the sphere surface.

    Args:
        cx, cy, cz: Sphere centre in lattice coordinates.
        R: Sphere radius in lattice units.
        n_theta: Number of azimuthal markers (around equator).
        n_phi: Number of polar markers (pole to pole).
        device: Torch device for the output tensors.

    Returns:
        Tuple ``(marker_x, marker_y, marker_z)`` each of shape ``(N,)``.
    """
    theta = torch.linspace(0, 2 * math.pi, n_theta, device=device, dtype=torch.float32)[:-1]
    phi = torch.linspace(0, math.pi, n_phi, device=device, dtype=torch.float32)
    th, ph = torch.meshgrid(theta, phi, indexing="ij")

    mx = (cx + R * torch.sin(ph) * torch.cos(th)).reshape(-1)
    my = (cy + R * torch.sin(ph) * torch.sin(th)).reshape(-1)
    mz = (cz + R * torch.cos(ph)).reshape(-1)
    return mx, my, mz


# --------------------------------------------------------------------------- #
# Moving / rotating body marker updates
# --------------------------------------------------------------------------- #


def update_moving_markers(
    marker_x0: torch.Tensor,
    marker_y0: torch.Tensor,
    marker_z0: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
    displacement: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate all markers by a rigid-body displacement.

    For a moving body with uniform velocity u_body over time dt, the
    displacement is (u_x*dt, u_y*dt, u_z*dt).  This updates marker positions
    without rebuilding the solid mask — the key IBM advantage for moving bodies.

    Args:
        marker_x0, marker_y0, marker_z0: Initial marker positions, shape (N,).
        cx, cy, cz: Body centre (not used for translation, kept for API symmetry).
        displacement: (dx, dy, dz) translation in lattice units.

    Returns:
        Updated (marker_x, marker_y, marker_z) each of shape (N,).
    """
    dx, dy, dz = displacement
    return (
        marker_x0 + dx,
        marker_y0 + dy,
        marker_z0 + dz,
    )


def update_rotating_markers(
    marker_x0: torch.Tensor,
    marker_y0: torch.Tensor,
    marker_z0: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
    angle: float,
    axis: str = "x",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate all markers about a body centre by *angle* radians.

    For a rotating body with angular velocity omega over time dt, the
    rotation angle is omega*dt.  The rotation is about the specified axis
    passing through (cx, cy, cz).

    Args:
        marker_x0, marker_y0, marker_z0: Initial marker positions, shape (N,).
        cx, cy, cz: Rotation centre (body centre).
        angle: Rotation angle in radians.
        axis: Rotation axis: ``'x'``, ``'y'``, or ``'z'``.

    Returns:
        Updated (marker_x, marker_y, marker_z) each of shape (N,).
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = marker_x0 - cx
    dy = marker_y0 - cy
    dz = marker_z0 - cz

    if axis == "x":
        # Rotate about x: y' = y*cos - z*sin, z' = y*sin + z*cos
        new_y = cy + dy * cos_a - dz * sin_a
        new_z = cz + dy * sin_a + dz * cos_a
        return marker_x0.clone(), new_y, new_z
    elif axis == "y":
        # Rotate about y: x' = x*cos + z*sin, z' = -x*sin + z*cos
        new_x = cx + dx * cos_a + dz * sin_a
        new_z = cz - dx * sin_a + dz * cos_a
        return new_x, marker_y0.clone(), new_z
    elif axis == "z":
        # Rotate about z: x' = x*cos - y*sin, y' = x*sin + y*cos
        new_x = cx + dx * cos_a - dy * sin_a
        new_y = cy + dx * sin_a + dy * cos_a
        return new_x, new_y, marker_z0.clone()
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")


# --------------------------------------------------------------------------- #
# IBM drag from marker forces (momentum exchange)
# --------------------------------------------------------------------------- #


def compute_ibm_drag_from_markers(
    marker_fx: torch.Tensor,
    marker_fy: torch.Tensor,
    marker_fz: torch.Tensor,
    dpS: float,
) -> tuple[float, float, float]:
    """Compute drag/lift coefficients from IBM marker forces.

    The total force on the body is the negative of the sum of IBM forces
    applied to the fluid (Newton's third law).  The drag coefficient is:

        Cd = -sum(F_marker) / dpS

    where dpS = 0.5 * rho * U^2 * A_frontal is the dynamic pressure × area.

    Args:
        marker_fx, marker_fy, marker_fz: Per-marker IBM forces, shape (N,).
            These are the forces applied TO the fluid; the body feels the
            negative.
        dpS: Dynamic pressure × frontal area (0.5 * u^2 * A).

    Returns:
        Tuple (Cd_x, Cd_y, Cd_z) — drag (x), lift (y), side (z) coefficients.
    """
    fx_body = -float(marker_fx.sum().item())
    fy_body = -float(marker_fy.sum().item())
    fz_body = -float(marker_fz.sum().item())
    if dpS == 0:
        return 0.0, 0.0, 0.0
    return fx_body / dpS, fy_body / dpS, fz_body / dpS


# --------------------------------------------------------------------------- #
# IBM-integrated LBM step (replaces lbm_step_correct for IBM)
# --------------------------------------------------------------------------- #


def _ibm_force_spread_3d_vec(
    marker_fx: torch.Tensor,
    marker_fy: torch.Tensor,
    marker_fz: torch.Tensor,
    marker_x: torch.Tensor,
    marker_y: torch.Tensor,
    marker_z: torch.Tensor,
    nz: int,
    ny: int,
    nx: int,
    kernel: str = "4pt",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized 3D force spreading (replaces Python loops).

    Spreads per-marker forces onto the Eulerian grid using the delta kernel.
    This is the vectorized counterpart of ``ibm_force_spread_3d`` in
    :mod:`ibm`, using ``index_put_`` with ``accumulate=True`` for
    scatter-add.
    """
    from .ibm import ibm_delta_4pt, ibm_delta_hat

    device = marker_fx.device
    n_markers = marker_x.shape[0]
    if n_markers == 0:
        z = torch.zeros((nz, ny, nx), dtype=marker_fx.dtype, device=device)
        return z, z.clone(), z.clone()

    delta_fn = ibm_delta_hat if kernel == "hat" else ibm_delta_4pt
    support = 2 if kernel == "hat" else 4
    half_s = support // 2

    ix0 = (torch.floor(marker_x) - half_s + 1).long()
    iy0 = (torch.floor(marker_y) - half_s + 1).long()
    iz0 = (torch.floor(marker_z) - half_s + 1).long()

    offsets = torch.arange(support, device=device)
    ix_all = (ix0.unsqueeze(1) + offsets.unsqueeze(0)) % nx
    iy_all = (iy0.unsqueeze(1) + offsets.unsqueeze(0)) % ny
    iz_all = (iz0.unsqueeze(1) + offsets.unsqueeze(0)) % nz

    rx_all = (ix0.unsqueeze(1) + offsets.unsqueeze(0)).float() - marker_x.unsqueeze(1)
    ry_all = (iy0.unsqueeze(1) + offsets.unsqueeze(0)).float() - marker_y.unsqueeze(1)
    rz_all = (iz0.unsqueeze(1) + offsets.unsqueeze(0)).float() - marker_z.unsqueeze(1)

    wx_all = delta_fn(rx_all)
    wy_all = delta_fn(ry_all)
    wz_all = delta_fn(rz_all)

    fx_grid = torch.zeros(nz, ny, nx, dtype=marker_fx.dtype, device=device)
    fy_grid = torch.zeros(nz, ny, nx, dtype=marker_fy.dtype, device=device)
    fz_grid = torch.zeros(nz, ny, nx, dtype=marker_fz.dtype, device=device)

    for di in range(support):
        for dj in range(support):
            for dk in range(support):
                w = wx_all[:, di] * wy_all[:, dj] * wz_all[:, dk]
                ix = ix_all[:, di]
                iy = iy_all[:, dj]
                iz = iz_all[:, dk]
                fx_grid.index_put_((iz, iy, ix), w * marker_fx, accumulate=True)
                fy_grid.index_put_((iz, iy, ix), w * marker_fy, accumulate=True)
                fz_grid.index_put_((iz, iy, ix), w * marker_fz, accumulate=True)

    return fx_grid, fy_grid, fz_grid


def ibm_step_correct(
    f: torch.Tensor,
    collide_fn: Callable,
    tau: float,
    solid: torch.Tensor,
    u_in: float,
    far_field_bc_fn: Callable,
    markers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    u_target_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    lattice: IBMLatticeName = "D3Q19",
    kernel: IBMKernelName = "4pt",
    correct_mass_fn: Callable | None = None,
    target_mass: float | None = None,
    step: int = 0,
    mass_interval: int = 200,
    ramp_steps: int = 1000,
    n_force_iter: int = 4,
    force_clip: float | None = 0.05,
    **collide_kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One LBM step with IBM direct forcing instead of bounce-back.

    **Stability measures** (based on mature IBM-LBM implementations):

    1. **NoDynamics** for solid cells — solid cells are restored to their
       pre-collision state after collision, preventing non-physical
       distributions that corrupt the IBM velocity interpolation.
    2. **Guo forcing factor** ``(1 − 1/(2τ))`` — the body force is scaled
       by this factor to account for discrete lattice effects.  Without it,
       the force overshoots by ~8× for typical τ≈0.57.
    3. **4-point Peskin kernel** (default) — smoother forces than the
       2-point hat kernel, with 4³=64 cell support vs 2³=8.
    4. **Velocity ramp** — the free-stream velocity and target velocity are
       linearly ramped from 0 to full over ``ramp_steps`` steps to avoid
       large initial forces.
    5. **Multi-direct forcing (MDM)** — the IBM force is applied
       ``n_force_iter`` times per step, re-interpolating the velocity after
       each application.  This improves no-slip enforcement (Lai & Peskin
       2000).
    6. **Force clamping** — per-marker force is clamped to
       ``[-force_clip, force_clip]`` to prevent runaway forces.

    Order of operations:
      1. Save pre-collision state
      2. Collision (all cells)
      3. NoDynamics: restore solid cells to pre-collision state
      4. IBM: interpolate velocity → compute force → spread → Guo correction
         (repeated ``n_force_iter`` times for multi-direct forcing)
      5. Streaming
      6. Far-field BC
      7. Mass correction (optional, every mass_interval steps)

    Args:
        f: Distribution tensor (Q, nz, ny, nx).
        collide_fn: Collision function (e.g., collide_mrt3d).
        tau: Relaxation time.
        solid: Boolean solid mask (nz, ny, nx) — used for NoDynamics and
            mass correction, NOT for bounce-back.
        u_in: Free-stream velocity for far-field BC.
        far_field_bc_fn: Far-field BC function ``fn(f, u_in) -> f``.
        markers: Tuple (marker_x, marker_y, marker_z) each of shape (N,).
        u_target_fn: Callable ``fn(step) -> (ut_x, ut_y, ut_z)`` returning
            per-marker target velocity tensors.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.
        kernel: Delta kernel: ``"4pt"`` (default, stable) or ``"hat"``.
        correct_mass_fn: Mass correction function (e.g., correct_mass3d).
        target_mass: Target total mass for correction.
        step: Current step number (for mass correction and velocity ramp).
        mass_interval: Mass correction interval (default 200).
        ramp_steps: Number of steps for linear velocity ramp (default 1000).
            Set to 0 to disable ramping.
        n_force_iter: Number of multi-direct forcing iterations (default 4).
            Each iteration re-interpolates velocity and applies a correction.
        force_clip: Maximum per-marker force magnitude (default 0.05).
            Set to None to disable clamping.
        **collide_kwargs: Additional collision parameters (e.g., C_s=0.05).

    Returns:
        Tuple ``(f_new, marker_forces)`` where:
        - ``f_new``: Updated distribution tensor.
        - ``marker_forces``: Tuple (marker_fx, marker_fy, marker_fz) of the
          total IBM forces at each marker (shape (N,)).  Use
          :func:`compute_ibm_drag_from_markers` to get Cd/Cl.
    """
    # ---- 1. Save pre-collision state (for NoDynamics) ----
    f_pre = f.clone()

    # ---- 2. Collision (all cells) ----
    f = collide_fn(f, tau=tau, **collide_kwargs)

    # ---- 3. NoDynamics: restore solid cells to pre-collision state ----
    # This prevents non-physical distributions in solid cells that would
    # corrupt the IBM velocity interpolation at the boundary markers.
    sm = solid.unsqueeze(0).expand_as(f)
    f = torch.where(sm, f_pre, f)

    # ---- 4. IBM direct forcing (with multi-direct forcing) ----
    lattice_name = _normalise_lattice(lattice)
    kernel_name = _normalise_kernel(kernel)

    marker_x, marker_y, marker_z = markers
    ut_x, ut_y, ut_z = u_target_fn(step)

    device = f.device
    ut_x = ut_x.to(device)
    ut_y = ut_y.to(device)
    ut_z = ut_z.to(device)
    marker_x = marker_x.to(device)
    marker_y = marker_y.to(device)
    marker_z = marker_z.to(device)

    # Velocity ramp: scale both target velocity and free-stream by ramp factor
    if ramp_steps > 0 and step < ramp_steps:
        ramp = float(step + 1) / float(ramp_steps)
        ut_x = ut_x * ramp
        ut_y = ut_y * ramp
        ut_z = ut_z * ramp

    nz, ny, nx = f.shape[1:]

    # Guo forcing factor: (1 − 1/(2τ))
    1.0 - 1.0 / (2.0 * tau)

    # Accumulate total marker forces for drag computation
    marker_fx_total = torch.zeros_like(ut_x)
    marker_fy_total = torch.zeros_like(ut_y)
    marker_fz_total = torch.zeros_like(ut_z)

    for _iter in range(n_force_iter):
        # 4a. Extract macroscopic velocity from current f
        rho, ux, uy, uz = macroscopic_velocity_3d(f, lattice=lattice_name)

        # 4b. Interpolate velocity at marker positions
        try:
            from .ibm_vec import ibm_velocity_interpolate_3d_vec

            u_mx, u_my, u_mz = ibm_velocity_interpolate_3d_vec(
                ux, uy, uz, marker_x, marker_y, marker_z, kernel=kernel_name
            )
        except Exception:
            from .ibm import ibm_velocity_interpolate_3d

            u_mx, u_my, u_mz = ibm_velocity_interpolate_3d(
                ux, uy, uz, marker_x, marker_y, marker_z, kernel=kernel_name
            )

        # 4c. Compute direct-forcing: F = ρ · (u_target − u_IB)
        #     (ρ ≈ 1.0 for incompressible LBM, but we use the mean density
        #      for correctness — avoids per-marker density interpolation cost)
        rho_mean = rho.mean()
        marker_fx = rho_mean * (ut_x - u_mx)
        marker_fy = rho_mean * (ut_y - u_my)
        marker_fz = rho_mean * (ut_z - u_mz)

        # 4d. Force clamping (safety)
        if force_clip is not None:
            marker_fx = torch.clamp(marker_fx, -force_clip, force_clip)
            marker_fy = torch.clamp(marker_fy, -force_clip, force_clip)
            marker_fz = torch.clamp(marker_fz, -force_clip, force_clip)

        # Accumulate for drag reporting
        marker_fx_total += marker_fx.detach()
        marker_fy_total += marker_fy.detach()
        marker_fz_total += marker_fz.detach()

        # 4e. Spread force to Eulerian grid (vectorized)
        fx_grid, fy_grid, fz_grid = _ibm_force_spread_3d_vec(
            marker_fx,
            marker_fy,
            marker_fz,
            marker_x,
            marker_y,
            marker_z,
            nz,
            ny,
            nx,
            kernel=kernel_name,
        )

        # 4f. Apply Guo body-force correction (with (1−1/2τ) factor)
        f = ibm_apply_body_force_3d_common(
            f, fx_grid, fy_grid, fz_grid, lattice=lattice_name, tau=tau
        )

    # ---- 5. Streaming ----
    if lattice_name == "D3Q19":
        from .solver3d import stream3d

        f = stream3d(f)
    else:
        from .d3q27 import stream27_roll

        f = stream27_roll(f)

    # ---- 6. Far-field BC ----
    # Scale u_in by ramp factor during startup
    u_in_eff = u_in
    if ramp_steps > 0 and step < ramp_steps:
        u_in_eff = u_in * float(step + 1) / float(ramp_steps)
    f = far_field_bc_fn(f, u_in_eff)

    # ---- 7. Mass correction ----
    if correct_mass_fn is not None and target_mass is not None:
        if step % mass_interval == 0:
            f = correct_mass_fn(f, target_mass)

    return f, (marker_fx_total, marker_fy_total, marker_fz_total)
