"""Common Immersed Boundary Method (IBM) module — solver-agnostic direct forcing.

This module extracts the validated IBM kernels
(:func:`tensorlbm.ibm.ibm_direct_forcing_3d`,
:func:`tensorlbm.ibm.ibm_apply_body_force_3d`) behind a lattice-neutral public
interface that can be inserted into **any** collision → stream → boundary loop
without binding to a specific solver.  It supports both the D3Q19 and D3Q27
lattices.

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

This module deliberately does **not** modify the solver hot path
(``solver3d.py`` / ``d3q27.py`` collision & streaming).  It only wraps the
existing IBM helpers and adds a D3Q27-aware Guo forcing application.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .ibm import (
    ibm_delta_4pt,
    ibm_delta_hat,
    ibm_direct_forcing_3d,
)

__all__ = [
    "IBMLatticeName",
    "IBMKernelName",
    "IBMCapabilityWithheldError",
    "ibm_direct_forcing_3d_common",
    "ibm_apply_body_force_3d_common",
    "derive_surface_markers_3d",
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
        from .d3q19 import C as C19, W as W19

        return _LatticeSpec(19, C19.to(device), W19.to(device))
    if lattice_u == "D3Q27":
        from .d3q27 import C as C27, W as W27

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
        (1 - pad[:, :, 1:-1, 1:-1, 2:])       # +x
        + (1 - pad[:, :, 1:-1, 1:-1, :-2])    # -x
        + (1 - pad[:, :, 1:-1, 2:, 1:-1])     # +y
        + (1 - pad[:, :, 1:-1, :-2, 1:-1])    # -y
        + (1 - pad[:, :, 2:, 1:-1, 1:-1])     # +z
        + (1 - pad[:, :, :-2, 1:-1, 1:-1])    # -z
    ).squeeze()  # (nz, ny, nx)
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
) -> torch.Tensor:
    """Apply a 3-D Guo body-force correction to a D3Q19 or D3Q27 distribution.

    Uses the first-order Guo (2002) forcing scheme::

        f_i ← f_i + w_i · 3 · (c_ix F_x + c_iy F_y + c_iz F_z)

    where the factor ``3 = 1/c_s²`` is identical for D3Q19 and D3Q27.

    Args:
        f:       Distribution tensor of shape ``(Q, nz, ny, nx)``.
        fx_grid: Eulerian x-force field of shape ``(nz, ny, nx)``.
        fy_grid: Eulerian y-force field of shape ``(nz, ny, nx)``.
        fz_grid: Eulerian z-force field of shape ``(nz, ny, nx)``.
        lattice: ``"D3Q19"`` or ``"D3Q27"``.

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
    forcing = w_view * 3.0 * (
        cx * fx_grid.unsqueeze(0) + cy * fy_grid.unsqueeze(0) + cz * fz_grid.unsqueeze(0)
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
    kernel: IBMKernelName = "hat",
    markers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the direct-forcing IBM body force and apply it to ``f``.

    This is the solver-agnostic public interface.  It:

    1. Extracts the macroscopic velocity from ``f`` using the lattice weights.
    2. Resolves marker positions (from ``markers`` or derived from ``mask``).
    3. Broadcasts ``u_target`` to per-marker target velocities.
    4. Calls the validated :func:`ibm_direct_forcing_3d` kernel.
    5. Applies the Guo body-force correction to ``f`` (D3Q19 or D3Q27 aware).

    Args:
        f:        Distribution tensor, shape ``(Q, nz, ny, nx)``.
        mask:     Solid mask, shape ``(nz, ny, nx)``; ``True`` inside the body.
        u_target: Target marker velocity — ``(3,)``, ``(3, nz, ny, nx)``, or
                  ``(N, 3)``.
        lattice:  ``"D3Q19"`` or ``"D3Q27"``.
        kernel:   Delta kernel: ``"hat"`` (2-point) or ``"4pt"`` (4-point).
        markers:  Optional explicit ``(marker_x, marker_y, marker_z)`` triple,
                  each of shape ``(N,)``.  When ``None``, surface markers are
                  derived from ``mask``.

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
        zero_force = torch.zeros(
            (3, nz, ny, nx), dtype=f.dtype, device=f.device
        )
        return zero_force, f.clone()

    # 4. Resolve per-marker target velocity.
    ut_x, ut_y, ut_z = _resolve_target_velocity(
        u_target, marker_x, marker_y, marker_z, (nz, ny, nx)
    )

    # 5. Direct forcing (validated kernel).
    fx_grid, fy_grid, fz_grid = ibm_direct_forcing_3d(
        ux, uy, uz, marker_x, marker_y, marker_z, ut_x, ut_y, ut_z, kernel=kernel_name
    )

    # 6. Apply Guo body-force correction (lattice-aware).
    f_corrected = ibm_apply_body_force_3d_common(
        f, fx_grid, fy_grid, fz_grid, lattice=lattice_name
    )

    force = torch.stack([fx_grid, fy_grid, fz_grid], dim=0)
    return force, f_corrected
