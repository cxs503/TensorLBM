"""Non-equilibrium extrapolation boundaries for external 3-D LBM flow.

Only populations entering the domain are reconstructed.  Their equilibrium
part imposes the far-field state while the non-equilibrium part is copied
from the adjacent interior plane.  Outgoing populations are left untouched,
avoiding the full-distribution equilibrium reset used by the legacy external
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class OpenBoundaryFaceDiagnostics:
    face: str
    updated_populations: int
    mass_delta: float
    momentum_delta: tuple[float, float, float]


@dataclass(frozen=True)
class OpenBoundaryDiagnostics:
    faces: tuple[OpenBoundaryFaceDiagnostics, ...]
    updated_populations: int
    mass_delta: float
    momentum_delta: tuple[float, float, float]
    face_sum_mass_closure_error: float
    face_sum_momentum_closure_error: tuple[float, float, float]
    finite: bool


def _lattice(
    q: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Callable[..., torch.Tensor]]:
    if q == 19:
        from .d3q19 import C, equilibrium3d

        return C.to(device=device), equilibrium3d
    if q == 27:
        from .d3q27 import C, equilibrium27

        return C.to(device=device), equilibrium27
    raise ValueError("only D3Q19 and D3Q27 are supported")


def non_equilibrium_far_field_bc_3d(
    f: torch.Tensor,
    *,
    u_in: float,
    rho_far: float = 1.0,
    uy_far: float = 0.0,
    uz_far: float = 0.0,
    faces: tuple[str, ...] = ("x-", "x+", "y-", "y+", "z-", "z+"),
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, OpenBoundaryDiagnostics]:
    """Apply incoming-only non-equilibrium extrapolation on selected faces.

    At ``x+`` the density is prescribed and velocity is extrapolated from the
    adjacent interior plane.  Other faces use the complete far-field state.
    """
    if f.ndim != 4:
        raise ValueError("f must have shape (Q,nz,ny,nx)")
    allowed = {"x-", "x+", "y-", "y+", "z-", "z+"}
    if not set(faces) <= allowed:
        raise ValueError("unknown far-field face")
    c, equilibrium = _lattice(f.shape[0], f.device, f.dtype)
    out = f.clone()
    face_diagnostics: list[OpenBoundaryFaceDiagnostics] = []

    def apply_face(
        face: str,
        boundary_slice: tuple[slice | int, slice | int, slice | int],
        interior_slice: tuple[slice | int, slice | int, slice | int],
        incoming: torch.Tensor,
        outlet: bool = False,
    ) -> None:
        interior_values = f[(slice(None),) + interior_slice]
        vector_shape = (f.shape[0],) + (1,) * (interior_values.ndim - 1)
        rho_i = interior_values.sum(dim=0).clamp_min(1e-12)
        ux_i = (interior_values * c[:, 0].to(f.dtype).view(vector_shape)).sum(dim=0) / rho_i
        uy_i = (interior_values * c[:, 1].to(f.dtype).view(vector_shape)).sum(dim=0) / rho_i
        uz_i = (interior_values * c[:, 2].to(f.dtype).view(vector_shape)).sum(dim=0) / rho_i
        local_eq = equilibrium(rho_i, ux_i, uy_i, uz_i, device=f.device)
        if outlet:
            target_ux = ux_i
            target_uy = uy_i
            target_uz = uz_i
        else:
            target_ux = torch.full_like(rho_i, u_in)
            target_uy = torch.full_like(rho_i, uy_far)
            target_uz = torch.full_like(rho_i, uz_far)
        target_rho = torch.full_like(rho_i, rho_far)
        target_eq = equilibrium(
            target_rho,
            target_ux,
            target_uy,
            target_uz,
            device=f.device,
        )
        directions = incoming.nonzero(as_tuple=False).flatten().tolist()
        previous = out[(directions,) + boundary_slice].clone() if return_diagnostics else None
        for direction in directions:
            out[(direction,) + boundary_slice] = (
                target_eq[direction] + interior_values[direction] - local_eq[direction]
            )
        if return_diagnostics:
            assert previous is not None
            updated = out[(directions,) + boundary_slice]
            delta = updated - previous
            population_delta = delta.sum(
                dim=tuple(range(1, delta.ndim)),
            )
            direction_tensor = torch.tensor(
                directions,
                device=f.device,
                dtype=torch.long,
            )
            direction_c = c[direction_tensor].to(dtype=f.dtype)
            momentum = (population_delta[:, None] * direction_c).sum(dim=0)
            face_diagnostics.append(
                OpenBoundaryFaceDiagnostics(
                    face=face,
                    updated_populations=delta.numel(),
                    mass_delta=float(population_delta.sum().item()),
                    momentum_delta=tuple(float(value.item()) for value in momentum),
                )
            )

    # Tensor spatial order is z,y,x.  c order is x,y,z.
    specs = {
        "x-": ((slice(None), slice(None), 0), (slice(None), slice(None), 1), c[:, 0] > 0, False),
        "x+": ((slice(None), slice(None), -1), (slice(None), slice(None), -2), c[:, 0] < 0, True),
        "y-": ((slice(None), 0, slice(None)), (slice(None), 1, slice(None)), c[:, 1] > 0, False),
        "y+": ((slice(None), -1, slice(None)), (slice(None), -2, slice(None)), c[:, 1] < 0, False),
        "z-": ((0, slice(None), slice(None)), (1, slice(None), slice(None)), c[:, 2] > 0, False),
        "z+": ((-1, slice(None), slice(None)), (-2, slice(None), slice(None)), c[:, 2] < 0, False),
    }
    for face in faces:
        boundary_slice, interior_slice, incoming, outlet = specs[face]
        apply_face(face, boundary_slice, interior_slice, incoming, outlet)
    if not return_diagnostics:
        return out
    face_mass_delta = sum(item.mass_delta for item in face_diagnostics)
    face_momentum_delta = tuple(
        sum(item.momentum_delta[axis] for item in face_diagnostics) for axis in range(3)
    )
    total_delta = out - f
    population_delta = total_delta.sum(dim=(1, 2, 3))
    mass_delta = float(population_delta.sum().item())
    momentum = (population_delta[:, None] * c.to(dtype=f.dtype)).sum(dim=0)
    momentum_delta = tuple(float(value.item()) for value in momentum)
    diagnostics = OpenBoundaryDiagnostics(
        faces=tuple(face_diagnostics),
        updated_populations=sum(item.updated_populations for item in face_diagnostics),
        mass_delta=mass_delta,
        momentum_delta=momentum_delta,
        face_sum_mass_closure_error=mass_delta - face_mass_delta,
        face_sum_momentum_closure_error=tuple(
            momentum_delta[axis] - face_momentum_delta[axis] for axis in range(3)
        ),
        finite=(
            torch.isfinite(
                torch.tensor(
                    (mass_delta, *momentum_delta),
                    dtype=torch.float64,
                )
            )
            .all()
            .item()
        ),
    )
    return out, diagnostics


__all__ = [
    "OpenBoundaryDiagnostics",
    "OpenBoundaryFaceDiagnostics",
    "non_equilibrium_far_field_bc_3d",
]
