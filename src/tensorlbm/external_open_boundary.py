"""Non-equilibrium extrapolation boundaries for external 3-D LBM flow.

Only populations entering the domain are reconstructed.  Their equilibrium
part imposes the far-field state while the non-equilibrium part is copied
from the adjacent interior plane.  Outgoing populations are left untouched,
avoiding the full-distribution equilibrium reset used by the legacy external
boundary.
"""
from __future__ import annotations

import torch


def _lattice(q: int, device: torch.device, dtype: torch.dtype):
    if q == 19:
        from .d3q19 import C, equilibrium3d, macroscopic3d
        return C.to(device=device), equilibrium3d, macroscopic3d
    if q == 27:
        from .d3q27 import C, equilibrium27, macroscopic27
        return C.to(device=device), equilibrium27, macroscopic27
    raise ValueError("only D3Q19 and D3Q27 are supported")


def non_equilibrium_far_field_bc_3d(
    f: torch.Tensor,
    *,
    u_in: float,
    rho_far: float = 1.0,
    uy_far: float = 0.0,
    uz_far: float = 0.0,
    faces: tuple[str, ...] = ("x-", "x+", "y-", "y+", "z-", "z+"),
) -> torch.Tensor:
    """Apply incoming-only non-equilibrium extrapolation on selected faces.

    At ``x+`` the density is prescribed and velocity is extrapolated from the
    adjacent interior plane.  Other faces use the complete far-field state.
    """
    if f.ndim != 4:
        raise ValueError("f must have shape (Q,nz,ny,nx)")
    allowed = {"x-", "x+", "y-", "y+", "z-", "z+"}
    if not set(faces) <= allowed:
        raise ValueError("unknown far-field face")
    c, equilibrium, macro = _lattice(f.shape[0], f.device, f.dtype)
    rho, ux, uy, uz = macro(f)
    local_eq = equilibrium(rho, ux, uy, uz, device=f.device)
    out = f.clone()

    def apply_face(
        face: str,
        boundary_slice: tuple[slice | int, slice | int, slice | int],
        interior_slice: tuple[slice | int, slice | int, slice | int],
        incoming: torch.Tensor,
        outlet: bool = False,
    ) -> None:
        interior_rho = rho[interior_slice]
        if outlet:
            target_ux = ux[interior_slice]
            target_uy = uy[interior_slice]
            target_uz = uz[interior_slice]
        else:
            target_ux = torch.full_like(interior_rho, u_in)
            target_uy = torch.full_like(interior_rho, uy_far)
            target_uz = torch.full_like(interior_rho, uz_far)
        target_rho = torch.full_like(interior_rho, rho_far)
        target_eq = equilibrium(
            target_rho, target_ux, target_uy, target_uz, device=f.device,
        )
        directions = incoming.nonzero(as_tuple=False).flatten().tolist()
        for direction in directions:
            out[(direction,) + boundary_slice] = (
                target_eq[direction]
                + f[(direction,) + interior_slice]
                - local_eq[(direction,) + interior_slice]
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
    return out


__all__ = ["non_equilibrium_far_field_bc_3d"]
