"""Conservative, optionally regularized population rescaling for LBM AMR."""

from __future__ import annotations

import math

import torch


def _lattice(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if f.shape[0] == 19:
        from .d3q19 import C, W
    elif f.shape[0] == 27:
        from .d3q27 import C, W
    else:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    return C.to(device=f.device, dtype=f.dtype), W.to(device=f.device, dtype=f.dtype)


def _macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if f.shape[0] == 19:
        from .d3q19 import macroscopic3d

        return macroscopic3d(f)
    if f.shape[0] == 27:
        from .d3q27 import macroscopic27

        return macroscopic27(f)
    raise ValueError("only D3Q19 and D3Q27 are supported")


def _equilibrium(
    q: int,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    if q == 19:
        from .d3q19 import equilibrium3d

        return equilibrium3d(rho, ux, uy, uz, device=rho.device)
    if q == 27:
        from .d3q27 import equilibrium27

        return equilibrium27(rho, ux, uy, uz, device=rho.device)
    raise ValueError("only D3Q19 and D3Q27 are supported")


def _remove_conserved_roundoff(f_neq: torch.Tensor) -> torch.Tensor:
    """Make the zeroth and first moments exactly zero to working precision."""
    c, _ = _lattice(f_neq)
    result = f_neq.clone()
    result[0] -= result.sum(dim=0)
    momentum = (result * c[:, 0, None, None, None]).sum(dim=0)
    result[1] -= 0.5 * momentum
    result[2] += 0.5 * momentum
    momentum = (result * c[:, 1, None, None, None]).sum(dim=0)
    result[3] -= 0.5 * momentum
    result[4] += 0.5 * momentum
    momentum = (result * c[:, 2, None, None, None]).sum(dim=0)
    result[5] -= 0.5 * momentum
    result[6] += 0.5 * momentum
    return result


def regularize_nonequilibrium_second_order(f_neq: torch.Tensor) -> torch.Tensor:
    """Project non-equilibrium populations onto the six stress moments.

    Grid transfer should not amplify non-hydrodynamic ghost modes.  This is
    the second-order Hermite regularization used by mature multi-domain LBM
    coupling schemes; it retains the full symmetric viscous stress tensor and
    removes higher-order kinetic content before coarse/fine rescaling.
    """
    if not isinstance(f_neq, torch.Tensor) or f_neq.ndim != 4:
        raise ValueError("f_neq must have shape (19|27,nz,ny,nx)")
    if not f_neq.is_floating_point():
        raise TypeError("f_neq must be floating point")
    c, w = _lattice(f_neq)
    q = f_neq.shape[0]
    cx = c[:, 0, None, None, None]
    cy = c[:, 1, None, None, None]
    cz = c[:, 2, None, None, None]
    pi_xx = (cx.square() * f_neq).sum(dim=0)
    pi_yy = (cy.square() * f_neq).sum(dim=0)
    pi_zz = (cz.square() * f_neq).sum(dim=0)
    pi_xy = (cx * cy * f_neq).sum(dim=0)
    pi_xz = (cx * cz * f_neq).sum(dim=0)
    pi_yz = (cy * cz * f_neq).sum(dim=0)
    cs2 = 1.0 / 3.0
    projected = (
        4.5
        * w.view(q, 1, 1, 1)
        * (
            (cx.square() - cs2) * pi_xx
            + (cy.square() - cs2) * pi_yy
            + (cz.square() - cs2) * pi_zz
            + 2.0 * cx * cy * pi_xy
            + 2.0 * cx * cz * pi_xz
            + 2.0 * cy * cz * pi_yz
        )
    )
    return _remove_conserved_roundoff(projected)


def rescale_nonequilibrium(
    f: torch.Tensor,
    *,
    tau_source: float,
    tau_target: float,
    spatial_ratio: float,
    regularize: bool = False,
) -> torch.Tensor:
    """Rescale stress under convective grid/time scaling.

    ``spatial_ratio`` is target spacing divided by source spacing.  The
    Chapman--Enskog non-equilibrium amplitude therefore scales as
    ``tau_target / (spatial_ratio * tau_source)``.  With ``regularize=True``,
    only the resolved second-order stress is transferred.
    """
    if not isinstance(f, torch.Tensor) or f.ndim != 4 or f.shape[0] not in (19, 27):
        raise ValueError("f must have shape (19|27,nz,ny,nx)")
    if not f.is_floating_point():
        raise TypeError("f must be floating point")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (
            tau_source,
            tau_target,
            spatial_ratio,
        )
    ):
        raise ValueError("tau values and spatial_ratio must be finite and positive")
    rho, ux, uy, uz = _macroscopic(f)
    equilibrium = _equilibrium(f.shape[0], rho, ux, uy, uz)
    non_equilibrium = f - equilibrium
    if regularize:
        non_equilibrium = regularize_nonequilibrium_second_order(non_equilibrium)
    scale = tau_target / (spatial_ratio * tau_source)
    return equilibrium + scale * non_equilibrium


__all__ = ["regularize_nonequilibrium_second_order", "rescale_nonequilibrium"]
