"""Moment-preserving positivity limiter for lattice populations."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PositivityDiagnostics:
    limited_cells: int
    total_cells: int
    limited_fraction: float
    minimum_alpha: float
    minimum_population_before: float
    minimum_population_after: float


def limit_nonequilibrium_for_positivity(
    f: torch.Tensor,
    *,
    floor: float = 1e-9,
) -> tuple[torch.Tensor, PositivityDiagnostics]:
    """Blend toward the local equilibrium without changing conserved moments.

    ``f_limited = f_eq + alpha (f-f_eq)``, where the largest cell-wise
    ``alpha <= 1`` satisfying every ``f_i >= floor`` is selected.  Since the
    non-equilibrium part has zero mass and momentum, the operation preserves
    the hydrodynamic moments up to floating-point roundoff.
    """
    if f.ndim != 4 or f.shape[0] not in (19, 27):
        raise ValueError("f must have shape (19|27,nz,ny,nx)")
    if floor < 0.0:
        raise ValueError("floor must be non-negative")
    if f.shape[0] == 19:
        from .d3q19 import equilibrium3d, macroscopic3d
        equilibrium, macro = equilibrium3d, macroscopic3d
    else:
        from .d3q27 import equilibrium27, macroscopic27
        equilibrium, macro = equilibrium27, macroscopic27
    minimum_before = float(f.min().item())
    total = int(f[0].numel())
    if minimum_before >= floor:
        return f, PositivityDiagnostics(
            limited_cells=0,
            total_cells=total,
            limited_fraction=0.0,
            minimum_alpha=1.0,
            minimum_population_before=minimum_before,
            minimum_population_after=minimum_before,
        )
    rho, ux, uy, uz = macro(f)
    feq = equilibrium(rho, ux, uy, uz, device=f.device)
    below = f < floor
    denominator = feq - f
    candidate = torch.where(
        below,
        (feq - floor) / denominator.clamp_min(1e-30),
        torch.ones_like(f),
    )
    alpha = candidate.amin(dim=0).clamp(0.0, 1.0)
    limited = alpha < 1.0
    out = feq + alpha.unsqueeze(0) * (f - feq)
    # A final maximum only absorbs last-bit roundoff and is not the limiter.
    out = out.clamp_min(floor)
    limited_cells = int(limited.sum().item())
    total = int(limited.numel())
    diagnostics = PositivityDiagnostics(
        limited_cells=limited_cells,
        total_cells=total,
        limited_fraction=limited_cells / total,
        minimum_alpha=float(alpha.min().item()),
        minimum_population_before=minimum_before,
        minimum_population_after=float(out.min().item()),
    )
    return out, diagnostics


__all__ = ["PositivityDiagnostics", "limit_nonequilibrium_for_positivity"]
