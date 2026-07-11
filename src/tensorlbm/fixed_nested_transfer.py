"""Conservative population transfer for a fixed 2:1 nested LBM grid.

This module deliberately provides only cell-centred population transfer.  It
has no patch management, ghost-layer policy, temporal interpolation, collision
coupling, or streaming/interface treatment.  In particular, it is *not* an
AMR implementation.

For a coarse cell of volume ``V_c`` and its eight fine children of volume
``V_f = V_c / 8``, restriction is the population-wise arithmetic mean.  Thus
it preserves the cell-integrated zeroth and first kinetic moments exactly:
``V_c sum_i f_i^c = sum_children V_f sum_i f_i^f`` and likewise for
``sum_i c_i f_i``.  Prolongation is piecewise-constant injection, the exact
right inverse of this restriction and therefore preserves uniform equilibria.
"""
from __future__ import annotations

import torch

_SUPPORTED_STENCILS = frozenset((19, 27))
_RATIO = 2


def _validate_populations(f: torch.Tensor, *, fine: bool) -> None:
    """Validate the common D3Q19/D3Q27 population layout."""
    kind = "fine" if fine else "coarse"
    if not isinstance(f, torch.Tensor):
        raise TypeError(f"{kind} populations must be a torch.Tensor")
    if f.ndim != 4:
        raise ValueError(f"{kind} populations must have shape (Q, nz, ny, nx)")
    if f.shape[0] not in _SUPPORTED_STENCILS:
        raise ValueError("only D3Q19 and D3Q27 populations are supported")
    if not f.is_floating_point():
        raise TypeError(f"{kind} populations must have a floating-point dtype")
    if fine and any(size % _RATIO for size in f.shape[1:]):
        raise ValueError("fine spatial dimensions must each be divisible by 2")


def prolongate_populations_2to1(f_coarse: torch.Tensor) -> torch.Tensor:
    """Piecewise-constantly inject D3Q19/D3Q27 populations onto a 2:1 grid.

    Each coarse population is copied to its eight children.  Accordingly, the
    fine-grid *integrated* mass and momentum are eight times coarse-grid sums,
    exactly matching the eight-fold change in number of unit-volume cells.
    The operation exactly preserves any spatially uniform (including moving)
    equilibrium and satisfies
    ``restrict_populations_2to1(prolongate_populations_2to1(f)) == f``.
    """
    _validate_populations(f_coarse, fine=False)
    return (
        f_coarse.repeat_interleave(_RATIO, dim=1)
        .repeat_interleave(_RATIO, dim=2)
        .repeat_interleave(_RATIO, dim=3)
    )


def restrict_populations_2to1(f_fine: torch.Tensor) -> torch.Tensor:
    """Conservatively restrict D3Q19/D3Q27 populations from a 2:1 grid.

    Restriction averages each population over every 2×2×2 child block.  When
    the fine-cell volume is one eighth of the coarse-cell volume, this exactly
    preserves the volume-weighted total mass and all three momentum components
    for arbitrary populations, without requiring an equilibrium reconstruction.
    """
    _validate_populations(f_fine, fine=True)
    q, nz_f, ny_f, nx_f = f_fine.shape
    return f_fine.reshape(q, nz_f // 2, 2, ny_f // 2, 2, nx_f // 2, 2).mean(dim=(2, 4, 6))


__all__ = ["prolongate_populations_2to1", "restrict_populations_2to1"]
