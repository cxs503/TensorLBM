"""Common AMR module — solver-agnostic refine/coarsen/halo exchange.

This module extracts the AMR patch mechanics from
:mod:`tensorlbm.adaptive_refinement` into a **common, solver-agnostic**
interface that can be combined with any collision operator or turbulence
model.  The key design decisions are:

* ``AMRPatch3D`` is a pure data container — it holds a distribution tensor,
  a bounding box, refinement ratio, VR level, and a lattice name.  It does
  not reference any solver.
* ``refine`` / ``coarsen`` are free functions that take a distribution
  tensor and return the refined/coarsened tensor.  They dispatch to the
  correct equilibrium/macroscopic functions based on the lattice name,
  enabling Filippova–Hänel (FH) second-order interface exchange for both
  D3Q19 and D3Q27.
* ``halo_exchange`` copies upsampled parent-level data into a patch's
  border cells, leaving the interior untouched.

Supported lattices: ``D3Q19``, ``D3Q27``.

The module does **not** modify any solver hot path.  It only provides
reusable AMR mechanics that a solver may call from its own adaptation
schedule.

References
----------
Lagrava D., Malaspinas O., Latt J., Chopard B. (2012)
    Advances in multi-domain lattice Boltzmann grid refinement.
    J. Comput. Phys. 231, 4808–4822.
Filippova O., Hänel D. (1998)
    Grid refinement for lattice-BGK models.
    J. Comput. Phys. 147, 219–228.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .refinement import BoxRegion, _coarse_to_fine_3d, _fine_to_coarse_3d

SUPPORTED_LATTICES: tuple[str, ...] = ("D3Q19", "D3Q27")


def _validate_lattice(lattice: str) -> str:
    """Return *lattice* if supported, else raise ValueError."""
    if lattice not in SUPPORTED_LATTICES:
        raise ValueError(
            f"Unsupported lattice {lattice!r}; supported: {SUPPORTED_LATTICES}"
        )
    return lattice


# ---------------------------------------------------------------------------
# Lattice dispatch helpers
# ---------------------------------------------------------------------------

def _macroscopic(lattice: str, f: torch.Tensor):
    """Dispatch to the correct macroscopic function for *lattice*."""
    if lattice == "D3Q19":
        from .d3q19 import macroscopic3d
        return macroscopic3d(f)
    elif lattice == "D3Q27":
        from .d3q27 import macroscopic27
        return macroscopic27(f)
    raise ValueError(f"Unsupported lattice: {lattice!r}")


def _equilibrium(lattice: str, rho, ux, uy, uz, *, device=None) -> torch.Tensor:
    """Dispatch to the correct equilibrium function for *lattice*."""
    if lattice == "D3Q19":
        from .d3q19 import equilibrium3d
        return equilibrium3d(rho, ux, uy, uz, device=device)
    elif lattice == "D3Q27":
        from .d3q27 import equilibrium27
        return equilibrium27(rho, ux, uy, uz, device=device)
    raise ValueError(f"Unsupported lattice: {lattice!r}")


# ---------------------------------------------------------------------------
# FH (Filippova–Hänel) coarse-to-fine / fine-to-coarse — lattice-dispatched
# ---------------------------------------------------------------------------

def _fh_coarse_to_fine_3d(
    f_coarse: torch.Tensor,
    lattice: str,
    tau_c: float,
    tau_f: float,
    ratio: int = 2,
) -> torch.Tensor:
    """Filippova–Hänel 2nd-order upsampling for 3-D lattices.

    Splits the distribution into equilibrium and non-equilibrium parts.
    The non-equilibrium part is rescaled by ``τ_f / τ_c`` to account for
    the changed relaxation time on the finer grid before interpolation.
    """
    rho, ux, uy, uz = _macroscopic(lattice, f_coarse)
    f_eq = _equilibrium(lattice, rho, ux, uy, uz, device=f_coarse.device)
    f_neq = f_coarse - f_eq

    scale = tau_f / tau_c
    f_rescaled = f_eq + scale * f_neq

    q, nz_c, ny_c, nx_c = f_rescaled.shape
    out = F.interpolate(
        f_rescaled.unsqueeze(0),
        size=(nz_c * ratio, ny_c * ratio, nx_c * ratio),
        mode="trilinear",
        align_corners=True,
    )
    return out.squeeze(0)


def _fh_fine_to_coarse_3d(
    f_fine: torch.Tensor,
    lattice: str,
    tau_f: float,
    tau_c: float,
    ratio: int = 2,
) -> torch.Tensor:
    """Filippova–Hänel 2nd-order restriction for 3-D lattices."""
    f_avg = _fine_to_coarse_3d(f_fine, ratio)
    rho, ux, uy, uz = _macroscopic(lattice, f_avg)
    f_eq = _equilibrium(lattice, rho, ux, uy, uz, device=f_avg.device)
    f_neq = f_avg - f_eq
    scale = tau_c / tau_f
    return f_eq + scale * f_neq


# ---------------------------------------------------------------------------
# Public refine / coarsen operations
# ---------------------------------------------------------------------------

def refine(
    f_coarse: torch.Tensor,
    *,
    lattice: str = "D3Q19",
    tau_c: float = 1.0,
    tau_f: float = 0.75,
    ratio: int = 2,
    use_fh: bool = True,
) -> torch.Tensor:
    """Refine a coarse distribution to a finer grid.

    This is a **solver-agnostic** operation: it takes a distribution tensor
    and returns the upsampled distribution.  When ``use_fh=True`` (default),
    the Filippova–Hänel second-order scheme is used, which rescales the
    non-equilibrium part by the relaxation-time ratio.  When ``use_fh=False``,
    plain trilinear interpolation is used.

    Args:
        f_coarse: Coarse distribution, shape ``(Q, nz, ny, nx)``.
        lattice:  Lattice name (``"D3Q19"`` or ``"D3Q27"``).
        tau_c:    Relaxation time on the coarse grid.
        tau_f:    Relaxation time on the fine grid.
        ratio:    Spatial refinement ratio (default 2).
        use_fh:   If True, use FH second-order exchange.

    Returns:
        Refined distribution, shape ``(Q, nz*r, ny*r, nx*r)``.
    """
    _validate_lattice(lattice)
    if use_fh:
        return _fh_coarse_to_fine_3d(f_coarse, lattice, tau_c, tau_f, ratio)
    return _coarse_to_fine_3d(f_coarse, ratio)


def coarsen(
    f_fine: torch.Tensor,
    *,
    lattice: str = "D3Q19",
    tau_f: float = 0.75,
    tau_c: float = 1.0,
    ratio: int = 2,
    use_fh: bool = True,
) -> torch.Tensor:
    """Coarsen a fine distribution to a coarser grid.

    This is a **solver-agnostic** operation: it takes a distribution tensor
    and returns the restricted (averaged) distribution.  When ``use_fh=True``
    (default), the Filippova–Hänel second-order scheme is used, which
    rescales the non-equilibrium part back to the coarser relaxation time.

    Args:
        f_fine:  Fine distribution, shape ``(Q, nz*r, ny*r, nx*r)``.
        lattice: Lattice name (``"D3Q19"`` or ``"D3Q27"``).
        tau_f:   Relaxation time on the fine grid.
        tau_c:   Relaxation time on the coarse grid.
        ratio:   Spatial refinement ratio (default 2).
        use_fh:  If True, use FH second-order exchange.

    Returns:
        Coarsened distribution, shape ``(Q, nz, ny, nx)``.
    """
    _validate_lattice(lattice)
    if use_fh:
        return _fh_fine_to_coarse_3d(f_fine, lattice, tau_f, tau_c, ratio)
    return _fine_to_coarse_3d(f_fine, ratio)


# ---------------------------------------------------------------------------
# Halo exchange between patches
# ---------------------------------------------------------------------------

def halo_exchange(
    patch_f: torch.Tensor,
    parent_f: torch.Tensor,
    *,
    box: BoxRegion,
    ratio: int,
    lattice: str = "D3Q19",
    tau_p: float = 1.0,
    tau_c: float = 0.75,
    use_fh: bool = True,
) -> None:
    """Overwrite fine-level border cells with upsampled parent values.

    This is an **in-place** operation on *patch_f*.  The interior cells
    (excluding a border of width *ratio*) are left untouched.  Only the
    border cells are overwritten with values upsampled from the parent-level
    distribution.

    Args:
        patch_f:  Fine-level distribution (modified in place).
        parent_f: Parent-level distribution.
        box:      Bounding box in parent-level grid coordinates.
        ratio:    Refinement ratio.
        lattice:  Lattice name (``"D3Q19"`` or ``"D3Q27"``).
        tau_p:    Relaxation time on the parent level.
        tau_c:    Relaxation time on the child (fine) level.
        use_fh:   If True, use FH second-order exchange.
    """
    _validate_lattice(lattice)
    b = box
    r = ratio
    f_parent_patch = parent_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1]
    if use_fh:
        f_up = _fh_coarse_to_fine_3d(f_parent_patch, lattice, tau_p, tau_c, r)
    else:
        f_up = _coarse_to_fine_3d(f_parent_patch, r)

    nz_f, ny_f, nx_f = patch_f.shape[1], patch_f.shape[2], patch_f.shape[3]
    border = torch.ones((nz_f, ny_f, nx_f), dtype=torch.bool, device=patch_f.device)
    border[r:-r, r:-r, r:-r] = False

    fz = min(f_up.shape[1], nz_f)
    fy = min(f_up.shape[2], ny_f)
    fx = min(f_up.shape[3], nx_f)
    patch_f[:, border[:fz, :fy, :fx]] = f_up[:, border[:fz, :fy, :fx]]


# ---------------------------------------------------------------------------
# AMRPatch3D — public data container
# ---------------------------------------------------------------------------

@dataclass
class AMRPatch3D:
    """A dynamically managed fine-resolution 3-D patch.

    This is a **pure data container** — it holds a distribution tensor and
    metadata but does not reference any solver.  It can be combined with
    any collision operator or turbulence model.

    Attributes:
        f:            Distribution tensor ``(Q, nz_f, ny_f, nx_f)``.
        box:          Bounding box in **parent-level** grid coordinates.
        ratio:        Refinement ratio relative to the parent level (default 2).
        level:        VR level index (0 = coarsest, 1 = first fine level, …).
        parent_level: Index of the parent level (``level - 1``).
        tau:          Relaxation time for this level's collisions.
        lattice:      Lattice name (``"D3Q19"`` or ``"D3Q27"``).
    """

    f: torch.Tensor
    box: BoxRegion
    ratio: int = 2
    level: int = 1
    parent_level: int = 0
    tau: float = 1.0
    lattice: str = "D3Q19"

    def __post_init__(self) -> None:
        _validate_lattice(self.lattice)

    @property
    def nz(self) -> int:
        return self.f.shape[1]

    @property
    def ny(self) -> int:
        return self.f.shape[2]

    @property
    def nx(self) -> int:
        return self.f.shape[3]

    @property
    def cells(self) -> int:
        """Total cell count in this patch."""
        return self.nz * self.ny * self.nx


__all__ = [
    "AMRPatch3D",
    "SUPPORTED_LATTICES",
    "refine",
    "coarsen",
    "halo_exchange",
]
