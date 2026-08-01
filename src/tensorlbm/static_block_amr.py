"""Conservative fixed 2:1 block refinement runtime for 3-D LBM.

This module supplies the missing end-to-end mechanics between the existing
population-transfer helpers and a real solver loop:

* one strictly interior fine block with a one-cell ghost layer;
* convective 2:1 spatial/temporal refinement (two fine substeps);
* time-interpolated coarse data at the fine ghost layer;
* Filippova-Haenel-style non-equilibrium population rescaling;
* restriction of the fine-owned volume; and
* a population-wise reflux correction on the adjacent coarse shell.

The runtime is collision/boundary agnostic.  A caller provides an ``advance``
callback and may therefore compose D3Q19/D3Q27, MRT/cumulant, wall models and
problem-specific physical boundaries without teaching this module about them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .fixed_nested_transfer import restrict_populations_2to1
from .refinement import BoxRegion

Advance3D = Callable[[torch.Tensor, float, int, int], torch.Tensor]
_SUPPORTED_Q = {19: "D3Q19", 27: "D3Q27"}


def convective_refined_tau(tau_coarse: float, ratio: int = 2) -> float:
    """Relaxation time preserving physical viscosity under ``dt,dx -> /r``.

    With convective scaling, ``nu_lattice`` grows by ``ratio`` on the fine
    level, hence ``tau_f - 1/2 = r (tau_c - 1/2)``.
    """
    if ratio != 2:
        raise ValueError("the production runtime currently supports ratio=2")
    if not tau_coarse > 0.5:
        raise ValueError("tau_coarse must be greater than 0.5")
    return 0.5 + ratio * (tau_coarse - 0.5)


def _macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if f.shape[0] == 19:
        from .d3q19 import macroscopic3d
        return macroscopic3d(f)
    if f.shape[0] == 27:
        from .d3q27 import macroscopic27
        return macroscopic27(f)
    raise ValueError("only D3Q19 and D3Q27 are supported")


def _equilibrium(
    q: int, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    if q == 19:
        from .d3q19 import equilibrium3d
        return equilibrium3d(rho, ux, uy, uz, device=rho.device)
    if q == 27:
        from .d3q27 import equilibrium27
        return equilibrium27(rho, ux, uy, uz, device=rho.device)
    raise ValueError("only D3Q19 and D3Q27 are supported")


def _rescale_nonequilibrium(
    f: torch.Tensor, *, tau_source: float, tau_target: float,
    spatial_ratio: float,
) -> torch.Tensor:
    """Rescale non-equilibrium stress between convectively scaled levels."""
    rho, ux, uy, uz = _macroscopic(f)
    feq = _equilibrium(f.shape[0], rho, ux, uy, uz)
    # fneq is proportional to tau times the lattice velocity gradient.  A
    # fine lattice gradient is 1/r of the coarse gradient.
    scale = tau_target / (spatial_ratio * tau_source)
    return feq + scale * (f - feq)


@dataclass(frozen=True)
class StaticBlockAMRConfig:
    """Configuration for one strictly interior 2:1 fine block."""

    box: BoxRegion
    tau_coarse: float
    ratio: int = 2
    ghost: int = 1
    reflux: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.box, BoxRegion):
            raise TypeError("box must be a BoxRegion")
        convective_refined_tau(self.tau_coarse, self.ratio)
        if self.ghost != 1:
            raise ValueError("the production runtime currently supports ghost=1")

    @property
    def tau_fine(self) -> float:
        return convective_refined_tau(self.tau_coarse, self.ratio)


@dataclass(frozen=True)
class PopulationRefluxLedger:
    """Population-wise coarse/fine replacement and reflux accounting."""

    replacement_mismatch: torch.Tensor
    applied_shell_correction: torch.Tensor
    shell_cells: int
    residual: torch.Tensor

    @property
    def mass_residual(self) -> float:
        return float(self.residual.sum().item())


def _validate_parent_and_box(f: torch.Tensor, config: StaticBlockAMRConfig) -> None:
    if not isinstance(f, torch.Tensor) or f.ndim != 4:
        raise ValueError("coarse populations must have shape (Q,nz,ny,nx)")
    if f.shape[0] not in _SUPPORTED_Q:
        raise ValueError("only D3Q19 and D3Q27 are supported")
    if not f.is_floating_point():
        raise TypeError("coarse populations must be floating point")
    nz, ny, nx = f.shape[1:]
    b = config.box
    if not (
        0 < b.x0 < b.x1 < nx - 1
        and 0 < b.y0 < b.y1 < ny - 1
        and 0 < b.z0 < b.z1 < nz - 1
    ):
        raise ValueError("fine block must be strictly interior with a coarse-cell margin")


def _sample_parent_with_ghost(
    parent: torch.Tensor,
    config: StaticBlockAMRConfig,
) -> torch.Tensor:
    """Piecewise-constant parent sampling on fine physical+ghost coordinates."""
    b, r, g = config.box, config.ratio, config.ghost
    z_f = torch.arange(b.z0 * r - g, b.z1 * r + g, device=parent.device)
    y_f = torch.arange(b.y0 * r - g, b.y1 * r + g, device=parent.device)
    x_f = torch.arange(b.x0 * r - g, b.x1 * r + g, device=parent.device)
    z_c = torch.div(z_f, r, rounding_mode="floor")
    y_c = torch.div(y_f, r, rounding_mode="floor")
    x_c = torch.div(x_f, r, rounding_mode="floor")
    sampled = parent[:, z_c[:, None, None], y_c[None, :, None], x_c[None, None, :]]
    return _rescale_nonequilibrium(
        sampled,
        tau_source=config.tau_coarse,
        tau_target=config.tau_fine,
        spatial_ratio=float(r),
    )


def _coarse_shell_mask(
    shape: tuple[int, int, int], box: BoxRegion, device: torch.device,
) -> torch.Tensor:
    """One-cell coarse shell outside a box, including edges and corners."""
    nz, ny, nx = shape
    shell = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    shell[
        box.z0 - 1:box.z1 + 1,
        box.y0 - 1:box.y1 + 1,
        box.x0 - 1:box.x1 + 1,
    ] = True
    shell[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1] = False
    return shell


class StaticBlockAMR3D:
    """One coarse grid plus one fixed, fine-owned, 2:1 nested block."""

    def __init__(
        self,
        coarse_f: torch.Tensor,
        config: StaticBlockAMRConfig,
        *,
        fine_solid: torch.Tensor | None = None,
    ) -> None:
        _validate_parent_and_box(coarse_f, config)
        self.coarse_f = coarse_f
        self.config = config
        self.fine_f = _sample_parent_with_ghost(coarse_f, config)
        physical_shape = self.physical_fine_shape
        if fine_solid is not None:
            if fine_solid.shape != physical_shape or fine_solid.dtype is not torch.bool:
                raise ValueError("fine_solid must be bool with the physical fine shape")
            if fine_solid.device != coarse_f.device:
                raise ValueError("fine_solid and coarse populations must share a device")
            self.fine_solid = fine_solid
            g = config.ghost
            self.fine_solid_with_ghost = torch.zeros(
                tuple(size + 2 * g for size in physical_shape),
                dtype=torch.bool,
                device=fine_solid.device,
            )
            self.fine_solid_with_ghost[g:-g, g:-g, g:-g] = fine_solid
        else:
            self.fine_solid = None
            self.fine_solid_with_ghost = None
        self.last_reflux: PopulationRefluxLedger | None = None

    @property
    def physical_fine_shape(self) -> tuple[int, int, int]:
        b, r = self.config.box, self.config.ratio
        return (
            (b.z1 - b.z0) * r,
            (b.y1 - b.y0) * r,
            (b.x1 - b.x0) * r,
        )

    @property
    def fine_physical(self) -> torch.Tensor:
        g = self.config.ghost
        return self.fine_f[:, g:-g, g:-g, g:-g]

    @property
    def total_allocated_cells(self) -> int:
        return int(self.coarse_f[0].numel() + self.fine_f[0].numel())

    @property
    def uniform_fine_equivalent_cells(self) -> int:
        return int(self.coarse_f[0].numel() * self.config.ratio**3)

    @property
    def cell_saving_fraction(self) -> float:
        return 1.0 - self.total_allocated_cells / self.uniform_fine_equivalent_cells

    def _fill_ghost(self, parent_time_state: torch.Tensor) -> None:
        sampled = _sample_parent_with_ghost(parent_time_state, self.config)
        g = self.config.ghost
        ghost_mask = torch.ones(
            self.fine_f.shape[1:], dtype=torch.bool, device=self.fine_f.device,
        )
        ghost_mask[g:-g, g:-g, g:-g] = False
        self.fine_f[:, ghost_mask] = sampled[:, ghost_mask]

    def _restrict_physical(self) -> torch.Tensor:
        restricted = restrict_populations_2to1(self.fine_physical)
        return _rescale_nonequilibrium(
            restricted,
            tau_source=self.config.tau_fine,
            tau_target=self.config.tau_coarse,
            spatial_ratio=1.0 / self.config.ratio,
        )

    def _replace_and_reflux(self, restricted: torch.Tensor) -> PopulationRefluxLedger:
        b = self.config.box
        old_patch = self.coarse_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1]
        mismatch = old_patch.sum(dim=(1, 2, 3)) - restricted.sum(dim=(1, 2, 3))
        self.coarse_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = restricted

        correction = torch.zeros_like(mismatch)
        shell_cells = 0
        if self.config.reflux:
            shell = _coarse_shell_mask(self.coarse_f.shape[1:], b, self.coarse_f.device)
            shell_cells = int(shell.sum().item())
            correction = mismatch / shell_cells
            self.coarse_f[:, shell] += correction[:, None]
        residual = mismatch - correction * shell_cells
        return PopulationRefluxLedger(mismatch, correction, shell_cells, residual)

    def step(self, advance: Advance3D) -> PopulationRefluxLedger:
        """Advance one coarse step and two time-interpolated fine substeps.

        ``advance(f, tau, level, substep)`` must return a new population
        tensor with the same shape.  ``level`` is 0/1; the coarse call uses
        ``substep=-1`` and fine calls use 0 and 1.
        """
        coarse_old = self.coarse_f.clone()
        coarse_new = advance(self.coarse_f, self.config.tau_coarse, 0, -1)
        if coarse_new.shape != self.coarse_f.shape:
            raise ValueError("advance changed the coarse population shape")
        self.coarse_f = coarse_new

        for substep in range(self.config.ratio):
            alpha_start = substep / self.config.ratio
            parent_start = torch.lerp(coarse_old, self.coarse_f, alpha_start)
            self._fill_ghost(parent_start)
            fine_new = advance(
                self.fine_f, self.config.tau_fine, 1, substep,
            )
            if fine_new.shape != self.fine_f.shape:
                raise ValueError("advance changed the fine population shape")
            self.fine_f = fine_new
            alpha_end = (substep + 1) / self.config.ratio
            parent_end = torch.lerp(coarse_old, self.coarse_f, alpha_end)
            self._fill_ghost(parent_end)

        self.last_reflux = self._replace_and_reflux(self._restrict_physical())
        return self.last_reflux


__all__ = [
    "Advance3D",
    "PopulationRefluxLedger",
    "StaticBlockAMR3D",
    "StaticBlockAMRConfig",
    "convective_refined_tau",
]
