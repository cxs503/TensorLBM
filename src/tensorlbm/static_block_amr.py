"""Conservative fixed 2:1 block refinement runtime for 3-D LBM.

This module supplies the missing end-to-end mechanics between the existing
population-transfer helpers and a real solver loop:

* one strictly interior fine block with a one-cell ghost layer;
* convective 2:1 spatial/temporal refinement (two fine substeps);
* time-interpolated coarse data at the fine ghost layer;
* Filippova-Haenel-style non-equilibrium population rescaling;
* restriction of the fine-owned volume; and
* a link-local kinetic flux register and exterior-interface reflux correction.

The runtime is collision/boundary agnostic.  A caller provides an ``advance``
callback and may therefore compose D3Q19/D3Q27, MRT/cumulant, wall models and
problem-specific physical boundaries without teaching this module about them.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .fixed_nested_transfer import restrict_populations_2to1
from .kinetic_flux_register import (
    KineticInterfaceTransfer,
    apply_face_local_reflux,
    build_kinetic_interface_links,
    observe_kinetic_interface_transfer,
)
from .refinement import BoxRegion


@dataclass(frozen=True)
class AMRAdvanceResult:
    """One level update plus its post-collision/pre-stream populations."""

    populations: torch.Tensor
    post_collision: torch.Tensor


Advance3D = Callable[
    [torch.Tensor, float, int, int], torch.Tensor | AMRAdvanceResult,
]
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
    maximum_reflux_correction_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.box, BoxRegion):
            raise TypeError("box must be a BoxRegion")
        convective_refined_tau(self.tau_coarse, self.ratio)
        if self.ghost != 1:
            raise ValueError("the production runtime currently supports ghost=1")
        if not 0.0 < self.maximum_reflux_correction_fraction <= 1.0:
            raise ValueError(
                "maximum_reflux_correction_fraction must lie in (0,1]",
            )

    @property
    def tau_fine(self) -> float:
        return convective_refined_tau(self.tau_coarse, self.ratio)


@dataclass(frozen=True)
class PopulationRefluxLedger:
    """Population-wise coarse/fine replacement and reflux accounting.

    ``shell_cells`` is retained for result-schema compatibility and now means
    the number of corrected exterior interface links.  Corrections are never
    distributed over an unrelated enclosing shell.
    """

    replacement_mismatch: torch.Tensor
    applied_shell_correction: torch.Tensor
    shell_cells: int
    residual: torch.Tensor
    limited_directions: int = 0
    raw_kinetic_mismatch: torch.Tensor | None = None

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
        coarse_owned = torch.zeros(
            coarse_f.shape[1:], dtype=torch.bool, device=coarse_f.device,
        )
        b = config.box
        coarse_owned[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = True
        fine_owned = torch.zeros(
            self.fine_f.shape[1:], dtype=torch.bool, device=coarse_f.device,
        )
        g = config.ghost
        fine_owned[g:-g, g:-g, g:-g] = True
        self.coarse_interface_links = build_kinetic_interface_links(
            coarse_owned, q=coarse_f.shape[0],
        )
        self.fine_interface_links = build_kinetic_interface_links(
            fine_owned, q=coarse_f.shape[0],
        )
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

    def _replace_without_reflux(self, restricted: torch.Tensor) -> PopulationRefluxLedger:
        b = self.config.box
        old_patch = self.coarse_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1]
        mismatch = old_patch.sum(dim=(1, 2, 3)) - restricted.sum(dim=(1, 2, 3))
        self.coarse_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = restricted
        return PopulationRefluxLedger(
            mismatch, torch.zeros_like(mismatch), 0, mismatch, 0, mismatch,
        )

    @staticmethod
    def _unpack_advance(
        result: torch.Tensor | AMRAdvanceResult,
        expected_shape: torch.Size,
        *,
        require_flux_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(result, AMRAdvanceResult):
            populations, post_collision = result.populations, result.post_collision
            if post_collision.shape != expected_shape:
                raise ValueError("post_collision changed the population shape")
        else:
            populations, post_collision = result, None
        if populations.shape != expected_shape:
            raise ValueError("advance changed the population shape")
        if require_flux_state and post_collision is None:
            raise TypeError(
                "reflux-enabled AMR advance must return AMRAdvanceResult with "
                "post-collision/pre-stream populations"
            )
        return populations, post_collision

    def step(self, advance: Advance3D) -> PopulationRefluxLedger:
        """Advance one coarse step and two time-interpolated fine substeps.

        With reflux enabled, ``advance(f, tau, level, substep)`` must return
        :class:`AMRAdvanceResult`, including the state after collision and
        before streaming.  A raw tensor remains accepted only when reflux is
        disabled. ``level`` is 0/1; the coarse call uses ``substep=-1`` and
        fine calls use 0 and 1.
        """
        coarse_old = self.coarse_f.clone()
        coarse_new, coarse_post = self._unpack_advance(
            advance(self.coarse_f, self.config.tau_coarse, 0, -1),
            self.coarse_f.shape, require_flux_state=self.config.reflux,
        )
        self.coarse_f = coarse_new
        coarse_transfer = (
            observe_kinetic_interface_transfer(
                coarse_post, self.coarse_interface_links,
            )
            if coarse_post is not None else None
        )
        fine_transfer: KineticInterfaceTransfer | None = None

        for substep in range(self.config.ratio):
            alpha_start = substep / self.config.ratio
            parent_start = torch.lerp(coarse_old, self.coarse_f, alpha_start)
            self._fill_ghost(parent_start)
            fine_new, fine_post = self._unpack_advance(
                advance(self.fine_f, self.config.tau_fine, 1, substep),
                self.fine_f.shape, require_flux_state=self.config.reflux,
            )
            if fine_post is not None:
                observed = observe_kinetic_interface_transfer(
                    fine_post, self.fine_interface_links,
                    cell_volume=1.0 / self.config.ratio**3,
                )
                fine_transfer = observed if fine_transfer is None else fine_transfer + observed
            self.fine_f = fine_new
            alpha_end = (substep + 1) / self.config.ratio
            parent_end = torch.lerp(coarse_old, self.coarse_f, alpha_end)
            self._fill_ghost(parent_end)

        restricted = self._restrict_physical()
        if not self.config.reflux:
            self.last_reflux = self._replace_without_reflux(restricted)
            return self.last_reflux
        if coarse_transfer is None or fine_transfer is None:
            raise RuntimeError("missing interface transfer for reflux")
        b = self.config.box
        self.coarse_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = restricted
        self.coarse_f, report = apply_face_local_reflux(
            self.coarse_f, self.coarse_interface_links,
            coarse_transfer, fine_transfer,
            maximum_correction_fraction=(
                self.config.maximum_reflux_correction_fraction
            ),
        )
        self.last_reflux = PopulationRefluxLedger(
            report.requested_inventory_correction,
            report.applied_inventory_correction,
            report.corrected_links,
            report.residual,
            report.limited_directions,
            report.raw_kinetic_mismatch,
        )
        return self.last_reflux


__all__ = [
    "Advance3D",
    "AMRAdvanceResult",
    "PopulationRefluxLedger",
    "StaticBlockAMR3D",
    "StaticBlockAMRConfig",
    "convective_refined_tau",
]
