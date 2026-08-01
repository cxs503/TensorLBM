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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .amr_interface_filter import (
    damp_interface_nonequilibrium,
    interface_shell_blend,
)
from .amr_population_transfer import rescale_nonequilibrium
from .fixed_nested_transfer import restrict_populations_2to1
from .kinetic_flux_register import (
    KineticInterfaceTransfer,
    apply_face_local_reflux,
    build_kinetic_interface_links,
    observe_kinetic_interface_transfer,
)
from .population_positivity import (
    PositivityDiagnostics,
    limit_nonequilibrium_for_positivity,
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


@dataclass(frozen=True)
class StaticBlockAMRConfig:
    """Configuration for one strictly interior 2:1 fine block."""

    box: BoxRegion
    tau_coarse: float
    ratio: int = 2
    ghost: int = 1
    reflux: bool = True
    maximum_reflux_correction_fraction: float = 0.2
    reflux_correction_stencil: str = "exterior_cells"
    regularize_restriction: bool = False
    regularize_prolongation: bool = False
    ghost_interpolation: str = "injection"
    enforce_transfer_positivity: bool = False
    interface_filter_width: int = 0
    interface_filter_strength: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.box, BoxRegion):
            raise TypeError("box must be a BoxRegion")
        convective_refined_tau(self.tau_coarse, self.ratio)
        if self.ghost != 1:
            raise ValueError("the production runtime currently supports ghost=1")
        if self.ghost_interpolation not in ("injection", "trilinear"):
            raise ValueError("ghost_interpolation must be injection or trilinear")
        if not 0.0 < self.maximum_reflux_correction_fraction <= 1.0:
            raise ValueError(
                "maximum_reflux_correction_fraction must lie in (0,1]",
            )
        if self.reflux_correction_stencil not in (
            "exterior_cells", "crossing_links",
        ):
            raise ValueError(
                "reflux_correction_stencil must be exterior_cells or crossing_links",
            )
        if self.interface_filter_width < 0:
            raise ValueError("interface_filter_width must be non-negative")
        if not 0.0 <= self.interface_filter_strength <= 1.0:
            raise ValueError("interface_filter_strength must lie in [0,1]")
        if (self.interface_filter_width == 0) != (
            self.interface_filter_strength == 0.0
        ):
            raise ValueError(
                "interface filter width and strength must both be zero or positive",
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
    restriction_limited_fraction: float = 0.0
    restriction_minimum_alpha: float = 1.0
    prolongation_limited_fraction: float = 0.0
    prolongation_minimum_alpha: float = 1.0

    @property
    def mass_residual(self) -> float:
        return float(self.residual.sum().item())


def _merge_reflux_ledgers(
    previous: PopulationRefluxLedger,
    current: PopulationRefluxLedger,
) -> PopulationRefluxLedger:
    """Accumulate repeated child-interface advances over one root step."""
    raw = None
    if previous.raw_kinetic_mismatch is not None:
        if current.raw_kinetic_mismatch is None:
            raise RuntimeError("nested reflux ledger lost raw kinetic mismatch")
        raw = previous.raw_kinetic_mismatch + current.raw_kinetic_mismatch
    elif current.raw_kinetic_mismatch is not None:
        raw = current.raw_kinetic_mismatch
    return PopulationRefluxLedger(
        replacement_mismatch=(
            previous.replacement_mismatch + current.replacement_mismatch
        ),
        applied_shell_correction=(
            previous.applied_shell_correction
            + current.applied_shell_correction
        ),
        shell_cells=previous.shell_cells + current.shell_cells,
        residual=previous.residual + current.residual,
        limited_directions=(
            previous.limited_directions + current.limited_directions
        ),
        raw_kinetic_mismatch=raw,
        restriction_limited_fraction=max(
            previous.restriction_limited_fraction,
            current.restriction_limited_fraction,
        ),
        restriction_minimum_alpha=min(
            previous.restriction_minimum_alpha,
            current.restriction_minimum_alpha,
        ),
        prolongation_limited_fraction=max(
            previous.prolongation_limited_fraction,
            current.prolongation_limited_fraction,
        ),
        prolongation_minimum_alpha=min(
            previous.prolongation_minimum_alpha,
            current.prolongation_minimum_alpha,
        ),
    )


@dataclass(frozen=True)
class _GhostSamplingPlan:
    target_flat: torch.Tensor
    z0: torch.Tensor
    y0: torch.Tensor
    x0: torch.Tensor
    z1: torch.Tensor | None = None
    y1: torch.Tensor | None = None
    x1: torch.Tensor | None = None
    wz: torch.Tensor | None = None
    wy: torch.Tensor | None = None
    wx: torch.Tensor | None = None


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
    return rescale_nonequilibrium(
        sampled,
        tau_source=config.tau_coarse,
        tau_target=config.tau_fine,
        spatial_ratio=float(r),
        regularize=config.regularize_prolongation,
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
        self.last_prolongation_positivity: PositivityDiagnostics | None = None
        self._maximum_prolongation_limited_fraction = 0.0
        self._minimum_prolongation_alpha = 1.0
        if config.enforce_transfer_positivity:
            self.fine_f, diagnostic = limit_nonequilibrium_for_positivity(
                self.fine_f,
            )
            self._record_prolongation_positivity(diagnostic)
        self._interface_filter_blend = interface_shell_blend(
            self.fine_f.shape[1:],
            ghost=config.ghost,
            width=config.interface_filter_width,
            strength=config.interface_filter_strength,
            device=self.fine_f.device,
            dtype=self.fine_f.dtype,
        )
        self._ghost_sampling_plan = self._build_ghost_sampling_plan()
        self.last_restriction_positivity: PositivityDiagnostics | None = None
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
            protected = self.fine_solid_with_ghost.clone()
            protected[1:] |= self.fine_solid_with_ghost[:-1]
            protected[:-1] |= self.fine_solid_with_ghost[1:]
            protected[:, 1:] |= self.fine_solid_with_ghost[:, :-1]
            protected[:, :-1] |= self.fine_solid_with_ghost[:, 1:]
            protected[:, :, 1:] |= self.fine_solid_with_ghost[:, :, :-1]
            protected[:, :, :-1] |= self.fine_solid_with_ghost[:, :, 1:]
            if bool((protected & (self._interface_filter_blend > 0.0)).any()):
                raise ValueError(
                    "interface filter overlaps the solid or its near-wall fluid",
                )
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

    def _record_prolongation_positivity(
        self,
        diagnostic: PositivityDiagnostics,
    ) -> None:
        if not all(math.isfinite(value) for value in (
            diagnostic.minimum_population_before,
            diagnostic.minimum_population_after,
            diagnostic.minimum_alpha,
        )):
            raise FloatingPointError("non-finite coarse-to-fine AMR prolongation")
        self.last_prolongation_positivity = diagnostic
        self._maximum_prolongation_limited_fraction = max(
            self._maximum_prolongation_limited_fraction,
            diagnostic.limited_fraction,
        )
        self._minimum_prolongation_alpha = min(
            self._minimum_prolongation_alpha,
            diagnostic.minimum_alpha,
        )

    def _reset_prolongation_positivity(self) -> None:
        self.last_prolongation_positivity = None
        self._maximum_prolongation_limited_fraction = 0.0
        self._minimum_prolongation_alpha = 1.0

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

    def _build_ghost_sampling_plan(self) -> _GhostSamplingPlan:
        """Cache the non-overlapping one-cell shell and its parent donors."""
        _, nz, ny, nx = self.fine_f.shape
        device = self.fine_f.device
        regions = (
            (
                torch.tensor((0, nz - 1), device=device),
                torch.arange(ny, device=device),
                torch.arange(nx, device=device),
            ),
            (
                torch.arange(1, nz - 1, device=device),
                torch.tensor((0, ny - 1), device=device),
                torch.arange(nx, device=device),
            ),
            (
                torch.arange(1, nz - 1, device=device),
                torch.arange(1, ny - 1, device=device),
                torch.tensor((0, nx - 1), device=device),
            ),
        )
        coordinates = [
            torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=-1).reshape(-1, 3)
            for z, y, x in regions
        ]
        local = torch.cat(coordinates, dim=0)
        target_flat = (local[:, 0] * ny + local[:, 1]) * nx + local[:, 2]
        b, r, g = self.config.box, self.config.ratio, self.config.ghost
        global_fine = torch.stack((
            b.z0 * r - g + local[:, 0],
            b.y0 * r - g + local[:, 1],
            b.x0 * r - g + local[:, 2],
        ), dim=1)
        if self.config.ghost_interpolation == "injection":
            donor = torch.div(global_fine, r, rounding_mode="floor")
            return _GhostSamplingPlan(
                target_flat, donor[:, 0], donor[:, 1], donor[:, 2],
            )

        # Coarse cell centres are at i+1/2.  Express a fine cell centre in
        # the corresponding coarse-index coordinate before linear blending.
        continuous = (global_fine.to(self.fine_f.dtype) + 0.5) / r - 0.5
        lower = torch.floor(continuous).to(torch.long)
        weight = continuous - lower.to(continuous.dtype)
        upper = lower + 1
        return _GhostSamplingPlan(
            target_flat,
            lower[:, 0], lower[:, 1], lower[:, 2],
            upper[:, 0], upper[:, 1], upper[:, 2],
            weight[:, 0], weight[:, 1], weight[:, 2],
        )

    def _fill_ghost(
        self,
        parent_time_state: torch.Tensor,
        *,
        tau_source: float | None = None,
        tau_target: float | None = None,
    ) -> None:
        plan = self._ghost_sampling_plan
        if self.config.ghost_interpolation == "injection":
            sampled = parent_time_state[:, plan.z0, plan.y0, plan.x0]
        else:
            assert all(value is not None for value in (
                plan.z1, plan.y1, plan.x1, plan.wz, plan.wy, plan.wx,
            ))
            z1, y1, x1 = plan.z1, plan.y1, plan.x1
            wz, wy, wx = plan.wz, plan.wy, plan.wx
            assert z1 is not None and y1 is not None and x1 is not None
            assert wz is not None and wy is not None and wx is not None
            wx = wx.unsqueeze(0)
            wy = wy.unsqueeze(0)
            wz = wz.unsqueeze(0)
            v00 = torch.lerp(
                parent_time_state[:, plan.z0, plan.y0, plan.x0],
                parent_time_state[:, plan.z0, plan.y0, x1], wx,
            )
            v01 = torch.lerp(
                parent_time_state[:, plan.z0, y1, plan.x0],
                parent_time_state[:, plan.z0, y1, x1], wx,
            )
            v10 = torch.lerp(
                parent_time_state[:, z1, plan.y0, plan.x0],
                parent_time_state[:, z1, plan.y0, x1], wx,
            )
            v11 = torch.lerp(
                parent_time_state[:, z1, y1, plan.x0],
                parent_time_state[:, z1, y1, x1], wx,
            )
            sampled = torch.lerp(
                torch.lerp(v00, v01, wy),
                torch.lerp(v10, v11, wy),
                wz,
            )
        sampled = rescale_nonequilibrium(
            sampled[:, None, None, :],
            tau_source=(
                self.config.tau_coarse if tau_source is None else tau_source
            ),
            tau_target=(self.config.tau_fine if tau_target is None else tau_target),
            spatial_ratio=float(self.config.ratio),
            regularize=self.config.regularize_prolongation,
        )[:, 0, 0, :]
        if self.config.enforce_transfer_positivity:
            sampled_4d, diagnostic = limit_nonequilibrium_for_positivity(
                sampled[:, None, None, :],
            )
            sampled = sampled_4d[:, 0, 0, :]
            self._record_prolongation_positivity(diagnostic)
        self.fine_f.reshape(self.fine_f.shape[0], -1)[:, plan.target_flat] = sampled

    def _restrict_physical(
        self,
        *,
        tau_source: float | None = None,
        tau_target: float | None = None,
    ) -> torch.Tensor:
        restricted = restrict_populations_2to1(self.fine_physical)
        restricted = rescale_nonequilibrium(
            restricted,
            tau_source=(self.config.tau_fine if tau_source is None else tau_source),
            tau_target=(
                self.config.tau_coarse if tau_target is None else tau_target
            ),
            spatial_ratio=1.0 / self.config.ratio,
            regularize=self.config.regularize_restriction,
        )
        self.last_restriction_positivity = None
        if self.config.enforce_transfer_positivity:
            restricted, diagnostic = limit_nonequilibrium_for_positivity(restricted)
            self.last_restriction_positivity = diagnostic
            if not all(math.isfinite(value) for value in (
                diagnostic.minimum_population_before,
                diagnostic.minimum_population_after,
                diagnostic.minimum_alpha,
            )):
                raise FloatingPointError("non-finite fine-to-coarse AMR restriction")
        return restricted

    def _filter_fine_interface(self, populations: torch.Tensor) -> torch.Tensor:
        return damp_interface_nonequilibrium(
            populations,
            self._interface_filter_blend,
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

    def step(
        self,
        advance: Advance3D,
        *,
        tau_pair: tuple[float, float] | None = None,
    ) -> PopulationRefluxLedger:
        """Advance one coarse step and two time-interpolated fine substeps.

        With reflux enabled, ``advance(f, tau, level, substep)`` must return
        :class:`AMRAdvanceResult`, including the state after collision and
        before streaming.  A raw tensor remains accepted only when reflux is
        disabled. ``level`` is 0/1; the coarse call uses ``substep=-1`` and
        fine calls use 0 and 1.
        """
        tau_coarse, tau_fine = (
            (self.config.tau_coarse, self.config.tau_fine)
            if tau_pair is None else tau_pair
        )
        self._reset_prolongation_positivity()
        if abs(
            tau_fine - convective_refined_tau(tau_coarse, self.config.ratio)
        ) > 1.0e-12:
            raise ValueError("dynamic tau_pair must preserve convective scaling")
        coarse_old = self.coarse_f.clone()
        coarse_new, coarse_post = self._unpack_advance(
            advance(self.coarse_f, tau_coarse, 0, -1),
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
            self._fill_ghost(
                parent_start, tau_source=tau_coarse, tau_target=tau_fine,
            )
            fine_new, fine_post = self._unpack_advance(
                advance(self.fine_f, tau_fine, 1, substep),
                self.fine_f.shape, require_flux_state=self.config.reflux,
            )
            fine_new = self._filter_fine_interface(fine_new)
            if fine_post is not None:
                observed = observe_kinetic_interface_transfer(
                    fine_post, self.fine_interface_links,
                    cell_volume=1.0 / self.config.ratio**3,
                )
                fine_transfer = observed if fine_transfer is None else fine_transfer + observed
            self.fine_f = fine_new
            alpha_end = (substep + 1) / self.config.ratio
            parent_end = torch.lerp(coarse_old, self.coarse_f, alpha_end)
            self._fill_ghost(
                parent_end, tau_source=tau_coarse, tau_target=tau_fine,
            )

        restricted = self._restrict_physical(
            tau_source=tau_fine, tau_target=tau_coarse,
        )
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
            correction_stencil=self.config.reflux_correction_stencil,
        )
        self.last_reflux = PopulationRefluxLedger(
            report.requested_inventory_correction,
            report.applied_inventory_correction,
            report.corrected_links,
            report.residual,
            report.limited_directions,
            report.raw_kinetic_mismatch,
            (
                self.last_restriction_positivity.limited_fraction
                if self.last_restriction_positivity is not None else 0.0
            ),
            (
                self.last_restriction_positivity.minimum_alpha
                if self.last_restriction_positivity is not None else 1.0
            ),
            self._maximum_prolongation_limited_fraction,
            self._minimum_prolongation_alpha,
        )
        return self.last_reflux


class NestedStaticBlockAMR3D:
    """Conservative hierarchy of strictly nested 2:1 static blocks.

    Each configuration describes one parent-to-child interface in the parent
    level's allocated coordinates.  A two-interface hierarchy therefore
    advances level 0 once, level 1 twice and level 2 four times per root step.
    Every interface retains the same time-interpolated ghost fill,
    non-equilibrium scaling, fine-volume restriction and face-local kinetic
    reflux contract as :class:`StaticBlockAMR3D`.

    This runtime deliberately requires reflux on every interface.  A nested
    production calculation must not silently mix conservative and
    replacement-only levels.
    """

    def __init__(
        self,
        coarse_f: torch.Tensor,
        configs: Sequence[StaticBlockAMRConfig],
        *,
        fine_solids: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        if not configs:
            raise ValueError("nested AMR requires at least one fine-level configuration")
        if any(not config.reflux for config in configs):
            raise ValueError("nested AMR requires reflux on every interface")
        if fine_solids is None:
            fine_solids = [None] * len(configs)
        if len(fine_solids) != len(configs):
            raise ValueError("fine_solids must have one entry per interface")

        self.interfaces: list[StaticBlockAMR3D] = []
        parent = coarse_f
        previous_tau_fine: float | None = None
        for level, (config, fine_solid) in enumerate(
            zip(configs, fine_solids, strict=True),
        ):
            if (
                previous_tau_fine is not None
                and abs(config.tau_coarse - previous_tau_fine) > 1.0e-12
            ):
                raise ValueError(
                    f"interface {level} tau_coarse must equal its parent tau_fine",
                )
            interface = StaticBlockAMR3D(
                parent, config, fine_solid=fine_solid,
            )
            self.interfaces.append(interface)
            parent = interface.fine_f
            previous_tau_fine = config.tau_fine
        self.last_reflux: tuple[PopulationRefluxLedger, ...] | None = None

    @property
    def coarse_f(self) -> torch.Tensor:
        return self.interfaces[0].coarse_f

    @coarse_f.setter
    def coarse_f(self, value: torch.Tensor) -> None:
        _validate_parent_and_box(value, self.interfaces[0].config)
        self.interfaces[0].coarse_f = value

    @property
    def finest_f(self) -> torch.Tensor:
        return self.interfaces[-1].fine_f

    @property
    def level_populations(self) -> tuple[torch.Tensor, ...]:
        return (self.coarse_f,) + tuple(
            interface.fine_f for interface in self.interfaces
        )

    @property
    def total_allocated_cells(self) -> int:
        return sum(int(level[0].numel()) for level in self.level_populations)

    @property
    def uniform_finest_equivalent_cells(self) -> int:
        ratio = self.interfaces[0].config.ratio
        return int(self.coarse_f[0].numel() * ratio ** (3 * len(self.interfaces)))

    @property
    def cell_saving_fraction(self) -> float:
        return 1.0 - self.total_allocated_cells / self.uniform_finest_equivalent_cells

    def restore_level_populations(
        self,
        populations: Sequence[torch.Tensor],
    ) -> None:
        """Restore all hierarchy levels while preserving shared parent state."""
        expected = self.level_populations
        if len(populations) != len(expected):
            raise ValueError("checkpoint must contain one population tensor per level")
        for level, (value, template) in enumerate(
            zip(populations, expected, strict=True),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != template.shape:
                raise ValueError(f"checkpoint level {level} has the wrong shape")
            if value.dtype != template.dtype or value.device != template.device:
                raise ValueError(
                    f"checkpoint level {level} must preserve dtype and device",
                )
        self.interfaces[0].coarse_f = populations[0]
        for level, interface in enumerate(self.interfaces):
            interface.fine_f = populations[level + 1]
            if level + 1 < len(self.interfaces):
                self.interfaces[level + 1].coarse_f = interface.fine_f

    def _advance_interface(
        self,
        interface_index: int,
        advance: Advance3D,
        coarse_substep: int,
        ledgers: list[PopulationRefluxLedger | None],
        tau_by_level: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interface = self.interfaces[interface_index]
        config = interface.config
        tau_coarse = tau_by_level[interface_index]
        tau_fine = tau_by_level[interface_index + 1]
        coarse_old = interface.coarse_f.clone()
        coarse_new, coarse_post = interface._unpack_advance(
            advance(
                interface.coarse_f,
                tau_coarse,
                interface_index,
                coarse_substep,
            ),
            interface.coarse_f.shape,
            require_flux_state=True,
        )
        assert coarse_post is not None
        interface.coarse_f = coarse_new
        coarse_transfer = observe_kinetic_interface_transfer(
            coarse_post, interface.coarse_interface_links,
        )
        fine_transfer: KineticInterfaceTransfer | None = None

        for local_substep in range(config.ratio):
            child_substep = (
                local_substep
                if coarse_substep < 0
                else coarse_substep * config.ratio + local_substep
            )
            alpha_start = local_substep / config.ratio
            parent_start = torch.lerp(
                coarse_old, interface.coarse_f, alpha_start,
            )
            interface._fill_ghost(
                parent_start,
                tau_source=tau_coarse,
                tau_target=tau_fine,
            )

            if interface_index + 1 < len(self.interfaces):
                child = self.interfaces[interface_index + 1]
                child.coarse_f = interface.fine_f
                fine_new, fine_post = self._advance_interface(
                    interface_index + 1,
                    advance,
                    child_substep,
                    ledgers,
                    tau_by_level,
                )
            else:
                fine_new, unpacked_post = interface._unpack_advance(
                    advance(
                        interface.fine_f,
                        tau_fine,
                        interface_index + 1,
                        child_substep,
                    ),
                    interface.fine_f.shape,
                    require_flux_state=True,
                )
                assert unpacked_post is not None
                fine_post = unpacked_post

            fine_new = interface._filter_fine_interface(fine_new)
            interface.fine_f = fine_new
            observed = observe_kinetic_interface_transfer(
                fine_post,
                interface.fine_interface_links,
                cell_volume=1.0 / config.ratio**3,
            )
            fine_transfer = (
                observed if fine_transfer is None else fine_transfer + observed
            )
            alpha_end = (local_substep + 1) / config.ratio
            parent_end = torch.lerp(coarse_old, interface.coarse_f, alpha_end)
            interface._fill_ghost(
                parent_end,
                tau_source=tau_coarse,
                tau_target=tau_fine,
            )

        if fine_transfer is None:
            raise RuntimeError("nested AMR omitted fine interface transfer")
        restricted = interface._restrict_physical(
            tau_source=tau_fine,
            tau_target=tau_coarse,
        )
        box = config.box
        interface.coarse_f[
            :, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1
        ] = restricted
        interface.coarse_f, report = apply_face_local_reflux(
            interface.coarse_f,
            interface.coarse_interface_links,
            coarse_transfer,
            fine_transfer,
            maximum_correction_fraction=config.maximum_reflux_correction_fraction,
            correction_stencil=config.reflux_correction_stencil,
        )
        ledger = PopulationRefluxLedger(
            report.requested_inventory_correction,
            report.applied_inventory_correction,
            report.corrected_links,
            report.residual,
            report.limited_directions,
            report.raw_kinetic_mismatch,
            (
                interface.last_restriction_positivity.limited_fraction
                if interface.last_restriction_positivity is not None else 0.0
            ),
            (
                interface.last_restriction_positivity.minimum_alpha
                if interface.last_restriction_positivity is not None else 1.0
            ),
            interface._maximum_prolongation_limited_fraction,
            interface._minimum_prolongation_alpha,
        )
        previous_ledger = ledgers[interface_index]
        if previous_ledger is not None:
            ledger = _merge_reflux_ledgers(previous_ledger, ledger)
        interface.last_reflux = ledger
        ledgers[interface_index] = ledger
        return interface.coarse_f, coarse_post

    def step(
        self,
        advance: Advance3D,
        *,
        tau_by_level: Sequence[float] | None = None,
    ) -> tuple[PopulationRefluxLedger, ...]:
        """Advance the complete hierarchy by one root-grid time step."""
        if tau_by_level is None:
            tau_by_level = (
                self.interfaces[0].config.tau_coarse,
                *(interface.config.tau_fine for interface in self.interfaces),
            )
        if len(tau_by_level) != len(self.interfaces) + 1:
            raise ValueError("tau_by_level must contain one value per hierarchy level")
        for level, (coarse_tau, fine_tau, interface) in enumerate(zip(
            tau_by_level[:-1],
            tau_by_level[1:],
            self.interfaces,
            strict=True,
        )):
            expected = convective_refined_tau(coarse_tau, interface.config.ratio)
            if abs(fine_tau - expected) > 1.0e-12:
                raise ValueError(
                    "dynamic tau chain violates convective scaling at "
                    f"interface {level}",
                )
        ledgers: list[PopulationRefluxLedger | None] = [
            None for _ in self.interfaces
        ]
        for interface in self.interfaces:
            interface._reset_prolongation_positivity()
        self._advance_interface(0, advance, -1, ledgers, tau_by_level)
        if any(ledger is None for ledger in ledgers):
            raise RuntimeError("nested AMR did not produce every reflux ledger")
        self.last_reflux = tuple(
            ledger for ledger in ledgers if ledger is not None
        )
        return self.last_reflux


__all__ = [
    "Advance3D",
    "AMRAdvanceResult",
    "PopulationRefluxLedger",
    "NestedStaticBlockAMR3D",
    "StaticBlockAMR3D",
    "StaticBlockAMRConfig",
    "convective_refined_tau",
]
