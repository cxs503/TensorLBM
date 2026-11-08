"""Two-dimensional Taylor-Green vortex on a 2:1 block-AMR grid.

Purpose: provide a *closed* conservation and accuracy test for the conservative
coarse/fine interface machinery in :mod:`tensorlbm.static_block_amr`, without
any boundary-condition or turbulence-modelling ambiguity.

The 2-D Taylor-Green vortex

    u_x = A cos(k x) sin(k y) exp(-2 nu k^2 t)
    u_y = -A sin(k x) cos(k y) exp(-2 nu k^2 t)
    p   = -(A^2 / 4) (cos 2kx + cos 2ky) exp(-4 nu k^2 t)

is an exact solution of the incompressible Navier-Stokes equations on a
periodic ``2*pi/k`` box.  It is run here as a quasi-2-D flow (``z``-invariant)
inside the 3-D D3Q19 block-AMR runtime, so the interface machinery exercises its
production code path.

Three conservation statements are measured, each with an independent ledger:

* **Mass** - total mass of the composite (coarse outside + fine inside, fine
  cells weighted ``1/8``) must not drift;
* **Momentum** - total momentum of the composite must only change by the
  physical viscous decay predicted by the exact solution;
* **Interface flux** - the population flux crossing the refinement interface as
  seen from the coarse level and as seen (volume-scaled) from the fine level must
  balance; the reflux correction is what closes the mismatch, so both the
  pre-correction mismatch (a modelling error of the prolongation) and the
  post-correction residual (the conservation violation) are reported separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .refinement import BoxRegion
from .solver3d import collide_bgk3d, stream3d
from .static_block_amr import (
    AMRAdvanceResult,
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
    convective_refined_tau,
)


@dataclass(frozen=True)
class TaylorGreenExact:
    """Exact 2-D Taylor-Green solution sampled on a lattice."""

    nx: int
    ny: int
    amplitude: float
    nu: float

    @property
    def kx(self) -> float:
        return 2.0 * math.pi / self.nx

    @property
    def ky(self) -> float:
        return 2.0 * math.pi / self.ny

    def decay(self, time: float) -> float:
        """Amplitude decay factor ``exp(-nu (kx^2 + ky^2) t)``."""
        return math.exp(-self.nu * (self.kx**2 + self.ky**2) * time)

    def velocity(
        self,
        time: float,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(ux, uy)`` on the ``(ny, nx)`` lattice at physical ``time``."""
        x = torch.arange(self.nx, device=device, dtype=dtype)
        y = torch.arange(self.ny, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        factor = self.amplitude * self.decay(time)
        ux = factor * torch.cos(self.kx * xx) * torch.sin(self.ky * yy)
        uy = -factor * torch.sin(self.kx * xx) * torch.cos(self.ky * yy)
        return ux, uy


def _sampled_equilibrium(
    config: TaylorGreenAMRConfig,
    *,
    nx: int,
    ny: int,
    nz: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    spacing: float = 1.0,
    time: float = 0.0,
) -> torch.Tensor:
    """Populations in local equilibrium on a lattice of arbitrary spacing.

    Cell centres sit at ``offset + (i + 0.5) * spacing`` in coarse lattice
    units, which is what a nested block needs when its fine spacing is ``0.5``.
    """
    dtype = config.torch_dtype()
    device = torch.device(config.device)
    i = torch.arange(nx, device=device, dtype=dtype)
    j = torch.arange(ny, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(offset_y + (j + 0.5) * spacing, offset_x + (i + 0.5) * spacing, indexing="ij")
    exact = config.exact
    factor = exact.amplitude * exact.decay(time)
    ux2 = factor * torch.cos(exact.kx * xx) * torch.sin(exact.ky * yy)
    uy2 = -factor * torch.sin(exact.kx * xx) * torch.cos(exact.ky * yy)
    ux = ux2.unsqueeze(0).expand(nz, ny, nx).contiguous()
    uy = uy2.unsqueeze(0).expand(nz, ny, nx).contiguous()
    rho = torch.ones((nz, ny, nx), device=device, dtype=dtype)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, ux, uy, zero, device=device)


def uniform_fine_initial(config: TaylorGreenAMRConfig) -> torch.Tensor:
    """Initial populations for the fully refined reference run."""
    if config.initialization == "exact":
        return _sampled_equilibrium(
            config,
            nx=2 * config.nx,
            ny=2 * config.ny,
            nz=2 * config.nz,
            offset_x=-0.5,
            offset_y=-0.5,
            spacing=0.5,
        )
    coarse = initial_populations(config)
    return (
        coarse.repeat_interleave(2, dim=1)
        .repeat_interleave(2, dim=2)
        .repeat_interleave(2, dim=3)
    )


def _seed_fine_block(solver: StaticBlockAMR3D, config: TaylorGreenAMRConfig) -> None:
    """Overwrite the fine block interior with a fine-resolution sample.

    Used by ``initialization="exact"`` so the fine level does not start from a
    piecewise-constant prolongation of the coarse level - the resulting
    interface-localised initial layer is then a property of the scheme rather
    than of the initial data.
    """
    box = config.box
    ratio = 2
    fine_nx = (box.x1 - box.x0) * ratio
    fine_ny = (box.y1 - box.y0) * ratio
    fine_nz = (box.z1 - box.z0) * ratio
    seeded = _sampled_equilibrium(
        config,
        nx=fine_nx,
        ny=fine_ny,
        nz=fine_nz,
        offset_x=box.x0 - 0.5,
        offset_y=box.y0 - 0.5,
        spacing=0.5,
    )
    ghost = solver.fine_f[:, :1, :1, :1]
    if seeded.shape != solver.fine_physical.shape:
        raise RuntimeError("seeded fine block shape mismatch")
    fine = solver.fine_f.clone()
    fine[:, 1:-1, 1:-1, 1:-1] = seeded.to(dtype=solver.fine_f.dtype)
    del ghost
    solver.fine_f = fine


@dataclass(frozen=True)
class TaylorGreenAMRConfig:
    """Configuration of the quasi-2-D Taylor-Green AMR study."""

    nx: int = 32
    ny: int = 32
    nz: int = 10
    box_x: tuple[int, int] = (8, 24)
    box_y: tuple[int, int] = (8, 24)
    box_z: tuple[int, int] = (2, 6)
    tau_coarse: float = 0.56
    amplitude: float = 0.03
    steps: int = 160
    device: str = "cpu"
    dtype: str = "float64"
    reflux: bool = True
    maximum_reflux_correction_fraction: float = 0.2
    regularize_prolongation: bool = False
    reflux_correction_stencil: str = "exterior_cells"
    interface_filter_width: int = 0
    interface_filter_strength: float = 0.0
    initialization: str = "prolongated"

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def box(self) -> BoxRegion:
        return BoxRegion(
            x0=self.box_x[0],
            x1=self.box_x[1],
            y0=self.box_y[0],
            y1=self.box_y[1],
            z0=self.box_z[0],
            z1=self.box_z[1],
        )

    @property
    def nu(self) -> float:
        return (self.tau_coarse - 0.5) / 3.0

    @property
    def exact(self) -> TaylorGreenExact:
        return TaylorGreenExact(self.nx, self.ny, self.amplitude, self.nu)

    def torch_dtype(self) -> torch.dtype:
        mapping = {"float32": torch.float32, "float64": torch.float64}
        try:
            return mapping[self.dtype]
        except KeyError as error:  # pragma: no cover - defensive
            raise ValueError(f"unsupported dtype {self.dtype!r}") from error

    def validate(self) -> None:
        if min(self.nx, self.ny, self.nz) < 8:
            raise ValueError("all directions need at least 8 cells")
        if self.tau_coarse <= 0.5:
            raise ValueError("tau_coarse must exceed 0.5")
        if not 0.0 < self.amplitude <= 0.1:
            raise ValueError("amplitude must lie in (0, 0.1] to stay incompressible")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if not 0.0 < self.maximum_reflux_correction_fraction <= 1.0:
            raise ValueError("maximum_reflux_correction_fraction must lie in (0, 1]")
        if self.initialization not in ("prolongated", "exact"):
            raise ValueError("initialization must be 'prolongated' or 'exact'")
        box = self.box
        limits = (
            (box.z0, box.z1, self.nz),
            (box.y0, box.y1, self.ny),
            (box.x0, box.x1, self.nx),
        )
        for lower, upper, size in limits:
            if not 1 < lower < upper < size - 2:
                raise ValueError("refinement box needs a two-cell exterior margin")


def _interface_momentum(
    inventory: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Momentum of a per-direction interface inventory vector (shape ``(q,)``)."""
    vector = inventory.reshape(-1)
    if vector.shape[0] != velocities.shape[0]:
        raise ValueError("inventory and lattice velocity table disagree on q")
    return (vector[:, None] * velocities).sum(dim=0)


@dataclass(frozen=True)
class InterfaceFluxLedger:
    """Per-step interface flux accounting (kinetic register + composite mass)."""

    raw_mismatch_l1: float
    applied_correction_l1: float
    residual_l1: float
    limited_directions: int
    composite_mass: float
    composite_momentum: float

    def to_dict(self) -> dict[str, float]:
        return {
            "raw_mismatch_l1": self.raw_mismatch_l1,
            "applied_correction_l1": self.applied_correction_l1,
            "residual_l1": self.residual_l1,
            "limited_directions": self.limited_directions,
            "composite_mass": self.composite_mass,
            "composite_momentum": self.composite_momentum,
        }


def initial_populations(
    config: TaylorGreenAMRConfig,
    *,
    rho0: float = 1.0,
) -> torch.Tensor:
    """Initial D3Q19 populations for the quasi-2-D Taylor-Green vortex."""
    dtype = config.torch_dtype()
    device = torch.device(config.device)
    ux, uy = config.exact.velocity(0.0, device=device, dtype=dtype)
    nz, ny, nx = config.shape_zyx
    full_ux = ux.unsqueeze(0).expand(nz, ny, nx).contiguous()
    full_uy = uy.unsqueeze(0).expand(nz, ny, nx).contiguous()
    rho = torch.full((nz, ny, nx), rho0, device=device, dtype=dtype)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, full_ux, full_uy, zero, device=device)


def _lattice_velocities(q: int, device: torch.device) -> torch.Tensor:
    from .d3q19 import C as C19
    from .d3q27 import C as C27

    table = C19 if q == 19 else C27
    return table.to(device=device, dtype=torch.float64)


def _advance(
    state: torch.Tensor,
    tau: float,
    level: int = 0,
    substep: int = 0,
) -> AMRAdvanceResult:
    """Periodic BGK advance; the AMR runtime supplies ``level``/``substep``."""
    del level, substep
    post = collide_bgk3d(state, tau=tau)
    return AMRAdvanceResult(stream3d(post), post)


def _composite_moments(state: torch.Tensor) -> tuple[float, float]:
    rho, ux, uy, uz = macroscopic3d(state)
    momentum = float(torch.sqrt((rho * ux) ** 2 + (rho * uy) ** 2 + (rho * uz) ** 2).sum().item())
    return float(rho.sum().item()), momentum


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = float(torch.sqrt(reference.square().mean()).item())
    if denominator == 0.0:
        raise ValueError("reference field is identically zero")
    return float(
        torch.sqrt((candidate - reference).square().mean()).item() / denominator,
    )


def _l2_error(
    field_ha: tuple[torch.Tensor, torch.Tensor],
    nx: int,
    ny: int,
) -> float:
    ux, uy = field_ha
    return float(
        torch.sqrt((ux.square() + uy.square()).mean()).item(),
    )


@dataclass
class TaylorGreenAMRResult:
    """Outcome of the Taylor-Green AMR conservation/accuracy study."""

    config: TaylorGreenAMRConfig
    ledger: list[InterfaceFluxLedger] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "nx": self.config.nx,
                "ny": self.config.ny,
                "nz": self.config.nz,
                "box": [self.config.box_x, self.config.box_y, self.config.box_z],
                "tau_coarse": self.config.tau_coarse,
                "nu": self.config.nu,
                "amplitude": self.config.amplitude,
                "steps": self.config.steps,
                "dtype": self.config.dtype,
                "reflux": self.config.reflux,
                "initialization": self.config.initialization,
                "maximum_reflux_correction_fraction": (
                    self.config.maximum_reflux_correction_fraction
                ),
            },
            "metrics": self.metrics,
            "ledger": [entry.to_dict() for entry in self.ledger],
        }


def run_taylor_green_amr(config: TaylorGreenAMRConfig) -> TaylorGreenAMRResult:
    """Run AMR and uniform-fine Taylor-Green vortices and compare them.

    Both runs start from the *same* discrete initial state (uniform fine is the
    2:1 repeated coarse state), so any difference at the end is produced by the
    interface treatment alone rather than by a different initial condition.
    """
    config.validate()
    dtype = config.torch_dtype()
    device = torch.device(config.device)
    coarse_initial = initial_populations(config)
    uniform_fine = uniform_fine_initial(config)
    solver = StaticBlockAMR3D(
        coarse_initial.clone(),
        StaticBlockAMRConfig(
            config.box,
            tau_coarse=config.tau_coarse,
            reflux=config.reflux,
            maximum_reflux_correction_fraction=config.maximum_reflux_correction_fraction,
            regularize_prolongation=config.regularize_prolongation,
            reflux_correction_stencil=config.reflux_correction_stencil,
            interface_filter_width=config.interface_filter_width,
            interface_filter_strength=config.interface_filter_strength,
        ),
    )
    if config.initialization == "exact":
        _seed_fine_block(solver, config)
    tau_fine = convective_refined_tau(config.tau_coarse)
    uniform_fine_coarse_baseline = coarse_initial.clone()

    initial_mass, initial_momentum = _composite_moments(solver.coarse_f)
    ledger: list[InterfaceFluxLedger] = []
    mass_drift: list[float] = []
    momentum_deviation: list[float] = []
    raw_mismatch_max = 0.0
    residual_max = 0.0
    applied_max = 0.0
    limited_total = 0

    for _ in range(config.steps):
        report = solver.step(_advance)
        uniform_fine_coarse_baseline = _advance(
            uniform_fine_coarse_baseline,
            config.tau_coarse,
        ).populations
        for _ in range(2):
            uniform_fine = _advance(uniform_fine, tau_fine).populations
        mass, momentum = _composite_moments(solver.coarse_f)
        raw = 0.0 if report.raw_kinetic_mismatch is None else float(
            report.raw_kinetic_mismatch.abs().sum().item(),
        )
        residual = float(report.residual.abs().sum().item())
        applied = float(report.applied_shell_correction.abs().sum().item())
        velocities = _lattice_velocities(report.residual.shape[0], report.residual.device)
        requested_vector = report.replacement_mismatch.reshape(-1)
        residual_vector = report.residual.reshape(-1)
        applied_vector = report.applied_shell_correction.reshape(-1)
        momentum_requested = float(
            (requested_vector[:, None] * velocities).abs().sum().item(),
        )
        momentum_residual = float(
            (residual_vector[:, None] * velocities).abs().sum().item(),
        )
        momentum_applied = float(
            (applied_vector[:, None] * velocities).abs().sum().item(),
        )
        momentum_scale = max(momentum_requested, 1e-300)
        momentum_residual_max = max(
            getattr(solver, "_momentum_residual_max", 0.0),
            momentum_residual / momentum_scale,
        )
        momentum_applied_ratio_min = min(
            getattr(solver, "_momentum_applied_min", float("inf")),
            momentum_applied / momentum_scale if momentum_requested > 0.0 else 1.0,
        )
        solver._momentum_residual_max = momentum_residual_max
        solver._momentum_applied_min = momentum_applied_ratio_min
        raw_mismatch_max = max(raw_mismatch_max, raw)
        residual_max = max(residual_max, residual)
        applied_max = max(applied_max, applied)
        limited_total += int(report.limited_directions)
        mass_drift.append(abs(mass - initial_mass) / initial_mass)
        momentum_deviation.append(abs(momentum - initial_momentum) / initial_momentum)
        ledger.append(
            InterfaceFluxLedger(
                raw_mismatch_l1=raw,
                applied_correction_l1=applied,
                residual_l1=residual,
                limited_directions=int(report.limited_directions),
                composite_mass=mass,
                composite_momentum=momentum,
            )
        )

    time = float(config.steps)
    exact = config.exact
    # Restrict the uniform-fine solution onto the coarse grid: this is the
    # apples-to-apples reference for the composite AMR state.
    from .fixed_nested_transfer import restrict_populations_2to1

    reference_coarse = restrict_populations_2to1(uniform_fine)
    rho_amr, ux_amr, uy_amr, _ = macroscopic3d(solver.coarse_f)
    rho_ref, ux_ref, uy_ref, _ = macroscopic3d(reference_coarse)
    ux_exact, uy_exact = exact.velocity(time, device=device, dtype=dtype)
    exact_amp = float(ux_exact.abs().max().item())

    restricted_baseline_rho, ux_baseline, uy_baseline, _ = macroscopic3d(
        uniform_fine_coarse_baseline,
    )
    del restricted_baseline_rho

    error_rms = torch.sqrt((ux_amr - ux_ref) ** 2 + (uy_amr - uy_ref) ** 2)
    amplitude = max(float(ux_amr.abs().max().item()), 1e-300)
    distances: list[int] = []
    rms_inside: list[float] = []
    box = config.box
    max_depth = min((box.x1 - box.x0) // 2, (box.y1 - box.y0) // 2)
    def shrunk(depth: int) -> torch.Tensor:
        mask = torch.zeros(config.shape_zyx, dtype=torch.bool, device=device)
        # The flow is z-invariant, so distance from the interface is measured in
        # the x-y plane where the vortex actually crosses it.
        z0, z1 = box.z0, box.z1
        y0, y1 = box.y0 + depth, box.y1 - depth
        x0, x1 = box.x0 + depth, box.x1 - depth
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return mask
        mask[z0:z1, y0:y1, x0:x1] = True
        return mask

    for depth in range(max_depth + 1):
        shell = shrunk(depth) & ~shrunk(depth + 1)
        if not bool(shell.any()):
            continue
        distances.append(depth)
        rms_inside.append(
            float(torch.sqrt(error_rms[shell].square().mean()).item()) / amplitude,
        )

    metrics: dict[str, Any] = {
        "time": time,
        "interface_error_profile": {
            "depth_from_interface_coarse_cells": distances,
            "rms_over_amplitude": rms_inside,
        },
        "exact_amplitude": exact_amp,
        "amr_amplitude": float(ux_amr.abs().max().item()),
        "amr_vs_uniform_fine_l2": _relative_l2(
            torch.stack([ux_ref, uy_ref]),
            torch.stack([ux_amr, uy_amr]),
        ),
        "amr_vs_exact_l2": _relative_l2(
            torch.stack([ux_exact.unsqueeze(0).expand_as(ux_amr), uy_exact.unsqueeze(0).expand_as(uy_amr)]),
            torch.stack([ux_amr, uy_amr]),
        ),
        "uniform_fine_vs_exact_l2": _relative_l2(
            torch.stack([ux_exact.unsqueeze(0).expand_as(ux_ref), uy_exact.unsqueeze(0).expand_as(uy_ref)]),
            torch.stack([ux_ref, uy_ref]),
        ),
        "coarse_only_vs_exact_l2": _relative_l2(
            torch.stack([ux_exact.unsqueeze(0).expand_as(ux_baseline), uy_exact.unsqueeze(0).expand_as(uy_baseline)]),
            torch.stack([ux_baseline, uy_baseline]),
        ),
        "relative_mass_drift_max": float(max(mass_drift)),
        "relative_momentum_deviation_max": float(max(momentum_deviation)),
        "initial_mass": initial_mass,
        "initial_momentum": initial_momentum,
        "interface_raw_mismatch_l1_max": raw_mismatch_max,
        "interface_residual_l1_max": residual_max,
        "interface_applied_correction_l1_max": applied_max,
        "interface_limited_directions_total": limited_total,
        "interface_residual_over_raw": (
            residual_max / raw_mismatch_max if raw_mismatch_max > 0.0 else 0.0
        ),
        "interface_momentum_residual_relative": getattr(
            solver,
            "_momentum_residual_max",
            0.0,
        ),
        "interface_momentum_applied_ratio_min": getattr(
            solver,
            "_momentum_applied_min",
            1.0,
        ),
        "reflux_enabled": config.reflux,
        "dtype": config.dtype,
        "density_peak_to_peak": float((rho_amr.max() - rho_amr.min()).item()),
        "cell_saving_fraction": solver.cell_saving_fraction,
        "allocated_cells": solver.total_allocated_cells,
    }
    return TaylorGreenAMRResult(config=config, ledger=ledger, metrics=metrics)


__all__ = [
    "InterfaceFluxLedger",
    "TaylorGreenAMRConfig",
    "TaylorGreenAMRResult",
    "TaylorGreenExact",
    "initial_populations",
    "run_taylor_green_amr",
]