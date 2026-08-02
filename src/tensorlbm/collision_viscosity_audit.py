"""Periodic shear-wave audit for recovered collision-model viscosity."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .cumulant import collide_cumulant_d3q19
from .d3q19 import equilibrium3d, macroscopic3d
from .entropic_kbc import collide_kbc_d3q19, collide_natural_kbc_d3q19
from .planar_d3q19 import collide_planar_cumulant_d3q19
from .solver3d import collide_bgk3d, stream3d


@dataclass(frozen=True)
class CollisionViscosityAuditConfig:
    collision_model: str
    tau: float = 0.8
    wavelength_cells: int = 32
    transverse_cells: int = 4
    amplitude: float = 1.0e-3
    steps: int = 200
    fit_start_step: int = 20
    maximum_relative_error_pct: float = 2.0
    kbc_max_iterations: int = 12
    wale_cw: float = 0.5
    vreman_cv: float = 0.025
    device: str = "cpu"
    dtype: str = "float64"
    natural_kbc_compute_dtype: str = "storage"

    def validate(self) -> None:
        if self.collision_model not in {
            "bgk", "cumulant", "planar_cumulant_d2q9",
            "cumulant_wale", "cumulant_vreman",
            "entropic_kbc", "natural_kbc",
        }:
            raise ValueError(
                "unsupported collision_model",
            )
        if not 0.5 < self.tau < 2.0:
            raise ValueError("tau must lie in (0.5,2)")
        if self.wavelength_cells < 16 or self.transverse_cells < 3:
            raise ValueError("shear-wave domain is too small")
        if not 0.0 < self.amplitude < 0.05:
            raise ValueError("amplitude must lie in (0,0.05)")
        if not 1 <= self.fit_start_step < self.steps:
            raise ValueError("fit_start_step must lie inside the trajectory")
        if self.maximum_relative_error_pct <= 0.0:
            raise ValueError("maximum_relative_error_pct must be positive")
        if self.kbc_max_iterations < 1:
            raise ValueError("kbc_max_iterations must be positive")
        if not 0.0 <= self.wale_cw <= 1.0:
            raise ValueError("wale_cw must lie in [0,1]")
        if not 0.0 <= self.vreman_cv <= 0.2:
            raise ValueError("vreman_cv must lie in [0,0.2]")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.natural_kbc_compute_dtype not in {"storage", "float64"}:
            raise ValueError(
                "natural_kbc_compute_dtype must be storage or float64",
            )
        if (
            self.natural_kbc_compute_dtype != "storage"
            and self.collision_model != "natural_kbc"
        ):
            raise ValueError(
                "float64 natural-KBC compute requires collision_model=natural_kbc",
            )


def _collide(
    populations: torch.Tensor,
    config: CollisionViscosityAuditConfig,
) -> torch.Tensor:
    if config.collision_model == "bgk":
        return collide_bgk3d(populations, config.tau)
    if config.collision_model == "cumulant":
        return collide_cumulant_d3q19(
            populations,
            tau=config.tau,
            C_s=0.0,
        )
    if config.collision_model == "planar_cumulant_d2q9":
        return collide_planar_cumulant_d3q19(populations, config.tau)
    if config.collision_model == "cumulant_wale":
        return collide_cumulant_d3q19(
            populations, tau=config.tau, C_w=config.wale_cw,
        )
    if config.collision_model == "cumulant_vreman":
        return collide_cumulant_d3q19(
            populations, tau=config.tau, C_v=config.vreman_cv,
        )
    if config.collision_model == "natural_kbc":
        compute_populations = (
            populations.double()
            if config.natural_kbc_compute_dtype == "float64"
            else populations
        )
        return collide_natural_kbc_d3q19(
            compute_populations, config.tau,
        ).to(dtype=populations.dtype)
    return collide_kbc_d3q19(
        populations,
        config.tau,
        max_iter=config.kbc_max_iterations,
    )


def run_collision_viscosity_audit(
    config: CollisionViscosityAuditConfig,
) -> dict[str, object]:
    """Measure viscosity from exponential decay of a transverse shear wave."""
    config.validate()
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)
    n = config.wavelength_cells
    shape = (config.transverse_cells, n, config.transverse_cells)
    coordinate = torch.arange(n, device=device, dtype=dtype)
    wave_number = 2.0 * math.pi / n
    mode = torch.sin(wave_number * coordinate).view(1, n, 1).expand(shape)
    rho = torch.ones(shape, device=device, dtype=dtype)
    zero = torch.zeros_like(rho)
    populations = equilibrium3d(
        rho,
        config.amplitude * mode,
        zero,
        zero,
        device=device,
    )
    samples: list[tuple[int, float]] = []
    denominator = mode.square().sum()
    for step in range(1, config.steps + 1):
        populations = stream3d(_collide(populations, config))
        if step >= config.fit_start_step:
            ux = macroscopic3d(populations)[1]
            amplitude = float(((ux * mode).sum() / denominator).item())
            samples.append((step, amplitude))

    steps = torch.tensor(
        [step for step, _ in samples],
        device=device,
        dtype=torch.float64,
    )
    log_amplitude = torch.log(torch.tensor(
        [max(abs(amplitude), 1.0e-300) for _, amplitude in samples],
        device=device,
        dtype=torch.float64,
    ))
    centered_steps = steps - steps.mean()
    slope = float((
        (centered_steps * (log_amplitude - log_amplitude.mean())).sum()
        / centered_steps.square().sum()
    ).item())
    recovered_viscosity = -slope / wave_number**2
    target_viscosity = (config.tau - 0.5) / 3.0
    relative_error_pct = (
        abs(recovered_viscosity - target_viscosity) / target_viscosity * 100.0
    )
    finite = bool(torch.isfinite(populations).all()) and all(
        math.isfinite(value)
        for value in (slope, recovered_viscosity, relative_error_pct)
    )
    admitted = finite and relative_error_pct <= config.maximum_relative_error_pct
    return {
        "schema": "tensorlbm-collision-viscosity-audit-v1",
        "configuration": asdict(config),
        "result": {
            "target_kinematic_viscosity": target_viscosity,
            "recovered_kinematic_viscosity": recovered_viscosity,
            "relative_error_pct": relative_error_pct,
            "decay_slope_per_step": slope,
            "initial_fitted_amplitude": samples[0][1],
            "final_amplitude": samples[-1][1],
            "minimum_population": float(populations.min().item()),
            "finite": finite,
        },
        "acceptance": {
            "maximum_relative_error_pct": config.maximum_relative_error_pct,
            "admitted": admitted,
        },
    }


__all__ = [
    "CollisionViscosityAuditConfig",
    "run_collision_viscosity_audit",
]
