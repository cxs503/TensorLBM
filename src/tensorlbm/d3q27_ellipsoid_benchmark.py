"""D3Q27 prolate spheroid (ellipsoid) benchmark for TensorLBM.

Runs the same 6:1 prolate spheroid physics as :mod:`tensorlbm.ellipsoid_benchmark`
but using the **D3Q27 lattice** (27 velocities) instead of D3Q19.

D3Q27 achieves 4th-order isotropy (vs 2nd-order for D3Q19), which reduces
numerical artefacts in curved-boundary flows and improves drag prediction.

Key differences from D3Q19:
- Collision: :func:`~tensorlbm.turbulence.collide_smagorinsky_mrt27`
- Streaming: :func:`~tensorlbm.d3q27.stream27`
- Boundaries: :func:`~tensorlbm.boundaries_d3q27.apply_zou_he_channel_boundaries_27`
- Forces: :func:`~tensorlbm.obstacles.compute_obstacle_forces_27`
- Mass correction: :func:`~tensorlbm.d3q27.correct_mass27`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .boundaries3d import make_channel_wall_mask_3d
from .boundaries_d3q27 import (
    apply_zou_he_channel_boundaries_27,
    make_channel_wall_mask_27,
)
from .d3q27 import (
    collide_mrt27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from .ellipsoid_benchmark import (
    build_ellipsoid_mask,
    ellipsoid_statistics,
    reference_ellipsoid_cd,
)
from .obstacles import compute_obstacle_forces_27
from .turbulence import collide_smagorinsky_mrt27
from .utils import resolve_device


@dataclass
class EllipsoidD3Q27Config:
    """Configuration for D3Q27 ellipsoid benchmark."""

    semi_major_a: float = 24.0
    semi_minor_b: float = 8.0
    alpha_deg: float = 0.0
    nx: int = 120
    ny: int = 64
    nz: int = 64
    u_in: float = 0.06
    re: float = 100.0
    n_steps: int = 4000
    warmup_steps: int = 2000
    smagorinsky_cs: float = 0.1
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def nu(self) -> float:
        return self.u_in * 2.0 * self.semi_minor_b / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def a_b_ratio(self) -> float:
        return self.semi_major_a / self.semi_minor_b


def run_ellipsoid_benchmark_d3q27(config: EllipsoidD3Q27Config) -> dict:
    """Run D3Q27 flow past a 6:1 prolate spheroid and report Cd, Cl.

    Uses Smagorinsky MRT (D3Q27) collision → stream → force measurement →
    Zou/He D3Q27 channel boundaries.
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    mask = build_ellipsoid_mask(
        config.nx, config.ny, config.nz,
        config.semi_major_a, config.semi_minor_b,
        config.alpha_deg, device=device,
    )
    wall_mask = make_channel_wall_mask_27(
        config.nz, config.ny, config.nx, mask, device=device,
    )

    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium27(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    diam = 2.0 * config.semi_minor_b
    dyn_pressure = 0.5 * config.u_in**2 * math.pi * config.semi_minor_b**2

    fx_list: list[float] = []
    fy_list: list[float] = []
    fz_list: list[float] = []

    print(f"D3Q27 Ellipsoid: a/b={config.a_b_ratio:.1f} "
          f"α={config.alpha_deg}° Re={config.re} tau={config.tau:.4f}")
    print(f"  Grid: {config.nx}×{config.ny}×{config.nz}  "
          f"steps={config.n_steps}  Cs={config.smagorinsky_cs}")
    print(f"  D={diam:.0f} lu  u_in={config.u_in}  device={device}")

    for step in range(1, config.n_steps + 1):
        # Collision — D3Q27 MRT with optional Smagorinsky
        if config.tau < 0.575 and config.smagorinsky_cs > 0:
            f = collide_smagorinsky_mrt27(f, tau=config.tau, C_s=config.smagorinsky_cs)
        elif config.tau < 0.575:
            f = collide_mrt27(f, tau=config.tau)
        else:
            from .d3q27 import collide_bgk27
            f = collide_bgk27(f, tau=config.tau)

        f = stream27(f)

        # Forces AFTER stream, BEFORE bounce-back
        fx, fy, fz = compute_obstacle_forces_27(f, mask)

        f = apply_zou_he_channel_boundaries_27(
            f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=mask,
        )

        if step % 200 == 0:
            f = correct_mass27(f, initial_mass)

        if step > config.warmup_steps:
            fx_list.append(float(fx.item()))
            fy_list.append(float(fy.item()))
            fz_list.append(float(fz.item()))

        if step % 500 == 0 or step == config.n_steps:
            n_samples = max(min(len(fx_list), 500), 1)
            cd_mean = sum(fx_list[-500:]) / n_samples / dyn_pressure
            cl_mean = sum(fy_list[-500:]) / n_samples / dyn_pressure
            cz_mean = sum(fz_list[-500:]) / n_samples / dyn_pressure
            print(f"  step {step:5d}: Cd={cd_mean:.4f}  Cl={cl_mean:.4f}  "
                  f"Cz={cz_mean:.4f}")

    n_total = max(len(fx_list), 1)
    cd_mean = sum(fx_list) / n_total / dyn_pressure
    cl_mean = sum(fy_list) / n_total / dyn_pressure
    cz_mean = sum(fz_list) / n_total / dyn_pressure

    ref = reference_ellipsoid_cd(config.re, config.alpha_deg)
    cd_err = abs(cd_mean - ref["cd"]) / max(abs(ref["cd"]), 1e-10) * 100

    stats = ellipsoid_statistics(
        config.nx, config.ny, config.nz,
        config.semi_major_a, config.semi_minor_b,
        config.alpha_deg, device=device,
    )

    print(f"\n  D3Q27 Results: 6:1 prolate spheroid  α={config.alpha_deg}°")
    print(f"  Cd_sim={cd_mean:.4f}  (ref {ref['cd']:.4f}, err {cd_err:.1f}%)")
    print(f"  Cl_sim={cl_mean:.4f}")
    print(f"  D={diam:.0f} lu  Re={config.re}  a/b={config.a_b_ratio:.1f}")

    return {
        "cd_sim": cd_mean, "cl_sim": cl_mean, "cz_sim": cz_mean,
        "cd_ref": ref["cd"], "cl_ref": ref["cl"],
        "cd_err_pct": cd_err,
        "alpha_deg": config.alpha_deg, "re": config.re,
        "a_b_ratio": config.a_b_ratio, "diameter_lu": diam,
        "lattice": "D3Q27",
        "geometry": stats,
    }


__all__ = [
    "EllipsoidD3Q27Config",
    "run_ellipsoid_benchmark_d3q27",
]
