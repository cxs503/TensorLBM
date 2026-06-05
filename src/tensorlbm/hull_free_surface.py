"""Free-surface ship hull flow using Color-Gradient multiphase LBM.

Couples the 3D Color-Gradient multiphase model with a Wigley hull
obstacle (bounce-back) to simulate wave-making resistance.

References
----------
Wigley (1934) Trans. Inst. Naval Archit. 76 57
Gunstensen et al. (1991) Phys. Rev. A 43 4320
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .boundaries3d import bounce_back_cells_3d
from .d3q19 import equilibrium3d
from .multiphase3d import color_gradient_step_3d
from .obstacles import compute_obstacle_forces_3d, wigley_hull_mask
from .ship_cad import series60_hull_mask, kcs_hull_mask
from .solver3d import stream3d

__all__ = ["HullFreeSurfaceConfig", "run_hull_free_surface"]

_HULL_BUILDERS = {
    "wigley": wigley_hull_mask,
    "series60": series60_hull_mask,
    "kcs": kcs_hull_mask,
}


@dataclass(frozen=True)
class HullFreeSurfaceConfig:
    """Configuration for the simplified hull free-surface benchmark.

    Args:
        nx: Number of cells in x.
        ny: Number of cells in y.
        nz: Number of cells in z.
        fill_fraction: Initial liquid fill fraction in z.
        re: Reynolds number used to set the relaxation time.
        u_in: Inlet velocity in the water region.
        n_steps: Number of time steps.
        output_interval: Drag sampling interval.
        device: Torch device string.
    """

    nx: int = 80
    ny: int = 32
    nz: int = 32
    hull_type: str = "wigley"
    fill_fraction: float = 0.5
    re: float = 100.0
    u_in: float = 0.05
    n_steps: int = 200
    output_interval: int = 50
    device: str = "cpu"


def _make_wall_mask(nz: int, ny: int, nx: int, obstacle: torch.Tensor) -> torch.Tensor:
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=obstacle.device)
    wall_mask[:, 0, :] = True
    wall_mask[:, -1, :] = True
    wall_mask[0, :, :] = True
    wall_mask[-1, :, :] = True
    wall_mask[obstacle] = False
    return wall_mask


def _apply_phase_inlet(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    water_slice: torch.Tensor,
    u_in: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rho_r_in = torch.where(water_slice, torch.ones_like(water_slice, dtype=torch.float32), 0.1)
    rho_b_in = torch.where(water_slice, 0.1, torch.ones_like(water_slice, dtype=torch.float32))
    ux_in = torch.where(water_slice, torch.full_like(rho_r_in, u_in), torch.zeros_like(rho_r_in))
    uy_in = torch.zeros_like(rho_r_in)
    uz_in = torch.zeros_like(rho_r_in)
    f_r[:, :, :, 0] = equilibrium3d(
        rho_r_in.unsqueeze(-1), ux_in.unsqueeze(-1), uy_in.unsqueeze(-1), uz_in.unsqueeze(-1)
    )[:, :, :, 0]
    f_b[:, :, :, 0] = equilibrium3d(
        rho_b_in.unsqueeze(-1), ux_in.unsqueeze(-1), uy_in.unsqueeze(-1), uz_in.unsqueeze(-1)
    )[:, :, :, 0]
    return f_r, f_b


def run_hull_free_surface(config: HullFreeSurfaceConfig) -> dict[str, object]:
    """Run a practical free-surface Wigley hull demonstration."""
    device = torch.device(config.device)
    nz, ny, nx = config.nz, config.ny, config.nx
    fill_height = max(int(config.fill_fraction * nz), 1)

    zz = torch.arange(nz, device=device).view(nz, 1, 1)
    water_mask = (zz < fill_height).expand(nz, ny, nx)

    hull_builder = _HULL_BUILDERS.get(config.hull_type, wigley_hull_mask)
    # Wider beam for fuller hull forms
    beam_scale = {"wigley": 0.25, "series60": 0.32, "kcs": 0.35}.get(config.hull_type, 0.25)
    hull = hull_builder(
        nx=nx, ny=ny, nz=nz,
        cx=0.45 * nx, cy=0.5 * (ny - 1),
        cz_keel=1.0,  # hull starts near bottom
        length=max(6.0, 0.35 * nx),
        beam=max(3.0, beam_scale * ny),
        draft=fill_height + 4,  # extend above water surface (~10-15% freeboard)
        device=device,
    )
    wall_mask = _make_wall_mask(nz, ny, nx, hull)
    solid_mask = wall_mask | hull

    rho_r = torch.where(water_mask, torch.ones((nz, ny, nx), device=device), 0.1)
    rho_b = torch.where(water_mask, 0.1, torch.ones((nz, ny, nx), device=device))
    ux0 = torch.where(water_mask, torch.full((nz, ny, nx), config.u_in, device=device), 0.0)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    f_r = equilibrium3d(rho_r, ux0, uy0, uz0)
    f_b = equilibrium3d(rho_b, ux0, uy0, uz0)

    # Precompute solid-cell equilibrium (zero velocity) for stability reset
    zero3d = torch.zeros((nz, ny, nx), device=device)
    f_r_solid_eq = equilibrium3d(rho_r, zero3d, zero3d, zero3d)
    f_b_solid_eq = equilibrium3d(rho_b, zero3d, zero3d, zero3d)

    nu = config.u_in * max(1.0, 0.35 * nx) / max(config.re, 1e-6)
    tau = 3.0 * nu + 0.5
    projected_area = max(float(hull.any(dim=2).float().sum().item()), 1.0)
    dyn_pressure = 0.5 * config.u_in**2 * projected_area
    drag_samples: list[float] = []

    water_slice = water_mask[:, :, 0]
    for step in range(1, config.n_steps + 1):
        # 1. CG collision
        f_r, f_b = color_gradient_step_3d(
            f_r, f_b, tau=tau, A=0.005, beta=0.7, solid_mask=solid_mask,
        )
        # 2. Stream
        f_r = stream3d(f_r)
        f_b = stream3d(f_b)
        # 3. Force diagnostic (before bounce-back)
        fx, _, _ = compute_obstacle_forces_3d(f_r + f_b, hull)
        # 4. Bounce-back on all solid cells
        f_r = bounce_back_cells_3d(f_r, solid_mask)
        f_b = bounce_back_cells_3d(f_b, solid_mask)
        # 5. Reset solid cells to equilibrium (prevents mass accumulation)
        f_r = torch.where(solid_mask.unsqueeze(0), f_r_solid_eq, f_r)
        f_b = torch.where(solid_mask.unsqueeze(0), f_b_solid_eq, f_b)
        # 6. Outlet: convective copy
        f_r[:, :, :, -1] = f_r[:, :, :, -2]
        f_b[:, :, :, -1] = f_b[:, :, :, -2]
        # 7. Inlet: reset to equilibrium
        f_r, f_b = _apply_phase_inlet(f_r, f_b, water_slice, config.u_in)

        if step % config.output_interval == 0 or step == config.n_steps:
            drag = float(fx.item()) / dyn_pressure if dyn_pressure != 0.0 else 0.0
            drag_samples.append(drag)
            print(f"  step={step:5d}  Cd={drag:.4f}")

    rho_r_final = f_r.sum(dim=0)
    rho_b_final = f_b.sum(dim=0)
    water_fraction = rho_r_final / torch.clamp(rho_r_final + rho_b_final, min=1e-12)
    wetted_fraction = (
        float((water_fraction[hull] > 0.5).float().mean().item()) if hull.any() else 0.0
    )
    mean_cd = float(sum(drag_samples) / len(drag_samples)) if drag_samples else 0.0
    return {"mean_cd": mean_cd, "wetted_fraction": wetted_fraction, "hull_type": config.hull_type, "config": asdict(config)}
