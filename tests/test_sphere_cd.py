"""Tests for sphere drag coefficient validation against Schiller-Naumann."""
from __future__ import annotations

import math
import torch
from tensorlbm.boundaries3d import apply_simple_channel_boundaries_3d, make_channel_wall_mask_3d, sphere_mask
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.obstacles import compute_obstacle_forces_3d


def _run_sphere_drag(re: float, nx: int = 80, ny: int = 40, nz: int = 40,
                     steps: int = 400, device: str = "cpu") -> float:
    radius = max(4.0, nx * 0.08)
    u_in = 0.06
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    dev = torch.device(device)
    mask = sphere_mask(nx, ny, nz, nx * 0.5, ny * 0.5, nz * 0.5, radius, device=dev)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, mask, device=dev)
    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=dev),
        torch.full((nz, ny, nx), u_in, device=dev),
        torch.zeros(nz, ny, nx, device=dev),
        torch.zeros(nz, ny, nx, device=dev),
        device=dev,
    )
    fx_list: list[float] = []
    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, mask)
        f = apply_simple_channel_boundaries_3d(f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=mask)
        if step > steps // 2:
            fx_list.append(float(fx.item()))
    fx_mean = sum(fx_list) / max(len(fx_list), 1)
    area = math.pi * radius**2
    return fx_mean / (0.5 * u_in**2 * area)


def _schiller_naumann(re: float) -> float:
    if re < 1e-6:
        return 100.0
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def test_sphere_cd_re50() -> None:
    cd = _run_sphere_drag(50)
    ref = _schiller_naumann(50)
    err = abs(cd - ref) / ref * 100
    assert err < 120, f"Cd={cd:.2f} vs ref={ref:.2f} err={err:.0f}%"


def test_sphere_cd_re100() -> None:
    cd = _run_sphere_drag(100)
    ref = _schiller_naumann(100)
    err = abs(cd - ref) / ref * 100
    assert err < 150, f"Cd={cd:.2f} vs ref={ref:.2f} err={err:.0f}%"


def test_sphere_cd_reasonable_range() -> None:
    cd = _run_sphere_drag(100)
    # Cd should be between 1 and 5 for Re=100
    assert 1.0 < cd < 5.0
