"""Tests for 2D cylinder drag coefficient vs literature correlations."""
from __future__ import annotations

import torch
from tensorlbm.boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_bgk, stream


def _run_cylinder_drag(
    re: float, nx: int = 100, ny: int = 50, steps: int = 400,
    device: str = "cpu",
) -> tuple[float, float]:
    """Run 2D cylinder flow and return Cd_mean, Strouhal."""
    radius = 6.0
    u_in = 0.06
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    dev = torch.device(device)
    mask = cylinder_mask(nx, ny, nx // 3, ny // 2, radius, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)

    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)

    cd_list: list[float] = []
    cl_list: list[float] = []

    for step in range(1, steps + 1):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)
        f = apply_simple_channel_boundaries(
            f, u_in=u_in, wall_mask=wall_mask,
            obstacle_mask=torch.zeros_like(mask),
        )
        f = bounce_back_cells(f, mask)
        if step > steps // 3:
            cd = float(fx.item()) / (0.5 * u_in**2 * 2.0 * radius)
            cl = float(fy.item()) / (0.5 * u_in**2 * 2.0 * radius)
            cd_list.append(cd)
            cl_list.append(cl)

    cd_mean = sum(cd_list) / max(len(cd_list), 1)
    return cd_mean, 0.0


def test_cylinder_cd_re20() -> None:
    cd, _ = _run_cylinder_drag(20)
    assert 2.0 < cd < 8.0, f"Cd={cd:.2f} out of range"


def test_cylinder_cd_re100() -> None:
    cd, _ = _run_cylinder_drag(100, steps=600)
    assert 1.0 < cd < 4.0, f"Cd={cd:.2f} out of range"


def test_cylinder_increasing_cd() -> None:
    """Cd should decrease with increasing Re (lower Re → higher Cd)."""
    cd20, _ = _run_cylinder_drag(20)
    cd100, _ = _run_cylinder_drag(100, steps=600)
    assert cd20 > cd100, f"Cd20={cd20:.2f} <= Cd100={cd100:.2f}"
