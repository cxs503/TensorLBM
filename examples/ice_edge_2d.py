#!/usr/bin/env python3
"""Ice-edge zone LBM simulation: simplified and stable version.

This version uses a 2D approximation (D2Q9) for faster development and testing,
then extends to 3D (D3Q19) for production runs.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use D2Q9 for 2D simulation (faster, more stable for development)
from tensorlbm.d2q9 import C as C2Q9, W as W2Q9, OPPOSITE as OPP2Q9

__all__ = ["run_ice_edge_2d", "IceEdge2DConfig"]


def _build_mirror_y_2d() -> list[int]:
    """Mirror table for y-axis specular reflection in D2Q9."""
    mirror = list(range(9))
    for i in range(9):
        cx, cy = C2Q9[i].tolist()
        for j in range(9):
            if C2Q9[j].tolist() == [cx, -cy]:
                mirror[i] = j
                break
    return mirror

MIRROR_Y_2D = _build_mirror_y_2d()


@dataclass(frozen=True)
class IceEdge2DConfig:
    """2D ice-edge zone configuration."""
    nx: int = 200
    ny: int = 40
    u0: float = 0.05
    re: float = 100.0
    ice_edge_frac: float = 0.5
    ls_ice: float = 2.0
    n_steps: int = 10000
    output_interval: int = 1000
    stats_start: int = 5000
    output_root: Path = Path("runs/ice_edge_zone")
    run_name: str | None = None
    device: str = "cpu"

    def __post_init__(self):
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def H(self) -> int:
        return self.ny - 2

    @property
    def nu(self) -> float:
        return self.u0 * self.H / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def x_edge(self) -> int:
        return max(1, int(self.nx * self.ice_edge_frac))

    @property
    def beta_ice(self) -> float:
        return self.ls_ice / (1.0 + self.ls_ice)


def _equilibrium_2d(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium."""
    device = rho.device
    c = C2Q9.to(device).float()
    w = W2Q9.to(device).float()

    u_sq = ux**2 + uy**2
    cu = c[:, 0].view(9,1,1) * ux.unsqueeze(0) + c[:, 1].view(9,1,1) * uy.unsqueeze(0)

    feq = w.view(9,1,1) * rho.unsqueeze(0) * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq.unsqueeze(0))
    return feq


def _stream_2d(f: torch.Tensor) -> torch.Tensor:
    """D2Q9 streaming. f shape: (9, ny, nx)"""
    ny, nx = f.shape[1], f.shape[2]
    f_new = torch.empty_like(f)

    for q in range(9):
        cx, cy = int(C2Q9[q, 0].item()), int(C2Q9[q, 1].item())
        if cx == 0 and cy == 0:
            f_new[q] = f[q]
            continue

        src = f[q]
        if cy > 0:
            tmp = torch.zeros_like(src)
            tmp[cy:, :] = src[:ny-cy, :]
            src = tmp
        elif cy < 0:
            tmp = torch.zeros_like(src)
            tmp[:ny+cy, :] = src[-cy:, :]
            src = tmp

        if cx > 0:
            tmp = torch.zeros_like(src)
            tmp[:, cx:] = src[:, :nx-cx]
            src = tmp
        elif cx < 0:
            tmp = torch.zeros_like(src)
            tmp[:, :nx+cx] = src[:, -cx:]
            src = tmp

        f_new[q] = src

    return f_new


def run_ice_edge_2d(config: IceEdge2DConfig) -> dict:
    """Run 2D ice-edge simulation."""
    device = torch.device(config.device)
    nx, ny = config.nx, config.ny
    omega = config.omega
    nu = config.nu
    u0 = config.u0
    x_edge = config.x_edge
    beta_ice = config.beta_ice

    run_name = config.run_name or f"ice_edge_2d_nx{nx}_ny{ny}_re{config.re:.0f}"
    run_dir = config.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[IceEdge2D] {run_name}: {nx}×{ny}, Re={config.re}, tau={config.tau:.3f}")

    # Initialize
    rho = torch.ones(ny, nx, device=device)
    ux = torch.full((ny, nx), u0, device=device)
    uy = torch.zeros(ny, nx, device=device)
    f = _equilibrium_2d(rho, ux, uy)

    # Time averaging
    u_sum = torch.zeros_like(ux)
    v_sum = torch.zeros_like(uy)
    p_sum = torch.zeros_like(rho)
    n_avg = 0

    opp = OPP2Q9.to(device)

    for step in range(1, config.n_steps + 1):
        # Collision
        rho = f.sum(dim=0)
        jx = (f * C2Q9.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
        jy = (f * C2Q9.to(device)[:, 1].view(9,1,1).float()).sum(dim=0)
        ux = jx / rho
        uy = jy / rho

        feq = _equilibrium_2d(rho, ux, uy)
        f = f - omega * (f - feq)

        # Streaming
        f = _stream_2d(f)

        # Boundary conditions
        # Inlet (x=0): velocity BC
        inlet_dirs = [1, 5, 8]  # cx > 0 in D2Q9
        for q in inlet_dirs:
            oq = int(opp[q].item())
            f[q, :, 0] = feq[q, :, 0] + f[oq, :, 0] - feq[oq, :, 0]

        # Outlet (x=nx-1): zero gradient
        f[:, :, -1] = f[:, :, -2]

        # Bottom wall (y=0): no-slip
        for q in range(9):
            oq = int(opp[q].item())
            if q != oq:
                f[q, 0, :] = f[oq, 0, :]

        # Top wall (y=ny-1): partial slip under ice, free-slip for open water
        # Ice region (x < x_edge)
        if beta_ice > 0 and x_edge > 0:
            for q in range(9):
                oq = int(opp[q].item())
                mq = MIRROR_Y_2D[q]
                f[q, ny-1, :x_edge] = (1.0 - beta_ice) * f[oq, ny-1, :x_edge] + beta_ice * f[mq, ny-1, :x_edge]

        # Open water (x >= x_edge): free-slip
        for q in range(9):
            mq = MIRROR_Y_2D[q]
            if q != mq:
                f[q, ny-1, x_edge:] = f[mq, ny-1, x_edge:]

        # Clamp density for stability
        rho = f.sum(dim=0)
        rho = torch.clamp(rho, min=0.1, max=3.0)
        # Renormalize
        f = f * (rho / f.sum(dim=0)).unsqueeze(0)

        # Time averaging
        if step >= config.stats_start and step % 2 == 0:
            rho = f.sum(dim=0)
            jx = (f * C2Q9.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
            jy = (f * C2Q9.to(device)[:, 1].view(9,1,1).float()).sum(dim=0)
            u_sum += jx / rho
            v_sum += jy / rho
            p_sum += rho / 3.0
            n_avg += 1

        if step % config.output_interval == 0:
            rho = f.sum(dim=0)
            jx = (f * C2Q9.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
            ux = jx / rho
            print(f"  Step {step}: u_max={ux.max():.4f}, rho_mean={rho.mean():.4f}")

    # Final fields
    rho = f.sum(dim=0)
    jx = (f * C2Q9.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
    jy = (f * C2Q9.to(device)[:, 1].view(9,1,1).float()).sum(dim=0)
    ux_final = jx / rho
    uy_final = jy / rho

    if n_avg > 0:
        u_mean = u_sum / n_avg
        v_mean = v_sum / n_avg
        p_mean = p_sum / n_avg
    else:
        u_mean = ux_final
        v_mean = uy_final
        p_mean = rho / 3.0

    # Wall shear stress
    tau_bottom = nu * (u_mean[1, :] - u_mean[0, :])
    tau_top = nu * (u_mean[-1, :] - u_mean[-2, :])

    # Drag coefficient
    F_bottom = tau_bottom.abs().sum().item()
    F_ice = tau_top[:x_edge].abs().sum().item()
    F_open = tau_top[x_edge:].abs().sum().item()
    F_total = F_bottom + F_ice + F_open
    A_ref = config.H
    C_D = F_total / (0.5 * u0**2 * A_ref)

    print(f"  C_D = {C_D:.6f} (bottom={F_bottom:.4f}, ice={F_ice:.4f}, open={F_open:.4f})")

    # Save results
    result = {
        "u_mean": u_mean.cpu().numpy(),
        "v_mean": v_mean.cpu().numpy(),
        "p_mean": p_mean.cpu().numpy(),
        "u_final": ux_final.cpu().numpy(),
        "tau_bottom": tau_bottom.cpu().numpy(),
        "tau_top": tau_top.cpu().numpy(),
        "C_D": C_D,
        "C_D_ice": F_ice / (0.5 * u0**2 * A_ref),
        "C_D_open": F_open / (0.5 * u0**2 * A_ref),
        "config": asdict(config),
    }

    np.savez_compressed(run_dir / "fields.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    with open(run_dir / "results.json", "w") as fp:
        json.dump({k: v for k, v in result.items() if not isinstance(v, np.ndarray)}, fp, indent=2, default=str)

    return result


if __name__ == "__main__":
    config = IceEdge2DConfig(
        nx=200, ny=40,
        u0=0.05, re=100.0,
        ice_edge_frac=0.5, ls_ice=2.0,
        n_steps=5000, output_interval=1000,
        stats_start=2000,
    )
    result = run_ice_edge_2d(config)
    print(f"\nFinal C_D = {result['C_D']:.6f}")
