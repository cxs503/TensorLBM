"""Ice-edge zone flow: pragmatic demonstration using TensorLBM.

This module demonstrates the key physics of ice-edge zone flow:
1. Transition from ice-covered (partial-slip) to open-water (free-slip) surface
2. Flow structure development at the ice edge
3. Drag coefficient quantification

Uses a simplified 2D approach for stability and speed, with clear documentation
of the physics and limitations.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.d2q9 import C, W, OPPOSITE

__all__ = ["IceEdgeDemo", "run_ice_edge_demo"]


@dataclass
class IceEdgeDemo:
    """Ice-edge zone demonstration configuration."""
    nx: int = 200
    ny: int = 40
    u0: float = 0.05
    re: float = 100.0
    ice_edge_frac: float = 0.5
    ls_ice: float = 2.0
    n_steps: int = 5000
    output_dir: Path = Path("runs/ice_edge_zone/demo")

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
        return int(self.nx * self.ice_edge_frac)

    @property
    def beta_ice(self) -> float:
        """Slip parameter: 0=no-slip, 1=free-slip."""
        return self.ls_ice / (1.0 + self.ls_ice)


def _equilibrium(rho, ux, uy, device):
    """D2Q9 equilibrium distribution."""
    c = C.to(device).float()
    w = W.to(device).float()
    u_sq = ux**2 + uy**2
    cu = c[:, 0].view(9,1,1) * ux.unsqueeze(0) + c[:, 1].view(9,1,1) * uy.unsqueeze(0)
    return w.view(9,1,1) * rho.unsqueeze(0) * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq.unsqueeze(0))


def _stream(f):
    """D2Q9 streaming with bounded y, periodic x."""
    ny, nx = f.shape[1], f.shape[2]
    f_new = torch.empty_like(f)
    for q in range(9):
        cx, cy = int(C[q, 0].item()), int(C[q, 1].item())
        src = f[q]
        # y-shift (bounded)
        if cy > 0:
            tmp = torch.zeros_like(src)
            tmp[cy:, :] = src[:ny-cy, :]
            src = tmp
        elif cy < 0:
            tmp = torch.zeros_like(src)
            tmp[:ny+cy, :] = src[-cy:, :]
            src = tmp
        # x-shift (periodic)
        if cx != 0:
            src = torch.roll(src, shifts=cx, dims=1)
        f_new[q] = src
    return f_new


def run_ice_edge_demo(config: IceEdgeDemo) -> dict:
    """Run ice-edge zone demonstration."""
    device = torch.device("cpu")
    nx, ny = config.nx, config.ny
    omega = config.omega
    nu = config.nu
    u0 = config.u0
    x_edge = config.x_edge
    beta_ice = config.beta_ice

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[IceEdge Demo] {nx}×{ny}, Re={config.re}, tau={config.tau:.3f}")
    print(f"[IceEdge Demo] Ice edge at x={x_edge}, beta_ice={beta_ice:.3f}")

    # Initialize
    rho = torch.ones(ny, nx, device=device)
    ux = torch.full((ny, nx), u0, device=device)
    uy = torch.zeros(ny, nx, device=device)
    f = _equilibrium(rho, ux, uy, device)

    opp = OPPOSITE.to(device)

    # Mirror table for y-reflection
    mirror_y = [0, 1, 4, 3, 2, 8, 7, 6, 5]

    # Time averaging
    u_sum = torch.zeros_like(ux)
    n_avg = 0

    t0 = time.time()
    for step in range(1, config.n_steps + 1):
        # Compute macroscopic
        rho = f.sum(dim=0)
        jx = (f * C.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
        jy = (f * C.to(device)[:, 1].view(9,1,1).float()).sum(dim=0)
        ux = jx / rho
        uy = jy / rho

        # Collision
        feq = _equilibrium(rho, ux, uy, device)
        f = f - omega * (f - feq)

        # Streaming
        f = _stream(f)

        # Boundary conditions
        # Bottom wall (y=0): no-slip
        for q in range(9):
            oq = int(opp[q].item())
            if q != oq:
                f[q, 0, :] = f[oq, 0, :]

        # Top wall (y=ny-1): partial slip under ice, free-slip for open water
        # Ice region (x < x_edge): partial slip
        for q in range(9):
            oq = int(opp[q].item())
            mq = mirror_y[q]
            # Only apply to directions pointing into fluid (cy < 0 for top wall)
            if C[q, 1].item() < 0:
                f[q, ny-1, :x_edge] = (1.0 - beta_ice) * f[oq, ny-1, :x_edge] + beta_ice * f[mq, ny-1, :x_edge]

        # Open water (x >= x_edge): free-slip
        for q in range(9):
            mq = mirror_y[q]
            if C[q, 1].item() < 0 and q != mq:
                f[q, ny-1, x_edge:] = f[mq, ny-1, x_edge:]

        # Inlet/outlet: periodic in x (already handled by streaming)
        # Add small relaxation to maintain mean flow
        if step % 100 == 0:
            u_mean_current = ux.mean().item()
            correction = u0 / max(u_mean_current, 1e-6)
            f = f * correction

        # Time averaging
        if step > config.n_steps // 2 and step % 2 == 0:
            rho = f.sum(dim=0)
            jx = (f * C.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
            u_sum += jx / rho
            n_avg += 1

        if step % 1000 == 0:
            rho = f.sum(dim=0)
            jx = (f * C.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
            ux = jx / rho
            print(f"  Step {step}: u_max={ux.max():.4f}, u_mean={ux.mean():.4f}")

    elapsed = time.time() - t0

    # Final fields
    if n_avg > 0:
        u_mean = u_sum / n_avg
    else:
        rho = f.sum(dim=0)
        jx = (f * C.to(device)[:, 0].view(9,1,1).float()).sum(dim=0)
        u_mean = jx / rho

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

    print(f"\n[Results] C_D = {C_D:.4f}")
    print(f"[Results] F_bottom={F_bottom:.4f}, F_ice={F_ice:.4f}, F_open={F_open:.4f}")
    print(f"[Results] Elapsed: {elapsed:.1f}s")

    # Generate figures
    fig_dir = config.output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    figures = _generate_figures(u_mean, tau_bottom, tau_top, x_edge, fig_dir)

    # Compute sha256
    h = hashlib.sha256()
    h.update(u_mean.cpu().numpy().tobytes())
    sha256 = h.hexdigest()

    result = {
        "u_mean": u_mean.cpu().numpy(),
        "tau_bottom": tau_bottom.cpu().numpy(),
        "tau_top": tau_top.cpu().numpy(),
        "C_D": C_D,
        "C_D_ice": F_ice / (0.5 * u0**2 * A_ref),
        "C_D_open": F_open / (0.5 * u0**2 * A_ref),
        "elapsed_s": elapsed,
        "sha256": sha256,
        "config": asdict(config),
        "figures": figures,
    }

    # Save
    np.savez_compressed(config.output_dir / "fields.npz",
                        u_mean=result["u_mean"],
                        tau_bottom=result["tau_bottom"],
                        tau_top=result["tau_top"])
    with open(config.output_dir / "results.json", "w") as fp:
        json.dump({k: v for k, v in result.items() if k not in ["u_mean", "tau_bottom", "tau_top"]},
                  fp, indent=2, default=str)

    return result


def _generate_figures(u_mean, tau_bottom, tau_top, x_edge, fig_dir):
    """Generate visualization figures."""
    figures = []

    # Figure 1: Velocity field
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(u_mean, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="u / U_0")
    ax.axvline(x=x_edge, color="red", linestyle="--", linewidth=2, label="Ice edge")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Streamwise velocity field")
    ax.legend()
    path = fig_dir / "velocity_field.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "velocity_field", "path": str(path)})

    # Figure 2: Wall shear stress
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(tau_bottom, "b-", linewidth=1.5, label="Bottom wall")
    axes[0].axvline(x=x_edge, color="red", linestyle="--", label="Ice edge")
    axes[0].set_ylabel("τ_w (bottom)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tau_top, "r-", linewidth=1.5, label="Top wall")
    axes[1].axvline(x=x_edge, color="red", linestyle="--", label="Ice edge")
    axes[1].set_xlabel("x (lattice units)")
    axes[1].set_ylabel("τ_w (top)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Wall shear stress distribution")
    plt.tight_layout()
    path = fig_dir / "wall_shear.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "wall_shear", "path": str(path)})

    # Figure 3: Velocity profiles
    fig, ax = plt.subplots(figsize=(8, 6))
    ny = u_mean.shape[0]
    y_vals = np.arange(ny)
    x_locs = [int(0.2*u_mean.shape[1]), x_edge, int(0.8*u_mean.shape[1])]
    for x_loc in x_locs:
        ax.plot(u_mean[:, x_loc], y_vals, label=f"x={x_loc}")
    ax.set_xlabel("u (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Velocity profiles at different x-locations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = fig_dir / "velocity_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "velocity_profiles", "path": str(path)})

    return figures


if __name__ == "__main__":
    config = IceEdgeDemo(
        nx=200, ny=40,
        u0=0.05, re=100.0,
        ice_edge_frac=0.5, ls_ice=2.0,
        n_steps=5000,
    )
    result = run_ice_edge_demo(config)
    print(f"\nFinal C_D = {result['C_D']:.4f}")
    print(f"Figures: {len(result['figures'])}")
