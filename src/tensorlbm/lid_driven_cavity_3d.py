"""3D Lid-Driven Cavity benchmark: D3Q27 vs D3Q19 comparison.

Runs a spanwise-periodic 3D lid-driven cavity at Re=400 with both D3Q27
and D3Q19 lattices, compares centreline velocity profiles against the
Ghia et al. (1982) reference, and produces comparison figures.

This is the IN14-D9 campaign's "顶盖驱动流 D3Q27 vs D3Q19 已验证锚带内对比".
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.boundary.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import C as C19, W as W19, equilibrium3d, macroscopic3d
from tensorlbm.solver import collide_mrt
from tensorlbm.d3q27 import (
    C as C27,
    OPPOSITE as OPPOSITE27,
    W as W27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27,
)

# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data (Re=400)
# ---------------------------------------------------------------------------

GHIA_RE400 = {
    "y": [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
        0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
    ],
    "u": [
        1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
        -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
        -0.08186, 0.00000,
    ],
    "x": [
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
        0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000,
    ],
    "v": [
        0.00000, 0.18360, 0.19713, 0.20920, 0.22980, 0.28003, 0.30174, 0.30203,
        0.05186, -0.38598, -0.44993, -0.38598, -0.22847, -0.19254, -0.15663,
        -0.12146, 0.00000,
    ],
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cavity3DConfig:
    """Configuration for the 3D lid-driven cavity benchmark."""
    nx: int = 64
    ny: int = 64
    nz: int = 8  # spanwise (periodic)
    u_lid: float = 0.06
    re: float = 400.0
    n_steps: int = 8000
    output_interval: int = 2000
    device: str = "cpu"

    @property
    def nu(self) -> float:
        return self.u_lid * self.nx / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 16 or self.ny < 16:
            raise ValueError("nx, ny must be >= 16")
        if self.nz < 2:
            raise ValueError("nz must be >= 2 for 3D")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau:.4f} <= 0.5; increase Re or reduce u_lid/nx")


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------


def make_cavity_wall_mask_3d(nz: int, ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Stationary walls: x=0, x=nx-1, y=0 (bottom). Top (y=ny-1) is the lid."""
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True   # x=0
    wall[:, :, -1] = True  # x=nx-1
    wall[:, 0, :] = True   # y=0
    return wall


def bounce_back_3d(f: torch.Tensor, wall_mask: torch.Tensor, opposite: torch.Tensor) -> torch.Tensor:
    """Half-way bounce-back for stationary walls (pre-streaming style)."""
    f_out = f.clone()
    for q in range(1, f.shape[0]):
        opp = int(opposite[q].item())
        f_out[q, wall_mask] = f[opp, wall_mask]
    return f_out


def stream_periodic_z(f: torch.Tensor) -> torch.Tensor:
    """Stream with periodic BC in z (spanwise)."""
    Q = f.shape[0]
    out = torch.empty_like(f)
    for q in range(Q):
        cx = int(C27[q, 0].item()) if Q == 27 else int(C19[q, 0].item())
        cy = int(C27[q, 1].item()) if Q == 27 else int(C19[q, 1].item())
        cz = int(C27[q, 2].item()) if Q == 27 else int(C19[q, 2].item())
        if cx == 0 and cy == 0 and cz == 0:
            out[q] = f[q]
        else:
            out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


def apply_moving_lid_3d(f: torch.Tensor, u_lid: float, Q: int) -> torch.Tensor:
    """Zou/He moving lid BC at y=ny-1."""
    if Q == 27:
        return zou_he_moving_lid_3d(f, u_lid=u_lid, uy_lid=0.0, uz_lid=0.0)
    # For D3Q19, use the same function (it works for both)
    return zou_he_moving_lid_3d(f, u_lid=u_lid, uy_lid=0.0, uz_lid=0.0)


# ---------------------------------------------------------------------------
# Solver step
# ---------------------------------------------------------------------------


def step_cavity_3d(
    f: torch.Tensor,
    wall_mask: torch.Tensor,
    u_lid: float,
    tau: float,
    Q: int,
) -> torch.Tensor:
    """One LBM step: collide → bounce-back → stream → moving lid."""
    if Q == 27:
        f = collide_mrt27(f, tau=tau)
        opposite = OPPOSITE27
    else:
        f = collide_mrt(f, tau=tau)
        from tensorlbm.d3q19 import OPPOSITE
        opposite = OPPOSITE

    # Bounce-back on stationary walls
    f = bounce_back_3d(f, wall_mask, opposite)

    # Stream (periodic in z)
    f = stream_periodic_z(f)

    # Moving lid
    f = apply_moving_lid_3d(f, u_lid, Q)

    return f


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------


def run_cavity_3d(config: Cavity3DConfig, Q: int) -> dict:
    """Run the 3D cavity simulation and return results."""
    device = torch.device(config.device)
    nz, ny, nx = config.nz, config.ny, config.nx
    wall_mask = make_cavity_wall_mask_3d(nz, ny, nx, device)

    # Initialize
    rho = torch.ones((nz, ny, nx), device=device)
    zeros = torch.zeros_like(rho)

    if Q == 27:
        f = equilibrium27(rho, zeros, zeros, zeros)
    else:
        f = equilibrium3d(rho, zeros, zeros, zeros)

    # Time stepping
    for step in range(config.n_steps):
        f = step_cavity_3d(f, wall_mask, config.u_lid, config.tau, Q)

    # Extract mid-plane (z = nz//2) centreline profiles
    z_mid = nz // 2
    rho_f, ux, uy, uz = (
        macroscopic27(f, device) if Q == 27 else macroscopic3d(f, device)
    )

    # u along vertical centreline (x = nx//2)
    x_mid = nx // 2
    u_vertical = ux[z_mid, :, x_mid].cpu().numpy()
    y_positions = np.linspace(0, 1, ny)

    # v along horizontal centreline (y = ny//2)
    y_mid = ny // 2
    v_horizontal = uy[z_mid, y_mid, :].cpu().numpy()
    x_positions = np.linspace(0, 1, nx)

    # Normalise by u_lid
    u_vertical_norm = u_vertical / config.u_lid
    v_horizontal_norm = v_horizontal / config.u_lid

    # Compute error vs Ghia
    u_ghia = np.array(GHIA_RE400["u"])
    y_ghia = np.array(GHIA_RE400["y"])
    v_ghia = np.array(GHIA_RE400["v"])
    x_ghia = np.array(GHIA_RE400["x"])

    # Interpolate simulation results to Ghia points
    u_sim_at_ghia = np.interp(y_ghia, y_positions[::-1], u_vertical_norm[::-1])
    v_sim_at_ghia = np.interp(x_ghia, x_positions, v_horizontal_norm)

    # L2 error
    u_l2 = float(np.sqrt(np.mean((u_sim_at_ghia - u_ghia) ** 2)))
    v_l2 = float(np.sqrt(np.mean((v_sim_at_ghia - v_ghia) ** 2)))
    u_max = float(np.max(np.abs(u_sim_at_ghia - u_ghia)))
    v_max = float(np.max(np.abs(v_sim_at_ghia - v_ghia)))

    return {
        "Q": Q,
        "y_positions": y_positions.tolist(),
        "u_vertical_norm": u_vertical_norm.tolist(),
        "x_positions": x_positions.tolist(),
        "v_horizontal_norm": v_horizontal_norm.tolist(),
        "u_l2_error": u_l2,
        "v_l2_error": v_l2,
        "u_max_error": u_max,
        "v_max_error": v_max,
        "ghia_y": y_ghia.tolist(),
        "ghia_u": u_ghia.tolist(),
        "ghia_x": x_ghia.tolist(),
        "ghia_v": v_ghia.tolist(),
        "u_sim_at_ghia": u_sim_at_ghia.tolist(),
        "v_sim_at_ghia": v_sim_at_ghia.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(results_27: dict, results_19: dict, output_dir: Path) -> list[dict]:
    """Generate comparison figures and return figure metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # Figure 1: u profile along vertical centreline
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results_27["u_vertical_norm"], results_27["y_positions"],
            "b-", label="D3Q27", linewidth=1.5)
    ax.plot(results_19["u_vertical_norm"], results_19["y_positions"],
            "r--", label="D3Q19", linewidth=1.5)
    ax.plot(results_27["ghia_u"], results_27["ghia_y"],
            "ko", markersize=4, label="Ghia et al. (1982)", fillstyle="none")
    ax.set_xlabel("u / u_lid")
    ax.set_ylabel("y / H")
    ax.set_title("Vertical Centreline Velocity Profile (Re=400)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    path_u = output_dir / "cavity_u_profile.png"
    fig.savefig(path_u, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "cavity_u_profile", "caption": "u/u_lid along vertical centreline", "path": str(path_u)})

    # Figure 2: v profile along horizontal centreline
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results_27["v_horizontal_norm"], results_27["x_positions"],
            "b-", label="D3Q27", linewidth=1.5)
    ax.plot(results_19["v_horizontal_norm"], results_19["x_positions"],
            "r--", label="D3Q19", linewidth=1.5)
    ax.plot(results_27["ghia_v"], results_27["ghia_x"],
            "ko", markersize=4, label="Ghia et al. (1982)", fillstyle="none")
    ax.set_xlabel("v / u_lid")
    ax.set_ylabel("x / H")
    ax.set_title("Horizontal Centreline Velocity Profile (Re=400)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path_v = output_dir / "cavity_v_profile.png"
    fig.savefig(path_v, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "cavity_v_profile", "caption": "v/u_lid along horizontal centreline", "path": str(path_v)})

    # Figure 3: Error comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["u L2", "v L2", "u max", "v max"]
    d27_vals = [results_27["u_l2_error"], results_27["v_l2_error"],
                results_27["u_max_error"], results_27["v_max_error"]]
    d19_vals = [results_19["u_l2_error"], results_19["v_l2_error"],
                results_19["u_max_error"], results_19["v_max_error"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, d27_vals, width, label="D3Q27", color="steelblue")
    ax.bar(x + width/2, d19_vals, width, label="D3Q19", color="coral")
    ax.set_ylabel("Error")
    ax.set_title("D3Q27 vs D3Q19 Error vs Ghia Reference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path_err = output_dir / "cavity_error_comparison.png"
    fig.savefig(path_err, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "cavity_error_comparison", "caption": "Error comparison bar chart", "path": str(path_err)})

    return figures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: str = "artifacts/cavity_3d_comparison") -> dict:
    """Run D3Q27 and D3Q19 cavity simulations and compare."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = Cavity3DConfig(nx=64, ny=64, nz=8, u_lid=0.06, re=400.0, n_steps=8000)
    config.validate()

    print(f"Running D3Q27 cavity (nx={config.nx}, ny={config.ny}, nz={config.nz}, Re={config.re})...")
    results_27 = run_cavity_3d(config, Q=27)

    print(f"Running D3Q19 cavity (nx={config.nx}, ny={config.ny}, nz={config.nz}, Re={config.re})...")
    results_19 = run_cavity_3d(config, Q=19)

    print("Generating comparison figures...")
    figures = plot_comparison(results_27, results_19, out)

    # Save results
    results = {
        "config": asdict(config),
        "d3q27": {
            "u_l2_error": results_27["u_l2_error"],
            "v_l2_error": results_27["v_l2_error"],
            "u_max_error": results_27["u_max_error"],
            "v_max_error": results_27["v_max_error"],
        },
        "d3q19": {
            "u_l2_error": results_19["u_l2_error"],
            "v_l2_error": results_19["v_l2_error"],
            "u_max_error": results_19["u_max_error"],
            "v_max_error": results_19["v_max_error"],
        },
        "figures": figures,
    }

    results_path = out / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"D3Q27: u_L2={results_27['u_l2_error']:.6f}, v_L2={results_27['v_l2_error']:.6f}")
    print(f"D3Q19: u_L2={results_19['u_l2_error']:.6f}, v_L2={results_19['v_l2_error']:.6f}")

    return results


if __name__ == "__main__":
    main()
