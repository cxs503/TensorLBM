from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tensorlbm import (
    apply_simple_channel_boundaries,
    collide_bgk,
    cylinder_mask,
    equilibrium,
    macroscopic,
    stream,
)

# Tunable demo parameters
NX = 320
NY = 100
U_IN = 0.08
RE = 100.0
RADIUS = 12.0
N_STEPS = 1200
OUTPUT_INTERVAL = 200
OUTPUT_DIR = Path("outputs")


def compute_vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    dudy = torch.zeros_like(uy)
    dvdx = torch.zeros_like(ux)
    dudy[1:-1, :] = 0.5 * (uy[2:, :] - uy[:-2, :])
    dvdx[:, 1:-1] = 0.5 * (ux[:, 2:] - ux[:, :-2])
    return dvdx - dudy


def main() -> None:
    device = torch.device("cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nu = U_IN * 2.0 * RADIUS / RE
    tau = 3.0 * nu + 0.5

    cx, cy = NX * 0.25, NY * 0.5
    obstacle = cylinder_mask(NX, NY, cx, cy, RADIUS, device=device)

    wall_mask = torch.zeros((NY, NX), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[obstacle] = False

    rho0 = torch.ones((NY, NX), device=device)
    ux0 = torch.full((NY, NX), U_IN, device=device)
    uy0 = torch.zeros((NY, NX), device=device)
    ux0[obstacle] = 0.0

    f = equilibrium(rho0, ux0, uy0, device=device)

    print(f"Running D2Q9 cylinder flow on CPU: NX={NX}, NY={NY}, tau={tau:.4f}, steps={N_STEPS}")

    for step in range(1, N_STEPS + 1):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
        f = apply_simple_channel_boundaries(f, u_in=U_IN, wall_mask=wall_mask, obstacle_mask=obstacle)

        if step % OUTPUT_INTERVAL == 0 or step == N_STEPS:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            mass = rho.sum().item()
            print(f"step={step:5d} mass={mass:.6f} max|u|={speed.max().item():.6f}")

            if step == N_STEPS:
                vort = compute_vorticity(ux, uy)
                speed_np = speed.detach().cpu().numpy()
                vort_np = vort.detach().cpu().numpy()
                obs_np = obstacle.detach().cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
                im0 = axes[0].imshow(speed_np, origin="lower", cmap="viridis")
                axes[0].contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
                axes[0].set_title("Velocity magnitude")
                plt.colorbar(im0, ax=axes[0], fraction=0.046)

                im1 = axes[1].imshow(vort_np, origin="lower", cmap="coolwarm")
                axes[1].contour(obs_np, levels=[0.5], colors="black", linewidths=0.7)
                axes[1].set_title("Vorticity")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                out = OUTPUT_DIR / "cylinder_flow_final.png"
                fig.savefig(out, dpi=160)
                plt.close(fig)
                print(f"Saved: {out}")


if __name__ == "__main__":
    main()
