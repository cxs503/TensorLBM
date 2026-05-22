from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


@dataclass
class RunConfig:
    nx: int = 320
    ny: int = 100
    u_in: float = 0.08
    re: float = 100.0
    tau: float | None = None
    radius: float = 12.0
    cx: float | None = None
    cy: float | None = None
    n_steps: int = 1200
    output_interval: int = 200
    log_interval: int = 50
    output_root: str = "outputs"
    run_name: str | None = None
    device: str = "cpu"


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="PyTorch D2Q9 cylinder-flow demo")
    parser.add_argument("--nx", type=int, default=RunConfig.nx, help="Grid size in x direction")
    parser.add_argument("--ny", type=int, default=RunConfig.ny, help="Grid size in y direction")
    parser.add_argument("--u-in", type=float, default=RunConfig.u_in, help="Inlet velocity")
    parser.add_argument("--re", type=float, default=RunConfig.re, help="Target Reynolds number")
    parser.add_argument("--tau", type=float, default=None, help="BGK relaxation parameter (overrides --re)")
    parser.add_argument("--radius", type=float, default=RunConfig.radius, help="Cylinder radius")
    parser.add_argument("--cx", type=float, default=None, help="Cylinder center x (default: 0.25*nx)")
    parser.add_argument("--cy", type=float, default=None, help="Cylinder center y (default: 0.5*ny)")
    parser.add_argument("--steps", type=int, default=RunConfig.n_steps, help="Number of simulation steps")
    parser.add_argument(
        "--output-interval",
        type=int,
        default=RunConfig.output_interval,
        help="Interval for saving flow images",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=RunConfig.log_interval,
        help="Interval for runtime diagnostics",
    )
    parser.add_argument("--output-root", type=str, default=RunConfig.output_root, help="Base output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name")
    parser.add_argument("--device", type=str, default=RunConfig.device, choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args(argv)
    return RunConfig(
        nx=args.nx,
        ny=args.ny,
        u_in=args.u_in,
        re=args.re,
        tau=args.tau,
        radius=args.radius,
        cx=args.cx,
        cy=args.cy,
        n_steps=args.steps,
        output_interval=args.output_interval,
        log_interval=args.log_interval,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
    )


def resolve_tau(config: RunConfig) -> float:
    tau = config.tau
    if tau is None:
        nu = config.u_in * 2.0 * config.radius / config.re
        tau = 3.0 * nu + 0.5
    if tau <= 0.5:
        raise ValueError(f"Unstable relaxation parameter tau={tau:.6f}; must be > 0.5")
    return tau


def make_run_dir(config: RunConfig) -> Path:
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if config.run_name:
        base = config.run_name
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = f"cylinder_nx{config.nx}_ny{config.ny}_steps{config.n_steps}_{ts}"

    run_dir = root / base
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{base}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    config: RunConfig,
    tau: float,
    cx: float,
    cy: float,
    device: torch.device,
) -> Path:
    metadata = {
        "config": asdict(config),
        "derived": {"tau": tau, "cylinder_center_x": cx, "cylinder_center_y": cy},
        "runtime": {"device": str(device), "created_utc": datetime.now(timezone.utc).isoformat()},
    }
    out = run_dir / "run_metadata.json"
    out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def compute_vorticity(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    dudy = torch.zeros_like(uy)
    dvdx = torch.zeros_like(ux)
    dudy[1:-1, :] = 0.5 * (uy[2:, :] - uy[:-2, :])
    dvdx[:, 1:-1] = 0.5 * (ux[:, 2:] - ux[:, :-2])
    return dvdx - dudy


def save_snapshot(run_dir: Path, step: int, speed: torch.Tensor, vort: torch.Tensor, obstacle: torch.Tensor) -> Path:
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

    out = run_dir / f"flow_step_{step:06d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)

    device_name = config.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    run_dir = make_run_dir(config)
    tau = resolve_tau(config)

    cx = config.cx if config.cx is not None else config.nx * 0.25
    cy = config.cy if config.cy is not None else config.ny * 0.5
    obstacle = cylinder_mask(config.nx, config.ny, cx, cy, config.radius, device=device)

    wall_mask = torch.zeros((config.ny, config.nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[obstacle] = False

    rho0 = torch.ones((config.ny, config.nx), device=device)
    ux0 = torch.full((config.ny, config.nx), config.u_in, device=device)
    uy0 = torch.zeros((config.ny, config.nx), device=device)
    ux0[obstacle] = 0.0

    f = equilibrium(rho0, ux0, uy0, device=device)
    meta_path = save_run_metadata(run_dir, config, tau=tau, cx=cx, cy=cy, device=device)

    print(
        f"Running D2Q9 cylinder flow: device={device.type}, NX={config.nx}, NY={config.ny}, "
        f"tau={tau:.4f}, steps={config.n_steps}"
    )
    print(f"Output directory: {run_dir}")
    print(f"Saved metadata: {meta_path}")

    for step in range(1, config.n_steps + 1):
        f = collide_bgk(f, tau=tau)
        f = stream(f)
        f = apply_simple_channel_boundaries(f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=obstacle)

        log_due = step % config.log_interval == 0 or step == config.n_steps
        output_due = step % config.output_interval == 0 or step == config.n_steps
        if log_due or output_due:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(obstacle, 0.0)
            uy = uy.masked_fill(obstacle, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            mass = rho.sum().item()
            if log_due:
                print(
                    f"step={step:5d} mass={mass:.6f} "
                    f"rho[min,max]=({rho.min().item():.6f},{rho.max().item():.6f}) "
                    f"mean|u|={speed.mean().item():.6f} max|u|={speed.max().item():.6f}"
                )

            if output_due:
                vort = compute_vorticity(ux, uy)
                out = save_snapshot(run_dir, step, speed, vort, obstacle)
                print(f"Saved: {out}")


if __name__ == "__main__":
    main()
