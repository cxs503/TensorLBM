"""2-D free-energy LBM benchmark: a liquid droplet impacts a quiescent liquid pool.

This is a reproducible exploratory benchmark, not a validated quantitative
reference case. It emits phase-field contours, a time-history CSV, and summary
metadata so that stability, conservation, and impact evolution are inspectable.

Run:
  PYTHONPATH=src python examples/droplet_impact.py --fast --overwrite
  PYTHONPATH=src python examples/droplet_impact.py --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import free_energy_step, init_free_energy_g
from tensorlbm.solver import stream


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=240)
    p.add_argument("--ny", type=int, default=160)
    p.add_argument("--radius", type=float, default=14.0)
    p.add_argument(
        "--gap",
        type=float,
        default=8.0,
        help="Initial gap between droplet bottom and pool interface (cells)",
    )
    p.add_argument(
        "--water-level",
        type=float,
        default=100.0,
        help="Initial pool interface y coordinate; y increases downward",
    )
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--output-interval", type=int, default=150)
    p.add_argument(
        "--g", type=float, default=2.0e-5, help="Downward body acceleration in lattice units"
    )
    p.add_argument("--tau-f", type=float, default=0.8)
    p.add_argument("--tau-g", type=float, default=0.8)
    p.add_argument("--A", type=float, default=0.04)
    p.add_argument("--B", type=float, default=0.04)
    p.add_argument("--kappa", type=float, default=0.02)
    p.add_argument("--Gamma", type=float, default=0.5)
    p.add_argument("--interface-width", type=float, default=3.0)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--output-root", default="outputs/droplet_impact")
    p.add_argument("--run-name", default=None)
    p.add_argument("--fast", action="store_true", help="Small smoke run (not for conclusions)")
    p.add_argument("--overwrite", action="store_true")
    return p


def save_contour(
    phi: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, step: int, path: Path, water_level: float
) -> None:
    field = phi.detach().cpu().numpy()
    u = ux.detach().cpu().numpy()
    v = uy.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    im = ax.imshow(
        field,
        origin="upper",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=(0, field.shape[1], field.shape[0], 0),
        interpolation="nearest",
    )
    yy, xx = torch.meshgrid(
        torch.arange(field.shape[0]), torch.arange(field.shape[1]), indexing="ij"
    )
    ax.contour(xx.numpy(), yy.numpy(), field, levels=[0.0], colors="black", linewidths=1.0)
    ax.axhline(water_level, color="gold", linestyle="--", linewidth=0.8, label="initial pool level")
    stride = max(4, min(field.shape) // 35)
    ax.quiver(
        xx[::stride, ::stride].numpy(),
        yy[::stride, ::stride].numpy(),
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="k",
        alpha=0.35,
        scale=1.5,
        width=0.0015,
    )
    ax.set(
        xlabel="x [lattice cells]",
        ylabel="y [lattice cells]",
        title=f"Droplet impact into a liquid pool — step {step}",
    )
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="order parameter φ (liquid ≈ +1, gas ≈ −1)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    if args.fast:
        args.nx, args.ny, args.n_steps, args.output_interval = 120, 90, 120, 30
        args.radius, args.water_level, args.gap = 9.0, 58.0, 5.0
    if args.nx < 64 or args.ny < 64:
        raise ValueError("nx and ny must both be >= 64")
    if args.output_interval < 1 or args.n_steps < 1:
        raise ValueError("n-steps and output-interval must be positive")
    if args.radius < 4 or args.radius + args.gap >= args.water_level:
        raise ValueError("droplet must fit above the pool and radius must be >= 4")
    if args.water_level >= args.ny - 8:
        raise ValueError("water-level must leave at least 8 cells below the pool")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    run_name = args.run_name or f"impact_{args.nx}x{args.ny}_s{args.n_steps}"
    out = Path(args.output_root) / run_name
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} already contains files; choose --run-name or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "contours").mkdir(exist_ok=True)

    ny, nx = args.ny, args.nx
    y = torch.arange(ny, dtype=torch.float32, device=device)
    x = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy, cx = args.water_level - args.radius - args.gap, nx / 2.0
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    pool_phi = torch.tanh((args.water_level - yy) / args.interface_width)
    drop_phi = torch.tanh((args.radius - r) / args.interface_width)
    # Union of pool and droplet phases; both represent the same liquid phase.
    phi0 = torch.maximum(pool_phi, drop_phi).clamp(-1.0, 1.0)
    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=device)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    g = init_free_energy_g(phi0, zero, zero)
    initial_mass = float(phi0.sum().item())
    started = time.perf_counter()
    rows: list[dict[str, float | int]] = []

    def record(step: int) -> None:
        phi = g.sum(dim=0)
        _rho, ux, uy = macroscopic(f)
        speed2 = ux.square() + uy.square()
        liquid_weight = ((phi + 1.0) * 0.5).clamp(0.0, 1.0)
        # Track the upper region containing the falling droplet; this metric is
        # explicitly not a conserved "droplet mass" after pool contact.
        top = yy < min(ny, args.water_level + 2.0 * args.radius)
        w = liquid_weight * top
        denom = w.sum().clamp_min(1e-12)
        cx_now = float((w * xx).sum().item() / denom.item())
        cy_now = float((w * yy).sum().item() / denom.item())
        mass = float(phi.sum().item())
        row = {
            "step": step,
            "time_lu": float(step),
            "phase_mass": mass,
            "relative_phase_mass_drift": abs(mass - initial_mass) / max(abs(initial_mass), 1e-12),
            "upper_region_liquid_centroid_x": cx_now,
            "upper_region_liquid_centroid_y": cy_now,
            "max_speed": float(torch.sqrt(speed2).max().item()),
            "kinetic_energy_proxy": float((0.5 * rho0 * speed2).sum().item()),
            "phi_min": float(phi.min().item()),
            "phi_max": float(phi.max().item()),
        }
        rows.append(row)
        save_contour(
            phi, ux, uy, step, out / "contours" / f"phase_{step:07d}.png", args.water_level
        )

    record(0)
    for step in range(1, args.n_steps + 1):
        f, g = free_energy_step(
            f,
            g,
            tau_f=args.tau_f,
            tau_g=args.tau_g,
            A=args.A,
            B=args.B,
            kappa=args.kappa,
            Gamma=args.Gamma,
            gx=0.0,
            gy=args.g,
            rho_heavy=1.0,
            rho_light=0.1,
        )
        f, g = stream(f), stream(g)
        if step % args.output_interval == 0 or step == args.n_steps:
            record(step)
            last = rows[-1]
            print(
                f"step={step:6d} mass_drift={last['relative_phase_mass_drift']:.3e} "
                f"max_u={last['max_speed']:.3e} "
                f"centroid_y={last['upper_region_liquid_centroid_y']:.2f}"
            )

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    elapsed = time.perf_counter() - started
    final = rows[-1]
    metadata = {
        "benchmark": "2d_free_energy_droplet_impact",
        "status": "exploratory_not_yet_reference_validated",
        "config": vars(args),
        "derived": {
            "initial_droplet_center_y": cy,
            "initial_droplet_center_x": cx,
            "initial_phase_mass": initial_mass,
            "elapsed_seconds_including_plotting": elapsed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "final_metrics": final,
        "diagnostic_notes": [
            "phase_mass is the signed order-parameter sum and is expected to be conserved approximately",
            "upper_region_liquid_centroid is a visualization diagnostic, not a droplet trajectory after coalescence",
            "kinetic_energy_proxy uses the uniform initial density and is not a calibrated physical energy",
            "quantitative validation requires mesh/time-step convergence and comparison against published experimental or numerical data",
        ],
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(steps, [row["relative_phase_mass_drift"] for row in rows], marker="o")
    axes[0, 0].set(title="Relative phase-mass drift", xlabel="step", ylabel="relative drift")
    axes[0, 1].plot(steps, [row["upper_region_liquid_centroid_y"] for row in rows], marker="o")
    axes[0, 1].set(title="Upper-region liquid centroid", xlabel="step", ylabel="y [cells]")
    axes[0, 2].plot(steps, [row["max_speed"] for row in rows], marker="o")
    axes[0, 2].set(title="Maximum speed", xlabel="step", ylabel="|u| [lu/step]")
    axes[1, 1].plot(steps, [row["kinetic_energy_proxy"] for row in rows], marker="o")
    axes[1, 1].set(title="Kinetic-energy proxy", xlabel="step", ylabel="sum(ρ|u|²/2)")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0,
        0.9,
        f"2-D free-energy droplet impact\nGrid: {nx} × {ny}\n"
        f"Steps: {args.n_steps}\nDevice: {device}\n"
        f"Final mass drift: {final['relative_phase_mass_drift']:.3e}\n"
        f"Elapsed: {elapsed:.1f} s\n\nExploratory result — not reference validated.",
        va="top",
        family="monospace",
    )
    fig.savefig(out / "diagnostic_curves.png", dpi=160)
    plt.close(fig)
    print(f"Artifacts written to: {out}")
    print("  contours/phase_*.png, history.csv, diagnostic_curves.png, run_metadata.json")


if __name__ == "__main__":
    main()
