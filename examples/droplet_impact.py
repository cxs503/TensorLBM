"""2-D free-energy LBM benchmark: a liquid droplet impacts a quiescent liquid pool.

This is a reproducible exploratory benchmark, not a validated quantitative
reference case. It emits phase-field contours, a time-history CSV, and summary
metadata so that stability, conservation, and impact evolution are inspectable.

GH #408 fixes applied:
  ① Periodic streaming → physical tank boundaries (bottom wall + side walls + outflow)
  ② Initial water line = static reference, separated from dynamic interface
  ③ Post-coalescence centroid: reports two-phase mass-weighted centroid
  ④ Boussinesq → cAC density-ratio pathway (density_ratio parameter)

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
from tensorlbm.boundary.boundaries import bounce_back_cells

# ---------------------------------------------------------------------------
# Boundary-condition helpers for physical tank
# ---------------------------------------------------------------------------


def _apply_tank_boundaries(
    f: torch.Tensor,
    g: torch.Tensor,
    wall_mask: torch.Tensor,
    top_bc: str = "pressure",
    rho_top: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply physical tank boundary conditions after streaming.

    Boundary layout (y increases downward):
      - Bottom (y=ny-1): bounce-back wall (solid floor)
      - Sides (x=0, x=nx-1): bounce-back walls
      - Top (y=0): pressure/outlet BC (open to atmosphere)

    Args:
        f: Momentum distribution (9, ny, nx)
        g: Phase-field distribution (9, ny, nx)
        wall_mask: Boolean mask of wall cells (ny, nx)
        top_bc: 'pressure' for fixed-density outlet, 'bounce_back' for closed top
        rho_top: Density at top boundary (for pressure BC)

    Returns:
        Updated (f, g) with boundary conditions applied.
    """
    # Bounce-back on walls (bottom + sides)
    f = bounce_back_cells(f, wall_mask)
    g = bounce_back_cells(g, wall_mask)

    # Top boundary: pressure outlet (open to atmosphere)
    if top_bc == "pressure":
        ny, nx = f.shape[1], f.shape[2]
        # After streaming at y=0:
        #   Known (pointing out of domain or tangential): 0,1,3,4,7,8
        #   Unknown (pointing into domain from outside): 2,5,6
        # Use zero-gradient extrapolation for unknown populations
        f_new = f.clone()
        f_new[2, 0, :] = f[4, 1, :]   # f2 from f4 one row below
        f_new[5, 0, :] = f[7, 1, :]   # f5 from f7 one row below
        f_new[6, 0, :] = f[8, 1, :]   # f6 from f8 one row below
        f = f_new

        # Same for g (phase field): zero-gradient extrapolation
        g_new = g.clone()
        g_new[2, 0, :] = g[4, 1, :]
        g_new[5, 0, :] = g[7, 1, :]
        g_new[6, 0, :] = g[8, 1, :]
        g = g_new

    return f, g


def _make_tank_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Create boolean mask for tank walls: bottom + left + right sides.

    Top is open (pressure outlet), not included in wall mask.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[-1, :] = True   # bottom wall
    mask[:, 0] = True    # left wall
    mask[:, -1] = True   # right wall
    return mask


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

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
        help="Initial pool interface y coordinate (static reference); y increases downward",
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
    p.add_argument(
        "--density-ratio",
        type=float,
        default=None,
        help="Target density ratio rho_heavy/rho_light. If set, overrides rho_heavy/rho_light "
             "with physically consistent values (cAC pathway).",
    )
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--output-root", default="outputs/droplet_impact")
    p.add_argument("--run-name", default=None)
    p.add_argument("--fast", action="store_true", help="Small smoke run (not for conclusions)")
    p.add_argument("--overwrite", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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
    # Static reference line (initial pool level) — distinct from dynamic interface
    ax.axhline(water_level, color="gold", linestyle="--", linewidth=0.8, label="initial pool level (static ref)")
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


# ---------------------------------------------------------------------------
# Initial phase field
# ---------------------------------------------------------------------------

def _initial_phase_field(
    xx: torch.Tensor,
    yy: torch.Tensor,
    radius: float,
    center_x: float,
    center_y: float,
    water_level: float,
    interface_width: float,
) -> torch.Tensor:
    """Initialize phase field with static pool and droplet.

    The water_level is a static reference; the actual interface evolves dynamically.
    """
    pool_phi = torch.tanh((yy - water_level) / interface_width)
    r = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    drop_phi = torch.tanh((radius - r) / interface_width)
    return torch.maximum(pool_phi, drop_phi).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Density ratio pathway (cAC connection)
# ---------------------------------------------------------------------------

def _resolve_density_params(
    density_ratio: float | None,
    rho_heavy_default: float,
    rho_light_default: float,
) -> tuple[float, float]:
    """Resolve heavy/light density parameters.

    If density_ratio is provided, compute rho_heavy/rho_light consistent with
    the target ratio while keeping the mean density near 1.0 (cAC pathway).
    This connects to LBM-P0-CAC-G for real water-air density ratios.

    For the Free-Energy model, practical density ratios are limited to ~10:1
    due to stability. For higher ratios, the Allen-Cahn (cAC) formulation
    should be used (see LBM-P0-CAC-G).

    Args:
        density_ratio: Target rho_heavy / rho_light, or None for defaults.
        rho_heavy_default: Default heavy phase density.
        rho_light_default: Default light phase density.

    Returns:
        (rho_heavy, rho_light) tuple.
    """
    if density_ratio is None:
        return rho_heavy_default, rho_light_default

    if density_ratio < 1.0:
        raise ValueError(f"density_ratio must be >= 1.0, got {density_ratio}")

    # cAC pathway: maintain mean density ~1.0 for numerical stability
    # rho_heavy = 2 * ratio / (1 + ratio)
    # rho_light = 2 / (1 + ratio)
    # This gives rho_heavy / rho_light = ratio and (rho_heavy + rho_light) / 2 = 1.0
    rho_heavy = 2.0 * density_ratio / (1.0 + density_ratio)
    rho_light = 2.0 / (1.0 + density_ratio)

    # Warn if ratio exceeds Free-Energy stability limit
    if density_ratio > 10.0:
        import warnings
        warnings.warn(
            f"density_ratio={density_ratio} exceeds typical Free-Energy stability limit (~10:1). "
            "Consider using the Allen-Cahn (cAC) formulation (LBM-P0-CAC-G) for higher ratios.",
            UserWarning,
        )

    return rho_heavy, rho_light


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

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
    phi0 = _initial_phase_field(xx, yy, args.radius, cx, cy, args.water_level, args.interface_width)
    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=device)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    g = init_free_energy_g(phi0, zero, zero)
    initial_mass = float(phi0.sum().item())

    # Resolve density parameters (cAC pathway)
    rho_heavy, rho_light = _resolve_density_params(
        args.density_ratio, rho_heavy_default=1.0, rho_light_default=0.1
    )

    # Physical tank boundary mask (bottom + sides; top is open)
    wall_mask = _make_tank_wall_mask(ny, nx, device)

    started = time.perf_counter()
    rows: list[dict[str, float | int]] = []

    def record(step: int) -> None:
        phi = g.sum(dim=0)
        _rho, ux, uy = macroscopic(f)
        speed2 = ux.square() + uy.square()
        liquid_weight = ((phi + 1.0) * 0.5).clamp(0.0, 1.0)

        # GH408-3: Two-phase mass-weighted centroid (valid before and after coalescence)
        # This reports the centroid of ALL liquid phase, not just the "droplet"
        w = liquid_weight
        denom = w.sum().clamp_min(1e-12)
        cx_now = float((w * xx).sum().item() / denom.item())
        cy_now = float((w * yy).sum().item() / denom.item())

        mass = float(phi.sum().item())
        row = {
            "step": step,
            "time_lu": float(step),
            "phase_mass": mass,
            "relative_phase_mass_drift": abs(mass - initial_mass) / max(abs(initial_mass), 1e-12),
            "liquid_centroid_x": cx_now,
            "liquid_centroid_y": cy_now,
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
            rho_heavy=rho_heavy,
            rho_light=rho_light,
        )
        f, g = stream(f), stream(g)

        # GH408-1: Apply physical tank boundaries (replaces periodic streaming)
        f, g = _apply_tank_boundaries(f, g, wall_mask, top_bc="pressure", rho_top=1.0)

        if step % args.output_interval == 0 or step == args.n_steps:
            record(step)
            last = rows[-1]
            print(
                f"step={step:6d} mass_drift={last['relative_phase_mass_drift']:.3e} "
                f"max_u={last['max_speed']:.3e} "
                f"centroid_y={last['liquid_centroid_y']:.2f}"
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
        "gh408_fixes": [
            "GH408-1: periodic streaming → physical tank boundaries (bottom+side walls + top pressure outlet)",
            "GH408-2: initial water level = static reference line, separate from dynamic interface",
            "GH408-3: centroid tracking → two-phase mass-weighted centroid (valid pre/post coalescence)",
            "GH408-4: density_ratio parameter pathway for cAC real water-air density ratio",
        ],
        "config": vars(args),
        "derived": {
            "initial_droplet_center_y": cy,
            "initial_droplet_center_x": cx,
            "initial_phase_mass": initial_mass,
            "rho_heavy_used": rho_heavy,
            "rho_light_used": rho_light,
            "density_ratio_actual": rho_heavy / rho_light if rho_light > 0 else float("inf"),
            "elapsed_seconds_including_plotting": elapsed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "final_metrics": final,
        "diagnostic_notes": [
            "phase_mass is the signed order-parameter sum and is expected to be conserved approximately",
            "liquid_centroid is the mass-weighted centroid of ALL liquid phase (valid before and after coalescence)",
            "kinetic_energy_proxy uses the uniform initial density and is not a calibrated physical energy",
            "quantitative validation requires mesh/time-step convergence and comparison against published experimental or numerical data",
            "Physical tank boundaries: bottom+side walls (bounce-back), top (pressure outlet)",
        ],
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(steps, [row["relative_phase_mass_drift"] for row in rows], marker="o")
    axes[0, 0].set(title="Relative phase-mass drift", xlabel="step", ylabel="relative drift")
    axes[0, 1].plot(steps, [row["liquid_centroid_y"] for row in rows], marker="o")
    axes[0, 1].set(title="Liquid mass-weighted centroid", xlabel="step", ylabel="y [cells]")
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
        f"Density ratio: {rho_heavy/rho_light:.1f}\n"
        f"Elapsed: {elapsed:.1f} s\n\n"
        f"GH #408 fixes applied.\nExploratory result — not reference validated.",
        va="top",
        family="monospace",
    )
    axes[1, 2].axis("off")
    fig.savefig(out / "diagnostic_curves.png", dpi=160)
    plt.close(fig)
    print(f"Artifacts written to: {out}")
    print("  contours/phase_*.png, history.csv, diagnostic_curves.png, run_metadata.json")


if __name__ == "__main__":
    main()
