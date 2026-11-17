"""2-D free-energy LBM benchmark: a liquid droplet impacts a quiescent liquid pool.

This is a reproducible exploratory benchmark, not a validated quantitative
reference case. It emits phase-field contours, a time-history CSV, and summary
metadata so that stability, conservation, and impact evolution are inspectable.

The default tank has impermeable walls, including its top. Optional pressure
reconstruction affects momentum only; the phase field remains impermeable.
The density-ratio option controls Boussinesq buoyancy, not a variable-density
or conservative Allen-Cahn (cAC) solver. Health gates do not validate impact physics.

Run:
  PYTHONPATH=src python examples/droplet_impact.py --fast --overwrite
  PYTHONPATH=src python examples/droplet_impact.py --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.boundary.tank import stream_tank
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import free_energy_step, init_free_energy_g
from tensorlbm.phasefield.operators import laplacian_2d

# ---------------------------------------------------------------------------
# Boundary-condition helpers for physical tank
# ---------------------------------------------------------------------------


# Kept as an import alias for callers of the example. Inputs must now be
# post-collision, BEFORE streaming; the function performs streaming itself.
_apply_tank_boundaries = stream_tank


def _make_tank_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Create boolean mask for tank walls: bottom + left + right sides.

    Top links are handled by stream_tank; top cells remain fluid.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[-1, :] = True  # bottom wall
    mask[:, 0] = True  # left wall
    mask[:, -1] = True  # right wall
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
    p.add_argument("--tau-g", type=float, default=1.0)
    p.add_argument("--A", type=float, default=0.004)
    p.add_argument("--B", type=float, default=0.004)
    p.add_argument("--kappa", type=float, default=None, help="Default: A*interface_width**2/2")
    p.add_argument("--Gamma", type=float, default=0.5)
    p.add_argument(
        "--interface-width",
        type=float,
        default=None,
        help="tanh width xi; default 3 or sqrt(2*kappa/A)",
    )
    p.add_argument("--geq-convention", choices=["lee_lin", "swift"], default="lee_lin")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument(
        "--density-ratio",
        type=float,
        default=None,
        help="Buoyancy density ratio only; this does not switch to a cAC solver.",
    )
    p.add_argument("--top-bc", choices=["bounce_back", "pressure"], default="bounce_back")
    p.add_argument("--rho-top", type=float, default=1.0)
    p.add_argument("--max-mass-drift", type=float, default=1e-4)
    p.add_argument("--max-speed", type=float, default=0.15)
    p.add_argument("--max-abs-phi", type=float, default=1.2)
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
    ax.axhline(
        water_level,
        color="gold",
        linestyle="--",
        linewidth=0.8,
        label="initial pool level (static ref)",
    )
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
# Effective density for Boussinesq buoyancy
# ---------------------------------------------------------------------------


def _resolve_density_params(
    density_ratio: float | None,
    rho_heavy_default: float,
    rho_light_default: float,
) -> tuple[float, float]:
    """Resolve buoyancy coefficients without changing the governing solver.

    This free-energy example uses uniform inertial density. These coefficients
    scale gravity only; their ratio is not a measured liquid/gas density ratio.
    """
    if density_ratio is None:
        return rho_heavy_default, rho_light_default

    if not math.isfinite(density_ratio) or density_ratio < 1.0:
        raise ValueError(f"density_ratio must be >= 1.0, got {density_ratio}")

    # Keep the mean of the two buoyancy coefficients equal to 1.0
    # rho_heavy = 2 * ratio / (1 + ratio)
    # rho_light = 2 / (1 + ratio)
    # This gives rho_heavy / rho_light = ratio and (rho_heavy + rho_light) / 2 = 1.0
    rho_heavy = 2.0 * density_ratio / (1.0 + density_ratio)
    rho_light = 2.0 / (1.0 + density_ratio)

    return rho_heavy, rho_light


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def _diagnostics(f, g, wall_mask, xx, yy, initial_mass, mass_scale, step):
    """Measure fluid cells only, using FP64 reductions for conserved totals."""
    phi = g.sum(dim=0)
    rho, ux, uy = macroscopic(f)
    active = ~wall_mask
    finite = bool(torch.isfinite(f[:, active]).all() and torch.isfinite(g[:, active]).all())
    if not finite:
        return {"step": step, "finite": False}
    speed2 = ux.square() + uy.square()
    w = ((phi + 1.0) * 0.5).clamp(0.0, 1.0).masked_fill(wall_mask, 0.0).double()
    denom = w.sum().clamp_min(1e-12)
    mass = float(phi[active].double().sum())
    return {
        "step": step,
        "finite": True,
        "time_lu": float(step),
        "phase_mass": mass,
        "relative_phase_mass_drift": abs(mass - initial_mass) / mass_scale,
        "liquid_centroid_x": float((w * xx).sum() / denom),
        "liquid_centroid_y": float((w * yy).sum() / denom),
        "max_speed": float(speed2[active].sqrt().max()),
        "min_density": float(rho[active].min()),
        "kinetic_energy_proxy": float((0.5 * rho[active] * speed2[active]).double().sum()),
        "phi_min": float(phi[active].min()),
        "phi_max": float(phi[active].max()),
    }


def _health_failures(metrics, args):
    if not metrics["finite"] or any(
        isinstance(value, float) and not math.isfinite(value) for value in metrics.values()
    ):
        return ["nonfinite_state"]
    checks = {
        "phase_mass_drift": metrics["relative_phase_mass_drift"] > args.max_mass_drift,
        "max_speed": metrics["max_speed"] > args.max_speed,
        "nonpositive_density": metrics["min_density"] <= 0.0,
        "phase_bounds": max(abs(metrics["phi_min"]), abs(metrics["phi_max"])) > args.max_abs_phi,
    }
    return [name for name, failed in checks.items() if failed]


def _resolve_phasefield_parameters(args):
    """Keep tanh initialization, free energy and mobility on one convention."""
    if args.A <= 0 or args.B <= 0 or not math.isclose(args.A, args.B, rel_tol=1e-12):
        raise ValueError("this +/-1 phase example requires positive A == B")
    if args.Gamma < 0:
        raise ValueError("Gamma must be nonnegative")
    if args.kappa is None:
        args.interface_width = 3.0 if args.interface_width is None else args.interface_width
        args.kappa = 0.5 * args.A * args.interface_width**2
    elif args.kappa <= 0:
        raise ValueError("kappa must be positive")
    xi = math.sqrt(2.0 * args.kappa / args.A)
    if args.interface_width is None:
        args.interface_width = xi
    if args.interface_width <= 0 or not math.isclose(args.interface_width, xi, rel_tol=1e-10):
        raise ValueError("interface-width must match sqrt(2*kappa/A)")


def main() -> None:
    args = parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    _resolve_phasefield_parameters(args)
    if args.fast:
        args.nx, args.ny, args.n_steps, args.output_interval = 120, 90, 120, 30
        args.radius, args.water_level, args.gap = 9.0, 58.0, 5.0
    if args.nx < 64 or args.ny < 64:
        raise ValueError("nx and ny must both be >= 64")
    if args.output_interval < 1 or args.n_steps < 1:
        raise ValueError("n-steps and output-interval must be positive")
    if (
        args.radius < 4
        or args.gap < 0
        or 2 * args.radius + args.gap + 2 >= args.water_level
        or 2 * args.radius + 4 >= args.nx
    ):
        raise ValueError("droplet must fit above the pool and radius must be >= 4")
    if args.water_level >= args.ny - 8:
        raise ValueError("water-level must leave at least 8 cells below the pool")
    for name in ("max_mass_drift", "max_speed", "max_abs_phi", "rho_top", "interface_width"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("tau_f", "tau_g"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.5:
            raise ValueError(f"{name} must be finite and > 0.5")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    run_name = args.run_name or f"impact_{args.nx}x{args.ny}_s{args.n_steps}"
    out = Path(args.output_root) / run_name
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} already contains files; choose --run-name or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    (out / "contours").mkdir(exist_ok=True)
    if args.overwrite:
        # Do not mix old successful frames with a shorter, failed rerun.
        for frame in (out / "contours").glob("phase_*.png"):
            frame.unlink()

    ny, nx = args.ny, args.nx
    dtype = getattr(torch, args.dtype)
    y = torch.arange(ny, dtype=dtype, device=device)
    x = torch.arange(nx, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy, cx = args.water_level - args.radius - args.gap, nx / 2.0
    phi0 = _initial_phase_field(xx, yy, args.radius, cx, cy, args.water_level, args.interface_width)
    rho0 = torch.ones((ny, nx), dtype=dtype, device=device)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    wall_mask = _make_tank_wall_mask(ny, nx, device)
    mu0 = (
        -args.A * phi0
        + args.B * phi0**3
        - args.kappa * laplacian_2d(phi0, boundary="no_flux", solid_mask=wall_mask)
    )
    g = init_free_energy_g(
        phi0,
        zero,
        zero,
        geq_convention=args.geq_convention,
        mu=mu0 if args.geq_convention == "lee_lin" else None,
        Gamma=args.Gamma,
    )
    initial_phi = g.sum(0)[~wall_mask].double()
    initial_mass = float(initial_phi.sum())
    # L1 normalization avoids singular relative error when signed phi sums to 0.
    mass_scale = max(float(initial_phi.abs().sum()), 1e-12)
    rho_heavy, rho_light = _resolve_density_params(args.density_ratio, 1.0, 0.1)
    started = time.perf_counter()
    rows = []
    failures = []
    max_observed = {"relative_phase_mass_drift": 0.0, "max_speed": 0.0, "abs_phi": 0.0}
    for step in range(args.n_steps + 1):
        if step:
            f, g = free_energy_step(
                f,
                g,
                tau_f=args.tau_f,
                tau_g=args.tau_g,
                A=args.A,
                B=args.B,
                kappa=args.kappa,
                Gamma=args.Gamma,
                geq_convention=args.geq_convention,
                gx=0.0,
                gy=args.g,
                rho_heavy=rho_heavy,
                rho_light=rho_light,
                boundary="no_flux",
                solid_mask=wall_mask,
            )
            f, g = stream_tank(f, g, wall_mask, top_bc=args.top_bc, rho_top=args.rho_top)
        metrics = _diagnostics(f, g, wall_mask, xx, yy, initial_mass, mass_scale, step)
        failures = _health_failures(metrics, args)
        if metrics["finite"] and not any(
            isinstance(v, float) and not math.isfinite(v) for v in metrics.values()
        ):
            for key in ("relative_phase_mass_drift", "max_speed"):
                max_observed[key] = max(max_observed[key], metrics[key])
            max_observed["abs_phi"] = max(
                max_observed["abs_phi"], abs(metrics["phi_min"]), abs(metrics["phi_max"])
            )
        if step % args.output_interval == 0 or step == args.n_steps or failures:
            # Strict JSON: invalid measurements become null, never NaN/Infinity.
            rows.append(
                {
                    k: None if isinstance(v, float) and not math.isfinite(v) else v
                    for k, v in metrics.items()
                }
            )
            if metrics["finite"] and not failures:
                _, ux, uy = macroscopic(f)
                save_contour(
                    g.sum(0),
                    ux,
                    uy,
                    step,
                    out / "contours" / f"phase_{step:07d}.png",
                    args.water_level,
                )
            print(
                f"step={step:6d} health={'failed' if failures else 'passed'} "
                f"mass_drift={metrics.get('relative_phase_mass_drift')} "
                f"max_u={metrics.get('max_speed')}"
            )
        if failures:
            break

    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    elapsed = time.perf_counter() - started
    metadata = {
        "benchmark": "2d_free_energy_droplet_impact",
        "status": "failed_health_gate" if failures else "exploratory_not_yet_reference_validated",
        "health": {
            "passed": not failures,
            "failures": failures,
            "checked_every_step": True,
            "completed_steps": step,
            "max_observed": max_observed,
            "thresholds": {
                "relative_phase_mass_drift": args.max_mass_drift,
                "max_speed": args.max_speed,
                "abs_phi": args.max_abs_phi,
                "min_density_exclusive": 0.0,
            },
        },
        "config": vars(args),
        "derived": {
            "equilibrium_interface_width": math.sqrt(2 * args.kappa / args.A),
            "surface_tension_analytic": (2 / 3) * math.sqrt(2 * args.kappa * args.A),
            "ch_mobility": args.Gamma * (args.tau_g - 0.5)
            if args.geq_convention == "lee_lin"
            else None,
            "legacy_extra_phi_diffusivity": (args.tau_g - 0.5) / 3
            if args.geq_convention == "swift"
            else 0.0,
            "initial_droplet_center_y": cy,
            "initial_droplet_center_x": cx,
            "initial_phase_mass": initial_mass,
            "phase_mass_normalization_l1": mass_scale,
            "rho_heavy_used": rho_heavy,
            "rho_light_used": rho_light,
            "buoyancy_density_ratio": rho_heavy / rho_light,
            "elapsed_seconds_including_contours": elapsed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "final_metrics": rows[-1],
        "diagnostic_notes": [
            "Conserved totals exclude solid storage; phase drift is normalized by initial sum(abs(phi)).",
            "Liquid centroid includes ALL liquid; kinetic energy is an uncalibrated lattice proxy.",
            "Momentum and phase stream without periodic wrap; scalar stencils use no-flux walls.",
            "Pressure top, if selected, fixes momentum density but keeps phase impermeable; not a two-phase outflow.",
            "Density ratio changes Boussinesq gravity only; no cAC or variable-inertial-density solver is selected.",
            "The lee_lin selector corrects phase moments; swift is retained only for legacy reproduction.",
            "Passing health gates does not validate impact, splash height, coexistence, or water/air density ratios.",
        ],
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    steps = [row["step"] for row in rows]
    for ax, key in zip(
        axes.flat,
        ("relative_phase_mass_drift", "liquid_centroid_y", "max_speed", "kinetic_energy_proxy"),
    ):
        ax.plot(steps, [row.get(key, float("nan")) for row in rows], marker="o")
        ax.set(title=key, xlabel="step")
    fig.suptitle("Exploratory health diagnostics — not physical reference validation")
    fig.savefig(out / "diagnostic_curves.png", dpi=160)
    plt.close(fig)
    print(f"Artifacts written to: {out}")
    if failures:
        raise RuntimeError(
            f"Droplet health gate failed at step {step}: {', '.join(failures)}; see {out / 'run_metadata.json'}"
        )


if __name__ == "__main__":
    main()
