"""Static 3-D bubble benchmark with Shan-Chen two-component (SCMC).

Usage
-----
    PYTHONPATH=src python examples/bubble_static_3d.py [options]
    PYTHONPATH=src python examples/bubble_static_3d.py --steps 200          # quick
    PYTHONPATH=src python examples/bubble_static_3d.py --steps 2000 --radii 8 12 16   # full sigma fit

Method
------
    For each radius R in cfg.radii we run SCMC to steady state, measure
    ΔP = (ρ_in - ρ_out)/3 inside r<0.5R and outside r>1.5R.  Linear fit
    ΔP vs 1/R through the origin gives σ_eff (Young-Laplace).  Maximum
    spurious velocity |u|_max is recorded as a quality metric.

Why SCMC (not SCMP):
    SCMP with Carnahan-Starling pseudopotential has a narrow stability
    window at large density ratios.  SCMC two-component with moderate
    density ratio is the proven testbed for Laplace-law benchmarks.

Outputs (to outputs/bubble_static_3d/):
    * radial_profile.png  - ρ(r) and p(r) from bubble centre to box edge
    * sigma_fit.json      - per-radius measurements + fitted σ_eff
    * console summary
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.multiphase3d import collide_sc_two_component_3d
from tensorlbm.solver3d import stream3d


@dataclass(frozen=True)
class BubbleStaticConfig:
    nx: int = 96
    ny: int = 96
    nz: int = 96
    radii: tuple[float, ...] = (8.0, 12.0, 16.0)
    G12: float = 0.9
    tau: float = 1.0
    rho_heavy: float = 0.7
    rho_light: float = 0.3
    n_steps: int = 2000
    output_root: str = "outputs"
    run_name: str = "bubble_static_3d"
    device: str = "cuda"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Static 3-D bubble benchmark (SCMC)")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--radii", type=float, nargs="+", default=[8.0, 12.0, 16.0])
    p.add_argument("--G12", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--rho-heavy", type=float, default=0.7)
    p.add_argument("--rho-light", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=2000, dest="n_steps")
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--run-name", default="bubble_static_3d")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p


def initial_distributions(nx, ny, nz, R, rho_heavy, rho_light, device):
    """Centred vapour sphere of radius R inside liquid."""
    z = torch.arange(nz, dtype=torch.float32, device=device)
    y = torch.arange(ny, dtype=torch.float32, device=device)
    x = torch.arange(nx, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    inside = r <= R
    zero = torch.zeros((nz, ny, nx), device=device)
    rho1 = torch.where(inside, torch.full_like(zero, rho_light),
                       torch.full_like(zero, rho_heavy))
    rho2 = torch.where(inside, torch.full_like(zero, rho_heavy),
                       torch.full_like(zero, rho_light))
    f1 = equilibrium3d(rho1, zero, zero, zero)
    f2 = equilibrium3d(rho2, zero, zero, zero)
    return f1, f2


def measure_pressure_jump(rho_total, R):
    """Match TensorLBM _measure_pressure_jump_3d convention.

    p_in = <ρ>_{r<0.5R},  p_out = <ρ>_{r>1.5R},  dp = p_in - p_out.
    Lattice pressure p = cs² * rho with cs² = 1/3.
    """
    cs2 = 1.0 / 3.0
    nz, ny, nx = rho_total.shape
    z = torch.arange(nz, dtype=torch.float32, device=rho_total.device)
    y = torch.arange(ny, dtype=torch.float32, device=rho_total.device)
    x = torch.arange(nx, dtype=torch.float32, device=rho_total.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    inside = r_field <= R * 0.5
    outside = r_field >= R * 1.5
    p_in = float((cs2 * rho_total[inside]).mean().item()) if inside.any() else float("nan")
    p_out = float((cs2 * rho_total[outside]).mean().item()) if outside.any() else float("nan")
    dp = p_in - p_out
    return p_in, p_out, dp


def measure_profile(rho_total, ux, uy, uz, cfg):
    """Radial profile along the +x axis (y=z=centre)."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    y0, z0 = ny // 2, nz // 2
    n_radius = nx // 2
    radii = np.arange(1, n_radius)
    rho_vals, p_vals, u_mag_vals = [], [], []
    line_rho = rho_total[z0, y0, :].cpu().numpy()
    line_ux = ux[z0, y0, :].cpu().numpy()
    line_uy = uy[z0, y0, :].cpu().numpy()
    line_uz = uz[z0, y0, :].cpu().numpy()
    for r in radii:
        idx = int(nx // 2 + r)
        rho_vals.append(float(line_rho[idx]))
        u_mag_vals.append(float(np.sqrt(line_ux[idx] ** 2 +
                                        line_uy[idx] ** 2 + line_uz[idx] ** 2)))
        p_vals.append(float(line_rho[idx]) / 3.0)
    return {
        "radii": radii,
        "rho": np.array(rho_vals),
        "u_mag": np.array(u_mag_vals),
        "p": np.array(p_vals),
    }


def fit_sigma_least_squares(radii, delta_ps):
    """σ_eff = |slope| of ΔP vs 1/R, fit through origin (least squares).

    SCMC heavy/light: ρ_heavy > ρ_light so dp = p_in - p_out < 0.
    Young-Laplace: σ_eff / R = |ΔP| → take |slope|.
    """
    if len(radii) < 2:
        return abs(delta_ps[0]) * radii[0] if radii else float("nan")
    inv_r = np.array([1.0 / r for r in radii])
    dp = np.array(delta_ps)
    num = float((inv_r * dp).sum())
    den = float((inv_r * inv_r).sum())
    slope = num / den if abs(den) > 1e-20 else float("nan")
    return abs(slope)


def plot_profile(profile, cfg, R_focus, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    r = profile["radii"]
    axes[0].plot(r, profile["rho"], "b-", lw=2, label=f"R0={R_focus}")
    axes[0].axhline(cfg.rho_heavy, color="b", ls=":", alpha=0.4, label="rho_heavy")
    axes[0].axhline(cfg.rho_light, color="r", ls=":", alpha=0.4, label="rho_light")
    axes[0].axvline(R_focus, color="k", ls="--", alpha=0.4)
    axes[0].set_xlabel("r (cells)")
    axes[0].set_ylabel("rho (lattice units)")
    axes[0].set_title(f"Radial density (after {cfg.n_steps} steps)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(r, profile["p"], "g-", lw=2)
    axes[1].set_xlabel("r (cells)")
    axes[1].set_ylabel("p = rho*cs^2")
    axes[1].set_title("Radial pressure")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    cfg = BubbleStaticConfig(
        nx=args.nx, ny=args.ny, nz=args.nz,
        radii=tuple(args.radii),
        G12=args.G12, tau=args.tau,
        rho_heavy=args.rho_heavy, rho_light=args.rho_light,
        n_steps=args.n_steps,
        output_root=args.output_root, run_name=args.run_name,
        device=args.device,
    )
    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")

    out_dir = Path(cfg.output_root) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config: {asdict(cfg)}")
    print(f"Output: {out_dir}")

    per_r = []
    R_focus_for_plot = cfg.radii[len(cfg.radii) // 2]  # middle radius for plot
    profile_focus = None

    for R in cfg.radii:
        # ensure R < nx/3 to leave room for transition band + outer zone
        if R * 1.5 >= cfg.nx / 2.0:
            print(f"  [skip] R={R}: too large for nx={cfg.nx}")
            continue
        print(f"\n--- R = {R} ---")
        f1, f2 = initial_distributions(cfg.nx, cfg.ny, cfg.nz, R,
                                       cfg.rho_heavy, cfg.rho_light, device)
        for step in range(cfg.n_steps):
            f1, f2 = collide_sc_two_component_3d(
                f1, f2, G_12=cfg.G12, tau1=cfg.tau, tau2=cfg.tau,
            )
            f1 = stream3d(f1)
            f2 = stream3d(f2)
            if (step + 1) % 500 == 0 or step == 0:
                rho_m, ux_m, uy_m, uz_m = macroscopic3d(f1 + f2)
                u_max = float(torch.sqrt(ux_m ** 2 + uy_m ** 2 + uz_m ** 2).max())
                rho_c = float(rho_m[cfg.nz // 2, cfg.ny // 2, cfg.nx // 2])
                print(f"  step {step+1:5d}/{cfg.n_steps}  "
                      f"rho_centre={rho_c:.4f}  |u|_max={u_max:.2e}")

        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f1 + f2)
        u_max_final = float(torch.sqrt(ux_f ** 2 + uy_f ** 2 + uz_f ** 2).max())
        p_in, p_out, dp = measure_pressure_jump(rho_f, R)
        sigma_eff_one = abs(dp) * R / 2.0  # single-radius estimate
        per_r.append({"R": R, "p_in": p_in, "p_out": p_out,
                      "dp": dp, "sigma_eff_one": sigma_eff_one,
                      "max_u": u_max_final})
        print(f"  R={R}  p_in={p_in:.5e}  p_out={p_out:.5e}  "
              f"dp={dp:.5e}  sigma(R)={sigma_eff_one:.5e}  |u|_max={u_max_final:.3e}")

        if R == R_focus_for_plot:
            profile_focus = measure_profile(rho_f, ux_f, uy_f, uz_f, cfg)

    radii_used = [d["R"] for d in per_r]
    deltas_used = [d["dp"] for d in per_r]
    sigma_fit = fit_sigma_least_squares(radii_used, deltas_used)
    mean_u = float(np.mean([d["max_u"] for d in per_r]))

    print("\n=== Surface tension fit (Young-Laplace) ===")
    print(f"  Per-radius results:")
    for d in per_r:
        print(f"    R={d['R']:5.1f}  dp={d['dp']:+.5e}  "
              f"sigma(R)={d['sigma_eff_one']:.5e}  |u|_max={d['max_u']:.3e}")
    print(f"  Fitted sigma_eff (ΔP = sigma_eff / R, least-squares through origin):")
    print(f"    sigma_eff = {sigma_fit:.6e}  (lattice units)")
    print(f"  Mean |u|_max across radii = {mean_u:.3e}")

    summary = {
        "config": {**asdict(cfg), "radii": list(cfg.radii)},
        "per_radius": per_r,
        "sigma_eff_fit": sigma_fit,
        "mean_max_spurious_u": mean_u,
        "note": "σ_eff from ΔP = σ_eff/R fit through origin.",
    }
    with open(out_dir / "sigma_fit.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {out_dir / 'sigma_fit.json'}")

    if profile_focus is not None:
        plot_path = out_dir / "radial_profile.png"
        plot_profile(profile_focus, cfg, R_focus_for_plot, plot_path)
        print(f"  Wrote {plot_path}")

    print("\n  >>> Use this sigma_eff in bubble_rp_validate.py:")
    print(f"  >>>    --sigma-eff {sigma_fit:.6e}")


if __name__ == "__main__":
    main()