#!/usr/bin/env python3
"""SDAA verification sweep for the permeability benchmark.

Runs a representative subset of the (phi, d) ladder on the available device,
collects results into scan_summary.json, and generates diagnostic plots.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# Add benchmarks dir to path for compile_route
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))

from compile_route import ensure_tensorlbm_importable  # noqa: E402
ensure_tensorlbm_importable()

import torch  # noqa: E402

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulent_channel import _apply_body_force_2d  # noqa: E402

# Constants (matching run.py)
TAU = 1.0
NU = (TAU - 0.5) / 3.0
RE_TARGET = 0.05

SANGANI_TABLE = [
    (0.05, 15.56), (0.10, 24.83), (0.20, 51.53), (0.30, 102.90),
    (0.40, 217.89), (0.50, 532.55), (0.60, 1.763e3), (0.70, 1.352e4),
    (0.75, 1.263e5),
]
F_TABLE = {phi: f for phi, f in SANGANI_TABLE}

# SDAA sweep matrix: smaller d values for CPU feasibility (reduced for quick run)
SWEEP_MATRIX = {
    0.3: [26],       # phi=0.3: 1 resolution
    0.5: [20, 40],   # phi=0.5: 2 resolutions
}

SAMPLE_EVERY = 200
DRIFT_SPAN = 10
DRIFT_TOL = 1e-5
STEADY_REPEATS = 3
POST_STEADY = 2000
MEAS_SAMPLES = 10
FORCE_SCALE = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cylinder_array_mask(d: int, phi: float, device: torch.device) -> torch.Tensor:
    r_real = d * math.sqrt(phi / math.pi)
    idx = np.arange(d, dtype=np.float64)
    dx = np.abs(idx - d / 2.0)
    dx = np.minimum(dx, d - dx)
    dy = dx.copy()
    dist2 = dx[None, :] ** 2 + dy[:, None] ** 2
    mask = dist2 <= r_real * r_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def run_single_case(phi: float, d: int, device: torch.device, max_steps: int = 30000) -> dict:
    """Run a single permeability case and return the result record."""
    f_table = F_TABLE[phi]
    r_real = d * math.sqrt(phi / math.pi)
    k_lu = d * d / f_table
    a_body = RE_TARGET * NU * NU / (k_lu * r_real) * FORCE_SCALE

    solid = cylinder_array_mask(d, phi, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    phi_actual = n_solid / (d * d)

    rho0 = torch.ones(d, d, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, u0, u0, device=device)
    mass0 = float(f.sum().item())
    del rho0, u0

    solid_b = solid.unsqueeze(0)

    def _step(f_t):
        f_t = collide_bgk(f_t, TAU)
        f_t = stream(f_t)
        f_t = torch.where(solid_b, f_t, _apply_body_force_2d(f_t, a_body))
        return bounce_back_cells(f_t, solid)

    hist_uxf = []
    hist_uyf = []
    hist_umax = []
    hist_uymax = []
    hist_mass = []
    recent_drifts = []
    steady_step = None
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % SAMPLE_EVERY == 0:
            _, ux, uy = macroscopic(f)
            uxf = float(ux[fluid].mean().item())
            uyf = float(uy[fluid].mean().item())
            umax = float(ux[fluid].abs().max().item())
            uymax = float(uy[fluid].abs().max().item())
            mass = float(f.sum().item())
            hist_uxf.append(uxf)
            hist_uyf.append(uyf)
            hist_umax.append(umax)
            hist_uymax.append(uymax)
            hist_mass.append(mass)

            if steady_step is None and len(hist_uxf) >= DRIFT_SPAN + 1:
                drift = abs(hist_uxf[-1] - hist_uxf[-1 - DRIFT_SPAN]) / abs(hist_uxf[-1])
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(
                    v < DRIFT_TOL for v in recent_drifts[-STEADY_REPEATS:]
                ):
                    steady_step = step

        if steady_step is not None and step >= steady_step + POST_STEADY:
            break

    wall = time.perf_counter() - t0
    steps_run = len(hist_uxf) * SAMPLE_EVERY

    # Measurement window
    n_meas = min(MEAS_SAMPLES, len(hist_uxf))
    uxf_meas = sum(hist_uxf[-n_meas:]) / n_meas
    uyf_meas = sum(hist_uyf[-n_meas:]) / n_meas
    umax_meas = sum(hist_umax[-n_meas:]) / n_meas
    uymax_meas = sum(hist_uymax[-n_meas:]) / n_meas

    k_sim = NU * uxf_meas / a_body
    k_ref = d * d / ((1.0 - phi) * f_table)
    err = abs(k_sim / k_ref - 1.0) * 100.0

    u_d_meas = uxf_meas * (1.0 - phi_actual)
    re_achieved = u_d_meas * r_real / NU
    ma = math.sqrt(3.0) * umax_meas
    mass_drift = max(abs(m / mass0 - 1.0) for m in hist_mass) if hist_mass else float("nan")

    return {
        "phi": phi,
        "d": d,
        "phi_actual": phi_actual,
        "k_sim": k_sim,
        "k_ref": k_ref,
        "err_pct": err,
        "Re_cell": re_achieved,
        "Ma": ma,
        "steps_run": steps_run,
        "steady_detected": steady_step is not None,
        "steady_step": steady_step,
        "wall_sec": wall,
        "mass_drift_rel": mass_drift,
        "uxf_final": uxf_meas,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SDAA sweep] device={device}, force_scale={FORCE_SCALE}")

    results = []
    for phi, d_list in SWEEP_MATRIX.items():
        for d in d_list:
            max_steps = min(20 * d * d, 50000)  # cap for CPU
            print(f"\n[SDAA sweep] phi={phi}, d={d}, max_steps={max_steps}")
            rec = run_single_case(phi, d, device, max_steps)
            results.append(rec)
            print(f"  -> k_sim={rec['k_sim']:.4f}, k_ref={rec['k_ref']:.4f}, "
                  f"err={rec['err_pct']:.2f}%, Re={rec['Re_cell']:.3f}, "
                  f"steady={rec['steady_detected']}, wall={rec['wall_sec']:.1f}s")

    # Write scan_summary.json
    summary_path = HERE / "scan_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[SDAA sweep] wrote {summary_path}")

    # Generate plots
    generate_plots(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("SDAA SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'phi':>5} {'d':>5} {'phi_act':>8} {'k_sim':>10} {'k_ref':>10} "
          f"{'err%':>8} {'Re':>6} {'Ma':>6} {'steady':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['phi']:5.1f} {r['d']:5d} {r['phi_actual']:8.4f} "
              f"{r['k_sim']:10.4f} {r['k_ref']:10.4f} {r['err_pct']:8.3f} "
              f"{r['Re_cell']:6.3f} {r['Ma']:6.4f} {str(r['steady_detected']):>7}")

    # Verdict
    pass_count = sum(1 for r in results if r['err_pct'] < 10.0)
    print(f"\nVerdict: {pass_count}/{len(results)} cases with err < 10%")
    finest_by_phi = {}
    for r in results:
        phi = r['phi']
        if phi not in finest_by_phi or r['d'] > finest_by_phi[phi]['d']:
            finest_by_phi[phi] = r
    for phi, r in sorted(finest_by_phi.items()):
        status = "PASS" if r['err_pct'] <= 3.0 else ("MARGINAL" if r['err_pct'] <= 10.0 else "FAIL")
        print(f"  phi={phi} finest d={r['d']}: err={r['err_pct']:.2f}% [{status}]")


def generate_plots(results: list[dict]):
    """Generate diagnostic plots for the report."""
    out_dir = HERE / "sdaa_plots"
    out_dir.mkdir(exist_ok=True)

    # Group by phi
    by_phi = {}
    for r in results:
        by_phi.setdefault(r['phi'], []).append(r)

    # Plot 1: Error convergence vs d
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for phi, recs in sorted(by_phi.items()):
        ds = [r['d'] for r in recs]
        errs = [r['err_pct'] for r in recs]
        ax = axes[0]
        ax.plot(ds, errs, 'o-', label=f'φ={phi}', markersize=8)
        ax.set_xlabel('Grid size d')
        ax.set_ylabel('Error vs k_ref (%)')
        ax.set_title('Permeability Error Convergence')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='3% threshold')

    # Plot 2: k_sim vs k_ref (parity plot)
    ax = axes[1]
    all_k_ref = [r['k_ref'] for r in results]
    all_k_sim = [r['k_sim'] for r in results]
    for phi, recs in sorted(by_phi.items()):
        krefs = [r['k_ref'] for r in recs]
        ksims = [r['k_sim'] for r in recs]
        ax.scatter(krefs, ksims, label=f'φ={phi}', s=80)
    lims = [min(all_k_ref + all_k_sim) * 0.5, max(all_k_ref + all_k_sim) * 1.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
    ax.set_xlabel('k_ref (Sangani & Acrivos 1982)')
    ax.set_ylabel('k_sim (LBM)')
    ax.set_title('Parity: k_sim vs k_ref')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "error_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SDAA sweep] saved {out_dir / 'error_convergence.png'}")

    # Plot 3: Velocity field snapshot for one case (smallest case for speed)
    first = results[0]
    phi, d = first['phi'], first['d']
    f_table = F_TABLE[phi]
    r_real = d * math.sqrt(phi / math.pi)
    k_lu = d * d / f_table
    a_body = RE_TARGET * NU * NU / (k_lu * r_real) * FORCE_SCALE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solid = cylinder_array_mask(d, phi, device)
    fluid = ~solid
    rho0 = torch.ones(d, d, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, u0, u0, device=device)
    solid_b = solid.unsqueeze(0)

    # Run enough steps to get a developed flow
    n_steps = min(10000, 20 * d * d)
    for step in range(n_steps):
        f = collide_bgk(f, TAU)
        f = stream(f)
        f = torch.where(solid_b, f, _apply_body_force_2d(f, a_body))
        f = bounce_back_cells(f, solid)

    _, ux, uy = macroscopic(f)
    ux_np = ux.cpu().numpy()
    uy_np = uy.cpu().numpy()
    solid_np = solid.cpu().numpy()
    ux_np[solid_np] = 0.0
    uy_np[solid_np] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(ux_np.T, origin='lower', cmap='RdBu_r')
    axes[0].set_title(f'u_x field (φ={phi}, d={d})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='u_x')

    speed = np.sqrt(ux_np**2 + uy_np**2)
    im1 = axes[1].imshow(speed.T, origin='lower', cmap='viridis')
    axes[1].set_title(f'|u| field (φ={phi}, d={d})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], label='|u|')

    plt.tight_layout()
    fig.savefig(out_dir / "velocity_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SDAA sweep] saved {out_dir / 'velocity_field.png'}")


if __name__ == "__main__":
    main()
