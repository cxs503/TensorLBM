#!/usr/bin/env python3
"""Independent reference evaluation for droplet spreading on a solid surface.

Uses the Color-Gradient (CG) LBM model with initial impact velocity (not
gravity-driven fall) to achieve inertial-regime Weber numbers.

References (independent of TensorLBM source code):
  - Clanet et al. (2004) J. Fluid Mech. 517:199-208
    beta_max = 0.5 * We^(1/4)
  - Inertial spreading law: beta(t) ~ We^(1/4) * (t/t_c)^(1/2), t_c = D0/U

Surface tension is measured via Laplace pressure test: sigma = DeltaP * R.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import color_gradient_step
from tensorlbm.solver import stream
from tensorlbm.boundaries import bounce_back_cells

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "reference_data"
OUT_DIR = HERE / "results_droplet"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def wall_mask(ny, nx, device):
    m = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    m[0, :] = True; m[-1, :] = True; m[:, 0] = True; m[:, -1] = True
    return m


def init_droplet_cg(nx, ny, radius, cx, cy, rho_h, rho_l, device, frac=0.05, uy_init=0.0):
    """Initialize CG distributions for a circular droplet with optional initial velocity."""
    y = torch.arange(ny, dtype=torch.float32, device=device)
    x = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)
    prof = 0.5 * (1.0 - torch.tanh((r - radius) / 3.0))
    rho_r = rho_h * prof + rho_h * frac * (1.0 - prof)
    rho_b = rho_l * frac * prof + rho_l * (1.0 - prof)
    zero = torch.zeros((ny, nx), device=device)
    uy = torch.full((ny, nx), uy_init, dtype=torch.float32, device=device)
    ux = torch.zeros_like(uy)
    return equilibrium(rho_r, ux, uy), equilibrium(rho_b, ux, uy)


def measure_laplace_sigma(nx, ny, radius, rho_h, rho_l, G, tau, device, steps=600):
    """Measure surface tension via Laplace pressure: sigma = DeltaP * R."""
    cx, cy = nx / 2.0, ny / 2.0
    f1, f2 = init_droplet_cg(nx, ny, radius, cx, cy, rho_h, rho_l, device)
    wm = wall_mask(ny, nx, device)
    A_surf = G * 0.04
    for step in range(steps):
        f1, f2 = color_gradient_step(f1, f2, tau=tau, A=A_surf, gy=0.0, solid_mask=wm)
        f1, f2 = stream(f1), stream(f2)
        f1 = bounce_back_cells(f1, wm); f2 = bounce_back_cells(f2, wm)
    rho_r = f1.sum(dim=0)
    rho_b = f2.sum(dim=0)
    p_in = float(rho_r[int(cy), int(cx)].item()) / 3.0
    p_out = float(rho_b[2, nx // 4].item()) / 3.0
    delta_p = p_in - p_out
    sigma = delta_p * radius
    return sigma, delta_p


def run_droplet_spreading(level_name, nx, ny, radius, gap, U0, n_steps,
                          rho_h=0.8, rho_l=0.4, G=0.9, tau=1.0):
    device = torch.device("cpu")
    A_surf = G * 0.04
    D0 = 2.0 * radius

    # Measure surface tension via Laplace test
    sigma, delta_p = measure_laplace_sigma(
        max(nx, 60), max(ny, 60), radius, rho_h, rho_l, G, tau, device, steps=500)
    print(f"  [{level_name}] Laplace: DeltaP={delta_p:.6f} sigma={sigma:.6f}")

    We = U0**2 * D0 / sigma if sigma > 0 else 0
    t_c = D0 / U0 if U0 > 1e-10 else float("inf")
    beta_max_ref = 0.5 * We**0.25 if We > 0 else 0
    print(f"  D0={D0:.1f} U0={U0} We={We:.2f} t_c={t_c:.1f} beta_max_ref={beta_max_ref:.3f}")

    # Initialize droplet just above wall with downward velocity
    cx = nx / 2.0
    cy = radius + gap
    f1, f2 = init_droplet_cg(nx, ny, radius, cx, cy, rho_h, rho_l, device, uy_init=U0)
    wm = wall_mask(ny, nx, device)

    rows = []
    t0 = time.perf_counter()
    measure_row = 2  # measure spreading at y=2 (first fully fluid row above wall)

    for step in range(n_steps + 1):
        if step > 0:
            f1, f2 = color_gradient_step(f1, f2, tau=tau, A=A_surf, gy=0.0, solid_mask=wm)
            f1, f2 = stream(f1), stream(f2)
            f1 = bounce_back_cells(f1, wm); f2 = bounce_back_cells(f2, wm)

        if step % max(5, n_steps // 40) == 0 or step == n_steps:
            rho_r = f1.sum(dim=0)
            rho_row = rho_r[measure_row, :].detach().cpu().numpy()
            threshold = (rho_h + rho_l) * 0.3
            liquid = rho_row > threshold
            if liquid.any():
                xs_l = np.where(liquid)[0]
                sw = float(xs_l[-1] - xs_l[0] + 1)
            else:
                sw = 0.0
            beta = sw / D0 if D0 > 0 else 0.0
            t_dim = step / t_c if t_c < 1e6 else 0.0
            _, ux, uy = macroscopic(f1 + f2)
            max_u = float(torch.sqrt(ux**2 + uy**2).max().item())
            rows.append({"step": step, "spread_width": round(sw, 2), "beta": round(beta, 4),
                         "t_dim": round(t_dim, 4), "max_u": round(max_u, 6)})
            if step % max(20, n_steps // 8) == 0:
                print(f"    step={step:5d} beta={beta:.3f} spread={sw:.1f} max_u={max_u:.4f}")

    elapsed = time.perf_counter() - t0
    betas = [r["beta"] for r in rows]
    beta_max_sim = max(betas) if betas else 0.0
    err_bm = abs(beta_max_sim - beta_max_ref) / beta_max_ref * 100 if beta_max_ref > 0 else None

    # Fit spreading exponent in the spreading phase
    spread_rows = [r for r in rows if 0.01 < r["beta"] < beta_max_sim * 0.95]
    if len(spread_rows) >= 3:
        ts = np.array([r["t_dim"] for r in spread_rows])
        bs = np.array([r["beta"] for r in spread_rows])
        mask = ts > 0
        if mask.sum() >= 2:
            n_fit, log_C = np.polyfit(np.log(ts[mask]), np.log(bs[mask]), 1)
            C_fit = math.exp(log_C)
        else:
            n_fit, C_fit = float("nan"), float("nan")
    else:
        n_fit, C_fit = float("nan"), float("nan")

    ref_exp = 0.5; ref_pre = We**0.25 if We > 0 else 0
    err_exp = abs(n_fit - ref_exp) / ref_exp * 100 if not math.isnan(n_fit) else None
    err_pre = abs(C_fit - ref_pre) / ref_pre * 100 if not math.isnan(C_fit) and ref_pre > 0 else None

    result = {
        "level": level_name, "nx": nx, "ny": ny, "radius": radius, "gap": gap,
        "D0": D0, "U0": U0, "sigma": round(sigma, 6),
        "We": round(We, 2), "t_c": round(t_c, 2), "n_steps": n_steps,
        "elapsed_s": round(elapsed, 1),
        "beta_max_sim": round(beta_max_sim, 4),
        "beta_max_ref_clanet": round(beta_max_ref, 4),
        "err_beta_max_pct": round(err_bm, 2) if err_bm is not None else None,
        "spreading_exponent_fit": round(float(n_fit), 4) if not math.isnan(n_fit) else None,
        "spreading_exponent_ref": ref_exp,
        "err_exponent_pct": round(err_exp, 2) if err_exp is not None else None,
        "spreading_prefactor_fit": round(float(C_fit), 4) if not math.isnan(C_fit) else None,
        "spreading_prefactor_ref": round(ref_pre, 4),
        "err_prefactor_pct": round(err_pre, 2) if err_pre is not None else None,
        "history": rows,
    }
    print(f"  beta_max: sim={beta_max_sim:.3f} ref={beta_max_ref:.3f} err={err_bm:.1f}%")
    if not math.isnan(n_fit):
        print(f"  exponent: fit={n_fit:.3f} ref={ref_exp:.3f} err={err_exp:.1f}%")
    print(f"  elapsed: {elapsed:.1f}s")
    return result


def evaluate():
    levels = [
        ("coarse_R8", 60, 40, 8.0, 1.0, 0.25, 200),
        ("fine_R16", 120, 80, 16.0, 2.0, 0.25, 400),
    ]

    cases = {}; error_table = []
    for ln, nx, ny, R, gap, U0, ns in levels:
        c = run_droplet_spreading(ln, nx, ny, R, gap, U0, ns)
        cases[ln] = c
        error_table.append({
            "level": ln, "We": c["We"],
            "beta_max_sim": c["beta_max_sim"], "beta_max_ref": c["beta_max_ref_clanet"],
            "err_beta_max_pct": c["err_beta_max_pct"],
            "exponent_fit": c["spreading_exponent_fit"], "exponent_ref": c["spreading_exponent_ref"],
            "err_exponent_pct": c["err_exponent_pct"],
            "prefactor_fit": c["spreading_prefactor_fit"], "prefactor_ref": c["spreading_prefactor_ref"],
            "err_prefactor_pct": c["err_prefactor_pct"],
        })

    cs = list(cases.values())
    conv = {}
    if len(cs) >= 2:
        conv = {"beta_max_coarse": cs[0]["beta_max_sim"], "beta_max_fine": cs[1]["beta_max_sim"],
                "err_beta_max_coarse": cs[0]["err_beta_max_pct"], "err_beta_max_fine": cs[1]["err_beta_max_pct"],
                "converging": (cs[1]["err_beta_max_pct"] or 999) < (cs[0]["err_beta_max_pct"] or 999)}

    results = {
        "benchmark": "droplet_spreading_solid_surface", "model": "cg",
        "references": {
            "clanet_2004": {"citation": "Clanet et al. (2004) J Fluid Mech 517:199-208", "figure": "Fig.8", "correlation": "beta_max = 0.5 * We^(1/4)"},
            "inertial_spreading_law": {"citation": "Dimensional analysis; Eggers et al. (2010) Phys Fluids 22:112101", "correlation": "beta(t) ~ We^(1/4) * (t/t_c)^(1/2)"},
            "laplace_test": {"description": "Surface tension measured via Laplace pressure test"},
        },
        "cases": cases, "error_table": error_table, "mesh_convergence": conv,
    }

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax = axes[0]
    for ln, case in cases.items():
        ts = [r["t_dim"] for r in case["history"]]; bs = [r["beta"] for r in case["history"]]
        ax.plot(ts, bs, "-", label=f"Sim {ln} (We={case['We']:.0f})", lw=1.5)
        We = case["We"]
        t_arr = np.array([t for t in ts if t > 0.01])
        if len(t_arr) > 0 and We > 0:
            ax.plot(t_arr, We**0.25 * np.sqrt(t_arr), "--", label=f"Law {ln}", alpha=0.7)
    ax.set(xlabel="t / t_c", ylabel="β = D(t)/D₀", title="Droplet spreading factor")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ref_We, ref_beta = [], []
    with (REF_DIR / "droplet_spreading_references.csv").open() as fh:
        for r in csv.DictReader(fh):
            ref_We.append(float(r["We"])); ref_beta.append(float(r["beta_max"]))
    We_arr = np.linspace(1, 500, 100)
    ax.plot(ref_We, ref_beta, "ks", ms=6, label="Clanet et al. (2004)")
    ax.plot(We_arr, 0.5*We_arr**0.25, "k-", label="0.5·We^(1/4)", alpha=0.5)
    for ln, case in cases.items():
        ax.plot(case["We"], case["beta_max_sim"], "o", ms=8, label=f"Sim {ln}")
    ax.set(xlabel="Weber number", ylabel="β_max", title="Max spreading vs Clanet (2004)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / "droplet_spreading_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = [r["level"] for r in error_table]
    x = range(len(labels)); w = 0.25
    ax.bar([i-w for i in x], [r["err_beta_max_pct"] or 0 for r in error_table], w, label="β_max err%", color="#4477AA")
    ax.bar([i for i in x], [r["err_exponent_pct"] or 0 for r in error_table], w, label="exponent err%", color="#EE6677")
    ax.bar([i+w for i in x], [r["err_prefactor_pct"] or 0 for r in error_table], w, label="prefactor err%", color="#FFBB33")
    ax.axhline(15, color="red", ls="--", lw=1, label="15% gate")
    ax.axhline(3, color="green", ls=":", lw=1, label="3% target")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set(ylabel="Error [%]", title="Droplet spreading error vs independent refs")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIG_DIR / "droplet_spreading_error_bars.png", dpi=150)
    plt.close(fig)

    (OUT_DIR / "error_table.json").write_text(json.dumps(results, indent=2))
    print("DONE")
    return results


if __name__ == "__main__":
    evaluate()
