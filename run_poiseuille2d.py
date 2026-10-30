#!/usr/bin/env python3
"""Fast verification: run H=20 only, combine with existing H=60 result."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parent
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "benchmarks"))

import torch

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.boundary.boundaries import zou_he_outlet_pressure

CS2 = 1.0 / 3.0
DEVICE = torch.device("cpu")


def zou_he_inlet_pressure(f: torch.Tensor, rho_in: float) -> torch.Tensor:
    f0, f2, f3, f4, f6, f7 = f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0], f[6, :, 0], f[7, :, 0]
    rho = torch.full_like(f0, rho_in)
    ux = 1.0 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho
    f_new = f.clone()
    f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux
    f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    return f_new


def run_case(H, tau=0.8, u_max_target=0.04, max_steps=5000, seed=0):
    torch.manual_seed(seed)
    ny = H + 2
    nx = 3 * H
    nu = (tau - 0.5) / 3.0
    delta_rho = 24.0 * nu * u_max_target / (CS2 * H)
    rho_in = 1.0 + delta_rho / 2.0
    rho_out = 1.0 - delta_rho / 2.0
    u_max_ana = delta_rho * CS2 * H * H / (8.0 * nu * nx)
    u_mean = (2.0 / 3.0) * u_max_ana
    Re = u_mean * H / nu
    Ma = u_max_ana / math.sqrt(CS2)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True
    wall[-1, :] = True

    xx = torch.arange(nx, device=DEVICE, dtype=torch.float32)
    rho0 = (rho_out + (rho_in - rho_out) * (1.0 - xx / (nx - 1))).view(1, nx).expand(ny, nx)
    f = equilibrium(rho0, torch.zeros((ny, nx), device=DEVICE), torch.zeros((ny, nx), device=DEVICE))

    def _step(f):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(DEVICE)], f)
        f = stream(f)
        f = zou_he_inlet_pressure(f, rho_in)
        return zou_he_outlet_pressure(f, rho_out)

    col = nx // 2
    t0 = time.time()
    umax_hist = []
    step = 0
    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            umax_hist.append(float(ux[:, col].max().item()))

    elapsed = time.time() - t0

    # Time-average profile
    prof_acc = torch.zeros(ny, device=DEVICE)
    n_avg = 50
    for _ in range(n_avg):
        f = _step(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= n_avg

    u_num = prof_acc[1:ny-1].cpu().numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = 4.0 * u_max_ana * (y_phys / H) * (1.0 - y_phys / H)

    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    imax = int(np.argmax(u_num))
    u_max_num = float(u_num[imax])
    u_max_ana_at_row = 4.0 * u_max_ana * (y_phys[imax] / H) * (1.0 - y_phys[imax] / H)
    u_max_err = abs(u_max_num - u_max_ana_at_row) / u_max_ana_at_row * 100.0

    mask = u_ana > 0.2 * u_max_ana
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)

    return {
        "H": H,
        "Re": round(Re, 2),
        "Ma": round(Ma, 5),
        "tau": tau,
        "n_steps": step,
        "steady": len(umax_hist) >= 10 and (max(umax_hist[-10:]) - min(umax_hist[-10:])) / max(abs(sum(umax_hist[-10:])/10), 1e-12) < 1e-5,
        "u_max_ana": round(u_max_ana, 8),
        "u_max_num": round(u_max_num, 8),
        "u_max_err_pct": round(u_max_err, 4),
        "l2_rel_err": round(l2_rel, 6),
        "max_rel_err_pct": round(max_rel, 4),
        "elapsed_s": round(elapsed, 1),
        "u_profile": [round(float(v), 8) for v in u_num],
        "u_analytic": [round(float(v), 8) for v in u_ana],
        "y_phys": [round(float(v), 4) for v in y_phys],
    }


def main():
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # Run H=20
    print("Running H=20...", flush=True)
    r20 = run_case(20, max_steps=5000)
    print(f"  H=20 Re={r20['Re']} steps={r20['n_steps']} "
          f"l2={r20['l2_rel_err']:.5f} umax_err={r20['u_max_err_pct']:+.4f}% "
          f"time={r20['elapsed_s']}s", flush=True)

    # Load existing H=60 result
    existing = json.loads((repo / "benchmarks/verified/poiseuille_2d/result.json").read_text())
    r60 = {
        "H": existing["H_eff"],
        "Re": round(existing["Re"], 2),
        "Ma": round(existing["Ma"], 5),
        "tau": existing["tau"],
        "n_steps": existing["n_steps"],
        "steady": existing["steady"],
        "u_max_ana": round(existing["u_max_ana"], 8),
        "u_max_num": round(existing["u_max_num"], 8),
        "u_max_err_pct": round(existing["u_max_err_pct"], 4),
        "l2_rel_err": round(existing["l2_rel_err"], 6),
        "max_rel_err_pct": round(existing["max_rel_err_central_pct"], 4),
        "elapsed_s": round(existing["elapsed_s"], 1),
        "u_profile": existing["u_profile"],
        "u_analytic": existing["u_analytic"],
        "y_phys": [round(float(i + 0.5), 4) for i in range(existing["H_eff"])],
    }

    results = [r20, r60]

    # Save results
    (out_dir / "poiseuille_2d_results.json").write_text(
        json.dumps(results, indent=2))

    # Summary
    summary = {
        "case": "B13_poiseuille_2d",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "zou_he_pressure_inlet/outlet + half-way bounce-back (pre-streaming)",
        "driving": "pressure_difference",
        "extrap": "none",
        "convergence": [
            {
                "H": r["H"],
                "Re": r["Re"],
                "l2_rel_err": r["l2_rel_err"],
                "u_max_err_pct": r["u_max_err_pct"],
                "max_rel_err_pct": r["max_rel_err_pct"],
                "n_steps": r["n_steps"],
                "steady": r["steady"],
            }
            for r in results
        ],
        "verdict": "PASS" if all(r["l2_rel_err"] < 0.01 and r["u_max_err_pct"] < 1.0 for r in results) else "FAIL",
    }
    (out_dir / "poiseuille_2d_report.json").write_text(
        json.dumps(summary, indent=2))

    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: velocity profile for H=60
        ax = axes[0]
        ax.plot(r60["u_analytic"], r60["y_phys"], "k-", linewidth=2, label="Analytic")
        ax.plot(r60["u_profile"], r60["y_phys"], "ro", markersize=4, label="LBM (H=60)")
        ax.set_xlabel("u (lattice units)")
        ax.set_ylabel("y (lattice units)")
        ax.set_title("Poiseuille 2D: Velocity Profile (H=60)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: convergence
        ax = axes[1]
        Hs = [r["H"] for r in results]
        l2s = [r["l2_rel_err"] for r in results]
        umax_errs = [r["u_max_err_pct"] for r in results]
        ax.semilogy(Hs, l2s, "bo-", linewidth=2, markersize=8, label="L2 relative error")
        ax.semilogy(Hs, umax_errs, "rs--", linewidth=2, markersize=8, label="u_max error (%)")
        ax.set_xlabel("H (grid resolution)")
        ax.set_ylabel("Error")
        ax.set_title("Grid Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(out_dir / "poiseuille_2d_profile.png", dpi=150)
        plt.close()
        print(f"Plots saved to {out_dir / 'poiseuille_2d_profile.png'}")
    except ImportError as e:
        print(f"matplotlib not available: {e}")

    print(f"\nResults saved to {out_dir}/")
    print(f"Verdict: {summary['verdict']}")
    return summary


if __name__ == "__main__":
    main()
