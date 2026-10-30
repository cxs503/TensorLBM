#!/usr/bin/env python3
"""SDAA sweep for 2D Couette flow verification — benchmarks/verified/couette_2d.

Runs the canonical Couette 2D benchmark on SDAA hardware, validates grid
convergence (H=40, 80) against the analytical linear profile u(y)=U0*y/H,
and produces an evidence JSON + comparison plots.

Usage:
    PYTHONPATH=src python scripts/sdaa_sweep_couette_2d.py
"""
from __future__ import annotations

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

# ── repo bootstrap ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from tensorlbm.d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk

CS2 = 1.0 / 3.0

# ── SDAA device check ───────────────────────────────────────────────────
SDAA_AVAILABLE = hasattr(torch, "sdaa") and torch.sdaa.is_available()
if SDAA_AVAILABLE:
    torch.sdaa.set_device(0)
    DEVICE = torch.device("sdaa:0")
    DEVICE_NAME = torch.sdaa.get_device_name(0)
    DEVICE_COUNT = torch.sdaa.device_count()
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "cpu"
    DEVICE_COUNT = 0

print(f"[sdaa_sweep] SDAA available: {SDAA_AVAILABLE}")
print(f"[sdaa_sweep] Device: {DEVICE}  name={DEVICE_NAME}  count={DEVICE_COUNT}")

# ── moving-wall bounce-back (copied from benchmarks/verified/couette_2d/run.py) ──
def moving_wall_bounce_back(
    f_pre: torch.Tensor, f: torch.Tensor, wall: torch.Tensor, u_wall: torch.Tensor
) -> torch.Tensor:
    opp = OPPOSITE.to(DEVICE)
    c = C.to(DEVICE)
    w = W.to(DEVICE)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run_couette_sdaa(
    H: int,
    tau: float = 0.8,
    U0: float = 0.05,
    min_steps: int = 10000,
    max_steps: int = 60000,
) -> dict:
    """Run one Couette case on SDAA and return measurements."""
    torch.manual_seed(0)
    ny = H + 2
    nx = H
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)
    Re = U0 * H / nu
    Ma = U0 / math.sqrt(CS2)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True
    wall[-1, :] = True
    u_wall = torch.zeros((ny, nx), device=DEVICE)
    u_wall[-1, :] = U0

    yy = torch.arange(ny, device=DEVICE, dtype=torch.float32)
    u0 = torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0).view(ny, 1).expand(ny, nx).clone()
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    f = equilibrium(torch.ones((ny, nx), device=DEVICE), u0, torch.zeros((ny, nx), device=DEVICE))
    initial_mass = float(f.sum().item())

    def _step(f):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall)
        from tensorlbm.solver import stream
        return stream(f)

    col = nx // 2
    t0 = time.time()
    u_top_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            u_top_hist.append(float(ux[ny - 2, col].item()))
            if step >= min_steps and len(u_top_hist) >= 10:
                recent = u_top_hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    # Time-average profile
    prof_acc = torch.zeros(ny, device=DEVICE)
    for _ in range(200):
        f = _step(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= 200.0

    u_num = prof_acc[1:ny - 1].cpu().numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * (y_phys / H_eff)

    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    max_abs_err = float(np.max(np.abs(u_num - u_ana)))
    mask = u_ana > 0.2 * U0
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)

    u_top_num = float(u_num[-1])
    u_top_ana = float(u_ana[-1])
    u_top_err = abs(u_top_num - u_top_ana) / u_top_ana * 100.0

    slope_num = float((u_num[-1] - u_num[0]) / (y_phys[-1] - y_phys[0]))
    tau_ana = nu * U0 / H_eff
    tau_num = slope_num * nu
    tau_err_pct = (tau_num - tau_ana) / tau_ana * 100.0

    return {
        "H": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "U0": U0,
        "Re": Re,
        "Ma": Ma,
        "n_steps": step,
        "steady": steady,
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_pct": max_rel,
        "u_top_row_num": u_top_num,
        "u_top_row_ana": u_top_ana,
        "u_top_row_err_pct": u_top_err,
        "tau_ana": tau_ana,
        "tau_num": tau_num,
        "tau_err_pct": tau_err_pct,
        "elapsed_s": round(elapsed, 1),
        "u_profile": [round(float(v), 8) for v in u_num],
        "u_analytic": [round(float(v), 8) for v in u_ana],
    }


def make_comparison_plot(results: list[dict], out_png: str) -> None:
    """Generate profile comparison + error plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        H = r["H"]
        u_num = np.array(r["u_profile"])
        u_ana = np.array(r["u_analytic"])
        y_phys = np.arange(1, len(u_num) + 1, dtype=np.float64) - 0.5
        H_eff = H

        ax = axes[0]
        ax.plot(u_ana, y_phys / H_eff, "--", label=f"Analytic (H={H})", alpha=0.7)
        ax.plot(u_num, y_phys / H_eff, "o-", markersize=3, label=f"SDAA LBM (H={H})")

        ax = axes[1]
        err = np.abs(u_num - u_ana) / np.maximum(u_ana, 1e-12) * 100.0
        ax.semilogy(y_phys / H_eff, err + 1e-12, "o-", markersize=3, label=f"H={H}")

    axes[0].set_xlabel("u / U0")
    axes[0].set_ylabel("y / H")
    axes[0].set_title("Velocity Profile: SDAA LBM vs Analytic")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("y / H")
    axes[1].set_ylabel("Relative Error (%)")
    axes[1].set_title("Pointwise Relative Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Couette 2D on SDAA ({DEVICE_NAME}) — D2Q9 BGK", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[sdaa_sweep] Plot saved: {out_png}")


def main() -> None:
    out_dir = REPO / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    H_list = [40, 80]
    results = []
    for H in H_list:
        print(f"\n[sdaa_sweep] Running H={H} ...", flush=True)
        r = run_couette_sdaa(H)
        results.append(r)
        print(
            f"  H={r['H']:3d} Re={r['Re']:7.2f} steps={r['n_steps']:6d} "
            f"l2_err={r['l2_rel_err']:.2e} max_rel={r['max_rel_err_pct']:.4f}% "
            f"u_top_err={r['u_top_row_err_pct']:+.4f}% tau_err={r['tau_err_pct']:+.4f}% "
            f"steady={r['steady']} elapsed={r['elapsed_s']:.1f}s",
            flush=True,
        )

    # Convergence check
    err_decreased = results[1]["max_rel_err_pct"] < results[0]["max_rel_err_pct"]
    all_pass = all(r["max_rel_err_pct"] < 3.0 for r in results)

    # Build evidence
    evidence = {
        "case": "couette_2d_sdaa_sweep",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "moving top wall + stationary bottom wall, half-way bounce-back",
        "reference": "analytical u(y) = U0*y/H",
        "device": {
            "type": "sdaa",
            "available": SDAA_AVAILABLE,
            "name": DEVICE_NAME,
            "count": DEVICE_COUNT,
            "device_str": str(DEVICE),
        },
        "threshold_pct": 3.0,
        "grids": {
            f"H{r['H']}": {
                "H": r["H"],
                "Re": r["Re"],
                "l2_rel_err": r["l2_rel_err"],
                "max_rel_err_pct": r["max_rel_err_pct"],
                "u_top_row_err_pct": r["u_top_row_err_pct"],
                "tau_err_pct": r["tau_err_pct"],
                "n_steps": r["n_steps"],
                "steady": r["steady"],
                "elapsed_s": r["elapsed_s"],
                "pass": r["max_rel_err_pct"] < 3.0,
            }
            for r in results
        },
        "convergence": {
            "max_rel_err_pct": [r["max_rel_err_pct"] for r in results],
            "err_decreased": err_decreased,
        },
        "all_criteria_pass": all_pass,
        "verdict": "PASS" if all_pass else "HONEST-FAIL",
    }

    evidence_path = out_dir / "sdaa_couette_2d_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    print(f"\n[sdaa_sweep] Evidence written: {evidence_path}")

    # Generate plots
    plot_path = out_dir / "sdaa_couette_2d_comparison.png"
    make_comparison_plot(results, str(plot_path))

    # Summary
    print("\n" + "=" * 60)
    print("[sdaa_sweep] SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["max_rel_err_pct"] < 3.0 else "FAIL"
        print(f"  H={r['H']:3d}: max_rel_err={r['max_rel_err_pct']:.4f}% [{status}]")
    print(f"  Convergence (err decreased): {err_decreased}")
    print(f"  Overall verdict: {evidence['verdict']}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
