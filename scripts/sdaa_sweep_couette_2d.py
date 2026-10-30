#!/usr/bin/env python
"""SDAA sweep for benchmarks/verified/couette_2d.

Runs the Couette 2D analytic-profile validation on the 'sdaa' device,
records device evidence, computes all criteria from the simulation output,
and writes artifacts/sdaa_couette_2d.json.

Criteria (from README / result.json):
  1. max_rel_err_masked_pct <= 3.0   (per-grid pointwise relative error)
  2. convergence: error decreases with finer grid (H=40 -> H=80)
  3. steady state achieved (steady == True)
  4. finite: all populations finite (finite == True)

Grid / steps: H=40 and H=80 (both grids to check convergence).
  Wall clock budget: ~5 min total (H=40 ~47s, H=80 ~190s).

Compile mode: 'eager' to avoid torch.compile / triton indirection on sdaa.
  TRITON_INTERPRET=1 is NOT needed because we use eager mode (no compilation).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# ---- bootstrap paths -------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
BENCH = REPO / "benchmarks"
BENCH_CASE = REPO / "benchmarks" / "verified" / "couette_2d"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BENCH))

import numpy as np
import torch

# ---- device evidence --------------------------------------------------------
SDAA_AVAILABLE = torch.sdaa.is_available()
SDAA_COUNT = torch.sdaa.device_count()
SDAA_DEVICE_NAME = torch.sdaa.get_device_name(0) if SDAA_COUNT > 0 else "N/A"
DEVICE = torch.device("sdaa:0")

print(f"[sdaa_sweep] device: {DEVICE}")
print(f"[sdaa_sweep] sdaa_available: {SDAA_AVAILABLE}")
print(f"[sdaa_sweep] sdaa_count: {SDAA_COUNT}")
print(f"[sdaa_sweep] device_name: {SDAA_DEVICE_NAME}")

# ---- import the case runner -------------------------------------------------
from compile_route import route_step  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_mrt, stream

CS2 = 1.0 / 3.0


def moving_wall_bounce_back(
    f_pre: torch.Tensor, f: torch.Tensor, wall: torch.Tensor, u_wall: torch.Tensor
) -> torch.Tensor:
    """Pre-streaming half-way bounce-back with moving-wall momentum injection."""
    opp = OPPOSITE.to(DEVICE)
    c = C.to(DEVICE)
    w = W.to(DEVICE)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run_case_sdaa(
    H: int,
    tau: float,
    U0: float,
    collision: str,
    min_steps: int,
    max_steps: int,
    compile_mode: str | None = "eager",
) -> dict:
    """Run one Couette case on sdaa device and return measurements."""
    torch.manual_seed(0)
    ny = H + 2
    nx = H
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0
    Re = U0 * H / nu
    Ma = U0 / math.sqrt(CS2)

    collide_fn = collide_mrt if collision == "mrt" else collide_bgk

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
        f = collide_fn(f, tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall)
        return stream(f)

    step_fn = route_step(_step, compile_mode, name=f"couette_2d_sdaa[H{H}]")

    col = nx // 2
    t0 = time.time()
    u_top_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
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

    # Time-average the profile over the last 200 steps
    prof_acc = torch.zeros(ny, device=DEVICE)
    for _ in range(200):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= 200.0

    rho, ux, _ = macroscopic(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy()
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

    mom_x = float((rho * ux).sum().item())

    return {
        "H_eff": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "U0": U0,
        "Re": Re,
        "Ma": Ma,
        "min_steps": min_steps,
        "n_steps": step,
        "compile_mode": compile_mode,
        "steady": steady,
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_masked_pct": max_rel,
        "u_top_row_num": u_top_num,
        "u_top_row_ana": u_top_ana,
        "u_top_row_err_pct": u_top_err,
        "tau_ana": tau_ana,
        "tau_num": tau_num,
        "tau_err_pct": tau_err_pct,
        "total_x_momentum": mom_x,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    t_start = time.time()

    # ---- configuration ------------------------------------------------------
    H_list = [40, 80]  # both grids for convergence check
    tau = 0.8
    U0 = 0.05
    collision = "bgk"
    min_steps = 10000
    max_steps = 60000
    compile_mode = "eager"  # avoid torch.compile/triton on sdaa

    # ---- triton workaround note ---------------------------------------------
    triton_interpret = os.environ.get("TRITON_INTERPRET", "not set")
    triton_note = (
        "TRITON_INTERPRET=1 not needed: compile_mode='eager' bypasses torch.compile entirely."
    )

    # ---- run ----------------------------------------------------------------
    results = []
    for H in H_list:
        print(f"\n[sdaa_sweep] Running H={H} ...", flush=True)
        r = run_case_sdaa(H, tau, U0, collision, min_steps, max_steps, compile_mode)
        results.append(r)
        print(
            f"[sdaa_sweep] H={r['H_eff']:3d} Re={r['Re']:7.2f} steps={r['n_steps']:6d} "
            f"l2_err={r['l2_rel_err']:.2e} max_rel={r['max_rel_err_masked_pct']:.4f}% "
            f"u_top_err={r['u_top_row_err_pct']:+.4f}% tau_err={r['tau_err_pct']:+.4f}% "
            f"steady={r['steady']} elapsed={r['elapsed_s']:.1f}s",
            flush=True,
        )

    elapsed_total = time.time() - t_start

    # ---- criteria evaluation ------------------------------------------------
    MAX_REL_THRESHOLD = 3.0  # percent

    criteria = []
    for r in results:
        H = r["H_eff"]
        # Criterion 1: max_rel_err_masked_pct <= 3%
        c1_pass = r["max_rel_err_masked_pct"] <= MAX_REL_THRESHOLD
        criteria.append({
            "H": H,
            "criterion": "max_rel_err_masked_pct <= 3%",
            "value": r["max_rel_err_masked_pct"],
            "threshold": MAX_REL_THRESHOLD,
            "pass": c1_pass,
        })
        # Criterion 2: steady state
        criteria.append({
            "H": H,
            "criterion": "steady == True",
            "value": r["steady"],
            "threshold": True,
            "pass": r["steady"],
        })
        # Criterion 3: finite
        criteria.append({
            "H": H,
            "criterion": "finite == True",
            "value": r["finite"],
            "threshold": True,
            "pass": r["finite"],
        })

    # Criterion 4: convergence (only meaningful with >= 2 grids)
    if len(results) >= 2:
        conv_pass = results[1]["max_rel_err_masked_pct"] < results[0]["max_rel_err_masked_pct"]
        criteria.append({
            "H": "40->80",
            "criterion": "convergence (error decreases with finer grid)",
            "value": f"{results[0]['max_rel_err_masked_pct']:.4f}% -> {results[1]['max_rel_err_masked_pct']:.4f}%",
            "threshold": "monotonic decrease",
            "pass": conv_pass,
        })
    else:
        criteria.append({
            "H": "N/A",
            "criterion": "convergence (error decreases with finer grid)",
            "value": "single grid only",
            "threshold": "monotonic decrease",
            "pass": False,
            "note": "Only H=40 run; convergence check requires >= 2 grids",
        })

    all_pass = all(c["pass"] for c in criteria)

    # ---- build artifact -----------------------------------------------------
    artifact = {
        "case": "couette_2d",
        "device_evidence": {
            "device": str(DEVICE),
            "sdaa_available": SDAA_AVAILABLE,
            "sdaa_count": SDAA_COUNT,
            "sdaa_device_name": SDAA_DEVICE_NAME,
        },
        "configuration": {
            "H_list": H_list,
            "tau": tau,
            "U0": U0,
            "collision": collision,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "compile_mode": compile_mode,
        },
        "triton_workaround": triton_note,
        "results": results,
        "criteria": criteria,
        "overall_pass": all_pass,
        "wall_clock_s": round(elapsed_total, 1),
    }

    out_path = REPO / "artifacts" / "sdaa_couette_2d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n[sdaa_sweep] Artifact written to {out_path}")
    print(f"[sdaa_sweep] Overall PASS: {all_pass}")
    print(f"[sdaa_sweep] Wall clock: {elapsed_total:.1f}s")

    for c in criteria:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['criterion']}: {c['value']} (threshold: {c['threshold']})")


if __name__ == "__main__":
    main()
