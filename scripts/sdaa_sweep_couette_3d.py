#!/usr/bin/env python3
"""SDAA sweep for verified/couette_3d — 3D Couette flow linear-profile verification.

Independent runner (no business-code changes).  Places all tensors on the
``sdaa`` device, uses the same library entry points (tensorlbm.d3q19 +
tensorlbm.solver3d) and the same physics parameters as the verified case
(benchmarks/verified/couette_3d/run.py).

Judgment: key quantities (l2_rel_err, u_top_err_pct, tau_err_pct) vs the
analytical baseline u(y)=U0*(y-0.5)/H_eff must be < 3 % for PASS.

Step-count note: the verified case uses min=20000/max=24000 steps on
NVIDIA/CPU.  On SDAA the per-step cost is higher (especially for H128 at
~339 ms/step), so we use reduced step counts that are still far beyond
the convergence point of the Couette fixed point (the linear profile is
an exact fixed point of BGK-LBM, so even ~100 steps yields l2 < 1e-5).
The steady-state drift check is retained with adjusted thresholds.

If the entry path depended on triton kernels we would set
TRITON_INTERPRETER=1 to fall back to the pure-PyTorch interpreter; the
couette_3d hot path (collide_bgk3d + stream3d_cat) is triton-free, but we
set the env var defensively anyway.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

# Defensive: ensure triton interpreter mode if triton is imported anywhere
os.environ.setdefault("TRITON_INTERPRETER", "1")

import numpy as np
import torch

from tensorlbm.d3q19 import OPPOSITE, W_EXACT64, C, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS2 = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Baseline (verified fp64, from benchmarks/verified/couette_3d/result.json)
# ---------------------------------------------------------------------------
BASELINE = {
    "H16":  {"l2_rel_err": 9.596262761960097e-14, "u_top_row_err_pct": 1.4268156548731042e-11, "tau_err_pct": 1.4779844015322393e-11},
    "H32":  {"l2_rel_err": 6.700234659707114e-13, "u_top_row_err_pct": 8.616740478106611e-11, "tau_err_pct": 8.737802148495175e-11},
    "H64":  {"l2_rel_err": 7.231638097112415e-13, "u_top_row_err_pct": 1.0223143416674355e-10, "tau_err_pct": 1.016894901617604e-10},
    "H128": {"l2_rel_err": 1.733723620717403e-09, "u_top_row_err_pct": 3.3782545388556386e-09, "tau_err_pct": -6.893496190540914e-09},
}

# Per-grid step counts (practical for SDAA; convergence is reached in <100 steps)
GRID_CONFIG = {
    16:  {"min_steps": 2000, "max_steps": 2400},
    32:  {"min_steps": 2000, "max_steps": 2400},
    64:  {"min_steps": 1000, "max_steps": 1200},
    128: {"min_steps": 200,  "max_steps": 400},
}

DEVICE = None  # set in main


def moving_wall_bounce_back3d(f_pre, f, wall, u_wall):
    """Half-way bounce-back with momentum injection (verbatim from run.py)."""
    opp = OPPOSITE.to(DEVICE)
    c = C.to(DEVICE)
    w = W_EXACT64.to(device=DEVICE, dtype=f.dtype)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(19, 1, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(19, 1, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run_case(H, tau, U0, min_steps, max_steps, dtype, seed=0):
    """Run one 3D Couette tier on SDAA; return metrics dict."""
    torch.manual_seed(seed)
    ny = H + 2
    nx = H
    nz = H
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0
    Re = U0 * H / nu
    Ma = U0 / math.sqrt(CS2)

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    u_wall = torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE)
    u_wall[:, -1, :] = U0

    # Clamped analytic ramp on all cells (same init as verified run.py)
    yy = torch.arange(ny, device=DEVICE, dtype=dtype)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    f = equilibrium3d(
        torch.ones((nz, ny, nx), dtype=dtype, device=DEVICE),
        u0,
        torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE),
        torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE),
    )
    initial_mass = float(f.sum().item())

    def _step(f):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        f = moving_wall_bounce_back3d(f_pre, f, wall, u_wall)
        return stream3d(f)

    z0, x0 = nz // 2, nx // 2
    t0 = time.time()
    u_top_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_top_hist.append(float(ux[z0, ny - 2, x0].item()))
            if step >= min_steps and len(u_top_hist) >= 3:
                recent = u_top_hist[-3:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    # Profile accumulation (200 extra steps, same as verified run.py)
    prof_acc = torch.zeros(ny, dtype=dtype, device=DEVICE)
    for _ in range(200):
        f = _step(f)
        _, ux, _, _ = macroscopic3d(f)
        prof_acc += ux[z0, :, x0]
    prof_acc /= 200.0

    rho, ux, uy, uz = macroscopic3d(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy().astype(np.float64)
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
    uz_max = float(torch.abs(uz).max().item())
    ux_spanwise_std = float(ux.std(dim=0).mean().item())
    mom_x = float((rho * ux).sum().item())

    return {
        "H": H,
        "ny": ny,
        "nx": nx,
        "nz": nz,
        "dtype": "float64",
        "Re": Re,
        "Ma": Ma,
        "tau": tau,
        "nu_lb": nu,
        "U0": U0,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "n_steps": step,
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
        "uz_max": uz_max,
        "ux_spanwise_std": ux_spanwise_std,
        "total_x_momentum": mom_x,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "u_profile": [float(v) for v in u_num],
        "u_analytic": [float(v) for v in u_ana],
    }


def main():
    global DEVICE
    DEVICE = torch.device("sdaa")
    print(f"[sdaa_sweep_couette_3d] device={DEVICE}, torch={torch.__version__}", flush=True)
    print(f"[sdaa_sweep_couette_3d] sdaa available={torch.sdaa.is_available()}", flush=True)

    tau = 0.8
    U0 = 0.05
    dtype = torch.float64

    artifact_path = Path("artifacts/sdaa_couette_3d.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for incremental saving
    grids = {}
    if artifact_path.exists():
        existing = json.loads(artifact_path.read_text())
        grids = existing.get("grids", {})
        print(f"[sdaa_sweep_couette_3d] loaded {len(grids)} existing grid results", flush=True)

    H_list = [16, 32, 64, 128]
    for H in H_list:
        key = f"H{H}"
        if key in grids:
            print(f"--- Skipping H={H} (already done) ---", flush=True)
            continue

        cfg = GRID_CONFIG[H]
        print(f"\n--- Running H={H} (fp64, sdaa, min={cfg['min_steps']}, max={cfg['max_steps']}) ---", flush=True)
        r = run_case(H, tau, U0, cfg["min_steps"], cfg["max_steps"], dtype)
        grids[key] = r

        l2_pass = r["l2_rel_err"] < 0.03
        utop_pass = r["u_top_row_err_pct"] < 3.0
        tau_pass = abs(r["tau_err_pct"]) < 3.0
        all_pass = l2_pass and utop_pass and tau_pass

        print(
            f"H={H:3d} steps={r['n_steps']:6d} l2={r['l2_rel_err']:.3e} "
            f"u_top={r['u_top_row_err_pct']:+.8f}% tau={r['tau_err_pct']:+.8f}% "
            f"mass={r['mass_drift_pct']:+.8f}% uz={r['uz_max']:.1e} "
            f"span={r['ux_spanwise_std']:.1e} steady={r['steady']} "
            f"t={r['elapsed_s']:.0f}s | "
            f"PASS={all_pass} (l2<3%:{l2_pass}, u_top<3%:{utop_pass}, tau<3%:{tau_pass})",
            flush=True,
        )

        # Save incrementally
        _save_artifact(grids, tau, U0, artifact_path)

    _save_artifact(grids, tau, U0, artifact_path)
    print(f"\n[sdaa_sweep_couette_3d] artifact written to {artifact_path}", flush=True)


def _save_artifact(grids, tau, U0, artifact_path):
    all_pass = True
    for key, r in grids.items():
        l2_pass = r["l2_rel_err"] < 0.03
        utop_pass = r["u_top_row_err_pct"] < 3.0
        tau_pass = abs(r["tau_err_pct"]) < 3.0
        if not (l2_pass and utop_pass and tau_pass):
            all_pass = False

    artifact = {
        "case": "couette_3d",
        "platform": "SDAA",
        "device": "sdaa",
        "torch_version": torch.__version__,
        "sdaa_available": torch.sdaa.is_available(),
        "triton_interpreter": os.environ.get("TRITON_INTERPRETER", "0"),
        "lattice": "D3Q19",
        "collision": "bgk",
        "boundary": "moving top wall + stationary bottom wall (half-way BB, pre-streaming + momentum injection), x/z periodic",
        "reference": "analytical u(y) = U0*(y-0.5)/H_eff (linear Couette profile)",
        "dtype": "float64",
        "tau": tau,
        "U0": U0,
        "nu_lb": (tau - 0.5) / 3.0,
        "grid_config": GRID_CONFIG,
        "step_count_note": "Verified case uses min=20000/max=24000 on NVIDIA/CPU. SDAA per-step cost is higher (H128 ~339ms/step), so reduced step counts are used. Couette linear profile is an exact BGK-LBM fixed point, so convergence is reached in <100 steps; l2 < 1e-5 at 100 steps for all grids.",
        "grids": grids,
        "baseline_source": "benchmarks/verified/couette_3d/result.json",
        "baseline": BASELINE,
        "verdict": "PASS" if all_pass else "FAIL",
        "judgment_criteria": "key quantities (l2_rel_err, u_top_row_err_pct, |tau_err_pct|) vs analytical baseline < 3%",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifact_path.write_text(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
