#!/usr/bin/env python
"""SDAA portability sweep: cylinder_re40_st (Re=20/40, steady, free-stream).

Adapts the W11-A2 formal protocol (run_formal_w11a2.py) to run on SDAA
device, verifying device portability of the TensorLBM D2Q9 MRT solver.

Protocol (frozen from W11-A2):
  §2 solver: collide_mrt(tau, tau_field=sponge) -> stream -> forces
     (post-stream, pre-BB) -> far_field_bc_2d
  §3 domain: LX=70D, UP=40D, H=20D
  §4 geometry: area-equivalent mask; nu = u_lb*D_eff/Re; Cd denom 0.5*u_lb^2*D_eff
  §5 ladder: u_lb = 0.05 fixed; tau follows
  §6 steady: t_c = D_eff/u_lb; plateau check from 60 t_c, every 10 t_c,
     |dm|/m < 0.15%; cap 150 t_c; Cd = mean last 20 t_c
  §7 stop-loss: wall > 17 h -> stop

Reference (Fornberg 1980 JFM 98:819):
  Re=20: Cd_ref = 2.0001
  Re=40: Cd_ref = 1.498

Verdict threshold: |err_pct| < 3%
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure TensorLBM src is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensorlbm.boundaries import (
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

# ── Frozen protocol parameters (W11-A2) ─────────────────────────────────────
LX_D, UP_D, H_D = 70.0, 40.0, 20.0
SPONGE_D, SPONGE_ALPHA = 10.0, 10.0
U_LB = 0.05
PLATEAU_REL = 0.0015
CAP_TC = 150.0
STOP_LOSS_S = 17.0 * 3600.0

# Reference values (Fornberg 1980)
REF_CD = {20.0: 2.0001, 40.0: 1.498}


def area_lock_mask(D: int, nx: int, ny: int, cy: float, cx: float, dev):
    """§4: bisection so R_area = sqrt(N_solid/pi) is closest to D/2."""
    R = D / 2.0
    lo, hi = R - 1.0, R + 1.0
    best = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        n = int(cylinder_mask(nx, ny, cx, cy, mid, dev).sum())
        ra = math.sqrt(n / math.pi)
        if best is None or abs(ra - R) < best[2]:
            best = (mid, n, abs(ra - R), ra)
        if ra > R:
            hi = mid
        else:
            lo = mid
    r_mask = best[0]
    solid = cylinder_mask(nx, ny, cx, cy, r_mask, dev)
    n = int(solid.sum())
    rows = torch.any(solid, dim=1).nonzero().flatten()
    r_row = (int(rows.max()) - int(rows.min()) + 1) / 2.0
    r_area = math.sqrt(n / math.pi)
    return solid, {
        "r_mask": float(r_mask),
        "n_solid": n,
        "R_row": float(r_row),
        "R_area": float(r_area),
        "R_eff": float(r_area),
        "D_eff_cells": 2.0 * float(r_area),
    }


def run_single(re_val: float, D_cells: int, device: torch.device) -> dict:
    """Run one (Re, D) case on the given device."""
    t0 = time.time()
    torch.manual_seed(0)

    nx, ny = int(round(LX_D * D_cells)), int(round(2 * H_D * D_cells))
    cx, cy = round(UP_D * D_cells), ny // 2
    solid, ginfo = area_lock_mask(D_cells, nx, ny, float(cy), float(cx), device)
    d_eff = ginfo["D_eff_cells"]
    nu = U_LB * d_eff / re_val
    tau = 0.5 + 3.0 * nu
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * D_cells), int(SPONGE_D * D_cells), power=2.0, device=device
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)
    dyn_p = 0.5 * U_LB**2 * d_eff

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full((ny, nx), U_LB, dtype=torch.float32, device=device)
    uy0 = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, uy0)
    del rho0, ux0, uy0

    t_c = max(1, int(round(d_eff / U_LB)))
    cap_steps = int(CAP_TC * t_c)
    check_from = int(60 * t_c)
    check_every = int(10 * t_c)
    last_n = int(20 * t_c)

    cd_hist: list[float] = []
    trace_cd: list[float] = []
    trace_t: list[int] = []
    plateau_at: int | None = None
    stop_reason = "cap"

    for step in range(1, cap_steps + 1):
        if time.time() - t0 > STOP_LOSS_S:
            stop_reason = "stop_loss"
            break
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx = compute_obstacle_forces(f, solid)[0].item()
        cd = fx / dyn_p
        cd_hist.append(cd)
        f = far_field_bc_2d(f, U_LB, solid)

        if len(cd_hist) >= last_n:
            mean_cd = float(np.mean(cd_hist[-last_n:]))
            trace_cd.append(mean_cd)
            trace_t.append(step)

            if step >= check_from and step % check_every == 0:
                if len(trace_cd) >= 2:
                    m_prev = trace_cd[-2]
                    m_curr = trace_cd[-1]
                    rel = abs(m_curr - m_prev) / max(abs(m_prev), 1e-12)
                    if rel < PLATEAU_REL:
                        plateau_at = step
                        stop_reason = "plateau"
                        break

    if plateau_at is None:
        plateau_at = cap_steps
    cd_final = float(np.mean(cd_hist[-last_n:])) if cd_hist else float("nan")
    wall_s = time.time() - t0

    cd_ref = REF_CD.get(re_val, float("nan"))
    err_pct = 100.0 * (cd_final - cd_ref) / cd_ref if cd_ref else float("nan")

    return {
        "re": re_val,
        "D": D_cells,
        "u_lb": U_LB,
        "ma": U_LB / math.sqrt(3.0),
        "tau": float(tau),
        "nu_lb": float(nu),
        "nx": nx,
        "ny": ny,
        "lx_d": LX_D,
        "up_d": UP_D,
        "h_d": H_D,
        "sponge_d": SPONGE_D,
        "sponge_alpha": SPONGE_ALPHA,
        "cx": cx,
        "cy": cy,
        **ginfo,
        "t_c_steps": t_c,
        "steps": plateau_at,
        "plateau_at": plateau_at,
        "cap_steps": cap_steps,
        "stop_reason": stop_reason,
        "cd": cd_final,
        "cd_last10tc": trace_cd[-1] if trace_cd else float("nan"),
        "cd_ref_primary": cd_ref,
        "err_pct": err_pct,
        "wall_s": wall_s,
        "trace_cd": trace_cd,
        "trace_t": trace_t,
    }


def verify_device() -> dict:
    """Verify SDAA device availability and properties."""
    info = {
        "sdaa_available": torch.sdaa.is_available(),
        "sdaa_device_count": torch.sdaa.device_count() if hasattr(torch.sdaa, "device_count") else None,
    }
    if info["sdaa_available"]:
        try:
            props = torch.sdaa.get_device_properties(0)
            info["device_name"] = props.name if hasattr(props, "name") else str(props)
            info["total_memory_miB"] = props.total_memory if hasattr(props, "total_memory") else None
        except Exception as e:
            info["device_props_error"] = str(e)
        # Quick tensor test
        try:
            t = torch.ones(100, device="sdaa:0")
            info["tensor_test"] = "OK"
            info["tensor_device"] = str(t.device)
        except Exception as e:
            info["tensor_test"] = f"FAIL: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=float, nargs="+", default=[20.0, 40.0], help="Reynolds numbers to sweep")
    ap.add_argument("--D", type=int, nargs="+", default=[40, 80], help="Grid resolutions (cells per D)")
    ap.add_argument("--device", default="sdaa:0", help="Torch device string")
    ap.add_argument("--out-dir", default="artifacts", help="Output directory")
    a = ap.parse_args()

    dev = torch.device(a.device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Device verification ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1: SDAA Device Verification")
    print("=" * 60)
    dev_info = verify_device()
    print(json.dumps(dev_info, indent=2))
    if not dev_info["sdaa_available"]:
        print("FATAL: SDAA not available — aborting.")
        return 1

    # ── Step 2: Grid convergence sweep ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Grid Convergence Sweep (SDAA)")
    print("=" * 60)

    all_results = []
    for re_val in a.re:
        for D in a.D:
            print(f"\n--- Re={re_val:.0f}, D={D} ---")
            r = run_single(re_val, D, dev)
            all_results.append(r)
            print(
                f"  cd={r['cd']:.6f}  ref={r['cd_ref_primary']:.4f}  "
                f"err={r['err_pct']:+.4f}%  steps={r['steps']}  "
                f"stop={r['stop_reason']}  wall={r['wall_s']:.1f}s"
            )

    # ── Step 3: Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Verdict (|err| < 3%)")
    print("=" * 60)

    verdict_rows = []
    all_pass = True
    for r in all_results:
        passed = abs(r["err_pct"]) < 3.0
        all_pass = all_pass and passed
        verdict_rows.append({
            "re": r["re"],
            "D": r["D"],
            "cd": r["cd"],
            "cd_ref": r["cd_ref_primary"],
            "err_pct": r["err_pct"],
            "passed": passed,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  Re={r['re']:.0f} D={r['D']:3d}: err={r['err_pct']:+.4f}%  [{status}]")

    overall_verdict = "PASS" if all_pass else "HONEST-FAIL"
    print(f"\n  Overall verdict: {overall_verdict}")

    # ── Step 4: Generate evidence ────────────────────────────────────────
    evidence = {
        "case": "cylinder_re40_st",
        "mission_id": 437,
        "branch": "agent/m437",
        "device_evidence": dev_info,
        "protocol": {
            "solver": "D2Q9 + collide_mrt + far_field_bc_2d",
            "domain": {"LX_D": LX_D, "UP_D": UP_D, "H_D": H_D},
            "geometry": "area-equivalent mask",
            "u_lb": U_LB,
            "sponge": {"width_D": SPONGE_D, "alpha": SPONGE_ALPHA, "power": 2.0},
            "steady_protocol": {
                "plateau_rel": PLATEAU_REL,
                "cap_tc": CAP_TC,
                "check_from_tc": 60,
                "check_every_tc": 10,
                "avg_last_tc": 20,
            },
        },
        "reference": {
            "source": "Fornberg 1980 JFM 98:819",
            "values": REF_CD,
        },
        "threshold_pct": 3.0,
        "results": verdict_rows,
        "overall_verdict": overall_verdict,
        "per_run_details": [
            {k: v for k, v in r.items() if k not in ("trace_cd", "trace_t")}
            for r in all_results
        ],
    }

    evidence_path = out_dir / "sdaa_cylinder_re40_st_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    print(f"\nEvidence written to: {evidence_path}")

    # ── Step 5: Generate convergence plot ────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"20.0": "#1f77b4", "40.0": "#ff7f0e"}
        markers = {40: "o", 80: "s"}

        for r in all_results:
            re_key = str(r["re"])
            color = colors.get(re_key, "#333333")
            tc_steps = r["t_c_steps"]
            t_tc = [t / tc_steps for t in r["trace_t"]]
            ax.plot(t_tc, r["trace_cd"], color=color, marker=markers.get(r["D"], "."),
                    label=f"Re={int(r['re'])} D={r['D']}", markersize=3, alpha=0.8)
            # Reference line
            ax.axhline(r["cd_ref_primary"], color=color, linestyle="--", alpha=0.4)

        ax.set_xlabel("Time (convective units t_c = D_eff / u_lb)")
        ax.set_ylabel("Drag coefficient Cd")
        ax.set_title("Cylinder steady flow — SDAA convergence (Re=20/40)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig_path = out_dir / "sdaa_cylinder_re40_st_convergence.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Convergence plot saved: {fig_path}")
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
