#!/usr/bin/env python3
"""SDAA sweep for benchmarks/verified/sod_shock_tube (isothermal D2Q9 Sod shock tube).

This script runs the exact same physics and metrics as the original run.py,
but places all tensor computation on the SDAA device.

No triton dependency in the core path (equilibrium/stream/collide_bgk/macroscopic
are pure torch ops), so TRITON_INTERPRET is not needed.

Validation criteria (from README, original NVIDIA-side verdict):
  - L2_rho_final <= 0.03 (3%)
  - |W_rel| <= 0.03 (3%)
  - |rho_mid_rel| <= 0.03 (3%)
  - |u_mid_rel| <= 0.03 (3%)
  - Monotone convergence: L2_rho not increasing with resolution
  - No NaN/negative density (first_bad_step is None)
"""

import json
import math
import os
import sys
import time

# Use local src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

CS = 1.0 / math.sqrt(3.0)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("sdaa")
DEVICE_NAME = torch.sdaa.get_device_name(0)
DEVICE_EVIDENCE = f"torch.sdaa.get_device_name(0) = {DEVICE_NAME!r}"

# ---------------------------------------------------------------------------
# Analytical helpers (identical to original run.py)
# ---------------------------------------------------------------------------

def iso_riemann(rhoL, rhoR, cs=CS):
    a = rhoL / rhoR
    f = lambda r: math.log(a / r) - (r - 1.0) / math.sqrt(r)
    lo, hi = 1.0 + 1e-12, a
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return r * rhoR, cs * (r - 1.0) / math.sqrt(r), cs * math.sqrt(r)


def ana_profile(x, t, rhoL, rhoR, rs, us, W, xd, cs=CS):
    s = (x - xd) / max(t, 1e-12)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    m1 = s < -cs
    m2 = (s >= -cs) & (s < us - cs)
    m3 = (s >= us - cs) & (s < W)
    m4 = s >= W
    rho[m1] = rhoL
    u[m1] = 0.0
    rho[m2] = rhoL * np.exp(-(s[m2] + cs) / cs)
    u[m2] = s[m2] + cs
    rho[m3] = rs
    u[m3] = us
    rho[m4] = rhoR
    u[m4] = 0.0
    return rho, u


# ---------------------------------------------------------------------------
# Simulation (all tensors on SDAA)
# ---------------------------------------------------------------------------

def run(nx, ny, rhoL, rhoR, tau, steps, snapshots, device):
    xd = nx // 2
    x = torch.arange(nx, dtype=torch.float32, device=device)
    rho = torch.where(x < xd, torch.full_like(x, rhoL), torch.full_like(x, rhoR))
    rho = rho.unsqueeze(0).expand(ny, nx).contiguous()
    ux = torch.zeros(ny, nx, dtype=torch.float32, device=device)
    uy = torch.zeros(ny, nx, dtype=torch.float32, device=device)
    f = equilibrium(rho, ux, uy)

    profs = {}
    bad = None
    for t in range(steps + 1):
        if t in snapshots:
            rr, uu, _ = macroscopic(f)
            # Transfer to CPU/numpy for storage
            profs[t] = (rr[0].detach().cpu().numpy().copy(),
                        uu[0].detach().cpu().numpy().copy())
        f = stream(f)
        f = collide_bgk(f, tau)
        if t % 50 == 0 and bad is None:
            rr, _, _ = macroscopic(f)
            if torch.isnan(rr).any() or (rr < 0).any():
                bad = t
    rr, _, _ = macroscopic(f)
    if bad is None and (torch.isnan(rr).any() or (rr < 0).any()):
        bad = steps
    return profs, bad


def metrics(profs, rhoL, rhoR, rs, us, W, xd, nx, tlist):
    x = np.arange(nx, dtype=np.float64)
    out = {}
    for t in tlist:
        rho, u = profs[t]
        rhoA, uA = ana_profile(x, t, rhoL, rhoR, rs, us, W, xd)
        lo, hi = int(math.ceil(CS * t)) + 1, int(nx - W * t) - 1
        mask = np.zeros(nx, bool)
        mask[max(lo, 0):min(hi, nx)] = True
        jump = rhoL - rhoR
        L2r = float(np.sqrt(np.mean((rho[mask] - rhoA[mask]) ** 2)) / jump)
        L2u = float(np.sqrt(np.mean((u[mask] - uA[mask]) ** 2)) / max(us, 1e-12))
        # smooth-region L2 (exclude shock transition +/-10 cells)
        wlo, whi = int(xd + 0.5 * W * t), int(xd + 1.5 * W * t)
        g = np.abs(np.gradient(rho))
        g[:wlo] = 0.0
        g[whi:] = 0.0
        xsh = int(np.argmax(g))
        mask2 = mask.copy()
        mask2[max(0, xsh - 10):min(nx, xsh + 10)] = False
        L2r_smooth = float(np.sqrt(np.mean((rho[mask2] - rhoA[mask2]) ** 2)) / jump)
        # local shock oscillation (window +-25)
        win = rho[max(0, xsh - 25):min(nx, xsh + 25)]
        osc = float(np.max(win) - rs) / jump
        # middle-state check
        tail = xd + (us - CS) * t
        mid = (x > tail + 10) & (x < xsh - 10)
        u_mid = float(u[mid].mean()) if mid.any() else float("nan")
        rho_mid = float(rho[mid].mean()) if mid.any() else float("nan")
        out[t] = dict(
            L2_rho=L2r,
            L2_u=L2u,
            L2_rho_smooth=L2r_smooth,
            shock_x=xsh,
            osc_frac=osc,
            rho_mid=rho_mid,
            u_mid=u_mid,
        )
    ts = sorted(out.keys())
    A = np.vstack([np.array(ts, float), np.ones(len(ts))]).T
    coeff, *_ = np.linalg.lstsq(A, [out[t]["shock_x"] for t in ts], rcond=None)
    W_fit = float(coeff[0])
    out["W_fit"] = W_fit
    out["W_ana"] = W
    out["W_rel"] = (W_fit - W) / W
    out["rho_mid_rel"] = (out[ts[-1]]["rho_mid"] - rs) / rs
    out["u_mid_rel"] = (out[ts[-1]]["u_mid"] - us) / us
    out["L2_rho_final"] = out[ts[-1]]["L2_rho"]
    out["L2_u_final"] = out[ts[-1]]["L2_u"]
    out["osc_frac_final"] = out[ts[-1]]["osc_frac"]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.time()
    print(f"[SDAA] Device: {DEVICE_EVIDENCE}")
    print(f"[SDAA] Device count: {torch.sdaa.device_count()}")

    # ---- Sod shock tube cases (same as original) ----
    sod_cases = []
    case_times = {}
    for nx, steps, snaps in [(2000, 400, [100, 200, 300, 400]),
                              (4000, 800, [200, 400, 600, 800])]:
        rhoL, rhoR = 1.0, 0.25
        t0 = time.time()
        profs, bad = run(nx, 4, rhoL, rhoR, 0.8, steps, snaps, DEVICE)
        torch.sdaa.synchronize()
        t1 = time.time()
        case_times[nx] = t1 - t0

        rs, us, W = iso_riemann(rhoL, rhoR)
        m = metrics(profs, rhoL, rhoR, rs, us, W, nx // 2, nx, snaps)
        per_t = {str(t): {k2: v2 for k2, v2 in m[t].items()} for t in snaps}
        sod_cases.append(
            dict(
                nx=nx,
                steps=steps,
                first_bad_step=bad,
                rho_star=rs,
                u_star=us,
                Ma_mid=us / CS,
                per_t=per_t,
                wall_time_s=round(t1 - t0, 3),
                **{k: v for k, v in m.items() if not isinstance(k, int)},
            )
        )
        print(
            f"  nx={nx}: bad={bad} L2r={m['L2_rho_final']:.5f} "
            f"W_fit={m['W_fit']:.4f} W_rel={m['W_rel']*100:.2f}% "
            f"rho_mid_rel={m['rho_mid_rel']*100:.3f}% u_mid_rel={m['u_mid_rel']*100:.3f}% "
            f"osc_frac={m['osc_frac_final']*100:.2f}% "
            f"time={t1-t0:.2f}s"
        )

    # ---- Acoustic supplementary ----
    acou = []
    for eps, steps, snaps in [(0.01, 400, [100, 200, 300, 400]),
                               (0.25, 800, [200, 400, 600, 800])]:
        nx = 4000
        rhoL, rhoR = 1.0 + eps, 1.0 - eps
        t0 = time.time()
        profs, bad = run(nx, 4, rhoL, rhoR, 0.8, steps, snaps, DEVICE)
        torch.sdaa.synchronize()
        t1 = time.time()
        rs, us, W = iso_riemann(rhoL, rhoR)
        m = metrics(profs, rhoL, rhoR, rs, us, W, nx // 2, nx, snaps)
        acou.append(
            dict(
                eps=eps,
                first_bad_step=bad,
                Ma_mid=us / CS,
                W_fit=m["W_fit"],
                W_ana=W,
                W_rel=m["W_rel"],
                u_mid_rel=m["u_mid_rel"],
                L2_rho_final=m["L2_rho_final"],
                wall_time_s=round(t1 - t0, 3),
            )
        )
        print(
            f"  acoustic eps={eps}: bad={bad} W_rel={m['W_rel']*100:.3f}% "
            f"u_mid_rel={m['u_mid_rel']*100:.3f}% L2r={m['L2_rho_final']:.5f} "
            f"time={t1-t0:.2f}s"
        )

    wall_total = time.time() - wall_start

    # ---- Verdict (same gates as original) ----
    gates = {}
    # Gate 1: stability (no NaN/negative density)
    gates["stability_no_nan_neg"] = all(c["first_bad_step"] is None for c in sod_cases)
    # Gate 2: L2_rho <= 3% for both resolutions
    gates["L2_rho_nx2000_le_3pct"] = sod_cases[0]["L2_rho_final"] <= 0.03
    gates["L2_rho_nx4000_le_3pct"] = sod_cases[1]["L2_rho_final"] <= 0.03
    # Gate 3: |W_rel| <= 3%
    gates["W_rel_nx2000_le_3pct"] = abs(sod_cases[0]["W_rel"]) <= 0.03
    gates["W_rel_nx4000_le_3pct"] = abs(sod_cases[1]["W_rel"]) <= 0.03
    # Gate 4: |rho_mid_rel| <= 3%
    gates["rho_mid_rel_nx2000_le_3pct"] = abs(sod_cases[0]["rho_mid_rel"]) <= 0.03
    gates["rho_mid_rel_nx4000_le_3pct"] = abs(sod_cases[1]["rho_mid_rel"]) <= 0.03
    # Gate 5: |u_mid_rel| <= 3%
    gates["u_mid_rel_nx2000_le_3pct"] = abs(sod_cases[0]["u_mid_rel"]) <= 0.03
    gates["u_mid_rel_nx4000_le_3pct"] = abs(sod_cases[1]["u_mid_rel"]) <= 0.03
    # Gate 6: monotone convergence
    gates["monotone_convergence"] = (
        sod_cases[1]["L2_rho_final"] <= sod_cases[0]["L2_rho_final"] * 1.02
    )

    all_pass = all(gates.values())

    # ---- Build output ----
    result = {
        "case": "sod_shock_tube_isothermal_d2q9",
        "device": DEVICE_NAME,
        "device_evidence": DEVICE_EVIDENCE,
        "device_count": torch.sdaa.device_count(),
        "triton_interpret": False,
        "triton_note": "No triton dependency in core path (equilibrium/stream/collide_bgk are pure torch ops)",
        "wall_clock_total_s": round(wall_total, 3),
        "case_wall_times": {str(k): v for k, v in case_times.items()},
        "verified": bool(all_pass),
        "reference": "isothermal Riemann solution (p=rho/3)",
        "tau": 0.8,
        "nu": 0.1,
        "ny": 4,
        "gates": {k: {"pass": bool(v), "gate": k} for k, v in gates.items()},
        "sod_cases": sod_cases,
        "acoustic_supplementary": acou,
        "note": "Reference is the ISOTHERMAL Riemann solution (p=rho/3). "
                "gamma=1.4 Sod solution is NOT applicable to D2Q9.",
    }

    # ---- Write artifacts ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sdaa_sod_shock_tube.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"\n[SDAA] Artifacts written to {out_path}")
    print(f"[SDAA] Wall clock: {wall_total:.2f}s")
    print(f"[SDAA] VERDICT: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for g, v in gates.items():
        print(f"  {g}: {'PASS' if v else 'FAIL'}")
    print("DONE")


if __name__ == "__main__":
    main()
