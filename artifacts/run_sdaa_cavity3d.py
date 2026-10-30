#!/usr/bin/env python3
"""SDAA re-run of cavity_3d_full benchmark — m216 下探补证。

Uses local TensorLBM source and SDAA device. Runs 64^3 and 96^3 grids,
produces artifacts/sdaa_cavity_3d_full_v2.json with full gate evidence.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Use local source
sys.path.insert(0, "/root/tp-agent/work/m216/src")

from tensorlbm.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.lid_driven_cavity import GHIA_RE1000
from tensorlbm.solver3d import collide_trt3d, stream3d

REF_U_MIN_3D_RE1000 = -0.2751
GHIA_RE1000_VORTEX = [0.5313, 0.5625]
THRESHOLD_PCT = 3.0  # frozen threshold


def stationary_pre_bounce3d(f_pre, f, wall):
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(nx, re=1000, u_lid=0.06, steps=100000, device=None,
             out_path=None, resid_interval=5000, min_resid=1e-9):
    nz = ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    nu = (tau - 0.5) / 3.0
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    u0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, u0, u0, u0)
    mass0 = float(rho0.sum().item())

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True
    wall[:, :, -1] = True
    wall[:, 0, :] = True
    wall[0, :, :] = True
    wall[-1, :, :] = True
    interior = ~wall
    interior[:, -1, :] = False

    _, ux_prev, uy_prev, uz_prev = macroscopic3d(f)
    ux_prev = ux_prev.detach().clone()
    uy_prev = uy_prev.detach().clone()
    uz_prev = uz_prev.detach().clone()
    last_resid = None
    t0 = time.time()
    for step in range(1, steps + 1):
        f_pre = f
        f = collide_trt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        f = zou_he_moving_lid_3d(f, u_lid)
        if step % resid_interval == 0:
            _, ux, uy, uz = macroscopic3d(f)
            du = torch.maximum(
                torch.maximum(
                    torch.abs(ux[interior] - ux_prev[interior]),
                    torch.abs(uy[interior] - uy_prev[interior]),
                ),
                torch.abs(uz[interior] - uz_prev[interior]),
            ).max().item()
            last_resid = du
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
            uz_prev = uz_prev.detach().clone()
            elapsed_now = time.time() - t0
            print(f"  step {step:7d} resid={du:.2e} t={elapsed_now:.0f}s", flush=True)
            if du < min_resid:
                break
    elapsed = time.time() - t0

    rho, ux, uy, uz = macroscopic3d(f)
    ux_w = ux.masked_fill(wall, 0.0)
    uy_w = uy.masked_fill(wall, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid
    uz_np = uz.detach().cpu().numpy() / u_lid

    ghia = GHIA_RE1000
    z0 = nz // 2
    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)

    u_cl_mid = ux_np[z0, :, x_mid]
    v_cl_mid = uy_np[z0, y_mid, :]

    span_symmetry = {
        "uz_midplane_max_abs": round(float(np.abs(uz_np[z0]).max()), 6),
        "uz_global_max_abs": round(float(np.abs(uz_np).max()), 6),
    }

    # 2D Ghia comparison
    u_gi = np.interp(ghia["y"], y_pos, u_cl_mid)
    v_gi = np.interp(ghia["x"], x_pos, v_cl_mid)
    dev = np.concatenate([np.abs(u_gi - np.array(ghia["u"])), np.abs(v_gi - np.array(ghia["v"]))])
    max_abs_dev_pct = round(100.0 * float(dev.max()), 4)
    rmse_u = round(float(np.sqrt(np.mean((u_gi - np.array(ghia["u"]))**2))), 5)
    rmse_v = round(float(np.sqrt(np.mean((v_gi - np.array(ghia["v"]))**2))), 5)

    # 3D reference
    u_min_3d = float(u_cl_mid.min())
    u_min_idx = int(np.argmin(u_cl_mid))
    u_min_err_pct = 100.0 * abs(u_min_3d - REF_U_MIN_3D_RE1000) / abs(REF_U_MIN_3D_RE1000)

    # 3D correction
    u_min_2d = -0.38289
    correction_pct = round(100.0 * (u_min_3d - u_min_2d) / abs(u_min_2d), 3)

    # Vortex
    speed2 = ux_np[z0]**2 + uy_np[z0]**2
    speed2[0, :] = speed2[-1, :] = np.inf
    speed2[:, 0] = speed2[:, -1] = np.inf
    xg = np.linspace(0.0, 1.0, nx)
    yg = np.linspace(0.0, 1.0, ny)
    in_region = (xg[None, :] >= 0.3) & (xg[None, :] <= 0.8) & (yg[:, None] >= 0.3) & (yg[:, None] <= 0.8)
    iy1, ix1 = np.unravel_index(np.argmin(np.where(in_region, speed2, np.inf)), speed2.shape)
    vortex = [round(ix1 / (nx - 1), 4), round(iy1 / (ny - 1), 4)]

    finite = bool(torch.isfinite(f).all().item())
    mass_drift = round(float(rho.sum().item() - mass0), 6)

    result = {
        "nx": nx, "nz": nz, "ny": ny,
        "re": re, "u_lid": u_lid,
        "tau": round(tau, 6), "nu": round(nu, 8),
        "steps": steps, "n_steps_run": step,
        "last_resid": last_resid,
        "elapsed_s": round(elapsed, 1),
        "finite": finite,
        "mass_drift": mass_drift,
        "u_min_3d_mid": round(u_min_3d, 5),
        "u_min_y_mid": round(float(y_pos[u_min_idx]), 4),
        "u_min_ref_ku": REF_U_MIN_3D_RE1000,
        "u_min_err_pct_vs_3d_ref": round(u_min_err_pct, 3),
        "u_min_3d_correction_pct": correction_pct,
        "u_mid": round(float(np.interp(0.5, y_pos, u_cl_mid)), 5),
        "v_mid": round(float(np.interp(0.5, x_pos, v_cl_mid)), 5),
        "max_abs_dev_pct_vs2d": max_abs_dev_pct,
        "rmse_u_vs2d": rmse_u,
        "rmse_v_vs2d": rmse_v,
        "primary_vortex": vortex,
        "span_symmetry": span_symmetry,
        "u_centerline_mid": [round(float(v), 6) for v in u_cl_mid],
        "v_centerline_mid": [round(float(v), 6) for v in v_cl_mid],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main():
    device = torch.device("sdaa:0")
    device_name = torch.sdaa.get_device_name(0)
    dev_nodes = sorted([str(p) for p in Path("/dev").glob("tcaicard*")])

    print(f"=== SDAA cavity_3d_full re-run (m216) ===")
    print(f"Device: {device_name}")
    print(f"SDAA device count: {torch.sdaa.device_count()}")
    print(f"/dev/tcaicard* nodes: {dev_nodes}")
    print()

    grids = {}
    for nx in [64, 96]:
        print(f"\n--- Running {nx}^3 ---")
        t_start = time.time()
        r = run_case(nx, steps=100000, device=device,
                     out_path=f"/root/tp-agent/work/m216/artifacts/case_{nx}3_sdaa.json")
        wall_total = time.time() - t_start
        r["wall_clock_total_s"] = round(wall_total, 1)
        grids[str(nx)] = r
        print(f"  => u_min_err = {r['u_min_err_pct_vs_3d_ref']}%  (threshold {THRESHOLD_PCT}%)")
        print(f"  => PASS" if r['u_min_err_pct_vs_3d_ref'] <= THRESHOLD_PCT else f"  => FAIL")

    # Convergence check
    err_64 = grids["64"]["u_min_err_pct_vs_3d_ref"]
    err_96 = grids["96"]["u_min_err_pct_vs_3d_ref"]
    monotone = err_96 < err_64

    # Gate evaluations
    gate_library_entry = True  # verified by import
    gate_direct_obs_64 = err_64 <= THRESHOLD_PCT
    gate_direct_obs_96 = err_96 <= THRESHOLD_PCT
    gate_monotone = monotone
    gate_reference_locked = True  # Ku 1987 pre-registered

    artifact = {
        "case": "cavity_3d_full",
        "mission": "m216",
        "rerun_purpose": "SDAA v2 下探补证 — WITHHELD→PASS/FAIL 判定",
        "device": {
            "torch.sdaa.get_device_name(0)": device_name,
            "torch.sdaa.device_count()": torch.sdaa.device_count(),
            "/dev/tcaicard*_nodes": dev_nodes,
        },
        "config": {
            "lattice": "D3Q19",
            "collision": "TRT (Λ=3/16)",
            "boundary": "V3-3D: pre-streaming 半程反弹(5静止壁) + zou_he_moving_lid_3d(顶盖)",
            "re": 1000,
            "u_lid": 0.06,
            "steps": 100000,
            "reference": "Ku, Hirsh & Taylor (1987) u_min(0.5,0.5) = -0.2751",
            "threshold_pct": THRESHOLD_PCT,
        },
        "grids": grids,
        "convergence": {
            "err_64_pct": err_64,
            "err_96_pct": err_96,
            "monotone_decrease": monotone,
        },
        "gates": {
            "library_entry": {"pass": gate_library_entry, "evidence": "from tensorlbm.boundaries3d/solver3d import OK"},
            "direct_observable_64": {"pass": gate_direct_obs_64, "value_pct": err_64, "threshold_pct": THRESHOLD_PCT},
            "direct_observable_96": {"pass": gate_direct_obs_96, "value_pct": err_96, "threshold_pct": THRESHOLD_PCT},
            "monotone_convergence": {"pass": gate_monotone, "evidence": f"{err_64}% → {err_96}%"},
            "reference_locked_first": {"pass": gate_reference_locked, "evidence": "Ku 1987 pre-registered"},
        },
        "overall_verdict": "PASS" if (gate_direct_obs_64 and gate_direct_obs_96 and gate_monotone) else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = "/root/tp-agent/work/m216/artifacts/sdaa_cavity_3d_full_v2.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(artifact, indent=2))
    print(f"\n=== Artifact written to {out_path} ===")
    print(f"Overall verdict: {artifact['overall_verdict']}")


if __name__ == "__main__":
    main()
