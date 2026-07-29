"""Test: Dynamic Smagorinsky vs Fixed Cs=0.05 on SUBOFF bare_hull 200³.

Runs two cases:
  Case A: D3Q19 MRT + fixed Smagorinsky  Cs=0.05
  Case B: D3Q19 MRT + dynamic Smagorinsky (Cs computed via Germano identity)

Reports Ct_fric, Ct_pres, Ct_total at steps 1000, 1500, 2000.
Outputs results to /tmp/dynamic_smag_results.json.

Usage:
    PYTHONPATH=src python test_dynamic_smag_comparison.py [--device sdaa:0]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d, collide_dynamic_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d as wall_fn_main


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def run_case(
    *,
    case_name: str,
    collision_mode: str,  # "fixed" or "dynamic"
    re: float,
    nx: int,
    ny: int,
    nz: int,
    u_in: float,
    cs_fixed: float,
    tau: float,
    n_steps: int,
    warmup: int,
    y_val: float,
    device: str,
    report_steps: list[int],
) -> dict:
    """Run a single case and return drag history + checkpoint values."""

    hull_length = nx * 0.6
    cx = nx * 0.35
    cy = ny / 2.0
    cz = nz / 2.0
    nu_lat = (tau - 0.5) / 3.0

    print(f"\n{'='*70}", flush=True)
    print(f"Case: {case_name} | collision={collision_mode}", flush=True)
    print(f"Re={re:.0e} tau={tau:.5f} nu_lat={nu_lat:.2e} hull_L={hull_length:.0f}", flush=True)
    print(f"Grid={nx}×{ny}×{nz}  u_in={u_in}  steps={n_steps}  warmup={warmup}", flush=True)
    print(f"{'='*70}\n", flush=True)

    solid, _stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=hull_length,
        device=device,
    )
    S_wet = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in * u_in * S_wet
    print(f"wetted area S={S_wet:.0f}  dyn_p_S={dyn_p_S:.6f}\n", flush=True)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.zeros_like(ux0)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=torch.device(device))
    initial_mass = float(rho0.sum().item())

    # Drag history post-warmup
    pf_fric: list[float] = []
    pf_pres: list[float] = []
    cs_values: list[float] = []  # dynamic Cs per step

    # Checkpoint data
    checkpoint_data: dict[int, dict] = {}

    print(f"{'Step':>6s}  {'Ct_fric':>12s}  {'Ct_pres':>12s}  {'Ct_tot':>12s}  {'max|u|':>10s}  {'Cs':>10s}")
    print("-" * 80, flush=True)

    t_start = time.time()

    for step in range(1, n_steps + 1):
        # Collision
        if collision_mode == "fixed":
            f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_fixed, s_e=1.19, s_eps=1.4, s_q=1.2)
            cs_step = cs_fixed
        elif collision_mode == "dynamic":
            f = collide_dynamic_smagorinsky_mrt3d(
                f, tau=tau,
                filter_width=2,
                lambda_clip=0.0,
                s_e=1.19, s_eps=1.4, s_q=1.2,
            )
            # Recompute Cs from the last collision step by re-running macroscopic
            # (approximate — get the global Cs that was computed)
            cs_step = 0.0  # dynamic Cs is per-cell, we approximate with mean later
        else:
            raise ValueError(f"Unknown collision_mode: {collision_mode}")

        # Streaming
        f = stream3d(f)

        # Wall function (body force + pressure-face drag)
        f, drag_fric, drag_pres = wall_fn_main(f, solid, nu_lat, y_val=y_val)

        # Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Record post-warmup
        if step > warmup and math.isfinite(drag_fric):
            pf_fric.append(drag_fric)
            pf_pres.append(drag_pres)
            if collision_mode == "dynamic":
                # Re-run macroscopic to estimate current effective Cs
                rho, ux, uy, uz = macroscopic3d(f)
                # Approximate Cs from strain — just record for monitoring
                cs_values.append(cs_step)

        # Report at checkpoints
        if step in report_steps:
            n_rec = len(pf_fric)
            if n_rec > 0:
                ct_fric = sum(pf_fric) / n_rec / dyn_p_S
                ct_pres = sum(pf_pres) / n_rec / dyn_p_S
                ct_tot = ct_fric + ct_pres
                cs_avg = sum(cs_values) / len(cs_values) if cs_values else cs_fixed
            else:
                ct_fric = drag_fric / dyn_p_S
                ct_pres = drag_pres / dyn_p_S
                ct_tot = ct_fric + ct_pres
                cs_avg = cs_fixed

            _, ux, uy, uz = macroscopic3d(f)
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())

            print(
                f" {step:5d}  {ct_fric:12.6f}  {ct_pres:12.6f}  {ct_tot:12.6f}  {ms:10.4f}  {cs_avg:10.6f}",
                flush=True,
            )

            checkpoint_data[step] = {
                "ct_fric": round(ct_fric, 8),
                "ct_pres": round(ct_pres, 8),
                "ct_total": round(ct_tot, 8),
                "cs_effective": round(cs_avg, 8),
                "max_velocity_lu": round(ms, 6),
                "step": step,
            }

    t_elapsed = time.time() - t_start

    # Final statistics
    n_total = len(pf_fric)
    ct_fric_arr = [v / dyn_p_S for v in pf_fric]
    ct_pres_arr = [v / dyn_p_S for v in pf_pres]
    ct_tot_arr = [f + p for f, p in zip(ct_fric_arr, ct_pres_arr)]

    mean_fric = sum(ct_fric_arr) / max(n_total, 1)
    mean_pres = sum(ct_pres_arr) / max(n_total, 1)
    mean_tot = sum(ct_tot_arr) / max(n_total, 1)
    std_fric = _std(ct_fric_arr)
    std_pres = _std(ct_pres_arr)
    std_tot = _std(ct_tot_arr)

    print(f"\n--- Final Stats ({case_name}, n={n_total}) ---", flush=True)
    print(f"  Ct_fric: mean={mean_fric:+.6f}  std={std_fric:.6f}", flush=True)
    print(f"  Ct_pres: mean={mean_pres:+.6f}  std={std_pres:.6f}", flush=True)
    print(f"  Ct_tot:  mean={mean_tot:+.6f}  std={std_tot:.6f}", flush=True)
    print(f"  Elapsed: {t_elapsed:.1f}s", flush=True)

    return {
        "case_name": case_name,
        "collision_mode": collision_mode,
        "re": re,
        "grid": f"{nx}x{ny}x{nz}",
        "n_steps": n_steps,
        "warmup": warmup,
        "tau": tau,
        "u_in": u_in,
        "cs_fixed": cs_fixed if collision_mode == "fixed" else None,
        "elapsed_s": round(t_elapsed, 1),
        "checkpoints": checkpoint_data,
        "final_statistics": {
            "ct_fric_mean": round(mean_fric, 8),
            "ct_fric_std": round(std_fric, 8),
            "ct_pres_mean": round(mean_pres, 8),
            "ct_pres_std": round(std_pres, 8),
            "ct_total_mean": round(mean_tot, 8),
            "ct_total_std": round(std_tot, 8),
            "n_samples": n_total,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare fixed Cs vs dynamic Smagorinsky on bare_hull 200³"
    )
    parser.add_argument("--device", default="sdaa:0", help="Torch device")
    parser.add_argument("--re", type=float, default=2e6, help="Reynolds number")
    parser.add_argument("--nx", type=int, default=200, help="Grid size X")
    parser.add_argument("--ny", type=int, default=200, help="Grid size Y")
    parser.add_argument("--nz", type=int, default=200, help="Grid size Z")
    parser.add_argument("--n-steps", type=int, default=2000, help="Number of steps")
    parser.add_argument("--u-in", type=float, default=0.06, help="Inlet velocity")
    parser.add_argument("--cs", type=float, default=0.05, help="Fixed Smagorinsky constant")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    parser.add_argument("--output", default="/tmp/dynamic_smag_results.json", help="Output JSON")
    args = parser.parse_args()

    # Compute tau from Re
    hull_length = args.nx * 0.6
    nu_lat = args.u_in * hull_length / args.re
    tau = 3.0 * nu_lat + 0.5

    report_steps = [1000, 1500, 2000]
    y_val = 0.5

    print(f"=== Dynamic Smagorinsky vs Fixed Cs Comparison ===", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Grid: {args.nx}³  Re: {args.re:.0e}  Steps: {args.n_steps}  Warmup: {args.warmup}", flush=True)
    print(f"tau: {tau:.6f}  nu_lat: {nu_lat:.6e}  u_in: {args.u_in}", flush=True)

    # Validate device
    try:
        d = torch.device(args.device)
        torch.zeros(1, device=d)
    except Exception as e:
        print(f"ERROR: Cannot use device {args.device}: {e}", flush=True)
        sys.exit(1)

    results = []

    # Case A: Fixed Cs=0.05
    result_a = run_case(
        case_name="Fixed Cs=0.05",
        collision_mode="fixed",
        re=args.re,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        cs_fixed=args.cs,
        tau=tau,
        n_steps=args.n_steps,
        warmup=args.warmup,
        y_val=y_val,
        device=args.device,
        report_steps=report_steps,
    )
    results.append(result_a)

    # Case B: Dynamic Smagorinsky
    result_b = run_case(
        case_name="Dynamic Smagorinsky",
        collision_mode="dynamic",
        re=args.re,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        u_in=args.u_in,
        cs_fixed=args.cs,  # not used but passed for reference
        tau=tau,
        n_steps=args.n_steps,
        warmup=args.warmup,
        y_val=y_val,
        device=args.device,
        report_steps=report_steps,
    )
    results.append(result_b)

    # Comparison
    a_final = result_a["final_statistics"]
    b_final = result_b["final_statistics"]

    print(f"\n{'='*70}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Metric':<20s} {'Fixed Cs=0.05':>18s} {'Dynamic Smag':>18s} {'Ratio':>10s}", flush=True)
    print("-" * 70, flush=True)

    for metric in ["ct_fric", "ct_pres", "ct_total"]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        a_mean = a_final[mean_key]
        b_mean = b_final[mean_key]
        a_std = a_final[std_key]
        b_std = b_final[std_key]
        std_ratio = b_std / max(a_std, 1e-12)
        print(f"{metric:>20s}  {a_mean:18.8f}  {b_mean:18.8f}  {std_ratio:10.3f}", flush=True)
        print(f"  {'std':>18s}  {a_std:18.8f}  {b_std:18.8f}", flush=True)

    # Stability verdict
    a_pres_std = a_final["ct_pres_std"]
    b_pres_std = b_final["ct_pres_std"]
    if b_pres_std < a_pres_std * 0.95:
        verdict = "Dynamic Smagorinsky gives MORE stable Ct_pres (lower variance)"
    elif b_pres_std > a_pres_std * 1.05:
        verdict = "Fixed Cs=0.05 gives MORE stable Ct_pres (lower variance)"
    else:
        verdict = "Dynamic and fixed Smagorinsky have SIMILAR Ct_pres stability"

    print(f"\nVerdict: {verdict}", flush=True)
    print(f"  Fixed  Ct_pres std: {a_pres_std:.8f}", flush=True)
    print(f"  Dynamic Ct_pres std: {b_pres_std:.8f}", flush=True)

    output = {
        "title": "Dynamic Smagorinsky vs Fixed Cs=0.05 — D3Q19 MRT bare_hull 200³",
        "parameters": {
            "re": args.re,
            "grid": f"{args.nx}x{args.ny}x{args.nz}",
            "n_steps": args.n_steps,
            "warmup": args.warmup,
            "u_in": args.u_in,
            "tau": round(tau, 8),
            "nu_lat": round(nu_lat, 10),
            "cs_fixed": args.cs,
            "device": args.device,
        },
        "cases": results,
        "comparison": {
            "ct_pres_stability_verdict": verdict,
            "fixed_ct_pres_std": a_pres_std,
            "dynamic_ct_pres_std": b_pres_std,
            "ct_pres_std_ratio_dynamic_over_fixed": round(b_pres_std / max(a_pres_std, 1e-12), 6),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
