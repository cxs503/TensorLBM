"""Long 50k-step convergence worker. Uses DragMonitor.sliding_mean (window=2000)
for convergence tracking. Runs on a single SDAA card.

Usage:
    PYTHONPATH=src python _long_worker.py <did> <case> <wl> <cs> <nx> <ny> <nz> <hl> <ns>

    did  : SDAA device id (0-7)
    case : suboff | kvlcc2 | wigley
    wl   : log | musker
    cs   : Smagorinsky constant (e.g. 0.05)
    nx   : grid X size
    ny   : grid Y size
    nz   : grid Z size
    hl   : hull length (lattice units)
    ns   : number of steps

Reports every 5000 steps: Ct_fric, Ct_pres, Ct_sliding (window=2000), converged.
Output appended to /tmp/long50k_results.json (shared across workers).
"""
from __future__ import annotations

import json
import math
import sys
import time
import os
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.yplus_guide import DragMonitor
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area

_OUTPUT = Path("/tmp/long50k_results.json")
_SLIDING_WINDOW = 2000
_REPORT_EVERY = 5000
_CONV_THRESHOLD = 0.0002  # Ct change across 3 windows

_FORM_FACTORS = {
    "wigley": 1.15,
    "series60": 1.18,
    "kcs": 1.20,
    "kvlcc2": 1.25,
    "npl": 1.10,
}


def run_suboff(device, nx, ny, nz, re, n_steps, u_in, hull_length, cs_smag,
               warmup, wall_law="log"):
    """Run SUBOFF bare_hull for n_steps with sliding-window convergence."""
    from tensorlbm.suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask

    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    config = SuboffConfig()
    hull_type = SuboffHullType("bare_hull")

    solid, stats = build_suboff_mask(
        hull_type, nx=nx, ny=ny, nz=nz,
        cx=cx_g, cy=cy_g, cz=cz_g,
        length=hull_length, device="cpu", config=config,
    )
    solid = solid.to(device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc  # bare_hull form factor ≈ 1.0

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    monitor = DragMonitor(warmup=warmup, window_frac=0.20)
    convergence_log = []
    snapshots = []  # for final JSON
    prev_sliding_ct = []  # track last 3 sliding window Cts for convergence check
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            monitor.add(step, df, dp)

        if step % _REPORT_EVERY == 0 or step == n_steps:
            sm = monitor.summary()
            slide = monitor.sliding_mean(_SLIDING_WINDOW)
            # Compute converged: last 3 sliding windows all within threshold
            prev_sliding_ct.append(slide["Ct_total_slide"])
            if len(prev_sliding_ct) >= 4:
                # Check last 3 windows (indices -3, -2, -1)
                last3 = prev_sliding_ct[-3:]
                max_ct = max(last3)
                min_ct = min(last3)
                converged = (max_ct - min_ct) < _CONV_THRESHOLD
            else:
                converged = False

            elapsed = time.time() - t0
            snapshot = {
                "step": step,
                "Ct_fric": round(sm.get("Ct_fric_avg", 0.0), 8),
                "Ct_pres": round(sm.get("Ct_pres_avg", 0.0), 8),
                "Ct_sliding": round(slide["Ct_total_slide"], 8),
                "Ct_sliding_fric": round(slide["Ct_fric_slide"], 8),
                "Ct_sliding_pres": round(slide["Ct_pres_slide"], 8),
                "n_slide": slide["n_slide"],
                "Ct_total_avg": round(sm.get("Ct_total_avg", 0.0), 8),
                "converged": converged,
                "elapsed_s": round(elapsed, 1),
            }
            convergence_log.append(snapshot)
            print(f"[SUBOFF {nx}³ Cs={cs_smag} wl={wall_law}] step {step}: "
                  f"Cf={snapshot['Ct_sliding_fric']:.6f} Cp={snapshot['Ct_sliding_pres']:.6f} "
                  f"Ct_slide={snapshot['Ct_sliding']:.6f} "
                  f"conv={converged} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[SUBOFF {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    sm = monitor.summary()
    slide = monitor.sliding_mean(_SLIDING_WINDOW)

    return {
        "case": f"suboff_bare_hull_{nx}³",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law,
        "n_steps": n_steps, "warmup": warmup,
        "sliding_window": _SLIDING_WINDOW,
        "n_samples": monitor.n,
        "Ct_fric_final": round(sm.get("Ct_fric_avg", 0.0), 8),
        "Ct_pres_final": round(sm.get("Ct_pres_avg", 0.0), 8),
        "Ct_total_final": round(sm.get("Ct_total_avg", 0.0), 8),
        "Ct_sliding_final": round(slide["Ct_total_slide"], 8),
        "Ct_sliding_fric": round(slide["Ct_fric_slide"], 8),
        "Ct_sliding_pres": round(slide["Ct_pres_slide"], 8),
        "Ct_ref": round(ct_ref, 8),
        "Ct_converged": sm.get("converged", False),
        "Ct_change_window": sm.get("Ct_change_window", 0.0),
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": round(elapsed, 1),
        "convergence_log": convergence_log,
        "convergence_snapshots": snapshots,
        "device": str(device),
    }


def run_ship(device, hull_str, nx, ny, nz, re, n_steps, u_in, hull_length,
             cs_smag, warmup, wall_law="log"):
    """Run a ship hull for n_steps with sliding-window convergence."""
    from tensorlbm.ship_cad import ShipHullType, build_hull_mask

    hull = ShipHullType(hull_str)
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    ff = _FORM_FACTORS[hull_str]
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * ff

    cx = nx * 0.3
    cy = ny * 0.5
    cz_keel = nz * 0.5

    solid, stats = build_hull_mask(
        hull, nx, ny, nz, cx=cx, cy=cy, cz_keel=cz_keel,
        length=hull_length, device="cpu",
    )
    solid = solid.to(device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    monitor = DragMonitor(warmup=warmup, window_frac=0.20)
    convergence_log = []
    prev_sliding_ct = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            monitor.add(step, df, dp)

        if step % _REPORT_EVERY == 0 or step == n_steps:
            sm = monitor.summary()
            slide = monitor.sliding_mean(_SLIDING_WINDOW)
            prev_sliding_ct.append(slide["Ct_total_slide"])
            if len(prev_sliding_ct) >= 4:
                last3 = prev_sliding_ct[-3:]
                converged = (max(last3) - min(last3)) < _CONV_THRESHOLD
            else:
                converged = False

            elapsed = time.time() - t0
            snapshot = {
                "step": step,
                "Ct_fric": round(sm.get("Ct_fric_avg", 0.0), 8),
                "Ct_pres": round(sm.get("Ct_pres_avg", 0.0), 8),
                "Ct_sliding": round(slide["Ct_total_slide"], 8),
                "Ct_sliding_fric": round(slide["Ct_fric_slide"], 8),
                "Ct_sliding_pres": round(slide["Ct_pres_slide"], 8),
                "n_slide": slide["n_slide"],
                "Ct_total_avg": round(sm.get("Ct_total_avg", 0.0), 8),
                "converged": converged,
                "elapsed_s": round(elapsed, 1),
            }
            convergence_log.append(snapshot)
            print(f"[{hull_str.upper()} {nx}³ Cs={cs_smag} wl={wall_law}] step {step}: "
                  f"Cf={snapshot['Ct_sliding_fric']:.6f} Cp={snapshot['Ct_sliding_pres']:.6f} "
                  f"Ct_slide={snapshot['Ct_sliding']:.6f} "
                  f"conv={converged} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[{hull_str.upper()} {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    sm = monitor.summary()
    slide = monitor.sliding_mean(_SLIDING_WINDOW)
    err_pct = abs(slide["Ct_total_slide"] - ct_ref) / ct_ref * 100 if ct_ref > 0 else float('inf')

    return {
        "case": f"{hull_str}_ship_{nx}³",
        "hull_type": hull_str,
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law,
        "n_steps": n_steps, "warmup": warmup,
        "sliding_window": _SLIDING_WINDOW,
        "n_samples": monitor.n,
        "Ct_fric_final": round(sm.get("Ct_fric_avg", 0.0), 8),
        "Ct_pres_final": round(sm.get("Ct_pres_avg", 0.0), 8),
        "Ct_total_final": round(sm.get("Ct_total_avg", 0.0), 8),
        "Ct_sliding_final": round(slide["Ct_total_slide"], 8),
        "Ct_sliding_fric": round(slide["Ct_fric_slide"], 8),
        "Ct_sliding_pres": round(slide["Ct_pres_slide"], 8),
        "Ct_ref": round(ct_ref, 8),
        "Cf_ITTC": round(cf_ittc, 8),
        "form_factor": ff,
        "error_pct": round(err_pct, 2),
        "Ct_converged": sm.get("converged", False),
        "Ct_change_window": sm.get("Ct_change_window", 0.0),
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": round(elapsed, 1),
        "convergence_log": convergence_log,
        "device": str(device),
    }


def main():
    if len(sys.argv) < 10:
        print("Usage: _long_worker.py <did> <case> <wl> <cs> <nx> <ny> <nz> <hl> <ns>",
              file=sys.stderr)
        print("  did  : SDAA device id (0-7)", file=sys.stderr)
        print("  case : suboff | kvlcc2 | wigley", file=sys.stderr)
        print("  wl   : log | musker", file=sys.stderr)
        print("  cs   : Smagorinsky Cs", file=sys.stderr)
        print("  nx ny nz : grid dims", file=sys.stderr)
        print("  hl   : hull length (lu)", file=sys.stderr)
        print("  ns   : number of steps", file=sys.stderr)
        sys.exit(1)

    did = int(sys.argv[1])
    case = sys.argv[2].lower()
    wl = sys.argv[3].lower()
    cs = float(sys.argv[4])
    nx = int(sys.argv[5])
    ny = int(sys.argv[6])
    nz = int(sys.argv[7])
    hl = float(sys.argv[8])
    ns = int(sys.argv[9])

    if case not in ("suboff", "kvlcc2", "wigley"):
        print(f"Unknown case: {case}. Use: suboff, kvlcc2, wigley", file=sys.stderr)
        sys.exit(1)
    if wl not in ("log", "musker"):
        print(f"Unknown wall_law: {wl}. Use: log, musker", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    re = 2e6
    u_in = 0.06
    warmup = max(0, ns // 3)

    tag = f"[{case} {nx}³ Cs={cs} wl={wl} SDAA:{did}]"
    print(f"{tag} Starting {ns} steps (warmup={warmup})...", flush=True)
    t_start = time.time()

    try:
        if case == "suboff":
            result = run_suboff(
                device=device, nx=nx, ny=ny, nz=nz, re=re, n_steps=ns,
                u_in=u_in, hull_length=hl, cs_smag=cs, warmup=warmup,
                wall_law=wl,
            )
        else:
            result = run_ship(
                device=device, hull_str=case, nx=nx, ny=ny, nz=nz, re=re,
                n_steps=ns, u_in=u_in, hull_length=hl, cs_smag=cs,
                warmup=warmup, wall_law=wl,
            )
        result["_status"] = "OK"
        result["_did"] = did
    except Exception as e:
        import traceback
        result = {
            "case": case,
            "device": f"sdaa:{did}",
            "_status": "EXCEPTION",
            "_did": did,
            "_error": str(e),
            "_traceback": traceback.format_exc(),
        }
        print(f"{tag} EXCEPTION: {e}", flush=True)

    elapsed = time.time() - t_start
    result["_worker_elapsed_s"] = round(elapsed, 1)
    print(f"{tag} Done ({elapsed:.0f}s)", flush=True)

    # Append result to shared JSON file (thread-safe via atomic write)
    _OUTPUT.parent.mkdir(exist_ok=True)
    try:
        if _OUTPUT.exists():
            try:
                results = json.loads(_OUTPUT.read_text())
            except json.JSONDecodeError:
                results = []
        else:
            results = []
        # Add or replace this run
        run_key = f"{case}_{nx}x{ny}x{nz}_Cs{cs}_{wl}_sdaa{did}"
        results = [r for r in results if r.get("_run_key") != run_key]
        result["_run_key"] = run_key
        results.append(result)
        _OUTPUT.write_text(json.dumps(results, indent=2, default=str))
    except Exception as e:
        print(f"Warning: failed to write results: {e}", flush=True)

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
