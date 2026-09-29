"""Long-step convergence worker — test if pressure oscillations dampen at 10k+ steps.

Usage:
    PYTHONPATH=src python longstep_worker.py <card_id> <case_name>

Where case_name is one of: suboff_200, kvlcc2_200, kvlcc2_320

Outputs Ct_fric, Ct_pres, Ct_total every 1000 steps.
Reports running-average drag (post-warmup).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area

_FORM_FACTORS = {
    ShipHullType.WIGLEY: 1.15,
    ShipHullType.SERIES60: 1.18,
    ShipHullType.KCS: 1.20,
    ShipHullType.KVLCC2: 1.25,
    ShipHullType.NPL: 1.10,
}


def run_suboff_long(device, nx, ny, nz, re, n_steps, u_in, hull_length, cs_smag,
                    warmup, wall_law="log", use_van_driest=False):
    """Run SUBOFF bare_hull for n_steps with detailed convergence output."""
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

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    fric_vals, pres_vals = [], []
    convergence_log = []  # snapshot every 1000 steps
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=use_van_driest)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 1000 == 0:
            n_samp = max(len(fric_vals), 1)
            cf = sum(fric_vals) / n_samp / dyn_p_S if fric_vals else 0.0
            cp = sum(pres_vals) / n_samp / dyn_p_S if pres_vals else 0.0
            ct = cf + cp
            elapsed = time.time() - t0
            snapshot = {
                "step": step,
                "Ct_fric": round(cf, 8),
                "Ct_pres": round(cp, 8),
                "Ct_total": round(ct, 8),
                "n_samples": n_samp,
                "elapsed_s": round(elapsed, 1),
            }
            convergence_log.append(snapshot)
            print(f"[SUBOFF {nx}³ sdaa:{device.index}] step {step}: "
                  f"Ct_fric={cf:.6f} Ct_pres={cp:.6f} Ct_total={ct:.6f} "
                  f"(samples={n_samp}, {elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[SUBOFF {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_samp = max(len(fric_vals), 1)
    cf_final = sum(fric_vals) / n_samp / dyn_p_S if fric_vals else 0.0
    cp_final = sum(pres_vals) / n_samp / dyn_p_S if pres_vals else 0.0
    ct_final = cf_final + cp_final

    # Compute running stats for last 2000 samples to check convergence
    if len(fric_vals) > 2000:
        cf_tail = sum(fric_vals[-2000:]) / 2000 / dyn_p_S
        cp_tail = sum(pres_vals[-2000:]) / 2000 / dyn_p_S
        ct_tail = cf_tail + cp_tail
        # Also check trend: compare first half vs second half of post-warmup
        mid = len(fric_vals) // 2
        cf_first = sum(fric_vals[:mid]) / mid / dyn_p_S
        cp_first = sum(pres_vals[:mid]) / mid / dyn_p_S
        cf_second = sum(fric_vals[mid:]) / (len(fric_vals) - mid) / dyn_p_S
        cp_second = sum(pres_vals[mid:]) / (len(pres_vals) - mid) / dyn_p_S
        trend = {
            "Ct_fric_last2k": round(cf_tail, 8),
            "Ct_pres_last2k": round(cp_tail, 8),
            "Ct_total_last2k": round(ct_tail, 8),
            "Ct_fric_first_half": round(cf_first, 8),
            "Ct_pres_first_half": round(cp_first, 8),
            "Ct_fric_second_half": round(cf_second, 8),
            "Ct_pres_second_half": round(cp_second, 8),
        }
    else:
        trend = {}

    return {
        "case": f"suboff_bare_hull_{nx}³_long",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup": warmup,
        "n_samples": n_samp,
        "Ct_fric": round(cf_final, 8),
        "Ct_pres": round(cp_final, 8),
        "Ct_total": round(ct_final, 8),
        "Ct_ref_ITTC": round(cf_ittc, 8),
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": round(elapsed, 1),
        "convergence_log": convergence_log,
        "convergence_trend": trend,
        "device": str(device),
    }


def run_ship_long(device, hull_str, nx, ny, nz, re, n_steps, u_in, hull_length,
                  cs_smag, warmup, wall_law="log", use_van_driest=False):
    """Run ship hull for n_steps with detailed convergence output."""
    hull = ShipHullType(hull_str)
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    ff = _FORM_FACTORS[hull]

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
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * ff

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    fric_vals, pres_vals = [], []
    convergence_log = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=use_van_driest)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 1000 == 0:
            n_samp = max(len(fric_vals), 1)
            cf = sum(fric_vals) / n_samp / dyn_p_S if fric_vals else 0.0
            cp = sum(pres_vals) / n_samp / dyn_p_S if pres_vals else 0.0
            ct = cf + cp
            elapsed = time.time() - t0
            err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float('inf')
            snapshot = {
                "step": step,
                "Ct_fric": round(cf, 8),
                "Ct_pres": round(cp, 8),
                "Ct_total": round(ct, 8),
                "error_pct": round(err_pct, 2),
                "n_samples": n_samp,
                "elapsed_s": round(elapsed, 1),
            }
            convergence_log.append(snapshot)
            print(f"[{hull.value} {nx}³ sdaa:{device.index}] step {step}: "
                  f"Ct_fric={cf:.6f} Ct_pres={cp:.6f} Ct_total={ct:.6f} "
                  f"err={err_pct:.1f}% (samples={n_samp}, {elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[{hull.value}] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_samp = max(len(fric_vals), 1)
    cf_final = sum(fric_vals) / n_samp / dyn_p_S if fric_vals else 0.0
    cp_final = sum(pres_vals) / n_samp / dyn_p_S if pres_vals else 0.0
    ct_final = cf_final + cp_final
    err_pct = abs(ct_final - ct_ref) / ct_ref * 100 if ct_ref > 0 else float('inf')

    # Convergence trend
    if len(fric_vals) > 2000:
        cf_tail = sum(fric_vals[-2000:]) / 2000 / dyn_p_S
        cp_tail = sum(pres_vals[-2000:]) / 2000 / dyn_p_S
        ct_tail = cf_tail + cp_tail
        mid = len(fric_vals) // 2
        cf_first = sum(fric_vals[:mid]) / mid / dyn_p_S
        cp_first = sum(pres_vals[:mid]) / mid / dyn_p_S
        cf_second = sum(fric_vals[mid:]) / (len(fric_vals) - mid) / dyn_p_S
        cp_second = sum(pres_vals[mid:]) / (len(pres_vals) - mid) / dyn_p_S
        trend = {
            "Ct_fric_last2k": round(cf_tail, 8),
            "Ct_pres_last2k": round(cp_tail, 8),
            "Ct_total_last2k": round(ct_tail, 8),
            "Ct_fric_first_half": round(cf_first, 8),
            "Ct_pres_first_half": round(cp_first, 8),
            "Ct_fric_second_half": round(cf_second, 8),
            "Ct_pres_second_half": round(cp_second, 8),
        }
    else:
        trend = {}

    return {
        "case": f"{hull.value}_ship_{nx}³_long",
        "hull_type": hull.value,
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup": warmup,
        "n_samples": n_samp,
        "Ct_fric": round(cf_final, 8),
        "Ct_pres": round(cp_final, 8),
        "Ct_total": round(ct_final, 8),
        "Ct_reference": round(ct_ref, 8),
        "error_pct": round(err_pct, 2),
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "Cf_ITTC": round(cf_ittc, 8), "form_factor": ff,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": round(elapsed, 1),
        "convergence_log": convergence_log,
        "convergence_trend": trend,
        "device": str(device),
    }


CASE_CONFIGS = {
    "suboff_200": {
        "fn": run_suboff_long,
        "kwargs": dict(nx=200, ny=80, nz=80, re=2e6, n_steps=10000,
                       u_in=0.06, hull_length=100.0, cs_smag=0.05, warmup=3333),
    },
    "kvlcc2_200": {
        "fn": run_ship_long,
        "kwargs": dict(hull_str="kvlcc2", nx=200, ny=60, nz=60, re=2e6, n_steps=10000,
                       u_in=0.06, hull_length=80.0, cs_smag=0.05, warmup=3333),
    },
    "kvlcc2_320": {
        "fn": run_ship_long,
        "kwargs": dict(hull_str="kvlcc2", nx=320, ny=96, nz=96, re=2e6, n_steps=5000,
                       u_in=0.06, hull_length=80.0, cs_smag=0.05, warmup=1666),
    },
}


def main():
    if len(sys.argv) < 3:
        print("Usage: longstep_worker.py <card_id> <case_name> [n_steps_override]",
              file=sys.stderr)
        sys.exit(1)

    card_id = int(sys.argv[1])
    case_name = sys.argv[2]

    if case_name not in CASE_CONFIGS:
        print(f"Unknown case: {case_name}. Choices: {list(CASE_CONFIGS.keys())}",
              file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"sdaa:{card_id}")
    torch.sdaa.set_device(device)

    cfg = CASE_CONFIGS[case_name]
    fn = cfg["fn"]
    kwargs = dict(cfg["kwargs"])

    # Allow n_steps override
    if len(sys.argv) >= 4:
        kwargs["n_steps"] = int(sys.argv[3])
        # Recalculate warmup
        kwargs["warmup"] = max(0, kwargs["n_steps"] // 3)

    tag = f"[{case_name} sdaa:{card_id}]"
    print(f"{tag} Starting {kwargs['n_steps']} steps (warmup={kwargs['warmup']})...",
          flush=True)
    t_start = time.time()

    try:
        result = fn(device=device, wall_law="log", use_van_driest=False, **kwargs)
        result["_status"] = "OK"
    except Exception as e:
        result = {
            "case": case_name,
            "device": f"sdaa:{card_id}",
            "_status": "EXCEPTION",
            "_error": str(e),
        }
        import traceback
        result["_traceback"] = traceback.format_exc()
        print(f"{tag} EXCEPTION: {e}", flush=True)

    elapsed = time.time() - t_start
    result["_launcher_elapsed_s"] = round(elapsed, 1)
    print(f"{tag} Done ({elapsed:.0f}s)", flush=True)

    # Output JSON
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
