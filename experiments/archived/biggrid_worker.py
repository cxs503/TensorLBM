"""Big-grid Musker vs log-law worker. Tests wall-law stability on 256³-384³ grids.

Usage: PYTHONPATH=src python biggrid_worker.py <card_id> <case_name> <wall_law>

Cases: suboff_256, suboff_320, suboff_384, kvlcc2_256
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch


def ittc_cf(re: float) -> float:
    return 0.075 / (math.log10(re) - 2.0) ** 2


def run_suboff_bare_hull(device, nx, ny, nz, re, n_steps, u_in, hull_length, cs_smag, wall_law):
    from tensorlbm.suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask
    from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.boundaries3d import far_field_bc_3d

    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    config = SuboffConfig()
    hull_type = SuboffHullType("bare_hull")
    solid, stats = build_suboff_mask(hull_type, nx=nx, ny=ny, nz=nz,
                                     cx=cx_g, cy=cy_g, cz=cz_g, length=hull_length,
                                     device="cpu", config=config)
    solid = solid.to(device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * 1.0  # bare_hull form factor ≈ 1.0

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    fric_vals, pres_vals, yplus_vals = [], [], []
    t0 = time.time()
    warmup = max(0, n_steps // 3)
    window_size = 500
    nan_at_step = -1

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=False)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
            win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
            n_s = max(len(win_fric), 1)
            cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
            cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
            ct = cf + cp
            err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0
            elapsed = time.time() - t0
            # Estimate y+ at near-wall cells
            from tensorlbm.d3q19 import macroscopic3d
            rho, ux, uy, uz = macroscopic3d(f)
            u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
            ut_est = torch.sqrt(nu_lat * u_mag / 0.5).clamp(min=1e-12)
            yp_est = 0.5 * ut_est / nu_lat
            near = torch.zeros_like(solid)
            for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
                near |= torch.roll(solid, sgn, dims=ax) & (~solid)
            yp_mean = float(yp_est[near].mean().item()) if near.any() else 0.0
            print(f"[SUBOFF {nx}³ {wall_law}] step {step}: Cf={cf:.5f} Cp={cp:.5f} Ct={ct:.5f} "
                  f"err={err:.1f}% y+_mean={yp_mean:.1f} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            nan_at_step = step
            print(f"[SUBOFF {nx}³ {wall_law}] NaN/Inf at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
    win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
    n_s = max(len(win_fric), 1)
    cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
    cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
    ct = cf + cp
    err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0
    finite = bool(torch.isfinite(f).all().item())

    return {
        "case": f"suboff_bare_hull_{nx}³_{wall_law}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law,
        "n_steps": n_steps, "warmup_start": warmup, "sliding_window": window_size,
        "n_samples_total": len(fric_vals), "n_samples_window": n_s,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_ref_ITTCx1k": ct_ref, "error_pct": err,
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": finite, "nan_at_step": nan_at_step,
        "wall_time_s": elapsed, "device": str(device),
    }


def run_kvlcc2(device, nx, ny, nz, re, n_steps, u_in, hull_length, cs_smag, wall_law):
    from tensorlbm.ship_cad import ShipHullType, build_hull_mask
    from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.boundaries3d import far_field_bc_3d

    hull = ShipHullType("kvlcc2")
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    ff = 1.25  # KVLCC2 form factor

    cx, cy, cz_keel = nx * 0.3, ny * 0.5, nz * 0.5
    solid, stats = build_hull_mask(hull, nx, ny, nz, cx=cx, cy=cy, cz_keel=cz_keel,
                                   length=hull_length, device="cpu")
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
    t0 = time.time()
    warmup = max(0, n_steps // 3)
    window_size = 500
    nan_at_step = -1

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=False)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
            win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
            n_s = max(len(win_fric), 1)
            cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
            cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
            ct = cf + cp
            err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0
            elapsed = time.time() - t0
            from tensorlbm.d3q19 import macroscopic3d
            rho, ux, uy, uz = macroscopic3d(f)
            u_mag = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
            ut_est = torch.sqrt(nu_lat * u_mag / 0.5).clamp(min=1e-12)
            yp_est = 0.5 * ut_est / nu_lat
            near = torch.zeros_like(solid)
            for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
                near |= torch.roll(solid, sgn, dims=ax) & (~solid)
            yp_mean = float(yp_est[near].mean().item()) if near.any() else 0.0
            print(f"[KVLCC2 {nx}³ {wall_law}] step {step}: Cf={cf:.5f} Cp={cp:.5f} Ct={ct:.5f} "
                  f"ref={ct_ref:.5f} err={err:.1f}% y+_mean={yp_mean:.1f} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            nan_at_step = step
            print(f"[KVLCC2 {nx}³ {wall_law}] NaN/Inf at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
    win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
    n_s = max(len(win_fric), 1)
    cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
    cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
    ct = cf + cp
    err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0
    finite = bool(torch.isfinite(f).all().item())

    return {
        "case": f"kvlcc2_ship_{nx}³_{wall_law}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law,
        "n_steps": n_steps, "warmup_start": warmup, "sliding_window": window_size,
        "n_samples_total": len(fric_vals), "n_samples_window": n_s,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_ref": ct_ref, "Cf_ITTC": cf_ittc, "form_factor": ff,
        "error_pct": err,
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": finite, "nan_at_step": nan_at_step,
        "wall_time_s": elapsed, "device": str(device),
    }


# ─── Case dispatch ───────────────────────────────────────────────────────────

CASE_CONFIGS = {
    "suboff_256": {
        "fn": run_suboff_bare_hull,
        "kwargs": dict(nx=256, ny=102, nz=102, re=2e6, n_steps=3000,
                       u_in=0.06, hull_length=102.0, cs_smag=0.05),
    },
    "suboff_320": {
        "fn": run_suboff_bare_hull,
        "kwargs": dict(nx=320, ny=128, nz=128, re=2e6, n_steps=3000,
                       u_in=0.06, hull_length=128.0, cs_smag=0.05),
    },
    "suboff_384": {
        "fn": run_suboff_bare_hull,
        "kwargs": dict(nx=384, ny=154, nz=154, re=2e6, n_steps=2000,
                       u_in=0.06, hull_length=154.0, cs_smag=0.05),
    },
    "kvlcc2_256": {
        "fn": run_kvlcc2,
        "kwargs": dict(nx=256, ny=77, nz=77, re=2e6, n_steps=3000,
                       u_in=0.06, hull_length=77.0, cs_smag=0.05),
    },
}


def main():
    if len(sys.argv) < 4:
        print("Usage: biggrid_worker.py <card_id> <case_name> <wall_law>", file=sys.stderr)
        print(f"  Cases: {list(CASE_CONFIGS.keys())}", file=sys.stderr)
        print(f"  wall_law: 'log', 'musker', 'reichardt', 'gradient', 'hybrid'", file=sys.stderr)
        sys.exit(1)

    card_id = int(sys.argv[1])
    case_name = sys.argv[2]
    wall_law = sys.argv[3]

    if case_name not in CASE_CONFIGS:
        print(f"Unknown case: {case_name}. Choices: {list(CASE_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)

    if wall_law not in ("log", "musker", "reichardt", "gradient", "hybrid"):
        print(f"Unknown wall_law: {wall_law}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"sdaa:{card_id}")
    torch.sdaa.set_device(device)

    cfg = CASE_CONFIGS[case_name]
    fn = cfg["fn"]
    kwargs = cfg["kwargs"]

    label = f"[{case_name} {wall_law} sdaa:{card_id}]"
    print(f"{label} Starting...", flush=True)
    t_start = time.time()

    try:
        result = fn(device=device, wall_law=wall_law, **kwargs)
        result["_status"] = "OK"
    except Exception as e:
        import traceback
        result = {
            "case": case_name,
            "wall_law": wall_law,
            "device": f"sdaa:{card_id}",
            "_status": "EXCEPTION",
            "_error": str(e),
            "_traceback": traceback.format_exc(),
        }
        print(f"{label} EXCEPTION: {e}", flush=True)

    elapsed = time.time() - t_start
    result["_launcher_elapsed_s"] = elapsed
    out = Path(f"/tmp/biggrid_{case_name}_{wall_law}_sdaa{card_id}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))

    print(f"{label} Done ({elapsed:.0f}s) → {out}", flush=True)
    # Emit JSON as final line for launcher parsing
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
