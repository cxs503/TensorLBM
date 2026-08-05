"""Fine-grid sign-fix re-test worker. Tests pressure-drag sign fix on 200³/320³ grids.

Usage: PYTHONPATH=src python fine_signfix_worker.py <card_id> <case_name>

Cases: suboff_320, suboff_200, kvlcc2_320, kvlcc2_200, flatplate_cs005_200, flatplate_cs0_200
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


def run_suboff_bare_hull(device, nx, ny, nz, re, n_steps, u_in, hull_length, cs_smag, wall_law, use_van_driest):
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

    fric_vals, pres_vals = [], []
    t0 = time.time()
    warmup = max(0, n_steps // 3)
    window_size = 500

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=use_van_driest)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            # Sliding window: last 500 samples
            win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
            win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
            n_s = max(len(win_fric), 1)
            cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
            cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
            ct = cf + cp
            err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0
            elapsed = time.time() - t0
            print(f"[SUBOFF {nx}³] step {step}: Cf={cf:.5f} Cp={cp:.5f} Ct={ct:.5f} err={err:.1f}% ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[SUBOFF {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
    win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
    n_s = max(len(win_fric), 1)
    cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
    cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
    ct = cf + cp
    err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0

    return {
        "case": f"suboff_bare_hull_{nx}³",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": warmup, "sliding_window": window_size,
        "n_samples_total": len(fric_vals), "n_samples_window": n_s,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_ref_ITTCx1k": ct_ref, "error_pct": err,
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed, "device": str(device),
    }


def run_kvlcc2(device, nx, ny, nz, re, n_steps, u_in, hull_length, cs_smag, wall_law, use_van_driest):
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

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=use_van_driest)
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
            print(f"[KVLCC2 {nx}³] step {step}: Cf={cf:.5f} Cp={cp:.5f} Ct={ct:.5f} ref={ct_ref:.5f} err={err:.1f}% ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[KVLCC2 {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
    win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
    n_s = max(len(win_fric), 1)
    cf = sum(win_fric) / n_s / dyn_p_S if win_fric else 0.0
    cp = sum(win_pres) / n_s / dyn_p_S if win_pres else 0.0
    ct = cf + cp
    err = abs(ct - ct_ref) / ct_ref * 100 if ct_ref else 0

    return {
        "case": f"kvlcc2_ship_{nx}³",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": warmup, "sliding_window": window_size,
        "n_samples_total": len(fric_vals), "n_samples_window": n_s,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_ref": ct_ref, "Cf_ITTC": cf_ittc, "form_factor": ff,
        "error_pct": err,
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed, "device": str(device),
    }


def run_flatplate(device, nx, ny, nz, re, n_steps, u_in, cs_smag, plate_pct, wall_law, use_van_driest):
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import correct_mass3d, stream3d
    from tensorlbm.turbulence import collide_smagorinsky_mrt3d
    from tensorlbm.wall_model import wall_function_3d
    from tensorlbm.boundaries3d import far_field_bc_3d

    L = float(nx) * plate_pct
    nu_lat = u_in * L / re
    tau = 3.0 * nu_lat + 0.5

    x_start = int((1.0 - plate_pct) * nx)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, x_start:] = True
    plate_area = (nx - x_start) * nz
    dyn_p_A = 0.5 * 1.0 * u_in ** 2 * plate_area
    cf_ittc = ittc_cf(re)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    fric_vals, pres_vals = [], []
    t0 = time.time()
    warmup = max(0, n_steps // 3)
    window_size = 500

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                     wall_law=wall_law, use_van_driest=use_van_driest)
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
            cf = sum(win_fric) / n_s / dyn_p_A if win_fric else 0.0
            cp = sum(win_pres) / n_s / dyn_p_A if win_pres else 0.0
            ct = cf + cp
            err_cf = abs(cf - cf_ittc) / cf_ittc * 100 if cf_ittc else 0
            elapsed = time.time() - t0
            print(f"[FLATPLATE {nx}³ Cs={cs_smag}] step {step}: Cf={cf:.5f} Cp={cp:.5f} Ct={ct:.5f} Cf_err={err_cf:.1f}% ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[FLATPLATE {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    win_fric = fric_vals[-window_size:] if len(fric_vals) >= window_size else fric_vals
    win_pres = pres_vals[-window_size:] if len(pres_vals) >= window_size else pres_vals
    n_s = max(len(win_fric), 1)
    cf = sum(win_fric) / n_s / dyn_p_A if win_fric else 0.0
    cp = sum(win_pres) / n_s / dyn_p_A if win_pres else 0.0
    ct = cf + cp
    err_cf = abs(cf - cf_ittc) / cf_ittc * 100 if cf_ittc else 0

    return {
        "case": f"flatplate_{nx}³_Cs{cs_smag}",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": warmup, "sliding_window": window_size,
        "n_samples_total": len(fric_vals), "n_samples_window": n_s,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Cf_ITTC": cf_ittc, "Cf_error_pct": err_cf,
        "plate_area": plate_area, "dyn_p_A": dyn_p_A,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed, "device": str(device),
    }


# ─── Case dispatch ───────────────────────────────────────────────────────────

CASE_CONFIGS = {
    "suboff_320": {
        "fn": run_suboff_bare_hull,
        "kwargs": dict(nx=320, ny=128, nz=128, re=2e6, n_steps=2000,
                       u_in=0.06, hull_length=128.0, cs_smag=0.05),
    },
    "suboff_200": {
        "fn": run_suboff_bare_hull,
        "kwargs": dict(nx=200, ny=80, nz=80, re=2e6, n_steps=5000,
                       u_in=0.06, hull_length=100.0, cs_smag=0.05),
    },
    "kvlcc2_320": {
        "fn": run_kvlcc2,
        "kwargs": dict(nx=320, ny=96, nz=96, re=2e6, n_steps=2000,
                       u_in=0.06, hull_length=96.0, cs_smag=0.05),
    },
    "kvlcc2_200": {
        "fn": run_kvlcc2,
        "kwargs": dict(nx=200, ny=60, nz=60, re=2e6, n_steps=3000,
                       u_in=0.06, hull_length=80.0, cs_smag=0.05),
    },
    "flatplate_cs005_200": {
        "fn": run_flatplate,
        "kwargs": dict(nx=200, ny=80, nz=80, re=2e6, n_steps=2000,
                       u_in=0.06, cs_smag=0.05, plate_pct=0.80),
    },
    "flatplate_cs0_200": {
        "fn": run_flatplate,
        "kwargs": dict(nx=200, ny=80, nz=80, re=2e6, n_steps=2000,
                       u_in=0.06, cs_smag=0.0, plate_pct=0.80),
    },
}


def main():
    if len(sys.argv) < 3:
        print("Usage: fine_signfix_worker.py <card_id> <case_name>", file=sys.stderr)
        print(f"  Cases: {list(CASE_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)

    card_id = int(sys.argv[1])
    case_name = sys.argv[2]

    if case_name not in CASE_CONFIGS:
        print(f"Unknown case: {case_name}. Choices: {list(CASE_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Always use log-law wall function (D3Q19 MRT+Smag standard)
    wall_law = "log"
    use_van_driest = False

    device = torch.device(f"sdaa:{card_id}")
    torch.sdaa.set_device(device)

    cfg = CASE_CONFIGS[case_name]
    fn = cfg["fn"]
    kwargs = cfg["kwargs"]

    label = f"[{case_name} sdaa:{card_id}]"
    print(f"{label} Starting...", flush=True)
    t_start = time.time()

    try:
        result = fn(device=device, wall_law=wall_law, use_van_driest=use_van_driest, **kwargs)
        result["_status"] = "OK"
        result["sign_fix"] = "pressure_drag_minus_sign"  # -p*(sp-sm)
    except Exception as e:
        import traceback
        result = {
            "case": case_name,
            "wall_law": wall_law,
            "device": f"sdaa:{card_id}",
            "_status": "EXCEPTION",
            "_error": str(e),
            "_traceback": traceback.format_exc(),
            "sign_fix": "pressure_drag_minus_sign",
        }
        print(f"{label} EXCEPTION: {e}", flush=True)

    elapsed = time.time() - t_start
    result["_launcher_elapsed_s"] = elapsed
    tag = case_name.replace("_cs005", "_Cs0.05").replace("_cs0", "_Cs0")
    out = Path(f"/tmp/fine_signfix_{tag}_sdaa{card_id}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))

    print(f"{label} Done ({elapsed:.0f}s) → {out}", flush=True)
    # Also emit JSON as final line for launcher parsing
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
