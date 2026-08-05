"""Improved wall-function benchmark worker — handles all 8 cases.

Usage:
    PYTHONPATH=src python improved_worker.py <card_id> <case_name> <improved:0|1>
    
Where case_name is one of:
    suboff_200, suboff_320, flatplate_320, cylinder, sphere,
    kvlcc2, wigley, flatplate_cs0
    
improved=0 → base (wall_law="log", use_van_driest=False)
improved=1 → improved (wall_law="musker", use_van_driest=True)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d, sphere_mask
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area


# ─── Flat plate helpers ──────────────────────────────────────────────────────

def build_plate_mask(nx, ny, nz, x_start, device):
    """Bottom-wall flat plate from x_start to nx."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, x_start:] = True
    return solid


# ─── Cylinder helpers ────────────────────────────────────────────────────────

def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """2D cylinder extruded along z."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


# ─── Ship hull helpers ───────────────────────────────────────────────────────

_FORM_FACTORS = {
    ShipHullType.WIGLEY: 1.15,
    ShipHullType.SERIES60: 1.18,
    ShipHullType.KCS: 1.20,
    ShipHullType.KVLCC2: 1.25,
    ShipHullType.NPL: 1.10,
}


def ittc_cf(re: float) -> float:
    """ITTC-1957 friction line."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


# ─── Simulation drivers ──────────────────────────────────────────────────────

def run_suboff(
    device: torch.device,
    nx: int, ny: int, nz: int,
    re: float, n_steps: int,
    u_in: float, hull_length: float,
    cs_smag: float,
    wall_law: str, use_van_driest: bool,
) -> dict:
    """Run SUBOFF bare_hull drag benchmark."""
    from tensorlbm.suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask

    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    config = SuboffConfig()
    hull_type = SuboffHullType("bare_hull")

    # Build mask on CPU first
    solid, stats = build_suboff_mask(
        hull_type, nx=nx, ny=ny, nz=nz,
        cx=cx_g, cy=cy_g, cz=cz_g,
        length=hull_length, device="cpu", config=config,
    )
    solid = solid.to(device)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * 1.0  # bare_hull form factor ≈1.0

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    fric_vals, pres_vals = [], []
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

        if step > max(0, n_steps // 3) and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            n_samples = max(len(fric_vals), 1)
            cf = sum(fric_vals) / n_samples / dyn_p_S if fric_vals else 0.0
            cp = sum(pres_vals) / n_samples / dyn_p_S if pres_vals else 0.0
            ct = cf + cp
            elapsed = time.time() - t0
            wall_label = f"musker+vd" if use_van_driest else wall_law
            print(f"[SUBOFF {nx}³ {wall_label}] step {step}: Cf={cf:.5f} Cp={cp:.5f} "
                  f"Ct={ct:.5f} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[SUBOFF {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_samples = max(len(fric_vals), 1)
    cf = sum(fric_vals) / n_samples / dyn_p_S if fric_vals else 0.0
    cp = sum(pres_vals) / n_samples / dyn_p_S if pres_vals else 0.0
    ct = cf + cp

    return {
        "case": f"suboff_bare_hull_{nx}³",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": max(0, n_steps // 3),
        "n_samples": n_samples,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_ref_ITTCx1k": cf_ittc,
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "device": str(device),
    }


def run_flatplate(
    device: torch.device,
    nx: int, ny: int, nz: int,
    re: float, n_steps: int,
    u_in: float, cs_smag: float,
    plate_pct: float,
    wall_law: str, use_van_driest: bool,
) -> dict:
    """Run flat plate Cf benchmark."""
    L = float(nx) * plate_pct
    nu_lat = u_in * L / re
    tau = 3.0 * nu_lat + 0.5

    x_start = int((1.0 - plate_pct) * nx)
    solid = build_plate_mask(nx, ny, nz, x_start, device)
    plate_area = (nx - x_start) * nz
    dyn_p_A = 0.5 * 1.0 * u_in ** 2 * plate_area
    cf_ittc = ittc_cf(re)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    samples = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, drag_f, _ = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                        wall_law=wall_law, use_van_driest=use_van_driest)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > max(0, n_steps // 3) and math.isfinite(drag_f):
            samples.append(drag_f)

        if step % 500 == 0 or step == n_steps:
            cf = (sum(samples) / max(len(samples), 1)) / dyn_p_A if samples else float('nan')
            err_pct = abs(cf - cf_ittc) / cf_ittc * 100.0 if cf_ittc > 0 else float('nan')
            elapsed = time.time() - t0
            wall_label = f"musker+vd" if use_van_driest else wall_law
            print(f"[FLATPLATE {nx}³ {wall_label}] step {step}: Cf={cf:.5f} "
                  f"err={err_pct:.1f}% ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[FLATPLATE {nx}³] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    cf_final = (sum(samples) / max(len(samples), 1)) / dyn_p_A if samples else float('nan')
    err_pct = abs(cf_final - cf_ittc) / cf_ittc * 100.0 if cf_ittc > 0 else float('nan')

    return {
        "case": f"flatplate_{nx}³" + ("_Cs0" if cs_smag == 0 else ""),
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": max(0, n_steps // 3),
        "n_samples": len(samples),
        "Cf": cf_final, "Cf_ITTC": cf_ittc, "error_pct": err_pct,
        "plate_area": plate_area, "dyn_p_A": dyn_p_A,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "device": str(device),
    }


def run_cylinder(
    device: torch.device,
    nx: int, ny: int, nz: int,
    diameter: float, re: float, n_steps: int,
    u_in: float, cs_smag: float,
    wall_law: str, use_van_driest: bool,
) -> dict:
    """Run cylinder drag benchmark."""
    nu_lat = u_in * diameter / re
    tau = 3.0 * nu_lat + 0.5
    radius = diameter / 2.0

    cx_cyl = nx * 0.25
    cy_cyl = ny * 0.5
    solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, radius, device)

    A_frontal = diameter * nz
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    cd_hist = []
    t0 = time.time()
    ref_cd_map = {100: 1.35, 200: 1.30, 500: 1.20}
    cd_ref = ref_cd_map.get(int(re), float('nan'))

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                                    wall_law=wall_law, use_van_driest=use_van_driest)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
        cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
        cd_total = cd_fric + cd_pres

        if step > max(0, n_steps // 3) and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        if step % 500 == 0 or step == n_steps:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float('nan')
            elapsed = time.time() - t0
            wall_label = f"musker+vd" if use_van_driest else wall_law
            print(f"[CYL Re={int(re)} {wall_label}] step {step}: Cd={cd_total:.4f} "
                  f"Cd_avg={cd_avg:.4f} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[CYL Re={int(re)}] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float('nan')
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0
    err_pct = abs(cd_mean - cd_ref) / cd_ref * 100 if cd_ref > 0 and math.isfinite(cd_mean) else float('nan')

    return {
        "case": f"cylinder_Re{int(re)}",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter, "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": max(0, n_steps // 3),
        "n_samples": len(cd_hist),
        "Cd_mean": cd_mean, "Cd_std": cd_std, "Cd_ref": cd_ref, "error_pct": err_pct,
        "frontal_area": A_frontal, "dyn_p": dyn_p,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "device": str(device),
    }


def run_sphere(
    device: torch.device,
    nx: int, ny: int, nz: int,
    radius: float, re: float, n_steps: int,
    u_in: float, cs_smag: float,
    wall_law: str, use_van_driest: bool,
) -> dict:
    """Run sphere drag benchmark."""
    diameter = 2.0 * radius
    nu_lat = u_in * diameter / re
    tau = 3.0 * nu_lat + 0.5

    cx_s, cy_s, cz_s = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx_s, cy_s, cz_s, radius, device=device)

    ref_area = math.pi * radius ** 2
    dyn_p = 0.5 * 1.0 * u_in ** 2 * ref_area
    cd_ref_map = {100: 1.09, 1000: 0.47, 10000: 0.40}
    cd_ref = cd_ref_map.get(int(re), float('nan'))

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    cd_hist = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f = stream3d(f)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu_lat, y_val=0.5,
                                                    wall_law=wall_law, use_van_driest=use_van_driest)
        f = far_field_bc_3d(f, u_in=u_in)
        f = bounce_back_cells_3d(f, solid)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
        cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
        cd_total = cd_fric + cd_pres

        if step > max(0, n_steps // 3) and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        if step % 500 == 0 or step == n_steps:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float('nan')
            elapsed = time.time() - t0
            wall_label = f"musker+vd" if use_van_driest else wall_law
            print(f"[SPHERE Re={int(re)} {wall_label}] step {step}: Cd={cd_total:.4f} "
                  f"Cd_avg={cd_avg:.4f} ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[SPHERE Re={int(re)}] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float('nan')
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0
    err_pct = abs(cd_mean - cd_ref) / cd_ref * 100 if cd_ref > 0 and math.isfinite(cd_mean) else float('nan')

    return {
        "case": f"sphere_Re{int(re)}",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter, "radius": radius, "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": max(0, n_steps // 3),
        "n_samples": len(cd_hist),
        "Cd_fric": sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float('nan'),
        "Cd_total": cd_mean, "Cd_std": cd_std, "Cd_ref": cd_ref, "error_pct": err_pct,
        "ref_area": ref_area, "dyn_p": dyn_p,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "device": str(device),
    }


def run_ship(
    device: torch.device,
    hull_str: str,
    nx: int, ny: int, nz: int,
    re: float, n_steps: int,
    u_in: float, hull_length: float,
    cs_smag: float,
    wall_law: str, use_van_driest: bool,
) -> dict:
    """Run ship hull drag benchmark."""
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

        if step > max(0, n_steps // 3) and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            n_samples = max(len(fric_vals), 1)
            cf = sum(fric_vals) / n_samples / dyn_p_S if fric_vals else 0.0
            cp = sum(pres_vals) / n_samples / dyn_p_S if pres_vals else 0.0
            ct = cf + cp
            err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float('inf')
            elapsed = time.time() - t0
            wall_label = f"musker+vd" if use_van_driest else wall_law
            print(f"[{hull.value} {wall_label}] step {step}: Cf={cf:.5f} Cp={cp:.5f} "
                  f"Ct={ct:.5f} ref={ct_ref:.5f} err={err_pct:.1f}% ({elapsed:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"[{hull.value}] NaN at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_samples = max(len(fric_vals), 1)
    cf = sum(fric_vals) / n_samples / dyn_p_S if fric_vals else 0.0
    cp = sum(pres_vals) / n_samples / dyn_p_S if pres_vals else 0.0
    ct = cf + cp
    err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float('inf')

    return {
        "case": f"{hull.value}_ship",
        "hull_type": hull.value,
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "Cs": cs_smag, "nu": nu_lat, "tau": tau,
        "wall_law": wall_law, "use_van_driest": use_van_driest,
        "n_steps": n_steps, "warmup_start": max(0, n_steps // 3),
        "n_samples": n_samples,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "Ct_reference": ct_ref, "error_pct": err_pct,
        "wetted_area": S, "dyn_p_S": dyn_p_S,
        "Cf_ITTC": cf_ittc, "form_factor": ff,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "device": str(device),
    }


# ─── Case dispatch ───────────────────────────────────────────────────────────

CASE_CONFIGS = {
    "suboff_200": {
        "fn": run_suboff,
        "kwargs": dict(nx=200, ny=80, nz=80, re=2e6, n_steps=2000,
                       u_in=0.06, hull_length=100.0, cs_smag=0.05),
    },
    "suboff_320": {
        "fn": run_suboff,
        "kwargs": dict(nx=320, ny=128, nz=128, re=2e6, n_steps=1500,
                       u_in=0.06, hull_length=100.0, cs_smag=0.05),
    },
    "flatplate_320": {
        "fn": run_flatplate,
        "kwargs": dict(nx=320, ny=128, nz=80, re=2e6, n_steps=2000,
                       u_in=0.06, cs_smag=0.05, plate_pct=0.80),
    },
    "cylinder": {
        "fn": run_cylinder,
        "kwargs": dict(nx=200, ny=80, nz=4, diameter=24.0, re=200.0, n_steps=2000,
                       u_in=0.08, cs_smag=0.05),
    },
    "sphere": {
        "fn": run_sphere,
        "kwargs": dict(nx=120, ny=60, nz=60, radius=12.0, re=100.0, n_steps=2000,
                       u_in=0.06, cs_smag=0.05),
    },
    "kvlcc2": {
        "fn": run_ship,
        "kwargs": dict(hull_str="kvlcc2", nx=200, ny=60, nz=60, re=2e6, n_steps=2000,
                       u_in=0.06, hull_length=80.0, cs_smag=0.05),
    },
    "wigley": {
        "fn": run_ship,
        "kwargs": dict(hull_str="wigley", nx=200, ny=60, nz=60, re=2e6, n_steps=2000,
                       u_in=0.06, hull_length=80.0, cs_smag=0.05),
    },
    "flatplate_cs0": {
        "fn": run_flatplate,
        "kwargs": dict(nx=400, ny=80, nz=80, re=2e6, n_steps=2000,
                       u_in=0.06, cs_smag=0.0, plate_pct=0.80),
    },
}


def main():
    if len(sys.argv) < 4:
        print("Usage: improved_worker.py <card_id> <case_name> <improved:0|1>", file=sys.stderr)
        sys.exit(1)

    card_id = int(sys.argv[1])
    case_name = sys.argv[2]
    improved = bool(int(sys.argv[3]))

    if case_name not in CASE_CONFIGS:
        print(f"Unknown case: {case_name}. Choices: {list(CASE_CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Set wall function parameters
    if improved:
        wall_law = "musker"
        use_van_driest = True
        label = "musker+vanDriest"
    else:
        wall_law = "log"
        use_van_driest = False
        label = "log-law (base)"

    device = torch.device(f"sdaa:{card_id}")
    torch.sdaa.set_device(device)

    cfg = CASE_CONFIGS[case_name]
    fn = cfg["fn"]
    kwargs = cfg["kwargs"]

    tag = f"[{case_name} {label} sdaa:{card_id}]"
    print(f"{tag} Starting...", flush=True)
    t_start = time.time()

    try:
        result = fn(device=device, wall_law=wall_law, use_van_driest=use_van_driest, **kwargs)
        result["_status"] = "OK"
    except Exception as e:
        result = {
            "case": case_name,
            "wall_law": wall_law,
            "use_van_driest": use_van_driest,
            "device": f"sdaa:{card_id}",
            "_status": "EXCEPTION",
            "_error": str(e),
        }
        import traceback
        result["_traceback"] = traceback.format_exc()
        print(f"{tag} EXCEPTION: {e}", flush=True)

    elapsed = time.time() - t_start
    result["_launcher_elapsed_s"] = elapsed
    print(f"{tag} Done ({elapsed:.0f}s)", flush=True)

    # Output JSON to stdout
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
