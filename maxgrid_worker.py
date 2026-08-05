"""MaxGrid universal worker — one script for all 8 benchmark configurations.

Each invocation runs ONE configuration on ONE SDAA card.
All use: D3Q19 MRT+Smag Cs=0.05 + wall_function_3d + far_field_bc_3d.
Only geometry and grid differ.

Usage:
    PYTHONPATH=src python maxgrid_worker.py \
        --case suboff_384 --device sdaa:0 --output /tmp/r_suboff_384.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from argparse import ArgumentParser

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, sphere_mask
from tensorlbm.wall_model import wall_function_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

# Ship hull imports (for KVLCC2, Wigley)
from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.suboff_resistance import _ittc57_friction_coefficient, _voxel_wetted_area
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask

# ---------------------------------------------------------------------------
# ITTC reference helpers
# ---------------------------------------------------------------------------

def ittc_cf(re: float) -> float:
    return 0.075 / (math.log10(re) - 2.0) ** 2


# Form factors (1+k) — for ship hulls
_FORM_FACTORS = {
    ShipHullType.WIGLEY: 1.15,
    ShipHullType.KVLCC2: 1.25,
}

# SUBOFF bare-hull form factor ≈ 1.0 (submarine body of revolution)
_SUBOFF_FORM_FACTOR = 1.0


# ---------------------------------------------------------------------------
# Case-specific runner functions
# ---------------------------------------------------------------------------

def run_suboff(
    device: str, nx: int, ny: int, nz: int,
    re: float = 2e6, u_in: float = 0.06,
    n_steps: int = 3000, warmup: int = 1000,
    smagorinsky_cs: float = 0.05,
) -> dict:
    """SUBOFF bare_hull benchmark."""
    dev = torch.device(device)

    # Hull length = 0.6 * nx (default SUBOFF convention)
    hull_length = nx * 0.6
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    # Hull placement
    cx = nx * 0.4
    cy = ny * 0.5
    cz = nz * 0.5

    solid, stats = build_suboff_mask(
        SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=hull_length,
        device="cpu",
    )
    solid = solid.to(dev)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    # ITTC reference Ct
    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * _SUBOFF_FORM_FACTOR

    # Initialize
    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full((nz, ny, nx), u_in)
    uy0 = torch.zeros(nz, ny, nx)
    uz0 = torch.zeros(nz, ny, nx)
    ux0[solid.cpu()] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    f = f.to(dev)
    initial_mass = float(f.sum().item())

    fric_vals, pres_vals = [], []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            cf_avg = sum(fric_vals) / max(len(fric_vals), 1) / dyn_p_S if fric_vals else 0.0
            cp_avg = sum(pres_vals) / max(len(pres_vals), 1) / dyn_p_S if pres_vals else 0.0
            ct_avg = cf_avg + cp_avg
            elapsed = time.time() - start_time
            print(f"[SUBOFF {nx}x{ny}x{nz}] step {step}: Ct_fric={cf_avg:.5f} "
                  f"Ct_pres={cp_avg:.5f} Ct_tot={ct_avg:.5f} (ref {ct_ref:.5f})  "
                  f"time={elapsed:.0f}s", file=sys.stderr, flush=True)
            if not torch.isfinite(f).all():
                print(f"[SUBOFF] NaN detected at step {step}", file=sys.stderr)
                break

    elapsed = time.time() - start_time
    cf = sum(fric_vals) / max(len(fric_vals), 1) / dyn_p_S if fric_vals else 0.0
    cp = sum(pres_vals) / max(len(pres_vals), 1) / dyn_p_S if pres_vals else 0.0
    ct = cf + cp
    err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float("inf")

    return {
        "case": f"suboff_{nx}x{ny}x{nz}",
        "device": device, "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "u_in": u_in, "nu": nu_lat, "tau": tau,
        "hull_length": hull_length, "C_s": smagorinsky_cs,
        "n_steps": n_steps, "warmup": warmup, "n_samples": len(fric_vals),
        "wetted_area": S, "Cf_ITTC": cf_ittc,
        "form_factor_1pk": _SUBOFF_FORM_FACTOR, "Ct_reference": ct_ref,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "stats": {k: v for k, v in stats.items() if not isinstance(v, torch.Tensor)},
    }


def run_ship_hull(
    device: str, hull_type_str: str,
    nx: int, ny: int, nz: int,
    re: float = 2e6, u_in: float = 0.06,
    hull_length: float = 80.0,
    n_steps: int = 2000, warmup: int = 667,
    smagorinsky_cs: float = 0.05,
) -> dict:
    """Generic ship hull benchmark (KVLCC2, Wigley)."""
    hull = ShipHullType(hull_type_str)
    dev = torch.device(device)
    ff = _FORM_FACTORS[hull]

    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    cx = nx * 0.3
    cy = ny * 0.5
    cz_keel = nz * 0.5

    solid, stats = build_hull_mask(
        hull, nx, ny, nz,
        cx=cx, cy=cy, cz_keel=cz_keel,
        length=hull_length, device="cpu",
    )
    solid = solid.to(dev)
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    cf_ittc = _ittc57_friction_coefficient(re)
    ct_ref = cf_ittc * ff

    rho0 = torch.ones((nz, ny, nx))
    ux0 = torch.full((nz, ny, nx), u_in)
    uy0 = torch.zeros(nz, ny, nx)
    uz0 = torch.zeros(nz, ny, nx)
    ux0[solid.cpu()] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    f = f.to(dev)
    initial_mass = float(f.sum().item())

    fric_vals, pres_vals = [], []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu_lat, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(df):
            fric_vals.append(df)
            pres_vals.append(dp)

        if step % 500 == 0 or step == n_steps:
            cf_avg = sum(fric_vals) / max(len(fric_vals), 1) / dyn_p_S if fric_vals else 0.0
            cp_avg = sum(pres_vals) / max(len(pres_vals), 1) / dyn_p_S if pres_vals else 0.0
            ct_avg = cf_avg + cp_avg
            elapsed = time.time() - start_time
            print(f"[{hull.value} {nx}x{ny}x{nz}] step {step}: Ct_tot={ct_avg:.5f} "
                  f"(ref {ct_ref:.5f})  time={elapsed:.0f}s",
                  file=sys.stderr, flush=True)
            if not torch.isfinite(f).all():
                print(f"[{hull.value}] NaN at step {step}", file=sys.stderr)
                break

    elapsed = time.time() - start_time
    cf = sum(fric_vals) / max(len(fric_vals), 1) / dyn_p_S if fric_vals else 0.0
    cp = sum(pres_vals) / max(len(pres_vals), 1) / dyn_p_S if pres_vals else 0.0
    ct = cf + cp
    err_pct = abs(ct - ct_ref) / ct_ref * 100 if ct_ref > 0 else float("inf")

    return {
        "case": f"{hull.value}_{nx}x{ny}x{nz}",
        "device": device, "grid": f"{nx}x{ny}x{nz}",
        "hull_type": hull.value, "Re": re, "u_in": u_in,
        "nu": nu_lat, "tau": tau, "hull_length": hull_length,
        "C_s": smagorinsky_cs,
        "n_steps": n_steps, "warmup": warmup, "n_samples": len(fric_vals),
        "wetted_area": S, "Cf_ITTC": cf_ittc,
        "form_factor_1pk": ff, "Ct_reference": ct_ref,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
        "stats": {k: v for k, v in stats.items() if not isinstance(v, torch.Tensor)},
    }


def run_flat_plate(
    device: str, nx: int, ny: int, nz: int,
    re: float = 2e6, u_in: float = 0.06,
    plate_pct: float = 0.80,
    n_steps: int = 2000, warmup: int = 667,
    smagorinsky_cs: float = 0.05,
) -> dict:
    """Flat plate Cf benchmark."""
    dev = torch.device(device)
    L = float(nx)
    nu_lat = u_in * L / re
    tau = 3.0 * nu_lat + 0.5

    x_start = int((1.0 - plate_pct) * nx)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev)
    solid[:, 0, x_start:] = True
    plate_area = (nx - x_start) * nz
    dyn_p_A = 0.5 * 1.0 * u_in ** 2 * plate_area

    cf_ittc = ittc_cf(re)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    initial_mass = float(rho0.sum().item())

    samples = []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)
        f = stream3d(f)
        f, drag_f, drag_p = wall_function_3d(f, solid, nu_lat, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > warmup and math.isfinite(drag_f):
            samples.append(drag_f)

        if step % 500 == 0 or step == n_steps:
            cf_avg = sum(samples) / max(len(samples), 1) / dyn_p_A if samples else 0.0
            err_avg = abs(cf_avg - cf_ittc) / cf_ittc * 100 if cf_ittc > 0 else 0.0
            elapsed = time.time() - start_time
            print(f"[FlatPlate Cs={smagorinsky_cs} {nx}x{ny}x{nz}] step {step}: "
                  f"Cf={cf_avg:.6f} (ITTC {cf_ittc:.5f}, err={err_avg:.1f}%) "
                  f"time={elapsed:.0f}s", file=sys.stderr, flush=True)
            if not torch.isfinite(f).all():
                print(f"[FlatPlate] NaN at step {step}", file=sys.stderr)
                break

    elapsed = time.time() - start_time
    cf_final = sum(samples) / max(len(samples), 1) / dyn_p_A if samples else 0.0
    err_pct = abs(cf_final - cf_ittc) / cf_ittc * 100 if cf_ittc > 0 else float("inf")

    return {
        "case": f"flatplate_Cs{smagorinsky_cs}_{nx}x{ny}x{nz}",
        "device": device, "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "u_in": u_in, "nu": nu_lat, "tau": tau,
        "C_s": smagorinsky_cs, "plate_pct": plate_pct,
        "plate_area_cells": plate_area,
        "n_steps": n_steps, "warmup": warmup, "n_samples": len(samples),
        "Cf_reference": cf_ittc, "Cf_final": cf_final,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }


def run_cylinder(
    device: str, nx: int, ny: int, nz: int,
    diameter: float, re: float = 200.0,
    u_in: float = 0.08,
    n_steps: int = 2000, warmup: int = 667,
    smagorinsky_cs: float = 0.05,
) -> dict:
    """Cylinder Cd benchmark (2D extruded)."""
    dev = torch.device(device)

    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    radius = diameter / 2.0
    cx_cyl = nx * 0.25
    cy_cyl = ny * 0.5
    cz_cyl = nz * 0.5

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx_cyl) ** 2 + (yy - cy_cyl) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()

    A_frontal = diameter * nz
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    cd_hist = []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)
        f = stream3d(f)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        cd_total = (drag_fric + drag_pres) / dyn_p if dyn_p > 0 else 0.0
        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else 0.0
            elapsed = time.time() - start_time
            print(f"[Cylinder Re={int(re)} D={diameter} {nx}x{ny}x{nz}] "
                  f"step {step}: Cd={cd_total:.4f} Cd_avg={cd_avg:.4f} "
                  f"time={elapsed:.0f}s", file=sys.stderr, flush=True)
            if not torch.isfinite(f).all():
                print(f"[Cylinder] NaN at step {step}", file=sys.stderr)
                break

    elapsed = time.time() - start_time
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0

    ref_cd = {100: 1.35, 200: 1.30, 500: 1.20}
    ref = ref_cd.get(int(re), float("nan"))
    err_pct = abs(cd_mean - ref) / ref * 100 if ref > 0 and math.isfinite(cd_mean) else float("nan")

    return {
        "case": f"cylinder_Re{int(re)}_D{int(diameter)}_{nx}x{ny}x{nz}",
        "device": device, "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "u_in": u_in, "nu": nu, "tau": tau,
        "diameter": diameter, "C_s": smagorinsky_cs,
        "n_steps": n_steps, "warmup": warmup, "n_samples": len(cd_hist),
        "Cd_mean": cd_mean, "Cd_std": cd_std,
        "Cd_ref": ref, "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }


def run_sphere(
    device: str, nx: int, ny: int, nz: int,
    diameter: float, re: float = 100.0,
    u_in: float = 0.06,
    n_steps: int = 2000, warmup: int = 667,
    smagorinsky_cs: float = 0.05,
) -> dict:
    """Sphere Cd benchmark."""
    dev = torch.device(device)

    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    radius = diameter / 2.0
    cx_sph = nx * 0.25
    cy_sph = ny * 0.5
    cz_sph = nz * 0.5

    solid = sphere_mask(nx, ny, nz, cx_sph, cy_sph, cz_sph, radius, dev)

    A_frontal = math.pi * radius ** 2
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    cd_hist = []
    start_time = time.time()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=smagorinsky_cs)
        f = stream3d(f)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        cd_total = (drag_fric + drag_pres) / dyn_p if dyn_p > 0 else 0.0
        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else 0.0
            elapsed = time.time() - start_time
            print(f"[Sphere Re={int(re)} D={diameter} {nx}x{ny}x{nz}] "
                  f"step {step}: Cd={cd_total:.4f} Cd_avg={cd_avg:.4f} "
                  f"time={elapsed:.0f}s", file=sys.stderr, flush=True)
            if not torch.isfinite(f).all():
                print(f"[Sphere] NaN at step {step}", file=sys.stderr)
                break

    elapsed = time.time() - start_time
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0

    # Schiller-Naumann correlation at Re=100: Cd ≈ 1.09
    ref_cd = {100: 1.09, 200: 0.77, 500: 0.55}
    ref = ref_cd.get(int(re), float("nan"))
    err_pct = abs(cd_mean - ref) / ref * 100 if ref > 0 and math.isfinite(cd_mean) else float("nan")

    return {
        "case": f"sphere_Re{int(re)}_D{int(diameter)}_{nx}x{ny}x{nz}",
        "device": device, "grid": f"{nx}x{ny}x{nz}",
        "Re": re, "u_in": u_in, "nu": nu, "tau": tau,
        "diameter": diameter, "C_s": smagorinsky_cs,
        "n_steps": n_steps, "warmup": warmup, "n_samples": len(cd_hist),
        "Cd_mean": cd_mean, "Cd_std": cd_std,
        "Cd_ref": ref, "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Configuration dispatch table
# ---------------------------------------------------------------------------

CONFIGS = {
    "suboff_384": {
        "runner": "suboff", "nx": 384, "ny": 144, "nz": 144,
        "re": 2e6, "u_in": 0.06, "n_steps": 3000, "warmup": 1000,
        "smagorinsky_cs": 0.05,
    },
    "suboff_448": {
        "runner": "suboff", "nx": 448, "ny": 168, "nz": 168,
        "re": 2e6, "u_in": 0.06, "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.05,
    },
    "kvlcc2": {
        "runner": "ship", "hull": "kvlcc2",
        "nx": 320, "ny": 96, "nz": 96,
        "re": 2e6, "u_in": 0.06, "hull_length": 96.0,
        "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.05,
    },
    "wigley": {
        "runner": "ship", "hull": "wigley",
        "nx": 320, "ny": 96, "nz": 96,
        "re": 2e6, "u_in": 0.06, "hull_length": 96.0,
        "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.05,
    },
    "flatplate_cs005": {
        "runner": "flatplate",
        "nx": 400, "ny": 80, "nz": 80,
        "re": 2e6, "u_in": 0.06, "plate_pct": 0.80,
        "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.05,
    },
    "cylinder_re200": {
        "runner": "cylinder",
        "nx": 320, "ny": 128, "nz": 4,
        "diameter": 32.0, "re": 200.0, "u_in": 0.08,
        "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.05,
    },
    "sphere_re100": {
        "runner": "sphere",
        "nx": 200, "ny": 100, "nz": 100,
        "diameter": 40.0, "re": 100.0, "u_in": 0.06,
        "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.05,
    },
    "flatplate_cs0": {
        "runner": "flatplate",
        "nx": 400, "ny": 80, "nz": 80,
        "re": 2e6, "u_in": 0.06, "plate_pct": 0.80,
        "n_steps": 2000, "warmup": 667,
        "smagorinsky_cs": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="MaxGrid universal benchmark worker")
    parser.add_argument("--case", type=str, required=True,
                        choices=list(CONFIGS.keys()),
                        help="Benchmark case identifier")
    parser.add_argument("--device", type=str, required=True,
                        help="Device string, e.g. sdaa:0")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    cfg = CONFIGS[args.case]
    runner = cfg.pop("runner")
    cfg["device"] = args.device

    if runner == "suboff":
        result = run_suboff(**cfg)
    elif runner == "ship":
        cfg["hull_type_str"] = cfg.pop("hull")
        result = run_ship_hull(**cfg)
    elif runner == "flatplate":
        result = run_flat_plate(**cfg)
    elif runner == "cylinder":
        result = run_cylinder(**cfg)
    elif runner == "sphere":
        result = run_sphere(**cfg)
    else:
        print(f"Unknown runner: {runner}", file=sys.stderr)
        sys.exit(1)

    # Add baselines for comparison
    baselines = {
        "suboff_384": 24.5,
        "suboff_448": 24.5,  # same case, finer grid
        "kvlcc2": 10.3,
        "wigley": 3.9,
        "flatplate_cs005": 36.6,
        "cylinder_re200": 8.1,
        "sphere_re100": 13.4,
        "flatplate_cs0": 18.0,
    }
    result["baseline_error_pct"] = baselines.get(args.case, None)
    result["grid_refinement_factor"] = None  # filled by launcher if needed

    with open(args.output, "w") as fp:
        json.dump(result, fp, indent=2, default=str)
    print(f"Result saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
