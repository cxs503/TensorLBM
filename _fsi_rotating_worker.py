#!/usr/bin/env python3
"""FSI Test 3: Rotating 3-blade turbine (fixed rotation speed).

SDAA:26  |  Re=1000, 3-blade turbine, fixed rotation speed

Pipeline (common-interface only):
  solid → get_near_wall_3d → SurfaceMesh.from_gradient →
  lbm_step_correct → drag_pressure + drag_friction → torque/power

The turbine rotates at a fixed angular speed.  The fluid exerts a torque
on the blades; the power coefficient is Cp = torque * omega / (0.5 * rho * U^3 * A).

Target: Cp within 30% of reference (typical Cp_max ≈ 0.4–0.5 for a 3-blade
turbine at tip-speed ratio λ ≈ 5–7).

Usage:
  python _fsi_rotating_worker.py <device_id> [output_path]
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch

# ---- Common interface imports (ONLY these modules) ----
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal


def build_turbine_mask(nx, ny, nz, cx, cy, R_blade, n_blades, blade_width, angle_deg, device):
    """3-blade turbine mask (rotated by angle_deg).

    Each blade is a rectangle from the hub to R_blade, with width blade_width.
    The mask is 2D (extruded along z).
    """
    yy, xx = torch.meshgrid(
        torch.arange(ny, device="cpu", dtype=torch.float32),
        torch.arange(nx, device="cpu", dtype=torch.float32),
        indexing="ij",
    )
    xx = xx - cx
    yy = yy - cy
    r = torch.sqrt(xx ** 2 + yy ** 2)

    # Rotate coordinates by -angle to check blade positions
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xr = xx * cos_a + yy * sin_a
    yr = -xx * sin_a + yy * cos_a

    mask = (r <= R_blade) & (r >= 0)  # start with full disk

    # Build blade mask: n_blades sectors
    blade = torch.zeros_like(mask)
    for i in range(n_blades):
        blade_angle = 2 * math.pi * i / n_blades
        # Rotate to blade frame
        cos_b = math.cos(blade_angle)
        sin_b = math.sin(blade_angle)
        xb = xr * cos_b + yr * sin_b
        yb = -xr * sin_b + yr * cos_b
        # Blade: radial extent [0, R_blade], tangential width [-w/2, w/2]
        blade |= (xb >= -R_blade) & (xb <= R_blade) & (abs(yb) <= blade_width / 2) & (r <= R_blade)

    # Hub
    hub_r = max(2, R_blade // 6)
    blade |= r <= hub_r

    solid = blade.unsqueeze(0).expand(nz, ny, nx).clone().to(device)
    return solid


def run_rotating(device_id, output_path=None):
    """Rotating turbine: Re=1000, 3-blade, fixed rotation speed."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # --- Grid & flow parameters ---
    nx, ny, nz = 300, 300, 4
    R_blade = 30                       # blade radius (lattice units)
    D = 2 * R_blade                     # rotor diameter
    cx = nx // 3
    cy = ny // 2
    n_blades = 3
    blade_width = 4
    Re = 1000
    u_in = 0.1
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 20000
    tag = f"[Rotating SDAA:{device_id}]"

    # Tip-speed ratio lambda = omega * R / U
    # Target lambda ≈ 5 for peak Cp
    tsr = 5.0
    omega = tsr * u_in / R_blade  # rad/step (lattice units)
    # Rotation period in steps
    rot_period = 2 * math.pi / omega if omega > 0 else n_steps
    # Steps per degree
    steps_per_deg = rot_period / 360.0

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R_blade} Re={Re} "
          f"u_in={u_in} nu={nu:.6e} tau={tau:.6f}", flush=True)
    print(f"{tag} TSR={tsr} omega={omega:.6f} rad/step "
          f"rot_period={rot_period:.1f} steps steps/deg={steps_per_deg:.2f}",
          flush=True)

    dpS = 0.5 * u_in ** 2 * D * nz   # frontal area = D × span (for Cd)
    # Power reference area: pi * R^2 (swept area)
    A_swept = math.pi * R_blade ** 2
    p_ref = 0.5 * u_in ** 3 * A_swept  # available power

    t0 = time.time()

    # --- Build initial turbine mask ---
    angle_deg = 0.0
    solid = build_turbine_mask(nx, ny, nz, cx, cy, R_blade, n_blades,
                               blade_width, angle_deg, device)

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # --- Common interface: SurfaceMesh.from_gradient ---
    mesh = SurfaceMesh.from_gradient(solid, near)
    print(f"{tag} SurfaceMesh.from_gradient built", flush=True)

    # --- Far-field BC ---
    bc_config = {'far_field_faces': ['y+', 'y-'], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    # --- Initialise flow ---
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # --- Histories ---
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    torque_hist, cp_hist = [], []

    # --- Main loop: rotate turbine, measure torque ---
    # We rebuild the mask every few steps (rotation is slow)
    rebuild_interval = max(1, int(steps_per_deg * 5))  # rebuild every 5 degrees

    for step in range(1, n_steps + 1):
        # Rotate turbine
        angle_deg = (omega * step * 180.0 / math.pi) % 360.0

        # Rebuild mask at new angle
        if step == 1 or step % rebuild_interval == 0:
            solid = build_turbine_mask(nx, ny, nz, cx, cy, R_blade, n_blades,
                                       blade_width, angle_deg, device)
            near = get_near_wall_3d(solid)
            mesh = SurfaceMesh.from_gradient(solid, near)

        # LBM step
        f = lbm_step_correct(
            f, collide_mrt3d, tau, solid, u_in,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
        )

        # Force measurement (drag integration)
        fx_p, fy_p, fz_p = drag_pressure_integration(
            f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
        fx_f, fy_f, fz_f = drag_friction_integration(
            f, mesh, dpS, nu, formula='standard')
        cd_p = fx_p
        cd_f = fx_f
        cd_tot = fx_p + fx_f
        cl = fy_p + fy_f

        # Torque: M_z = sum (r × F) ≈ Cl * R_blade (transverse force × lever arm)
        # In lattice units: torque = Cl * dpS * R_blade
        torque = cl * dpS * R_blade
        # Power = torque * omega
        power = torque * omega
        # Cp = power / (0.5 * rho * U^3 * A)
        cp = power / p_ref if p_ref > 0 else 0.0

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        torque_hist.append(torque)
        cp_hist.append(cp)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 2000 == 0:
            n_avg = min(500, len(cd_tot_hist))
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            tq_avg = sum(torque_hist[-n_avg:]) / n_avg
            cp_avg = sum(cp_hist[-n_avg:]) / n_avg
            print(f"{tag} step={step}/{n_steps} angle={angle_deg:.1f}° "
                  f"Cd={at:.4f} Cl={cl_avg:.4f} torque={tq_avg:.2f} "
                  f"Cp={cp_avg:.4f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # --- Post-processing ---
    n_half = len(cp_hist) // 2
    cp_arr = np.array(cp_hist[n_half:])
    cd_arr = np.array(cd_tot_hist[n_half:])
    tq_arr = np.array(torque_hist[n_half:])

    cp_mean = float(np.mean(cp_arr))
    cp_max = float(np.max(cp_arr))
    cd_mean = float(np.mean(cd_arr))
    torque_mean = float(np.mean(tq_arr))

    # Reference: Cp_max ≈ 0.45 for 3-blade turbine at TSR=5 (Betz limit = 0.593)
    cp_ref = 0.45
    err = abs(cp_mean - cp_ref) / cp_ref * 100 if cp_ref > 0 else float('inf')

    print(f"\n{tag} === FINAL ===  Cp_mean={cp_mean:.4f} Cp_max={cp_max:.4f} "
          f"(ref={cp_ref}) err={err:.1f}% Cd_mean={cd_mean:.4f} "
          f"torque_mean={torque_mean:.2f} time={elapsed:.0f}s", flush=True)

    result_dict = {
        "case": "rotating_turbine_Re1000",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "R_blade": R_blade, "n_blades": n_blades,
        "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "TSR": tsr, "omega": omega,
        "n_steps": n_steps,
        "Cp_mean": cp_mean,
        "Cp_max": cp_max,
        "Cd_mean": cd_mean,
        "torque_mean": torque_mean,
        "Cp_ref": cp_ref,
        "error_pct": float(err),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_gradient",
            "lbm_step_correct.lbm_step_correct",
            "drag_pressure.drag_pressure_integration",
            "drag_pressure.drag_friction_integration",
            "postprocess.detect_strouhal",
        ],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_rotating(dev, out)
