"""RANS Spalart-Allmaras benchmark worker — tests SA model on separation flows.

D3Q19 MRT + SA RANS (collide_rans_mrt3d + SASolver) + wallfn log-law.

Usage:
    python rans_sa_worker.py <device_id> <case> <output_path>
    
    case: "cylinder", "sphere", "square_prism", "backward_step"

Output: JSON result dict with Cd (or xr/h for step) and status.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.rans_common import collide_rans_mrt3d
from tensorlbm.rans_ke import SASolver
from tensorlbm.wall_model import wall_function_3d, compute_wall_distance_fmm


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device):
    """3D sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius**2


def build_square_prism_mask(nx, ny, nz, cx, cy, side, device):
    """Square prism extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    half = side / 2.0
    square = (xx >= cx - half) & (xx <= cx + half) & (yy >= cy - half) & (yy <= cy + half)
    return square.unsqueeze(0).expand(nz, ny, nx).clone()


def build_backward_step_mask(nx, ny, nz, step_h, full_H, device):
    """Backward-facing step: solid in lower-left region, then step down.
    
    Geometry: full_H = step_h + channel_h_after
    Solid from x=0 to x=nx*0.2 (inlet region), y=0 to y=step_h.
    After x=nx*0.2, step drops — free channel for y>0 to y=full_H.
    Actually: step region from x_start to end-of-step at some x,
    then after the step, free flow.
    
    Standard setup: solid occupies y=0..step_h for first 20% of domain,
    full channel height full_H = step_h + upper_h.
    """
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    step_x = int(nx * 0.2)  # step edge at 20%
    # Solid before step: y=0..step_h-1
    solid[:, :step_h, :step_x] = True
    # Solid after step: no floor  (free channel)
    # Top wall: y = full_H-1
    solid[:, full_H - 1 :, :] = True
    # Bottom wall after step: y=0 is free
    # Side walls in z (for 3D): z=0 and z=nz-1
    solid[0, :, :] = True
    solid[-1, :, :] = True
    return solid


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def compute_reattachment_length(ux, solid, step_x, ny, step_h):
    """Estimate reattachment length xr/h from the velocity field.
    
    Find the first x > step_x where ux < 0 stops and returns positive
    near the bottom wall (y ~ 1 cell above floor).
    """
    # Look at bottom wall cells after step (y=2, to avoid floor BC effects)
    y_probe = min(2, ny - 2)
    # Get velocity profile along x at y_probe
    ux_profile = ux[0, y_probe, step_x:]  # take first z-slice
    # Find where velocity changes from negative to positive
    if ux_profile.numel() == 0:
        return float("nan")

    ux_np = ux_profile.cpu().numpy()
    # Simple sign-change detection
    for i in range(len(ux_np) - 1):
        if ux_np[i] <= 0 and ux_np[i + 1] > 0:
            xr_cells = i + 0.5  # interpolate
            return xr_cells / step_h  # xr/h
    # If no sign change, return domain length
    return float(ux_profile.numel()) / step_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device_id = int(sys.argv[1])
    case = sys.argv[2]
    output_path = sys.argv[3]

    # ===================================================================
    # Case-specific parameters
    # ===================================================================
    if case == "cylinder":
        nx, ny, nz = 200, 80, 4
        diameter = 24.0
        radius = diameter / 2.0
        u_in = 0.08
        re = 200.0
        ref_cd = 1.30
        cx_geom = nx * 0.25
        cy_geom = ny * 0.5
        cz_geom = nz * 0.5
        A_frontal = diameter * nz
        build_mask = lambda: build_cylinder_mask(nx, ny, nz, cx_geom, cy_geom, radius, device)

    elif case == "sphere":
        nx, ny, nz = 200, 100, 100
        diameter = 40.0
        radius = diameter / 2.0
        u_in = 0.08
        re = 1000.0
        ref_cd = 0.47
        cx_geom = nx * 0.25
        cy_geom = ny * 0.5
        cz_geom = nz * 0.5
        A_frontal = math.pi * radius**2
        build_mask = lambda: build_sphere_mask(nx, ny, nz, cx_geom, cy_geom, cz_geom, radius, device)

    elif case == "square_prism":
        nx, ny, nz = 200, 80, 4
        diameter = 30.0
        side = 30.0
        u_in = 0.08
        re = 22000.0
        ref_cd = 2.1
        cx_geom = nx * 0.25
        cy_geom = ny * 0.5
        A_frontal = side * nz
        build_mask = lambda: build_square_prism_mask(nx, ny, nz, cx_geom, cy_geom, side, device)

    elif case == "backward_step":
        nx, ny, nz = 200, 80, 4
        step_h = 20
        full_H = 60  # step_h + channel_h_after = 20 + 40 = 60
        u_in = 0.08
        re = 5000.0
        diameter = step_h  # use step_h as characteristic length
        ref_cd = float("nan")  # no Cd for step; we measure xr/h
        A_frontal = step_h * nz  # frontal for reference area
        build_mask = lambda: build_backward_step_mask(nx, ny, nz, step_h, full_H, device)

    else:
        raise ValueError(f"Unknown case: {case}")

    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} SA-RANS {case} Re={int(re)}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} nu={nu:.6e} tau={tau:.6f}",
          flush=True)

    t0 = time.time()

    # Build geometry
    solid = build_mask()
    dyn_p = 0.5 * 1.0 * u_in**2 * A_frontal

    # Initialize flow
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    # Compute wall distance for SA
    print(f"{tag} computing wall distance...", flush=True)
    t_wd = time.time()
    wall_dist = compute_wall_distance_fmm(solid, max_iter=200)
    print(f"{tag} wall distance done ({time.time() - t_wd:.1f}s)", flush=True)

    # Initialize SA model
    sa = SASolver(nu=nu, nu_t_max=0.5)
    sa.initialize(ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), nu_tilde_0=10.0 * nu)
    # Set nu_tilde=0 at solid walls
    sa._nu_tilde[solid] = 0.0

    n_steps = 2000
    warmup = 300

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    cd_hist = []
    xr_hist = []
    status = "running"

    for step in range(1, n_steps + 1):
        # 1. Compute nu_t from SA model
        nu_t = sa.compute_nu_t(mask=solid)

        # 2. Collision: MRT + SA RANS
        f = collide_rans_mrt3d(f, tau=tau, nu_t=nu_t)

        # 3. Stream
        f = stream3d(f)

        # 4. Wall function (log-law body force + drag computation)
        f, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=0.5, wall_law="log"
        )

        # 5. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 6. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # 7. Advance SA model
        _, ux, uy, uz = macroscopic3d(f)
        nu_t = sa.step(ux, uy, uz, wall_dist, mask=solid)

        # Compute Cd
        cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
        cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
        cd_total = cd_fric + cd_pres

        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        # For backward step, also track reattachment
        if case == "backward_step" and step > warmup:
            step_x = int(nx * 0.2)
            xrh = compute_reattachment_length(ux, solid, step_x, ny, step_h)
            if math.isfinite(xrh):
                xr_hist.append(xrh)

        # Check divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            status = "diverged"
            break

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            elapsed = time.time() - t0
            extra = ""
            if case == "backward_step" and xr_hist:
                xr_avg = sum(xr_hist) / len(xr_hist)
                extra = f" xr/h={xr_avg:.2f}"
            print(f"{tag} step={step} Cd={cd_total:.4f} Cd_avg={cd_avg:.4f}{extra} "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Final statistics
    if cd_hist:
        cd_mean = sum(cd_hist) / len(cd_hist)
        n = len(cd_hist)
        cd_std = math.sqrt(sum((c - cd_mean) ** 2 for c in cd_hist) / max(n - 1, 1)) if n > 1 else 0.0
    else:
        cd_mean = float("nan")
        cd_std = 0.0

    err_pct = (
        abs(cd_mean - ref_cd) / ref_cd * 100
        if math.isfinite(ref_cd) and ref_cd > 0 and math.isfinite(cd_mean)
        else float("nan")
    )

    xr_mean = float("nan")
    xr_std = 0.0
    if xr_hist:
        xr_mean = sum(xr_hist) / len(xr_hist)
        n_xr = len(xr_hist)
        xr_std = math.sqrt(sum((x - xr_mean) ** 2 for x in xr_hist) / max(n_xr - 1, 1)) if n_xr > 1 else 0.0

    if status != "diverged":
        status = "ok"

    result = {
        "case": f"{case}_Re{int(re)}",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": "MRT+SA-RANS",
        "boundary": "wall_function_3d(log)+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter if case != "backward_step" else step_h,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "cd_samples": len(cd_hist),
        "xr_h_mean": xr_mean,
        "xr_h_std": xr_std,
        "finite": status == "ok",
        "status": status,
        "elapsed_s": elapsed,
    }

    print(f"{tag} DONE Cd={cd_mean:.4f}±{cd_std:.4f} (ref={ref_cd}) "
          f"err={err_pct if math.isfinite(err_pct) else 'N/A'}% "
          f"xr/h={xr_mean:.2f} status={status} time={elapsed:.0f}s",
          flush=True)

    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
