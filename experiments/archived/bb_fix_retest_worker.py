#!/usr/bin/env python3
"""BB fix retest via common interface — cylinder/sphere/SUBOFF/NACA. SDAA 8-11.

ALL retests use common interface ONLY:
- lbm_step_correct() (has BB fix with f_pre)
- drag_pressure_integration + drag_friction_integration
- SurfaceMesh.from_cylinder/from_sphere/from_suboff/from_naca
- far_field_bc_3d, detect_strouhal
NO custom code. Common modules only.

Universal flow:
  1. solid = build_geometry()
  2. near = get_near_wall_3d(solid)
  3. mesh = SurfaceMesh.from_xxx(solid, near, ...)
  4. for step: f = lbm_step_correct(f, solid, collide_fn, tau, far_field_bc_fn, u_in, step, ...)
  5. Cd_p,_,_ = drag_pressure_integration(f, mesh, dpS, extrap='quadratic')
  6. Cd_f,_,_ = drag_friction_integration(f, mesh, dpS, nu)
  7. St = detect_strouhal(cl_hist, dt, D, u_in)

TEST 1: Cylinder Re=200 D=48 (SDAA:8)   — from_cylinder, 5000 steps
TEST 2: Sphere   Re=100 D=40 (SDAA:9)   — from_sphere,   3000 steps
TEST 3: SUBOFF   Re=1000 L=80 (SDAA:10) — from_suboff,   5000 steps
TEST 4: NACA0012 Re=1000 6L  (SDAA:11)  — from_naca,    10000 steps

Usage:
  PYTHONPATH=src python bb_fix_retest_worker.py <test_name> <device_id> <output_json>
  test_name: cylinder | sphere | suboff | naca
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.postprocess import detect_strouhal
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device):
    """Vectorized sphere mask: (i-cx)^2+(j-cy)^2+(k-cz)^2 < R^2."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < R ** 2


def build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device):
    """Build NACA 0012 solid mask (2D extruded in z, nz=4).

    Standard NACA 4-digit symmetric thickness:
      yt = 5*t*(0.2969*sqrt(xc) - 0.1260*xc - 0.3516*xc^2 + 0.2843*xc^3 - 0.1015*xc^4)
    with t=0.12 for NACA 0012.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xc = (xx - x_le) / chord
    xc = xc.clamp(min=1e-12, max=1.0)
    t = 0.12
    half_t = 5.0 * t * (
        0.2969 * torch.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    ) * chord
    in_chord = (xx >= x_le) & (xx <= x_le + chord)
    in_profile = (yy >= y_c - half_t) & (yy <= y_c + half_t)
    solid = in_chord & in_profile
    # Close LE/TE gaps
    le_col = int(x_le)
    te_col = int(x_le + chord)
    yc_int = int(y_c)
    solid[:, :, le_col] |= (yy[:, :, le_col] == yc_int)
    solid[:, :, te_col] |= (yy[:, :, te_col] == yc_int)
    return solid


# ---------------------------------------------------------------------------
# Common simulation runner using lbm_step_correct (BB fix)
# ---------------------------------------------------------------------------

def run_simulation(
    device_id, tag, solid, mesh, dpS, nu, u_in, tau, cs_smag,
    n_steps, warmup, ref_cd, ref_name, cl_hist_for_st=None,
    st_params=None, output_path=None,
):
    """Run LBM simulation using lbm_step_correct (BB fix) and compute drag.

    Uses common interface ONLY:
      - lbm_step_correct() for the main loop (has BB fix with f_pre)
      - drag_pressure_integration(extrap='quadratic')
      - drag_friction_integration(formula='standard')
      - detect_strouhal() for Strouhal number
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    nz, ny, nx = solid.shape

    t0 = time.time()

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time()-t0:.1f}s), initial_mass={initial_mass}",
          flush=True)

    # Histories
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []

    for step in range(1, n_steps + 1):
        # --- COMMON INTERFACE: lbm_step_correct (has BB fix with f_pre) ---
        f = lbm_step_correct(
            f,
            collide_fn=collide_smagorinsky_mrt3d,
            tau=tau,
            solid=solid,
            u_in=u_in,
            far_field_bc_fn=far_field_bc_3d,
            correct_mass_fn=correct_mass3d,
            target_mass=initial_mass,
            step=step,
            mass_interval=200,
            C_s=cs_smag,
        )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # --- COMMON INTERFACE: drag_pressure_integration (extrap='quadratic') ---
        cdp_x, cdp_y, _ = drag_pressure_integration(
            f, mesh, dpS, extrap='quadratic'
        )
        # --- COMMON INTERFACE: drag_friction_integration ---
        cdf_x, cdf_y, _ = drag_friction_integration(f, mesh, dpS, nu)

        cd_tot = cdp_x + cdf_x
        cl = cdp_y + cdf_y

        if step > warmup:
            if math.isfinite(cd_tot):
                cd_p_hist.append(cdp_x)
                cd_f_hist.append(cdf_x)
                cd_tot_hist.append(cd_tot)
            if math.isfinite(cl):
                cl_hist.append(cl)

        if step % 500 == 0 or step == n_steps:
            elapsed = time.time() - t0
            n_avg = min(500, len(cd_tot_hist))
            if n_avg > 0:
                cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
                cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
                cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
                cl_avg = sum(cl_hist[-n_avg:]) / n_avg
                print(
                    f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                    f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f} ({elapsed:.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t0

    # Final averages (last 50% of post-warmup samples)
    n_final = max(1, len(cd_tot_hist) // 2)
    cd_p_mean = sum(cd_p_hist[-n_final:]) / n_final if cd_p_hist else float("nan")
    cd_f_mean = sum(cd_f_hist[-n_final:]) / n_final if cd_f_hist else float("nan")
    cd_tot_mean = sum(cd_tot_hist[-n_final:]) / n_final if cd_tot_hist else float("nan")

    err_pct = abs(cd_tot_mean - ref_cd) / ref_cd * 100 if (
        ref_cd > 0 and math.isfinite(cd_tot_mean)
    ) else float("nan")

    # Strouhal detection (if applicable)
    st_result = None
    if st_params is not None and len(cl_hist) > 100:
        st_result = detect_strouhal(
            cl_hist,
            sample_rate=1.0,
            u_ref=st_params['u_ref'],
            length_ref=st_params['length_ref'],
            method='auto',
            min_cycles=3,
        )

    result = {
        "case": tag,
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "Cs": cs_smag,
        "n_steps": n_steps,
        "warmup": warmup,
        "n_samples": len(cd_tot_hist),
        "Cd_pressure": cd_p_mean,
        "Cd_friction": cd_f_mean,
        "Cd_total": cd_tot_mean,
        "Cd_ref": ref_cd,
        "ref_name": ref_name,
        "Cd_err_pct": err_pct,
        "Strouhal": st_result,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
        "bb_fix": True,
        "step_fn": "lbm_step_correct",
        "extrap": "quadratic",
        "friction_formula": "standard",
    }

    print(
        f"{tag} DONE Cd_p={cd_p_mean:.4f} Cd_f={cd_f_mean:.4f} "
        f"Cd_tot={cd_tot_mean:.4f} (ref={ref_cd:.4f}) err={err_pct:.1f}% "
        f"St={st_result} time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# TEST 1: Cylinder Re=200 D=48 (SDAA:8)
# ---------------------------------------------------------------------------

def run_cylinder(device_id, output_path):
    """Cylinder Re=200, D=48, 5000 steps — from_cylinder, far_field_bc."""
    nx, ny, nz = 600, 200, 4
    diameter = 48.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re  # 0.0192
    tau = 3.0 * nu + 0.5       # 0.5576
    cs_smag = 0.05
    n_steps = 5000
    warmup = 1000

    cx_c = nx * 0.25
    cy_c = ny * 0.5

    A_frontal = diameter * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal
    ref_cd = 1.30
    ref_name = "cylinder Re=200 (experimental ~1.30-1.33)"

    tag = f"[BBfix SDAA:{device_id} Cylinder Re=200 D=48]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} "
          f"nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} n_steps={n_steps}", flush=True)

    # 1. Build geometry
    solid = build_cylinder_mask(nx, ny, nz, cx_c, cy_c, radius, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # 2. Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # 3. Surface mesh (from_cylinder)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_c, cy_c, radius, axis='z')
    print(f"{tag} mesh built ({time.time()-time.time():.1f}s)", flush=True)

    # 4-7. Run simulation
    st_params = {'u_ref': u_in, 'length_ref': diameter}
    return run_simulation(
        device_id, tag, solid, mesh, dpS, nu, u_in, tau, cs_smag,
        n_steps, warmup, ref_cd, ref_name, st_params=st_params,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# TEST 2: Sphere Re=100 D=40 (SDAA:9)
# ---------------------------------------------------------------------------

def run_sphere(device_id, output_path):
    """Sphere Re=100, D=40, 3000 steps — from_sphere, far_field_bc."""
    D = 40.0
    R = D / 2.0
    nx, ny, nz = 120, 120, 120
    u_in = 0.08
    Re = 100.0
    nu = u_in * D / Re  # 0.032
    tau = 3.0 * nu + 0.5  # 0.596
    cs_smag = 0.05
    n_steps = 3000
    warmup = 600

    cx = nx * 0.25
    cy = ny * 0.5
    cz = nz * 0.5

    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2
    ref_cd = 1.09
    ref_name = "sphere Re=100 (Henderson empirical)"

    tag = f"[BBfix SDAA:{device_id} Sphere Re=100 D=40]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} R={R} u_in={u_in} "
          f"nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} n_steps={n_steps}", flush=True)

    # 1. Build geometry
    solid = build_sphere_solid(nx, ny, nz, cx, cy, cz, R, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # 2. Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # 3. Surface mesh (from_sphere)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"{tag} mesh built", flush=True)

    # 4-7. Run simulation
    return run_simulation(
        device_id, tag, solid, mesh, dpS, nu, u_in, tau, cs_smag,
        n_steps, warmup, ref_cd, ref_name,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# TEST 3: SUBOFF Re=1000 L=80 (SDAA:10)
# ---------------------------------------------------------------------------

def run_suboff(device_id, output_path):
    """SUBOFF Re=1000, L=80, 5000 steps — from_suboff, far_field_bc."""
    L = 80
    nx, ny, nz = 200, 80, 80
    u_in = 0.06
    Re = 1000.0
    nu = u_in * L / Re  # 0.0048
    tau = 3.0 * nu + 0.5  # 0.5144
    cs_smag = 0.05
    n_steps = 5000
    warmup = 1000

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5

    # dpS uses wetted area (pi * D * L) — consistent with previous SUBOFF tests
    dpS = 0.5 * u_in ** 2 * math.pi * D * L
    # Blasius laminar Cf = 1.328/sqrt(Re) = 0.042
    ref_cd = 1.328 / math.sqrt(Re)
    ref_name = f"Blasius laminar Cf=1.328/sqrt(Re)={ref_cd:.4f}"

    tag = f"[BBfix SDAA:{device_id} SUBOFF Re=1000 L=80]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} nx={nx} ny={ny} nz={nz} L={L} R_max={radius:.4f} D={D:.4f} "
          f"u_in={u_in} nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} "
          f"n_steps={n_steps} dpS={dpS:.6e}", flush=True)

    # 1. Build geometry
    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config, device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    # 2. Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # 3. Surface mesh (from_suboff)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
    print(f"{tag} mesh built", flush=True)

    # 4-7. Run simulation
    return run_simulation(
        device_id, tag, solid, mesh, dpS, nu, u_in, tau, cs_smag,
        n_steps, warmup, ref_cd, ref_name,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# TEST 4: NACA 0012 Re=1000 6L (SDAA:11)
# ---------------------------------------------------------------------------

def run_naca(device_id, output_path):
    """NACA 0012 Re=1000, 6L domain, 10000 steps — from_naca, far_field_bc."""
    chord = 100
    nx, ny, nz = 600, 300, 4  # 6L domain (6 chord x 3 chord)
    u_in = 0.05
    Re = 1000.0
    nu = u_in * chord / Re  # 0.005
    tau = 3.0 * nu + 0.5    # 0.515
    cs_smag = 0.05
    n_steps = 10000
    warmup = 2000

    x_le = int(nx * 0.25)  # 1.5 chords from inlet
    y_c = ny // 2           # centered

    dpS = 0.5 * u_in ** 2 * chord * nz
    ref_cd = 0.05
    ref_name = "NACA 0012 Re=1000 (laminar friction ref ~0.05)"

    tag = f"[BBfix SDAA:{device_id} NACA0012 Re=1000 6L]"
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    print(f"{tag} chord={chord} nx={nx} ny={ny} nz={nz} u_in={u_in} "
          f"nu={nu:.6f} tau={tau:.6f} Cs={cs_smag} n_steps={n_steps}", flush=True)

    # 1. Build geometry (NACA 0012, symmetric, 0° AoA)
    solid = build_naca0012_mask(nx, ny, nz, x_le, y_c, chord, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # 2. Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # 3. Surface mesh (from_naca with m=0 for NACA 0012 symmetric)
    mesh = SurfaceMesh.from_naca(solid, near, x_le, y_c, chord, m=0.0, p=0.40, t=0.12)
    print(f"{tag} mesh built", flush=True)

    # 4-7. Run simulation
    return run_simulation(
        device_id, tag, solid, mesh, dpS, nu, u_in, tau, cs_smag,
        n_steps, warmup, ref_cd, ref_name,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: PYTHONPATH=src python bb_fix_retest_worker.py "
              "<test_name> <device_id> <output_json>")
        print("  test_name: cylinder | sphere | suboff | naca")
        sys.exit(1)

    test_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if test_name == "cylinder":
        run_cylinder(device_id, output_path)
    elif test_name == "sphere":
        run_sphere(device_id, output_path)
    elif test_name == "suboff":
        run_suboff(device_id, output_path)
    elif test_name == "naca":
        run_naca(device_id, output_path)
    else:
        print(f"Unknown test: {test_name}")
        print("  test_name: cylinder | sphere | suboff | naca")
        sys.exit(1)


if __name__ == "__main__":
    main()
