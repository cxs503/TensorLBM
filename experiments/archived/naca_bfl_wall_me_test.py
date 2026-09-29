"""NACA 0012 BFL + wall-surface ME test.

Compares wall-surface momentum exchange with pressure integration
on a NACA 0012 airfoil.

Main loop (verified-correct):
  collide → NoDynamics → stream → BFL(after stream) → compute drag → far_field_bc → correct_mass

Drag computed AFTER BFL, BEFORE far_field_bc.

Two ME variants compared:
  - Cd_me_wall: drag_momentum_exchange_wall (drag_momentum_wall.py) — task-specified
  - Cd_me_bfl:  drag_momentum_exchange_bfl  (wall_surface_bfl.py) — correct Ladd formula

Pressure integration:
  - Cd_p: drag_pressure_integration (SurfaceMesh.from_gradient)
  - Cd_f: drag_friction_integration
  - Cd_pf = Cd_p + Cd_f
"""
import sys
import math
import time
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.wall_surface_bfl import bouzidi_bounce_back_wallsurface, drag_momentum_exchange_bfl
from tensorlbm.solver3d import collide_bgk3d, collide_mrt3d, stream3d, correct_mass3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
    get_near_wall_2d,
)
from tensorlbm.drag_momentum_wall import drag_momentum_exchange_wall


# ============================================================================
# NACA 0012 geometry
# ============================================================================

def naca_yt(xc):
    """NACA 0012 half-thickness. xc ∈ [0, 1].

    y_t = 0.6 * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x² + 0.2843*x³ - 0.1015*x⁴)
    """
    xc = np.asarray(xc, dtype=np.float64)
    return 0.6 * (
        0.2969 * np.sqrt(np.maximum(xc, 0.0))
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    )


def is_inside_naca_np(x, y, x_le, y_c, chord):
    """Vectorized inside test for NACA 0012 airfoil."""
    xc = (x - x_le) / chord
    yt = chord * naca_yt(xc)
    y_upper = y_c + yt
    y_lower = y_c - yt
    return (xc >= 0.0) & (xc <= 1.0) & (y >= y_lower) & (y <= y_upper)


def build_naca_solid(nx, ny, nz, x_le, y_c, chord, device):
    """Build NACA 0012 solid mask (2D extruded in z)."""
    xx, yy = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        indexing="xy",
    )
    inside = is_inside_naca_np(xx, yy, x_le, y_c, chord)
    le_col = int(round(x_le))
    te_col = int(round(x_le + chord))
    yc_int = int(round(y_c))
    if 0 <= le_col < nx and 0 <= yc_int < ny:
        inside[yc_int, le_col] = True
    if 0 <= te_col < nx and 0 <= yc_int < ny:
        inside[yc_int, te_col] = True
    solid = torch.from_numpy(inside).to(device).unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


# ============================================================================
# BFL q-value computation for NACA 0012
# ============================================================================

def compute_q_naca(solid, x_le, y_c, chord, device):
    """Compute BFL q-values for NACA 0012 airfoil using vectorized bisection.

    For each near-wall fluid cell, for each lattice direction d:
      - Find intersection of the line (cell + t*c_d) with the NACA surface
      - q = t (fractional distance, 0=fluid cell, 1=solid cell)
    """
    c = C.to(device).float()
    nz, ny, nx = solid.shape

    fluid_boundary_mask = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    for d in range(19):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        dcz = float(c[d, 2].item())

        if dcx == 0.0 and dcy == 0.0 and dcz == 0.0:
            continue
        if dcx == 0.0 and dcy == 0.0:
            continue

        nb_solid = torch.roll(
            solid, shifts=(-int(dcz), -int(dcy), -int(dcx)), dims=(0, 1, 2)
        )
        boundary = (~solid) & nb_solid

        if not boundary.any():
            continue

        boundary_2d = boundary[0]
        if not boundary_2d.any():
            continue

        y_idx, x_idx = np.where(boundary_2d.cpu().numpy())
        x_f = x_idx.astype(np.float64)
        y_f = y_idx.astype(np.float64)

        lo = np.zeros(len(x_f), dtype=np.float64)
        hi = np.ones(len(x_f), dtype=np.float64)

        for _ in range(60):
            mid = (lo + hi) / 2.0
            x_mid = x_f + mid * dcx
            y_mid = y_f + mid * dcy
            inside = is_inside_naca_np(x_mid, y_mid, x_le, y_c, chord)
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)

        q_vals = np.clip((lo + hi) / 2.0, 0.01, 0.99).astype(np.float32)

        q_2d = np.zeros((ny, nx), dtype=np.float32)
        q_2d[y_idx, x_idx] = q_vals
        q_torch = torch.from_numpy(q_2d).to(device)

        for z in range(nz):
            fluid_boundary_mask[d, z] = boundary_2d
            q_field[d, z] = torch.where(boundary_2d, q_torch, q_field[d, z])

    return fluid_boundary_mask, q_field


# ============================================================================
# Main
# ============================================================================

def main():
    device_id = 4
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    chord = 100.0
    Re = 1e5
    u_in = 0.05
    nu = u_in * chord / Re
    tau = 3.0 * nu + 0.5
    nx, ny, nz = 600, 200, 4
    n_steps = 3000
    win = 300
    ref_cd = 0.012

    x_le = nx * 0.25
    y_c = ny / 2.0

    ref_area = chord * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * ref_area

    tag = f"[SDAA:{device_id}] NACA0012"
    print(f"{tag} nx={nx} ny={ny} nz={nz} chord={chord} Re={Re:.0e} u_in={u_in}", flush=True)
    print(f"  nu={nu:.6e} tau={tau:.6f} x_le={x_le} y_c={y_c}", flush=True)
    print(f"  ref_area={ref_area:.1f} dpS={dpS:.6f} ref_Cd={ref_cd}", flush=True)

    # Geometry diagnostics
    for xc_frac in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        yt = chord * naca_yt(np.array([xc_frac]))[0]
        print(f"  NACA yt(xc={xc_frac:.1f}) = {yt:.4f} lu", flush=True)

    # Build solid mask
    solid = build_naca_solid(nx, ny, nz, x_le, y_c, chord, device)
    n_solid = int(solid.sum().item())
    print(f"  solid cells: {n_solid} ({n_solid // nz} per z-layer)", flush=True)

    # Compute BFL q-values
    t_q = time.time()
    fbm, qf = compute_q_naca(solid, x_le, y_c, chord, device)
    n_links = int(fbm.sum().item())
    q_min = float(qf[fbm].min().item()) if n_links > 0 else 0.0
    q_max = float(qf[fbm].max().item()) if n_links > 0 else 0.0
    q_mean = float(qf[fbm].mean().item()) if n_links > 0 else 0.0
    print(f"  BFL q-field: {n_links} links, q=[{q_min:.4f}, {q_max:.4f}], mean={q_mean:.4f} ({time.time()-t_q:.1f}s)", flush=True)

    # Surface mesh for pressure integration
    near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"  near-wall cells: {n_near} ({n_near // nz} per z-layer)", flush=True)
    mesh = SurfaceMesh.from_gradient(solid, near)

    # Initialize flow field
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(rho0.sum().item())

    f_eq_solid = equilibrium3d(
        rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0),
        device=device,
    )

    # Main loop
    t0 = time.time()
    cd_me_bfl_hist = []
    cd_me_wall_hist = []
    cd_p_hist = []
    cd_f_hist = []
    diverged = False

    for step in range(1, n_steps + 1):
        # 1. Collide (MRT for stability at low tau)
        f = collide_mrt3d(f, tau=tau)
        # 2. NoDynamics: reset solid to equilibrium at rest
        f[:, solid] = f_eq_solid[:, solid]
        # 3. Save pre-stream (post-collision, post-NoDynamics)
        f_pre = f.clone()
        # 4. Stream
        f = stream3d(f)
        # 5. BFL (after stream) — use wall_surface_bfl (correct formula)
        f = bouzidi_bounce_back_wallsurface(f, f_pre, fbm, qf)
        # 6. Compute drag (AFTER BFL, BEFORE far_field_bc)
        #    Correct ME: F = (fp_d + f_bc) * c_d  [wall_surface_bfl.py]
        cd_me_bfl = drag_momentum_exchange_bfl(f, f_pre, fbm, qf, dpS)
        #    Task-specified ME: F = 2 * f_d_wall * c_d  [drag_momentum_wall.py]
        cd_me_wall = drag_momentum_exchange_wall(f, f_pre, fbm, qf, dpS)
        #    Pressure + friction integration
        cd_p, _ = drag_pressure_integration(f, mesh, dpS)
        cd_f = drag_friction_integration(f, mesh, dpS, nu)
        # 7. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=None)
        # 8. Mass correction
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        # Collect drag (last win steps)
        if step > n_steps - win:
            if math.isfinite(cd_me_bfl):
                cd_me_bfl_hist.append(cd_me_bfl)
            if math.isfinite(cd_me_wall):
                cd_me_wall_hist.append(cd_me_wall)
            if math.isfinite(cd_p):
                cd_p_hist.append(cd_p)
            if math.isfinite(cd_f):
                cd_f_hist.append(cd_f)

        # Progress report
        if step % 300 == 0 or step == n_steps:
            def avg(lst):
                return sum(lst) / max(len(lst), 1) if lst else float("nan")
            elapsed = time.time() - t0
            print(
                f"{tag} step={step:4d} "
                f"Cd_me_bfl={avg(cd_me_bfl_hist):.6f} "
                f"Cd_me_wall={avg(cd_me_wall_hist):.6f} "
                f"Cd_p={avg(cd_p_hist):.6f} "
                f"Cd_f={avg(cd_f_hist):.6f} "
                f"Cd_pf={avg(cd_p_hist)+avg(cd_f_hist) if cd_p_hist and cd_f_hist else float('nan'):.6f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    # Final results
    def avg(lst):
        return sum(lst) / max(len(lst), 1) if lst else float("nan")

    cd_me_bfl_final = avg(cd_me_bfl_hist)
    cd_me_wall_final = avg(cd_me_wall_hist)
    cd_p_final = avg(cd_p_hist)
    cd_f_final = avg(cd_f_hist)
    cd_pf_final = cd_p_final + cd_f_final if math.isfinite(cd_p_final) and math.isfinite(cd_f_final) else float("nan")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"NACA 0012 BFL + Wall-Surface ME Results")
    print(f"{'=' * 70}")
    print(f"  Cd_me_bfl  (correct Ladd ME)   = {cd_me_bfl_final:.6f}")
    print(f"  Cd_me_wall (task-specified ME)  = {cd_me_wall_final:.6f}")
    print(f"  Cd_p       (pressure integ)     = {cd_p_final:.6f}")
    print(f"  Cd_f       (friction integ)     = {cd_f_final:.6f}")
    print(f"  Cd_pf      (p+f integration)    = {cd_pf_final:.6f}")
    print(f"  Cd_ref     (laminar Re=1e5)     = {ref_cd:.6f}")
    if math.isfinite(cd_pf_final) and math.isfinite(cd_me_bfl_final):
        diff = abs(cd_me_bfl_final - cd_pf_final)
        diff_pct = diff / max(abs(cd_pf_final), 1e-10) * 100
        print(f"  ME_bfl vs PF diff = {diff:.6f} ({diff_pct:.2f}%)")
    if math.isfinite(cd_me_bfl_final):
        print(f"  ME_bfl vs ref err = {abs(cd_me_bfl_final - ref_cd) / ref_cd * 100:.2f}%")
    if math.isfinite(cd_pf_final):
        print(f"  PF vs ref err     = {abs(cd_pf_final - ref_cd) / ref_cd * 100:.2f}%")
    print(f"  n_samples = {len(cd_me_bfl_hist)}")
    print(f"  elapsed = {elapsed:.0f}s")
    print(f"  diverged = {diverged}")

    result = {
        "case": "NACA0012_BFL_wall_ME",
        "device_id": device_id,
        "grid": f"{nx}x{ny}x{nz}",
        "chord": chord,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "win": win,
        "n_samples": len(cd_me_bfl_hist),
        "n_solid": n_solid,
        "n_boundary_links": n_links,
        "q_min": q_min,
        "q_max": q_max,
        "q_mean": q_mean,
        "Cd_me_bfl": cd_me_bfl_final,
        "Cd_me_wall": cd_me_wall_final,
        "Cd_p": cd_p_final,
        "Cd_f": cd_f_final,
        "Cd_pf": cd_pf_final,
        "Cd_ref": ref_cd,
        "ME_bfl_vs_PF_diff_pct": (
            abs(cd_me_bfl_final - cd_pf_final) / max(abs(cd_pf_final), 1e-10) * 100
            if math.isfinite(cd_pf_final) and math.isfinite(cd_me_bfl_final)
            else None
        ),
        "ME_bfl_vs_ref_err_pct": (
            abs(cd_me_bfl_final - ref_cd) / ref_cd * 100
            if math.isfinite(cd_me_bfl_final) else None
        ),
        "PF_vs_ref_err_pct": (
            abs(cd_pf_final - ref_cd) / ref_cd * 100
            if math.isfinite(cd_pf_final) else None
        ),
        "elapsed_s": elapsed,
        "diverged": diverged,
        "finite": bool(torch.isfinite(f).all().item()) if not diverged else False,
    }

    out_path = Path(f"/root/TensorLBM_dev/naca_bfl_wall_me_results_sdaa{device_id}.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
