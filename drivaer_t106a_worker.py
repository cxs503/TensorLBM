#!/usr/bin/env python3
"""DrivAer simplified + T106A cascade benchmarks — unified pressure integration.

Both benchmarks use:
  - SurfaceMesh.from_gradient  (generic normal from solid mask gradient)
  - Verified main loop: NoDynamics + half-way BB + far_field_bc_3d
  - MRT + Smagorinsky LES (Cs=0.05)
  - drag_pressure_integration + drag_friction_integration (unified)

BENCHMARK 1: DrivAer simplified car body (Re=1000)
  - Geometry: rectangular box L=60, W=20, H=15 with rounded front + slanted rear
  - Grid: nx=300, ny=100, nz=80
  - u_in=0.05, tau=0.509, 10000 steps
  - Reference: Cd ≈ 0.3 (experimental, simplified)
  - Measure: Cd, Cl

BENCHMARK 2: T106A turbine cascade (Re=1000, simplified 2D)
  - Geometry: simplified cambered curved blade with stagger
  - Grid: nx=400, ny=200, nz=4
  - u_in=0.05, tau=0.515, 10000 steps
  - Reference: pressure loss coefficient ≈ 0.05
  - Measure: pressure loss, turning angle

Usage:
  python drivaer_t106a_worker.py <benchmark> <device_id> <output_path>
  benchmark: drivaer | t106a
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)


# ---------------------------------------------------------------------------
# DrivAer simplified geometry: rectangular box with rounded front + slanted rear
# ---------------------------------------------------------------------------
def build_drivaer_solid(nx, ny, nz, L, W, H, slant_deg, x_start, cy, cz, device):
    """Build DrivAer simplified car body solid mask (nz, ny, nx).

    The body is a rectangular box of length L, width W, height H with:
      - Rounded front: the front quarter has a smooth elliptical top transition
        from y_base to y_top (reduces front drag vs flat Ahmed front)
      - Slanted rear: the rear portion has a slanted top (Ahmed-like wake)

    Coordinate convention: x=streamwise, y=vertical, z=spanwise.
    """
    slant_rad = math.radians(slant_deg)
    L_front = L * 0.25          # front rounding portion
    L_slant = L * 0.30          # rear slant portion
    L_flat = L - L_front - L_slant  # middle flat portion
    y_base = cy - H / 2.0       # bottom of body
    y_top = cy + H / 2.0        # top of flat portion
    z_min = cz - W / 2.0
    z_max = cz + W / 2.0
    x_front_end = x_start + L_front
    x_slant_start = x_start + L_front + L_flat
    x_end = x_start + L

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # x within body extent
    in_x = (xx >= x_start) & (xx <= x_end)
    # z within width
    in_z = (zz >= z_min) & (zz <= z_max)

    # y top depends on x position:
    # Front region: quarter-ellipse from y_base to y_top
    #   y_top_local = y_base + (y_top - y_base) * sqrt(1 - (1 - t)^2)
    #   where t = (x - x_start) / L_front, t in [0, 1]
    # Flat region: y_top
    # Slant region: y_top - (x - x_slant_start) * tan(slant)
    t_front = (xx - x_start) / L_front  # 0 at front, 1 at front_end
    # Quarter-ellipse: smooth transition from y_base to y_top
    y_front = y_base + (y_top - y_base) * torch.sqrt(
        torch.clamp(1.0 - (1.0 - t_front) ** 2, min=0.0)
    )
    y_flat = torch.full_like(xx, y_top)
    y_slant = y_top - (xx - x_slant_start) * math.tan(slant_rad)

    y_top_local = torch.where(
        xx <= x_front_end,
        y_front,
        torch.where(xx <= x_slant_start, y_flat, y_slant),
    )
    in_y = (yy >= y_base) & (yy <= y_top_local)

    solid = in_x & in_y & in_z
    return solid


# ---------------------------------------------------------------------------
# T106A simplified blade geometry: cambered airfoil with stagger
# ---------------------------------------------------------------------------
def naca_half_thickness(x_norm, t_max=0.12):
    """NACA 4-digit half-thickness distribution (normalised by chord).

    y_t = (t_max/0.2) * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2
                          + 0.2843*x^3 - 0.1015*x^4)
    """
    a = x_norm.clamp(min=1e-12, max=1.0)
    return (t_max / 0.2) * (
        0.2969 * torch.sqrt(a)
        - 0.1260 * a
        - 0.3516 * a * a
        + 0.2843 * a * a * a
        - 0.1015 * a * a * a * a
    )


def build_t106a_solid(nx, ny, nz, chord, camber, t_max, stagger_deg,
                       x_le, y_le, device):
    """Build simplified T106A turbine cascade blade solid mask (nz, ny, nx).

    The blade is a cambered airfoil in the x-y plane, extruded in z.
    The camber line is a parabolic arc; thickness is NACA 4-digit.
    The blade is rotated by the stagger angle.

    Args:
        chord: Chord length in lattice units.
        camber: Max camber in lattice units (at mid-chord).
        t_max: Max thickness / chord fraction (e.g. 0.12).
        stagger_deg: Stagger angle in degrees.
        x_le, y_le: Leading edge position in domain.
    """
    stagger_rad = math.radians(stagger_deg)
    cos_s = math.cos(stagger_rad)
    sin_s = math.sin(stagger_rad)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Transform from domain (x, y) to blade local (xi, eta)
    # Forward:  x = x_le + xi*cos - eta*sin,  y = y_le + xi*sin + eta*cos
    # Inverse:  xi = (x-x_le)*cos + (y-y_le)*sin,  eta = -(x-x_le)*sin + (y-y_le)*cos
    dx = xx - x_le
    dy = yy - y_le
    xi = dx * cos_s + dy * sin_s       # along chord
    eta = -dx * sin_s + dy * cos_s     # perpendicular to chord

    # Normalised chord position
    x_norm = xi / chord  # 0 at LE, 1 at TE

    # Camber line (parabolic arc): eta_c = 4 * camber * x_norm * (1 - x_norm)
    eta_c = 4.0 * camber * x_norm * (1.0 - x_norm)

    # Half-thickness (in lattice units)
    half_t = chord * naca_half_thickness(x_norm, t_max)

    # Upper and lower surfaces
    eta_upper = eta_c + half_t
    eta_lower = eta_c - half_t

    # Solid: within chord extent and between surfaces
    in_chord = (x_norm >= 0.0) & (x_norm <= 1.0)
    in_profile = (eta >= eta_lower) & (eta <= eta_upper)

    solid = in_chord & in_profile

    # Close LE/TE: mark the nearest cell at LE and TE
    le_xi = 0.0
    te_xi = chord
    # LE position in domain
    le_x = x_le + le_xi * cos_s
    le_y = y_le + le_xi * sin_s
    te_x = x_le + te_xi * cos_s
    te_y = y_le + te_xi * sin_s
    # Camber at LE and TE is 0, so just mark single cell
    for cx, cy_ in [(int(round(le_x)), int(round(le_y))),
                     (int(round(te_x)), int(round(te_y)))]:
        if 0 <= cx < nx and 0 <= cy_ < ny:
            solid[:, cy_, cx] = True

    return solid


# ---------------------------------------------------------------------------
# Generic run function (used by both benchmarks)
# ---------------------------------------------------------------------------
def run_benchmark(
    device_id,
    tag,
    nx, ny, nz,
    solid,
    u_in,
    tau,
    n_steps,
    dpS,
    cd_ref,
    ref_name,
    measure_cascade=False,
    x_inlet_meas=None,
    x_outlet_meas=None,
    output_path=None,
):
    """Run LBM simulation with unified pressure integration and return results."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    nu = (tau - 0.5) / 3.0
    cs_smag = 0.05

    solid = solid.to(device)
    nz, ny, nx = solid.shape

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag} "
        f"dpS={dpS:.6e} Cd_ref={cd_ref}",
        flush=True,
    )

    t0 = time.time()

    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Precompute near-wall mask and surface mesh (from_gradient)
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    mesh = SurfaceMesh.from_gradient(solid, near)

    # Normal statistics
    nx_n_vals = mesh.nx_n[near]
    ny_n_vals = mesh.ny_n[near]
    nz_n_vals = mesh.nz_n[near]
    print(
        f"{tag} normal stats: "
        f"nx_n=[{float(nx_n_vals.min()):.3f}, {float(nx_n_vals.max()):.3f}] "
        f"ny_n=[{float(ny_n_vals.min()):.3f}, {float(ny_n_vals.max()):.3f}] "
        f"nz_n=[{float(nz_n_vals.min()):.3f}, {float(nz_n_vals.max()):.3f}]",
        flush=True,
    )
    norm_check = torch.sqrt(mesh.nx_n ** 2 + mesh.ny_n ** 2 + mesh.nz_n ** 2)
    norm_near = norm_check[near]
    print(
        f"{tag} |n| stats: min={float(norm_near.min()):.6f} "
        f"max={float(norm_near.max()):.6f} mean={float(norm_near.mean()):.6f}",
        flush=True,
    )
    n_nx_nonzero = int((nx_n_vals.abs() > 1e-6).sum().item())
    n_ny_nonzero = int((ny_n_vals.abs() > 1e-6).sum().item())
    n_nz_nonzero = int((nz_n_vals.abs() > 1e-6).sum().item())
    print(
        f"{tag} 3D check: nx_n nonzero={n_nx_nonzero}/{n_near} "
        f"ny_n nonzero={n_ny_nonzero}/{n_near} nz_n nonzero={n_nz_nonzero}/{n_near}",
        flush=True,
    )

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialize flow field: uniform flow, zero inside solid
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}",
          flush=True)

    # History
    cd_p_hist = []
    cd_f_hist = []
    cd_tot_hist = []
    cl_hist = []
    fz_hist = []
    cp_loss_hist = []
    turn_angle_hist = []

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (MRT + Smagorinsky LES)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (without obstacle_mask → don't touch solid)
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation (unified pressure integration)
        fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS)
        fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

        cd_p = fx_p
        cd_f = fx_f
        cd_tot = cd_p + cd_f
        cl = fy_p + fy_f
        fz_tot = fz_p + fz_f

        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        fz_hist.append(fz_tot)

        # 9. Cascade measurements (T106A only)
        if measure_cascade and step % 100 == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            p = (rho - 1.0) / 3.0
            # Total pressure: p_t = p + 0.5 * (u^2 + v^2 + w^2)
            u_mag2 = ux ** 2 + uy ** 2 + uz ** 2
            p_t = p + 0.5 * u_mag2

            # Inlet plane measurement
            xi_in = x_inlet_meas
            # Average over y (exclude 5 cells from top/bottom) and all z
            y_slice = slice(5, ny - 5)
            p_t_in = float(p_t[:, y_slice, xi_in].mean().item())
            u_in_avg = float(ux[:, y_slice, xi_in].mean().item())
            v_in_avg = float(uy[:, y_slice, xi_in].mean().item())
            theta_in = math.degrees(math.atan2(v_in_avg, u_in_avg))

            # Outlet plane measurement
            xi_out = x_outlet_meas
            p_t_out = float(p_t[:, y_slice, xi_out].mean().item())
            u_out_avg = float(ux[:, y_slice, xi_out].mean().item())
            v_out_avg = float(uy[:, y_slice, xi_out].mean().item())
            theta_out = math.degrees(math.atan2(v_out_avg, u_out_avg))

            # Pressure loss coefficient
            cp_loss = (p_t_in - p_t_out) / (0.5 * u_in ** 2)
            # Turning angle
            turn_angle = theta_out - theta_in

            cp_loss_hist.append(cp_loss)
            turn_angle_hist.append(turn_angle)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_avg = sum(cl_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            extra = ""
            if measure_cascade and cp_loss_hist:
                cp_loss_avg = sum(cp_loss_hist[-5:]) / min(5, len(cp_loss_hist))
                turn_avg = sum(turn_angle_hist[-5:]) / min(5, len(turn_angle_hist))
                extra = f" Cp_loss={cp_loss_avg:.4f} turn={turn_avg:.2f}deg"
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.4f} Cd_f={cd_f_avg:.4f} "
                f"Cd_tot={cd_tot_avg:.4f} Cl={cl_avg:.6f}{extra} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last 1000 steps or all if fewer)
    n_final = min(1000, len(cd_tot_hist))
    cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
    cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
    cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final
    cl_final = sum(cl_hist[-n_final:]) / n_final
    fz_final = sum(fz_hist[-n_final:]) / n_final

    err_pct = abs(cd_tot_final - cd_ref) / cd_ref * 100 if cd_ref > 0 else float("nan")

    result = {
        "case": tag,
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_solid": n_solid,
        "n_near": n_near,
        "Cs": cs_smag,
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cl": cl_final,
        "fz": fz_final,
        "Cd_ref": cd_ref,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "normal_stats": {
            "nx_n_min": float(nx_n_vals.min()),
            "nx_n_max": float(nx_n_vals.max()),
            "ny_n_min": float(ny_n_vals.min()),
            "ny_n_max": float(ny_n_vals.max()),
            "nz_n_min": float(nz_n_vals.min()),
            "nz_n_max": float(nz_n_vals.max()),
            "n_norm_min": float(norm_near.min()),
            "n_norm_max": float(norm_near.max()),
            "n_norm_mean": float(norm_near.mean()),
            "nx_n_nonzero": n_nx_nonzero,
            "ny_n_nonzero": n_ny_nonzero,
            "nz_n_nonzero": n_nz_nonzero,
        },
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    # Add cascade measurements if applicable
    cp_loss_final = None
    turn_angle_final = None
    if measure_cascade and cp_loss_hist:
        n_cascade = min(len(cp_loss_hist), 20)  # last 20 measurements
        cp_loss_final = sum(cp_loss_hist[-n_cascade:]) / n_cascade
        turn_angle_final = sum(turn_angle_hist[-n_cascade:]) / n_cascade
        result["pressure_loss"] = cp_loss_final
        result["turning_angle_deg"] = turn_angle_final
        result["pressure_loss_ref"] = 0.05
        result["pressure_loss_ref_name"] = "T106A LPT cascade (Re=1000, typical)"

    print(
        f"{tag} DONE Cd_p={cd_p_final:.4f} Cd_f={cd_f_final:.4f} "
        f"Cd_tot={cd_tot_final:.4f} Cl={cl_final:.6f} "
        f"(ref={cd_ref:.4f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )
    if cp_loss_final is not None:
        print(
            f"{tag} CASCADE pressure_loss={cp_loss_final:.4f} "
            f"turning_angle={turn_angle_final:.2f}deg",
            flush=True,
        )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage: python drivaer_t106a_worker.py <benchmark> <device_id> <output_path>")
        print("  benchmark: drivaer | t106a")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "drivaer":
        # DrivAer simplified car body (Re=1000)
        nx, ny, nz = 300, 100, 80
        L, W, H = 60, 20, 15
        slant_deg = 25.0
        u_in = 0.05
        tau = 0.509
        n_steps = 10000
        cd_ref = 0.30
        ref_name = "DrivAer simplified experimental (Re=1000)"

        x_start = nx * 0.25   # 75
        cy = ny * 0.5          # 50
        cz = nz * 0.5          # 40

        # Frontal area = W * H
        dpS = 0.5 * u_in ** 2 * W * H

        device = torch.device(f"sdaa:{device_id}")
        torch.sdaa.set_device(device)
        solid = build_drivaer_solid(
            nx, ny, nz, L, W, H, slant_deg, x_start, cy, cz, device
        )
        tag = f"drivaer_re1000_sdaa{device_id}"

        run_benchmark(
            device_id, tag, nx, ny, nz, solid,
            u_in, tau, n_steps, dpS, cd_ref, ref_name,
            output_path=output_path,
        )

    elif benchmark == "t106a":
        # T106A simplified turbine cascade (Re=1000, 2D)
        nx, ny, nz = 400, 200, 4
        chord = 100
        camber = 15       # 15% of chord (lattice units)
        t_max = 0.12      # 12% thickness
        stagger_deg = 30.0
        u_in = 0.05
        tau = 0.515
        n_steps = 10000
        cd_ref = 0.0      # drag not primary metric for cascade
        ref_name = "T106A LPT cascade (Re=1000, simplified)"

        x_le = 100        # leading edge at 25% of nx
        y_le = 100        # center of ny

        # For drag: frontal area = max_thickness * nz (quasi-2D)
        max_thickness = chord * t_max  # 12 lattice units
        dpS = 0.5 * u_in ** 2 * max_thickness * nz

        # Measurement planes
        x_inlet_meas = 50    # before blade
        x_outlet_meas = 350  # after blade

        device = torch.device(f"sdaa:{device_id}")
        torch.sdaa.set_device(device)
        solid = build_t106a_solid(
            nx, ny, nz, chord, camber, t_max, stagger_deg,
            x_le, y_le, device
        )
        tag = f"t106a_re1000_sdaa{device_id}"

        run_benchmark(
            device_id, tag, nx, ny, nz, solid,
            u_in, tau, n_steps, dpS, cd_ref, ref_name,
            measure_cascade=True,
            x_inlet_meas=x_inlet_meas,
            x_outlet_meas=x_outlet_meas,
            output_path=output_path,
        )

    else:
        print(f"Unknown benchmark: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
