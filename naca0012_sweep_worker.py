"""NACA 0012 airfoil fine angle sweep — D3Q19 MRT+Smag+wallfn+farfield.

Usage:
    PYTHONPATH=src python naca0012_sweep_worker.py <device_id> <angle_deg>

Writes result to /tmp/naca0012_results/result_<device_id>_a<angle>.json
"""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C19
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d, free_slip_cells_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

KAPPA, B_CONST = 0.41, 5.0

# ──────────────────── NACA 0012 GEOMETRY BUILDER ────────────────────

def naca0012_half_thickness(x_over_c: torch.Tensor, t: float = 0.12) -> torch.Tensor:
    """NACA 4-digit symmetrical half-thickness distribution.

    y = t/0.2 * c * (0.2969*sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)^2
                      + 0.2843*(x/c)^3 - 0.1015*(x/c)^4)

    Args:
        x_over_c: Normalised chord position ∈ [0, 1], trailing edge at 1.
        t: Maximum thickness / chord (0.12 for NACA 0012).
    Returns:
        Half-thickness / chord at each point.
    """
    a = x_over_c.clamp(min=1e-12)
    y = (t / 0.2) * (
        0.2969 * torch.sqrt(a)
        - 0.1260 * a
        - 0.3516 * a * a
        + 0.2843 * a * a * a
        - 0.1015 * a * a * a * a
    )
    return y


def build_naca0012_mask(
    nx: int, ny: int, nz: int,
    cx_le: float, cz_center: float,
    chord: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a 3D extruded NACA 0012 solid mask.

    The airfoil profile lies in the x-z plane, extruded across the full y span.
    The chord extends from cx_le to cx_le + chord along x.
    Leading edge at (cx_le, cz_center), trailing edge at (cx_le + chord, cz_center).
    The mask is True for solid cells.

    Args:
        nx, ny, nz: Grid dimensions.
        cx_le: x-coordinate of leading edge.
        cz_center: z-coordinate of chord line (centerline).
        chord: Chord length in lattice units.
        device: Target device.
    Returns:
        Boolean tensor of shape (nz, ny, nx).
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Normalised chord position
    x_norm = (xx - cx_le) / chord  # 0 at LE, 1 at TE

    # Half-thickness (normalised by chord)
    half_t = chord * naca0012_half_thickness(x_norm)  # (nz, ny, nx) physical lu

    # Airfoil upper and lower surfaces in z
    z_upper_px = cz_center + half_t  # upper surface z-coordinate (float)
    z_lower_px = cz_center - half_t  # lower surface z-coordinate (float)

    # Mark cells where the cell-center z is within [z_lower, z_upper]
    # Only within the chord extent: 0 ≤ x_norm ≤ 1
    in_chord = (x_norm >= 0.0) & (x_norm <= 1.0)
    in_profile = (zz >= z_lower_px) & (zz <= z_upper_px)

    solid = in_chord & in_profile

    # Fill the leading/trailing edge gaps: cells at LE/TE where half_t→0.
    # The NACA formula gives half_t=0 exactly at x_norm=0 and x_norm=1.
    # We mark the nearest integer cell as solid for a closed body.
    le_col = int(cx_le)
    te_col = int(cx_le + chord)
    cz_int = int(cz_center)
    solid[:, :, le_col] |= (zz[:, :, le_col] == cz_int)
    solid[:, :, te_col] |= (zz[:, :, te_col] == cz_int)
    return solid


# ──────────────────── WALL FUNCTION (inline) ────────────────────

def wallfn(f: torch.Tensor, solid: torch.Tensor, nu: float, y_val: float = 0.5):
    """Apply log-law wall function body force + return friction/pressure forces.

    Returns (f_corrected, friction_force_x, pressure_force_x, friction_force_z, pressure_force_z).
    """
    device = f.device
    c = C19.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

    # Compute u_tau via Newton iteration on log-law
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu
    turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone()
        vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly / KAPPA + B_CONST) - vm
            fp = (ly / KAPPA + B_CONST) + 1.0 / KAPPA
            uu = (uu - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu

    tw = ut * ut
    ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx_body = coef * (ux * ium)
    fy_body = coef * (uy * ium)
    fz_body = coef * (uz * ium)

    # D3Q19 Guo body force
    w19 = torch.tensor([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=f.dtype, device=device).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx * ux + cy * uy + cz * uz
    forcing = w19 * (1.0 + cu / cs2) * (cx * fx_body + cy * fy_body + cz * fz_body) / cs2
    f = f + forcing

    # Friction drag: wall shear in x-direction integrated over near-wall cells
    df = (tw * (ux * ium) * near.to(f.dtype)).sum().item()
    # Friction lift: wall shear in z-direction
    lf = (tw * (uz * ium) * near.to(f.dtype)).sum().item()

    # Pressure drag: pressure difference across solid in x-direction
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()

    # Pressure lift: pressure difference across solid in z-direction
    sp_z = torch.roll(solid, 1, dims=0)
    sm_z = torch.roll(solid, -1, dims=0)
    lp = (p * (sp_z.to(f.dtype) - sm_z.to(f.dtype)) * fluid.to(f.dtype)).sum().item()

    return f, df, dp, lf, lp


# ──────────────────── MAIN ────────────────────

def main():
    did = int(sys.argv[1])
    alpha_deg = float(sys.argv[2])
    Cs = 0.05

    # ── Physics parameters ──
    u_in = 0.06
    chord = 80.0
    re = 3e6
    nu = u_in * chord / re
    tau = 3.0 * nu + 0.5

    # ── Grid ──
    nx, ny, nz = 200, 80, 80

    # ── Device ──
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    alpha_rad = math.radians(alpha_deg)
    ux_in = u_in * math.cos(alpha_rad)
    uz_in = u_in * math.sin(alpha_rad)

    tag = f"[SDAA:{did}] NACA0012 α={alpha_deg}° Cs={Cs} {nx}x{ny}x{nz} Re={re:.0e}"
    print(f"{tag}", flush=True)
    print(f"  tau={tau:.10f} nu={nu:g} ux_in={ux_in:.6f} uz_in={uz_in:.6f}", flush=True)

    n_steps = 1500
    warmup = 500
    t0 = time.time()

    # ── Geometry ──
    cx_le = 40.0  # leading edge x
    cz_center = nz / 2.0  # chord line at vertical center
    solid = build_naca0012_mask(nx, ny, nz, cx_le, cz_center, chord, device)
    n_solid = solid.sum().item()
    print(f"  solid cells: {n_solid}", flush=True)

    # Reference area for Cd: chord × span (ny cells)
    ref_area = chord * float(ny)  # 80 * 80 = 6400
    dyn_press = 0.5 * 1.0 * u_in * u_in  # 0.5 * ρ * U², ρ=1
    dpS_drag = dyn_press * ref_area
    print(f"  ref_area={ref_area:.1f} dyn_press_ref={dyn_press:.6f}", flush=True)

    # ── Symmetry wall mask for y faces ──
    sym_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    sym_mask[:, 0, :] = True
    sym_mask[:, -1, :] = True

    # ── Initial conditions ──
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), ux_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.full((nz, ny, nx), uz_in, device=device)
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    uz0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    im = float(rho0.sum().item())

    fric_drag, pres_drag, fric_lift, pres_lift = [], [], [], []

    # ── Custom far-field BC with angled inflow ──
    def far_field_angled(f_in, ux_fs, uz_fs):
        """Far-field BC: angled inlet at x=0, equilibrium at z faces, symmetry at y faces."""
        rho1 = torch.ones((f_in.shape[1], f_in.shape[2], f_in.shape[3]),
                          dtype=f_in.dtype, device=f_in.device)
        feq = equilibrium3d(rho1,
                            torch.full_like(rho1, ux_fs),
                            torch.full_like(rho1, 0.0),
                            torch.full_like(rho1, uz_fs),
                            device=f_in.device)
        f_out = f_in.clone()
        f_out[:, :, :, 0] = feq[:, :, :, 0]        # inlet (free stream)
        f_out[:, :, :, -1] = f_out[:, :, :, -2]    # outlet (zero gradient)
        f_out[:, :, 0, :] = feq[:, :, 0, :]        # z- far-field
        f_out[:, :, -1, :] = feq[:, :, -1, :]      # z+ far-field
        # y faces: free-slip (symmetry) — handled separately below
        f_out = bounce_back_cells_3d(f_out, solid)
        return f_out

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
        f = stream3d(f)
        f, df, dp, lf, lp = wallfn(f, solid, nu)
        f = far_field_angled(f, ux_in, uz_in)
        f = free_slip_cells_3d(f, sym_mask, axis=1)  # symmetry on y faces

        if step % 100 == 0:
            f = correct_mass3d(f, im)

        if step > warmup and math.isfinite(df):
            fric_drag.append(df)
            pres_drag.append(dp)
            fric_lift.append(lf)
            pres_lift.append(lp)

        if step % 500 == 0 or step == n_steps:
            cf = (sum(fric_drag) / max(len(fric_drag), 1)) / dpS_drag if fric_drag else 0
            cp = (sum(pres_drag) / max(len(pres_drag), 1)) / dpS_drag if pres_drag else 0
            cd = cf + cp
            clf = (sum(fric_lift) / max(len(fric_lift), 1)) / dpS_drag if fric_lift else 0
            clp = (sum(pres_lift) / max(len(pres_lift), 1)) / dpS_drag if pres_lift else 0
            cl = clf + clp
            elapsed = time.time() - t0
            print(f"  step={step:4d} Cd={cd:.6f} Cf={cf:.6f} Cp={cp:.6f} Cl={cl:.6f} n={len(fric_drag)} ({elapsed:.0f}s)",
                  flush=True)

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

    cf = (sum(fric_drag) / max(len(fric_drag), 1)) / dpS_drag if fric_drag else 0
    cp = (sum(pres_drag) / max(len(pres_drag), 1)) / dpS_drag if pres_drag else 0
    cd = cf + cp
    clf = (sum(fric_lift) / max(len(fric_lift), 1)) / dpS_drag if fric_lift else 0
    clp = (sum(pres_lift) / max(len(pres_lift), 1)) / dpS_drag if pres_lift else 0
    cl = clf + clp

    # Experimental reference (Abbott 1959)
    ref_data = {0: (0.0080, 0.0), 1: (None, None), 2: (0.0095, None),
                3: (None, None), 4: (0.0125, None), 5: (None, None),
                6: (0.0180, None), 8: (0.0260, None), 10: (0.040, None), 12: (0.080, None)}
    ref_cd = ref_data.get(int(alpha_deg), (None, None))[0]
    err = abs(cd - ref_cd) / ref_cd * 100 if ref_cd else None

    result = {
        "case": "NACA0012",
        "device_id": did,
        "alpha_deg": alpha_deg,
        "Cs": Cs,
        "Re": re,
        "grid": f"{nx}x{ny}x{nz}",
        "chord_lu": chord,
        "u_in": u_in,
        "ux_in": ux_in,
        "uz_in": uz_in,
        "nu": nu,
        "tau": tau,
        "solid_cells": n_solid,
        "ref_area": ref_area,
        "steps_total": n_steps,
        "warmup": warmup,
        "n_averaged": len(fric_drag),
        "Cd_fric": cf,
        "Cd_pres": cp,
        "Cd_total": cd,
        "Cl_fric": clf,
        "Cl_pres": clp,
        "Cl_total": cl,
        "Cd_experimental": ref_cd,
        "error_pct": err,
        "elapsed_s": time.time() - t0,
        "finite": bool(torch.isfinite(f).all().item()),
    }

    out_file = Path(f"/tmp/naca0012_results/result_{did:02d}_a{int(alpha_deg)}.json")
    out_dir = out_file.parent
    out_dir.mkdir(exist_ok=True)
    out_file.write_text(json.dumps(result))
    print(f"\nDONE Cd={cd:.6f} Cf={cf:.6f} Cp={cp:.6f} Cl={cl:.6f} ref={ref_cd} err={err} elapsed={result['elapsed_s']:.0f}s",
          flush=True)
    print(f"Output: {out_file}", flush=True)


if __name__ == "__main__":
    main()
