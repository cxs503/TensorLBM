"""NACA 0012 v3 — larger domain, more steps, library wall_function_3d.

Usage:
    PYTHONPATH=src python naca0012_worker_v3.py <device_id> <angle_deg> <Cs>

Key changes from v2:
- Larger domain: 256x8x128 (was 200x8x80), better far-field spacing
- More steps: 4000 (was 2000)
- Same wall_function_3d + MRT+Smag + far_field + free-slip y
"""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, free_slip_cells_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d


def naca0012_half_thickness(x_over_c: torch.Tensor, t: float = 0.12) -> torch.Tensor:
    a = x_over_c.clamp(min=1e-12)
    return (t / 0.2) * (
        0.2969 * torch.sqrt(a) - 0.1260 * a - 0.3516 * a * a
        + 0.2843 * a * a * a - 0.1015 * a * a * a * a)


def build_naca0012_mask(nx, ny, nz, cx_le, cz_center, chord, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij")
    x_norm = (xx - cx_le) / chord
    half_t = chord * naca0012_half_thickness(x_norm)
    z_upper = cz_center + half_t
    z_lower = cz_center - half_t
    solid = (x_norm >= 0.0) & (x_norm <= 1.0) & (zz >= z_lower) & (zz <= z_upper)
    le_col = int(cx_le)
    te_col = int(cx_le + chord)
    cz_int = int(cz_center)
    solid[:, :, le_col] |= (zz[:, :, le_col] == cz_int)
    solid[:, :, te_col] |= (zz[:, :, te_col] == cz_int)
    return solid


def main():
    did = int(sys.argv[1])
    alpha_deg = float(sys.argv[2])
    Cs = float(sys.argv[3])

    u_in = 0.06
    chord = 80.0
    re = 3e6
    nu = u_in * chord / re
    tau = 3.0 * nu + 0.5

    # Larger domain
    nx, ny, nz = 256, 8, 128

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    alpha_rad = math.radians(alpha_deg)
    ux_in = u_in * math.cos(alpha_rad)
    uz_in = u_in * math.sin(alpha_rad)

    tag = f"[SDAA:{did}] NACA0012 α={alpha_deg}° Cs={Cs} {nx}x{ny}x{nz} Re={re:.0e}"
    print(f"{tag}", flush=True)
    print(f"  tau={tau:.10f} nu={nu:g} ux_in={ux_in:.6f} uz_in={uz_in:.6f}", flush=True)

    n_steps = 4000
    warmup = 1000
    t0 = time.time()

    # Geometry — position airfoil with room upstream and downstream
    cx_le = 48.0
    cz_center = nz / 2.0
    solid = build_naca0012_mask(nx, ny, nz, cx_le, cz_center, chord, device)
    n_solid = solid.sum().item()
    print(f"  solid cells: {n_solid}", flush=True)

    ref_area = chord * float(ny)
    dyn_press = 0.5 * 1.0 * u_in * u_in
    dpS = dyn_press * ref_area
    print(f"  ref_area={ref_area:.1f} dpS={dpS:.6f}", flush=True)

    sym_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    sym_mask[:, 0, :] = True
    sym_mask[:, -1, :] = True

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), ux_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.full((nz, ny, nx), uz_in, device=device)
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    uz0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    im = float(rho0.sum().item())

    fric_drag, pres_drag = [], []

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
        f = stream3d(f)
        f, df, dp = wall_function_3d(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=ux_in, uz=uz_in)
        f = free_slip_cells_3d(f, sym_mask, axis=1)

        if step % 100 == 0:
            f = correct_mass3d(f, im)

        if step > warmup and math.isfinite(df):
            fric_drag.append(df)
            pres_drag.append(dp)

        if step % 1000 == 0 or step == n_steps:
            cf = (sum(fric_drag) / max(len(fric_drag), 1)) / dpS if fric_drag else 0
            cp = (sum(pres_drag) / max(len(pres_drag), 1)) / dpS if pres_drag else 0
            cd = cf + cp
            elapsed = time.time() - t0
            print(f"  step={step:4d} Cd={cd:.6f} Cf={cf:.6f} Cp={cp:.6f} n={len(fric_drag)} ({elapsed:.0f}s)",
                  flush=True)

        if not torch.isfinite(f).all():
            print(f"  DIVERGED at step {step}", flush=True)
            break

    cf = (sum(fric_drag) / max(len(fric_drag), 1)) / dpS if fric_drag else 0
    cp = (sum(pres_drag) / max(len(pres_drag), 1)) / dpS if pres_drag else 0
    cd = cf + cp

    ref_cd = {0: 0.0080, 2: 0.0095, 4: 0.0125}.get(int(alpha_deg), None)
    err = abs(cd - ref_cd) / ref_cd * 100 if ref_cd else None

    result = {
        "case": "NACA0012", "device_id": did, "alpha_deg": alpha_deg,
        "Cs": Cs, "Re": re, "grid": f"{nx}x{ny}x{nz}", "chord_lu": chord,
        "u_in": u_in, "ux_in": ux_in, "uz_in": uz_in, "nu": nu, "tau": tau,
        "solid_cells": n_solid, "ref_area": ref_area,
        "steps_total": n_steps, "warmup": warmup, "n_averaged": len(fric_drag),
        "Cd_fric": cf, "Cd_pres": cp, "Cd_total": cd,
        "Cd_experimental": ref_cd, "error_pct": err,
        "elapsed_s": time.time() - t0,
        "finite": bool(torch.isfinite(f).all().item()),
    }

    out_dir = Path("/tmp/naca0012_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"result_{did:02d}_a{int(alpha_deg)}_cs{Cs}.json"
    out_file.write_text(json.dumps(result))
    print(f"\nDONE Cd={cd:.6f} Cf={cf:.6f} Cp={cp:.6f} ref={ref_cd} err={err:.2f}% elapsed={result['elapsed_s']:.0f}s",
          flush=True)
    print(f"Output: {out_file}", flush=True)


if __name__ == "__main__":
    main()
