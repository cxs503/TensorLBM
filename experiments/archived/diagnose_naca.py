"""Quick diagnosis: test NACA 0012 geometry + basic flow."""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, stream3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d

# Import our geometry builder
from naca0012_worker import build_naca0012_mask

KAPPA, B_CONST = 0.41, 5.0


def wallfn(f, solid, nu, y_val=0.5):
    """Wall function with body force. Returns (f, df, dp, rho_mean)."""
    from tensorlbm.d3q19 import C as C19
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
    fx = coef * (ux * ium)
    fy = coef * (uy * ium)
    fz = coef * (uz * ium)

    w19 = torch.tensor([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=f.dtype, device=device).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx * ux + cy * uy + cz * uz
    forcing = w19 * (1.0 + cu / cs2) * (cx * fx + cy * fy + cz * fz) / cs2
    f = f + forcing

    df = (tw * (ux * ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2)
    sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


def main():
    did = 0  # use CPU for diagnosis
    device = torch.device("cpu")

    # Parameters
    u_in = 0.06
    chord = 80.0
    re = 3e6
    nu = u_in * chord / re
    tau = 3.0 * nu + 0.5
    Cs = 0.05

    nx, ny, nz = 200, 8, 80
    cx_le = 40.0
    cz_center = nz / 2.0

    print(f"τ={tau:.10f}, ν={nu:g}, Re={re:.0f}")
    print(f"Grid: {nx}x{ny}x{nz}")

    # Build geometry
    solid = build_naca0012_mask(nx, ny, nz, cx_le, cz_center, chord, device)
    print(f"Solid cells: {solid.sum().item()}")

    # Check geometry: count cells at mid-y slice
    mid_y = ny // 2
    slice_2d = solid[:, mid_y, :]
    print(f"Airfoil cells at mid-y: {slice_2d.sum().item()}")
    # Print a few rows to verify
    print("Airfoil mask (z=center±5, x=LE...LE+10):")
    cz_i = int(cz_center)
    for z in range(cz_i - 3, cz_i + 4):
        row = ''.join('#' if slice_2d[z, int(cx_le + i)] else '.' for i in range(12))
        print(f"  z={z}: {row}")

    # Compute CD at alpha=0 with far_field_bc_3d
    alpha_deg = 0
    alpha_rad = math.radians(alpha_deg)
    ux_in = u_in * math.cos(alpha_rad)
    uz_in = u_in * math.sin(alpha_rad)

    # Initial conditions
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), ux_in, device=device)
    uy0 = torch.zeros_like(ux0)
    uz0 = torch.full((nz, ny, nx), uz_in, device=device)
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    uz0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    im = float(rho0.sum().item())

    n_steps = 200
    warmup = 50
    fric, pres = [], []

    print(f"\nRunning {n_steps} steps (diagnostic)...")
    for step in range(1, n_steps + 1):
        f = collide_mrt3d(f, tau)  # plain MRT, no Smag for diagnostic
        f = stream3d(f)
        f, df, dp = wallfn(f, solid, nu)
        f = far_field_bc_3d(f, u_in=ux_in, uz=uz_in)
        if step % 50 == 0:
            f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df)
            pres.append(dp)

        if step % 50 == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            solid_f = solid.float()
            p_field = (rho - 1.0) / 3.0
            print(f"  step={step:3d} rho_range=[{rho.min().item():.6f}, {rho.max().item():.6f}] "
                  f"ux_range=[{ux[~solid].min().item():.4f}, {ux[~solid].max().item():.4f}] "
                  f"df={df:.6f} dp={dp:.6f}", flush=True)

    ref_area = chord * ny
    dyn_press = 0.5 * 1.0 * u_in * u_in
    dpS = dyn_press * ref_area

    cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
    cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
    cd = cf + cp
    print(f"\nFinal: Cd={cd:.6f} Cf={cf:.6f} Cp={cp:.6f}")
    print(f"Raw forces: friction={sum(fric)/len(fric):.6f}, pressure={sum(pres)/len(pres):.6f}")
    print(f"dpS={dpS:.6f}")


if __name__ == "__main__":
    main()
