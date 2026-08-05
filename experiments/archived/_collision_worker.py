"""Single SDAA worker: test different collision operators on bare_hull + wall_function."""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C19
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.solver3d import correct_mass3d, stream3d

# All available D3Q19 collision operators
from tensorlbm.turbulence import collide_smagorinsky_mrt3d, collide_smagorinsky_bgk3d
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.solver3d import collide_bgk3d, collide_mrt3d, collide_trt3d, collide_rlbm3d
from tensorlbm.entropic_kbc import collide_kbc_d3q19
from tensorlbm.cascaded_collision import collide_cascaded_d3q19

KAPPA = 0.41; B_CONST = 5.0
REF_CT = 0.00405


def wallfn(f, solid, nu, y_val=0.5):
    fluid = ~solid; near = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    rho, ux, uy, uz = macroscopic3d(f)
    um = torch.sqrt(ux*ux + uy*uy + uz*uz).clamp(min=1e-12)
    ut = torch.sqrt(nu * um / y_val).clamp(min=1e-12)
    yp = y_val * ut / nu; turb = (yp > 11.6) & near
    if turb.any():
        uu = ut[turb].clone(); vm = um[turb]
        for _ in range(8):
            ly = torch.log(y_val * uu / nu)
            fv = uu * (ly/KAPPA + B_CONST) - vm
            fp = (ly/KAPPA + B_CONST) + 1.0/KAPPA
            uu = (uu - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        ut[turb] = uu
    tw = ut * ut; ium = 1.0 / um
    coef = -(tw / y_val) * near.to(f.dtype)
    fx = coef*(ux*ium); fy = coef*(uy*ium); fz = coef*(uz*ium)
    # D3Q19 Guo body force (inline, with velocity correction 1+c·u/cs²)
    # — matches historical CUMULANT D3Q27 impl, just D3Q19 weights
    device = f.device
    c19 = C19.to(device).float()
    cx19 = c19[:, 0].view(19, 1, 1, 1)
    cy19 = c19[:, 1].view(19, 1, 1, 1)
    cz19 = c19[:, 2].view(19, 1, 1, 1)
    w19 = torch.tensor(
        [1/3] + [1/18]*6 + [1/36]*12,
        dtype=f.dtype, device=device,
    ).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    cu = cx19*ux + cy19*uy + cz19*uz
    forcing = w19 * (1.0 + cu / cs2) * (cx19*fx + cy19*fy + cz19*fz) / cs2
    f = f + forcing
    df = (tw * (ux*ium) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    dp = (p * (sp.to(f.dtype) - sm.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, df, dp


def collide(collision, f, tau, Cs):
    """Dispatch to collision operator."""
    if collision == "MRT+Smag":
        return collide_smagorinsky_mrt3d(f, tau=tau, C_s=Cs)
    if collision == "BGK+Smag":
        return collide_smagorinsky_bgk3d(f, tau=tau, C_s=Cs)
    if collision == "CUMULANT":
        return collide_cumulant_d3q19(f, tau)
    if collision == "KBC":
        return collide_kbc_d3q19(f, tau)
    if collision == "RLBM":
        return collide_rlbm3d(f, tau)
    if collision == "TRT":
        return collide_trt3d(f, tau)
    if collision == "CM":
        return collide_cascaded_d3q19(f, tau)
    if collision == "BGK":
        return collide_bgk3d(f, tau)
    raise ValueError(collision)


def main():
    did = int(sys.argv[1])
    collision = sys.argv[2]
    nx = int(sys.argv[3]); ny = int(sys.argv[4]); nz = int(sys.argv[5])
    hull_length = float(sys.argv[6])
    n_steps = int(sys.argv[7])
    Cs = float(sys.argv[8])

    u_in, re = 0.06, 2e6
    nu = u_in * hull_length / re; tau = 3.0 * nu + 0.5
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[{did:02d}] {collision:<10} {nx}³ Cs={Cs}"
    print(f"{tag} tau={tau:.6f}", flush=True)

    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
        length=hull_length, device=device,
    )
    S = _voxel_wetted_area(solid, 1.0); dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    warmup = n_steps // 3
    fric, pres = [], []
    t0 = time.time()
    final_step = 0
    all_finite = True

    for step in range(1, n_steps + 1):
        try:
            f = collide(collision, f, tau, Cs)
        except Exception as e:
            print(f"{tag} COLLISION ERROR at {step}: {e}", flush=True)
            break
        f = stream3d(f)
        f, df, dp = wallfn(f, solid, nu, y_val=0.5)
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0: f = correct_mass3d(f, im)
        if step > warmup and math.isfinite(df):
            fric.append(df); pres.append(dp)
        final_step = step

        if not torch.isfinite(f).all():
            print(f"{tag} DIV at {step}", flush=True)
            all_finite = False
            break

        if step % 200 == 0:
            cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
            cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
            print(f"{tag} step={step} Ct={cf+cp:.4f} ({time.time()-t0:.0f}s)", flush=True)

    cf = (sum(fric) / max(len(fric), 1)) / dpS if fric else 0
    cp = (sum(pres) / max(len(pres), 1)) / dpS if pres else 0
    ct = cf + cp

    result = {
        "collision": collision, "grid": f"{nx}x{ny}x{nz}",
        "Cs": Cs, "tau": tau,
        "Ct_fric": cf, "Ct_pres": cp, "Ct_total": ct,
        "error_pct": abs(ct - REF_CT) / REF_CT * 100,
        "steps": final_step, "finite": all_finite,
        "elapsed_s": time.time() - t0,
    }
    print(f"{tag} DONE Ct={ct:.5f} err={result['error_pct']:.1f}%", flush=True)
    Path(f"/tmp/collision_large/result_{did:02d}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
