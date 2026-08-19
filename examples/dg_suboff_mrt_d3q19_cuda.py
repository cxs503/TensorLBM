"""D3Q19 MRT / D3Q27 Cumulant LBM SUBOFF wall-function solver (CUDA port).

Ported from dg_suboff_mrt_d3q19.py for NVIDIA CUDA GPUs.
Changes: sdaa:0 -> cuda:0, added JSON result output.

Usage:
    # D3Q19 MRT (default)
    PYTHONPATH=src python examples/dg_suboff_mrt_d3q19_cuda.py --device cuda:0

    # D3Q27 Cumulant (high-Re accurate path, Ct ~ 0.004)
    PYTHONPATH=src python examples/dg_suboff_mrt_d3q19_cuda.py --device cuda:0 --lattice d3q27 --steps 600

    # Large grid test (Plan A: D=24.3)
    PYTHONPATH=src python examples/dg_suboff_mrt_d3q19_cuda.py --device cuda:0 --lattice d3q27 \
        --nx 520 --ny 208 --nz 208 --hull 260 --re 1000 --steps 5000 --warmup 2000
"""
from __future__ import annotations
import math, time, argparse, json, os, torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C19
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.d3q27 import (
    equilibrium27, macroscopic27, C as C27, W as W27, OPPOSITE as OPPOSITE27,
    correct_mass27,
)
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area

KAPPA = 0.41
B_CONST = 5.0

_C19_SHIFTS = [(int(C19[q, 0]), int(C19[q, 1]), int(C19[q, 2])) for q in range(19)]

_C27_SHIFTS = [(int(C27[q, 0]), int(C27[q, 1]), int(C27[q, 2])) for q in range(27)]


def stream19_roll(f: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(f)
    for q in range(19):
        sx, sy, sz = _C19_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def stream27_roll(f: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _C27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def far_field_bc_19(f, u_in=0.06):
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium3d(rho1, torch.full_like(rho1, u_in),
                        torch.zeros_like(rho1), torch.zeros_like(rho1))
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, 0, :, :] = feq[:, 0, :, :]
    f[:, -1, :, :] = feq[:, -1, :, :]
    f[:, :, 0, :] = feq[:, :, 0, :]
    f[:, :, -1, :] = feq[:, :, -1, :]
    return f


def far_field_bc_27(f, u_in=0.06):
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(rho1, torch.full_like(rho1, u_in),
                        torch.zeros_like(rho1), torch.zeros_like(rho1))
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, 0, :, :] = feq[:, 0, :, :]
    f[:, -1, :, :] = feq[:, -1, :, :]
    f[:, :, 0, :] = feq[:, :, 0, :]
    f[:, :, -1, :] = feq[:, :, -1, :]
    return f


def wall_function_19(f, solid, nu, y_val=0.5):
    device = f.device
    c = C19.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    fluid = ~solid
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    near = nbrs

    rho, ux, uy, uz = macroscopic3d(f)
    u_mag = torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
    u_tau = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu
    turb = (y_plus > 11.6) & near
    if bool(turb.any()):
        ut = u_tau[turb].clone(); um = u_mag[turb]
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu)
            fv = ut * (lyp/KAPPA + B_CONST) - um
            fp = (lyp/KAPPA + B_CONST) + 1.0/KAPPA
            ut = (ut - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)
    fx = coef * (ux * inv_umag); fy = coef * (uy * inv_umag); fz = coef * (uz * inv_umag)

    w19 = torch.tensor(
        [1/3] + [1/18]*6 + [1/36]*12,
        dtype=f.dtype, device=device,
    ).view(19, 1, 1, 1)
    cs2 = 1.0/3.0
    cu = cx*ux + cy*uy + cz*uz
    forcing = w19 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
    f = f + forcing
    drag_fric = (tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    drag_pres = (p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, drag_fric, drag_pres


def wall_function_27(f, solid, nu, y_val=0.5):
    device = f.device
    c = C27.to(device).float()
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    fluid = ~solid
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    near = nbrs

    rho, ux, uy, uz = macroscopic27(f)
    u_mag = torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
    u_tau = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
    y_plus = y_val * u_tau / nu
    turb = (y_plus > 11.6) & near
    if bool(turb.any()):
        ut = u_tau[turb].clone(); um = u_mag[turb]
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu)
            fv = ut * (lyp/KAPPA + B_CONST) - um
            fp = (lyp/KAPPA + B_CONST) + 1.0/KAPPA
            ut = (ut - fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau[turb] = ut
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    coef = -(tau_w / y_val) * near.to(f.dtype)
    fx = coef * (ux * inv_umag); fy = coef * (uy * inv_umag); fz = coef * (uz * inv_umag)

    w27 = W27.to(device).to(f.dtype).view(27, 1, 1, 1)
    cs2 = 1.0/3.0
    cu = cx*ux + cy*uy + cz*uz
    forcing = w27 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
    f = f + forcing
    drag_fric = (tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item()
    p = (rho - 1.0) / 3.0
    sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
    drag_pres = (p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item()
    return f, drag_fric, drag_pres


def run(re=2e6, hull_length=96.0, nx=192, ny=72, nz=72, u_in=0.06,
        n_steps=600, warmup=200, y_val=0.5, device="cuda:0", lattice="d3q19",
        output_json=None):
    if lattice not in ("d3q19", "d3q27"):
        raise ValueError(f"unknown lattice {lattice!r}; expected 'd3q19' or 'd3q27'")
    use_d3q27 = lattice == "d3q27"

    dev = torch.device(device)
    nu_lat = u_in * hull_length / re; tau = 3.0*nu_lat + 0.5
    cx_g, cy_g, cz_g = nx*0.35, ny/2.0, nz/2.0

    solid, _ = build_suboff_mask(hull_type="full", nx=nx,ny=ny,nz=nz,
                                 cx=cx_g,cy=cy_g,cz=cz_g,length=hull_length,device=str(dev))
    solid = solid.to(dev)
    S = _voxel_wetted_area(solid, 1.0); dyn_p_S = 0.5*1.0*u_in**2*S

    rho0 = torch.ones(nz,ny,nx,device=dev)
    ux0 = torch.full((nz,ny,nx),u_in,device=dev); ux0[solid]=0
    if use_d3q27:
        f = equilibrium27(rho0,ux0,torch.zeros_like(ux0),torch.zeros_like(ux0))
        collide = lambda f: collide_cumulant_d3q27(f, tau=tau)
        stream = stream27_roll
        wall_fn = wall_function_27
        far_field = far_field_bc_27
        correct_mass = correct_mass27
        op_name = "D3Q27 Cumulant"
    else:
        f = equilibrium3d(rho0,ux0,torch.zeros_like(ux0),torch.zeros_like(ux0))
        collide = lambda f: collide_mrt3d(f, tau=tau)
        stream = stream19_roll
        wall_fn = wall_function_19
        far_field = far_field_bc_19
        correct_mass = correct_mass3d
        op_name = "D3Q19 MRT"
    initial_mass = float(rho0.sum().item())

    # Grid resolution metric D = hull_length / max(ny, nz)
    D = hull_length / max(ny, nz)
    print(f"{op_name}: Re={re:.0e} tau={tau:.5f} grid={nx}x{ny}x{nz} cells={nx*ny*nz:,} D={D:.1f}")
    print(f"Device: {device} | Experimental AFF-8 Ct ~ 0.004\n")

    fric=[];pres=[];t0=time.time();t_step_total=0.0
    for step in range(1,n_steps+1):
        ts=time.time()
        f = collide(f)
        f = stream(f)
        f,df,dp = wall_fn(f, solid, nu_lat, y_val=y_val)
        f = far_field(f, u_in=u_in)
        if step%100==0: f = correct_mass(f, initial_mass)
        t_step_total += time.time()-ts
        if step>warmup and math.isfinite(df): fric.append(df);pres.append(dp)
        if step%100==0 or step==n_steps:
            cf=sum(fric)/max(len(fric),1)/dyn_p_S; cp=sum(pres)/max(len(pres),1)/dyn_p_S
            avg=t_step_total/step; mlups=nx*ny*nz/avg/1e6
            print(f"  step {step:4d}: Ct_f={cf:.4f} Ct_p={cp:.4f} Ct={cf+cp:.4f} "
                  f"{avg*1000:.0f}ms/step {mlups:.1f}MLUPS")

    cf=sum(fric)/max(len(fric),1)/dyn_p_S; cp=sum(pres)/max(len(pres),1)/dyn_p_S
    total=time.time()-t0; avg=t_step_total/n_steps; mlups=nx*ny*nz/avg/1e6
    ct_total = cf+cp
    print(f"\nFinal: Ct_fric={cf:.4f} Ct_pres={cp:.4f} Ct_total={ct_total:.4f}")
    print(f"  (exp ~0.004, ratio {ct_total/0.004:.2f}x)")
    print(f"Perf: {avg*1000:.0f}ms/step | {mlups:.1f}MLUPS | {total:.1f}s | {op_name}")

    result = {
        "operator": op_name,
        "Re": re,
        "grid": f"{nx}x{ny}x{nz}",
        "nx": nx, "ny": ny, "nz": nz,
        "D": round(D, 1),
        "hull_length": hull_length,
        "tau": round(tau, 5),
        "nu_lat": nu_lat,
        "u_in": u_in,
        "n_steps": n_steps,
        "warmup": warmup,
        "y_val": y_val,
        "device": device,
        "Ct_fric": round(cf, 4),
        "Ct_pres": round(cp, 4),
        "Ct_total": round(ct_total, 4),
        "Ct_exp": 0.004,
        "error_pct": round(abs(ct_total - 0.004) / 0.004 * 100, 1) if re >= 1e5 else None,
        "ms_per_step": round(avg*1000, 1),
        "MLUPS": round(mlups, 1),
        "total_time_s": round(total, 1),
    }
    if output_json:
        with open(output_json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Results saved to {output_json}")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description="SUBOFF wall-function solver CUDA (D3Q19 MRT / D3Q27 Cumulant)")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--lattice",default="d3q19",choices=["d3q19","d3q27"],
                   help="d3q19 = MRT (default); d3q27 = Cumulant (high-Re accurate, Ct~0.004)")
    p.add_argument("--nx",type=int,default=192)
    p.add_argument("--ny",type=int,default=72)
    p.add_argument("--nz",type=int,default=72)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--warmup",type=int,default=200)
    p.add_argument("--hull",type=float,default=96.0)
    p.add_argument("--re",type=float,default=2e6)
    p.add_argument("--u_in",type=float,default=0.06)
    p.add_argument("--y_val",type=float,default=0.5)
    p.add_argument("--output_json",type=str,default=None,
                   help="Save results to JSON file")
    a=p.parse_args()
    run(nx=a.nx,ny=a.ny,nz=a.nz,n_steps=a.steps,warmup=a.warmup,
        hull_length=a.hull,device=a.device,lattice=a.lattice,
        re=a.re,u_in=a.u_in,y_val=a.y_val,output_json=a.output_json)
