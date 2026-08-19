#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""High-Re quantitative benchmark: 2D cylinder Re=3900 (classic LES case).

Literature (Beaudan & Moin 1994; Kravchenko & Moin 2000; Franke et al.):
  Cd ≈ 0.98-1.04, St ≈ 0.21-0.22  (subcritical, 2D LES at Re=3900)
Uses the high-Re fixes: tau_eff upper clamp + MRT+Smag LES.

2D D2Q9 MRT+Smagorinsky: D=40 cells, u=0.1, tau=0.506 (Re=u·D/ν).
"""
import sys, os, time, json, math
sys.path.insert(0, "/DATA/cxs_host/TensorLBM/src")

import torch
from tensorlbm.d2q9 import equilibrium, C as C2D
from tensorlbm.turbulence import collide_smagorinsky_mrt


def cylinder_mask(ny, nx, cx, cy, radius, device):
    yy, xx = torch.meshgrid(torch.arange(ny, device=device),
                            torch.arange(nx, device=device), indexing="ij")
    return ((xx - cx)**2 + (yy - cy)**2).sqrt() < radius


def run_cylinder(re=3900, nx=400, ny=160, D=40, u_in=0.1,
                 n_steps=40000, device="cuda:0"):
    """2D cylinder at high Re with MRT+Smag LES (D2Q9)."""
    dev = torch.device(device)
    radius = D / 2.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    cs2 = 1.0 / 3.0

    cx, cy = nx * 0.2, ny / 2.0
    solid = cylinder_mask(ny, nx, cx, cy, radius, dev)
    # surface: solid cells with fluid neighbour
    fluid = ~solid
    surface = solid & (torch.roll(fluid, 1, dims=0) | torch.roll(fluid, -1, dims=0)
                       | torch.roll(fluid, 1, dims=1) | torch.roll(fluid, -1, dims=1))

    dyn_p = 0.5 * u_in**2 * D  # 2D frontal area = D
    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)
    initial_mass = float(f.sum().item())

    REF_CD, REF_ST = 1.00, 0.21
    print(f"\n{'='*64}")
    print(f"Cylinder Re={re}  grid {nx}x{ny}  D={D}  u={u_in}  tau={tau:.4f}")
    print(f"nu={nu:.6f}  ref Cd={REF_CD}  ref St={REF_ST}")
    print(f"{'='*64}")

    cd_list, cl_list = [], []
    t_shed = []
    cl_prev = 0.0
    cdev = C2D.to(dev).float()
    t0 = time.time()

    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_smagorinsky_mrt(f, tau, C_s=0.12)
        f = torch.where(solid.unsqueeze(0), before, collided)
        from tensorlbm.solver import stream
        f = stream(f)
        # MEM: Ladd 2·Σ c·f on solid (after stream, before BB)
        fx = 2.0 * (cdev[:, 0].view(9, 1, 1) * f * solid.unsqueeze(0)).sum().item()
        fy = 2.0 * (cdev[:, 1].view(9, 1, 1) * f * solid.unsqueeze(0)).sum().item()
        # far-field BC with internal bounce-back on the obstacle
        from tensorlbm.boundaries import far_field_bc_2d
        f = far_field_bc_2d(f, u_in, obstacle_mask=solid)
        if step % 1000 == 0:
            f = f * (initial_mass / f.sum().item())
        if step > n_steps // 3:
            cd = fx / dyn_p
            cl = -fy / dyn_p
            cd_list.append(cd)
            cl_list.append(cl)
            if cl_prev * cl < 0:
                t_shed.append(step)
            cl_prev = cl
        if step % 4000 == 0 or step == n_steps:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            el = time.time() - t0
            print(f"  step {step:5d}: Cd={cd_avg:.4f} (ref {REF_CD}, "
                  f"err {abs(cd_avg-REF_CD)/REF_CD*100:.1f}%), {el:.0f}s")

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    err = abs(cd_avg - REF_CD) / REF_CD * 100
    st = float("nan")
    if len(t_shed) > 2:
        periods = [t_shed[i+1] - t_shed[i] for i in range(len(t_shed) - 1)]
        T = sum(periods) / len(periods)
        st = D / (T * u_in)
    print(f"\n  FINAL: Cd={cd_avg:.4f} (ref {REF_CD}, err {err:.1f}%)  "
          f"St={st:.4f} (ref {REF_ST})")
    return {"re": re, "cd": cd_avg, "cd_ref": REF_CD, "err_pct": err,
            "st": st, "st_ref": REF_ST, "tau": tau, "steps": n_steps}


if __name__ == "__main__":
    device = "cuda:0"
    r = run_cylinder(re=3900, n_steps=40000, device=device)
    print("\n" + "=" * 64)
    print("SUMMARY (2D cylinder Re=3900, MRT+Smag, tau_eff ≤ 1.0)")
    print("=" * 64)
    print(f"  Re={r['re']}: Cd={r['cd']:.4f} ref={r['cd_ref']:.4f} "
          f"err={r['err_pct']:.1f}%  St={r['st']:.4f} ref={r['st_ref']}")
    with open("/tmp/highre_cylinder_results.json", "w") as fp:
        json.dump(r, fp, indent=2)
    print("saved /tmp/highre_cylinder_results.json")
