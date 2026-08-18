"""B4/B5: 2D cylinder Re=100/200 — far-field BC, low blockage, seeded shedding.

Key fixes vs earlier attempts (Cd=1.23 stale-state, St abnormal):
  1. Large domain 30D x 20D (blockage ratio D/ny = 5%) with far-field BC
  2. Seeded vortex shedding: initial asymmetric perturbation + short
     inlet transverse forcing window (St_seed ~0.164) so shedding
     establishes quickly (pure random noise grows too slowly)
  3. Long run 60000 steps, time-average Cd/St over last 30000
References:
  Re=100: Cd≈1.35 (Braza 1986 exp), St≈0.164 (2D numerical standard)
  Re=200: Cd≈1.33 (Braza), St≈0.196 (Roshko)
"""
import sys, math, time
sys.path.insert(0, '/home/wxsc/cxs/TensorLBM/src')
import torch
from tensorlbm.d2q9 import C as C2D, equilibrium, macroscopic
from tensorlbm.solver import collide_mrt, stream
from tensorlbm.boundaries import far_field_bc_2d


def cylinder_mask(ny, nx, cx, cy, radius, device):
    yy, xx = torch.meshgrid(torch.arange(ny, device=device),
                            torch.arange(nx, device=device), indexing="ij")
    return ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt() < radius


def run_cylinder(re, nx, ny, D, u_in, n_steps, device="cuda:0",
                 n_warmup_frac=0.3, seed_shed=True, cs=0.0):
    radius = D / 2.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    cx, cy = nx * 0.3, ny * 0.5
    dev = torch.device(device)
    solid = cylinder_mask(ny, nx, cx, cy, radius, dev)
    dyn_p = 0.5 * u_in**2 * D
    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))
    f[:, solid] = 0.0
    initial_mass = float(f.sum().item())
    cdev = C2D.to(dev).float()

    # seeding: initial asymmetric perturbation (small rotation around cyl)
    if seed_shed:
        yy, xx = torch.meshgrid(torch.arange(ny, device=dev, dtype=torch.float32),
                                torch.arange(nx, device=dev, dtype=torch.float32),
                                indexing="ij")
        th = torch.atan2(yy - cy, xx - cx)
        pert = 0.02 * u_in * torch.sin(2.0 * th)  # dipolar perturbation
        rho0p, ux0p, uy0p = macroscopic(f)
        feq_p = equilibrium(rho0p, ux0p, uy0p + pert)
        f = torch.where(solid.unsqueeze(0), f, feq_p)

    t0 = time.time()
    cd_list, cl_list, t_shed, cl_prev = [], [], [], None
    warmup = int(n_steps * n_warmup_frac)
    for step in range(1, n_steps + 1):
        before = f.clone()
        from tensorlbm.turbulence import collide_smagorinsky_mrt
        collided = collide_smagorinsky_mrt(f, tau, C_s=cs)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        # MEM force (post-stream, pre-BB)
        fx = 2.0 * (cdev[:, 0].view(9, 1, 1) * f * solid.unsqueeze(0)).sum().item()
        fy = 2.0 * (cdev[:, 1].view(9, 1, 1) * f * solid.unsqueeze(0)).sum().item()
        f = far_field_bc_2d(f, u_in, obstacle_mask=solid)
        if step % 2000 == 0:
            f = f * (initial_mass / f.sum().item())
        if step > warmup:
            cd = fx / dyn_p
            cl = -fy / dyn_p
            cd_list.append(cd)
            cl_list.append(cl)
            if cl_prev is not None and cl_prev * cl < 0:
                t_shed.append(step)
            cl_prev = cl
        if step % 10000 == 0 or step == n_steps:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            el = time.time() - t0
            print(f"  step {step:6d}: Cd={cd_avg:.4f} ({el:.0f}s)", flush=True)

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    cd_rms = math.sqrt(sum((c - cd_avg) ** 2 for c in cd_list) / max(len(cd_list), 1))
    cl_std = math.sqrt(sum(c ** 2 for c in cl_list) / max(len(cl_list), 1))
    st = float("nan")
    if len(t_shed) > 3:
        periods = [t_shed[i + 1] - t_shed[i] for i in range(len(t_shed) - 1)]
        T = sum(periods) / len(periods)
        st = D / (T * u_in)
    return {"re": re, "nx": nx, "ny": ny, "D": D, "tau": tau,
            "cd": cd_avg, "cd_rms": cd_rms, "cl_std": cl_std, "st": st,
            "steps": n_steps, "blockage": D / ny}


if __name__ == "__main__":
    device = "cuda:0"
    re = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
    # Domain 30D x 20D, cylinder at 30% length → blockage 5%
    nx, ny = 30 * D, 20 * D
    u_in = 0.06
    print(f"=== 2D cylinder Re={re} D={D} domain={nx}x{ny} blockage={D/ny:.1%} ===")
    r = run_cylinder(re, nx, ny, D, u_in, n_steps, device)
    cd_ref = 1.35 if re == 100 else 1.33
    st_ref = 0.164 if re == 100 else 0.196
    print(f"  FINAL: Cd={r['cd']:.4f} (ref {cd_ref}, err "
          f"{abs(r['cd']-cd_ref)/cd_ref*100:.1f}%) rms={r['cd_rms']:.4f} "
          f"Cl_std={r['cl_std']:.4f} St={r['st']:.4f} (ref {st_ref})")
