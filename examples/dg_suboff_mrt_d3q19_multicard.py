"""Multi-card parallel D3Q19 MRT SUBOFF solver.

Splits grid along X across N SDAA cards with halo exchange.
Each card runs MRT collision + torch.roll streaming locally.

D3Q19 version of dg_suboff_cumulant_d3q27_multicard.py — same multi-card
logic, only the lattice interface is swapped (D3Q27 Cumulant -> D3Q19 MRT).

Usage:
    torchrun --nproc_per_node=4 examples/dg_suboff_mrt_d3q19_multicard.py
    torchrun --nproc_per_node=8 examples/dg_suboff_mrt_d3q19_multicard.py --nx 640
"""
from __future__ import annotations
import math, time, argparse, os, torch
import torch.distributed as dist
from tensorlbm.d3q19 import C as C19

KAPPA = 0.41
B_CONST = 5.0

# D3Q19 velocity shifts for torch.roll streaming — generated from C19 to
# guarantee ordering consistency between streaming and the wall-function
# cx/cy/cz (avoids the D3Q27 C-vs-shift mismatch bug).
_C19_SHIFTS = [(int(C19[q, 0]), int(C19[q, 1]), int(C19[q, 2])) for q in range(19)]

def _setup():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_offset = int(os.environ.get("SDAA_DEVICE_OFFSET", 0))
    if world_size > 1:
        dist.init_process_group("tccl", rank=rank, world_size=world_size)
    device = torch.device(f"sdaa:{local_rank + device_offset}")
    torch.sdaa.set_device(device)
    return rank, world_size, device

def _cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def stream19_roll(f):
    out = torch.empty_like(f)
    for q in range(19):
        sx, sy, sz = _C19_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out

def halo_exchange(f_local, rank, world_size):
    if world_size == 1:
        return
    left_interior = f_local[:, :, :, 1:2].contiguous()
    right_interior = f_local[:, :, :, -2:-1].contiguous()
    # Use all_gather instead of isend/irecv (TCCL point-to-point unreliable for large tensors)
    right_gather = [torch.empty_like(right_interior) for _ in range(world_size)]
    dist.all_gather(right_gather, right_interior)
    left_halo = right_gather[(rank - 1) % world_size]
    left_gather = [torch.empty_like(left_interior) for _ in range(world_size)]
    dist.all_gather(left_gather, left_interior)
    right_halo = left_gather[(rank + 1) % world_size]
    f_local[:, :, :, 0:1] = left_halo
    f_local[:, :, :, -1:] = right_halo

def run_multicard(nx=384, ny=160, nz=160, n_steps=1000, warmup=300,
                  re=2e6, hull_length=160.0, u_in=0.06, y_val=0.5):
    rank, world_size, device = _setup()
    is_main = rank == 0
    with open(f"/tmp/mc_{rank}.log","w") as _f: _f.write("setup done\n")
    assert nx % world_size == 0
    nx_local = nx // world_size
    nx_halo = nx_local + 2
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    if is_main:
        print(f"D3Q19 MRT Multi-card: {world_size} cards")
        print(f"Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} cells ({nx*ny*nz/1e6:.1f}M)")
        print(f"Per card: {nx_local}x{ny}x{nz} = {nx_local*ny*nz:,} cells")
        print(f"Re={re:.0e} tau={tau:.5f} | Experimental AFF-8 Ct ~ 0.004\n")

    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C19
    from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
    from tensorlbm.suboff_cad import build_suboff_mask
    from tensorlbm.suboff_resistance import _voxel_wetted_area

    # Build mask on CPU, slice, transfer to SDAA
    cx_global = nx * 0.35
    x_start = rank * nx_local
    x_end = x_start + nx_local
    full_solid, _ = build_suboff_mask(
        hull_type="full", nx=nx, ny=ny, nz=nz,
        cx=cx_global, cy=ny/2.0, cz=nz/2.0,
        length=hull_length, device="cpu")
    left_halo_idx = (x_start - 1) % nx
    right_halo_idx = x_end % nx
    solid = torch.zeros(nz, ny, nx_halo, dtype=torch.bool, device=device)
    solid[:, :, 1:-1] = full_solid[:, :, x_start:x_end].to(device)
    solid[:, :, 0] = full_solid[:, :, left_halo_idx].to(device)
    solid[:, :, -1] = full_solid[:, :, right_halo_idx].to(device)
    del full_solid
    with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write("mask done\n")

    S_local = _voxel_wetted_area(solid[:, :, 1:-1], 1.0)
    S_tensor = torch.tensor([S_local], device=device, dtype=torch.float32)
    dist.all_reduce(S_tensor, op=dist.ReduceOp.SUM)
    with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write("before all_reduce\n")
    S = float(S_tensor.item())
    dyn_p_S = 0.5 * 1.0 * u_in**2 * S

    # D3Q19 constants
    c = C19.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    w19 = torch.tensor([1/3]+[1/18]*6+[1/36]*12,
                       dtype=torch.float32, device=device).view(19, 1, 1, 1)
    cs2 = 1.0/3.0
    fluid = ~solid
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    near = nbrs

    # Init
    rho0 = torch.ones(nz, ny, nx_halo, device=device)
    ux0 = torch.full((nz, ny, nx_halo), u_in, device=device); ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    def wall_fn_19(f, nu, y_val=0.5):
        rho, ux, uy, uz = macroscopic3d(f)
        with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write("  wall_fn: after macro\n")
        u_mag = torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
        u_tau = torch.sqrt(nu*u_mag/y_val).clamp(min=1e-12)
        y_plus = y_val*u_tau/nu; turb = (y_plus>11.6)&near
        with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write("  wall_fn: before turb.any\n")
        # Vectorized Newton iteration (no advanced indexing — avoids SDAA→CPU sync deadlock under TCCL)
        ut = u_tau.clone(); um = u_mag
        for _ in range(8):
            lyp = torch.log(y_val*ut/nu); fv = ut*(lyp/KAPPA+B_CONST)-um
            fp = (lyp/KAPPA+B_CONST)+1.0/KAPPA; ut = (ut-fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau = torch.where(turb, ut, u_tau)
        tau_w = u_tau*u_tau; inv_umag = 1.0/u_mag; coef = -(tau_w/y_val)*near.to(f.dtype)
        fx = coef*(ux*inv_umag); fy = coef*(uy*inv_umag); fz = coef*(uz*inv_umag)
        cu = cx*ux + cy*uy + cz*uz
        forcing = w19 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
        f = f + forcing
        df = (tau_w*(ux*inv_umag)*near.to(f.dtype)).sum()
        p = (rho-1.0)/3.0
        sp = torch.roll(solid, 1, dims=2); sm = torch.roll(solid, -1, dims=2)
        dp = (p*(sm.to(f.dtype)-sp.to(f.dtype))*fluid.to(f.dtype)).sum()
        return f, df, dp

    def far_field_19(f, u_in=0.06):
        nz, ny, nx_l = f.shape[1], f.shape[2], f.shape[3]
        rho1 = torch.ones(nz, ny, nx_l, dtype=f.dtype, device=f.device)
        feq = equilibrium3d(rho1, torch.full_like(rho1, u_in),
                            torch.zeros_like(rho1), torch.zeros_like(rho1))
        f = f.clone()
        f[:, :, :, 0] = feq[:, :, :, 0]; f[:, :, :, -1] = f[:, :, :, -2]
        f[:, 0, :, :] = feq[:, 0, :, :]; f[:, -1, :, :] = feq[:, -1, :, :]
        f[:, :, 0, :] = feq[:, :, 0, :]; f[:, :, -1, :] = feq[:, :, -1, :]
        return f

    fric_list = []; pres_list = []
    t0 = time.time(); t_step_total = 0.0
    total_cells = nx * ny * nz

    with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write("before loop\n")
    for step in range(1, n_steps + 1):
        ts = time.time()
        f = collide_mrt3d(f, tau=tau)
        # Exchange post-collision populations before local periodic streaming.
        # Exchanging after torch.roll only repairs halo cells; it leaves the
        # first/last physical x planes contaminated by a same-rank wraparound.
        if step <= 3:
            with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write(f"step {step} before halo\n")
        halo_exchange(f, rank, world_size)
        f = stream19_roll(f)
        if step <= 3:
            with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write(f"step {step} after halo\n")
        f, df_local, dp_local = wall_fn_19(f, nu_lat, y_val=y_val)
        if step <= 3:
            with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write(f"step {step} after wall_fn\n")
        f = far_field_19(f, u_in=u_in)
        if step <= 3:
            with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write(f"step {step} after far_field\n")
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)
        t_step_total += time.time() - ts

        if step > warmup:
            drag_tensor = torch.tensor([df_local, dp_local], device=device, dtype=torch.float32)
            if step <= 3:
                with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write(f"step {step} before drag all_reduce\n")
            dist.all_reduce(drag_tensor, op=dist.ReduceOp.SUM)
            if step <= 3:
                with open(f"/tmp/mc_{rank}.log","a") as _f: _f.write(f"step {step} after drag all_reduce\n")
            fric_list.append(float(drag_tensor[0].item()))
            pres_list.append(float(drag_tensor[1].item()))

        if step % 100 == 0 or step == n_steps:
            cf = sum(fric_list)/max(len(fric_list),1)/dyn_p_S
            cp = sum(pres_list)/max(len(pres_list),1)/dyn_p_S
            avg = t_step_total/step; mlups = total_cells/avg/1e6
            if is_main:
                print(f"  step {step:4d}: Ct_f={cf:.4f} Ct_p={cp:.4f} Ct={cf+cp:.4f} "
                      f"{avg*1000:.0f}ms/step {mlups:.1f}MLUPS", flush=True)

    cf = sum(fric_list)/max(len(fric_list),1)/dyn_p_S
    cp = sum(pres_list)/max(len(pres_list),1)/dyn_p_S
    total = time.time()-t0; avg = t_step_total/n_steps; mlups = total_cells/avg/1e6

    if is_main:
        print(f"\n{'='*60}")
        print(f"Final: Ct_fric={cf:.4f} Ct_pres={cp:.4f} Ct_total={cf+cp:.4f}")
        print(f"  (exp ~0.004, ratio {(cf+cp)/0.004:.2f}x)")
        print(f"Perf: {avg*1000:.0f}ms/step | {mlups:.1f}MLUPS | {total:.1f}s")
        print(f"Cards: {world_size} | D3Q19 MRT | Grid: {nx}x{ny}x{nz}")
        print(f"{'='*60}")

    _cleanup()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=384)
    p.add_argument("--ny", type=int, default=160)
    p.add_argument("--nz", type=int, default=160)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--hull", type=float, default=160.0)
    a = p.parse_args()
    run_multicard(nx=a.nx, ny=a.ny, nz=a.nz, n_steps=a.steps, warmup=a.warmup, hull_length=a.hull)
