"""Multi-card parallel D3Q27 Cumulant SUBOFF solver.

Splits grid along X across N SDAA cards with halo exchange.
Each card runs Cumulant collision + torch.roll streaming locally.

Usage:
    torchrun --nproc_per_node=4 examples/dg_suboff_cumulant_d3q27_multicard.py
    torchrun --nproc_per_node=8 examples/dg_suboff_cumulant_d3q27_multicard.py --nx 640
"""
from __future__ import annotations
import math, time, argparse, os, torch
import torch.distributed as dist
from tensorlbm.d3q27 import C as C27
from tensorlbm.suboff_farfield import build_suboff_far_field_metadata

KAPPA = 0.41
B_CONST = 5.0
# A voxel wall needs enough nodes across its curvature to distinguish physical
# pressure drag from the staircase form drag of the digital surface.  This is
# deliberately a conservative *benchmark* gate, not a solver-stability limit.
_MIN_ABSOLUTE_CT_DIAMETER_CELLS = 24.0
_SUBOFF_L_OVER_D = 8.57


def validate_suboff_voxel_resolution(hull_length: float) -> None:
    """Reject grids that cannot make an absolute smooth-SUBOFF Ct claim.

    ``SuboffConfig`` has L/D=8.57.  At fewer than 24 voxels over D, the
    D3Q27 halfway wall resolves an AFF-8 appendage/hull surface as a stepped
    bluff body; its pressure force is therefore not comparable with the
    smooth-model experimental total-resistance coefficient.
    """
    diameter_cells = float(hull_length) / _SUBOFF_L_OVER_D
    if diameter_cells < _MIN_ABSOLUTE_CT_DIAMETER_CELLS:
        raise ValueError(
            "absolute SUBOFF Ct requires a voxel diameter of at least "
            f"{_MIN_ABSOLUTE_CT_DIAMETER_CELLS:g} cells; got "
            f"{diameter_cells:.2f} (hull_length={hull_length:g}). "
            "Increase --hull to at least 206 while scaling the domain, or "
            "treat this run as a qualitative coarse-grid diagnostic."
        )


def suboff_far_field_plan(*, nx, ny, nz, hull_length, u_in, transient_steps=None):
    """Return reporting-only physical-domain/outlet observability metadata.

    This mirrors the geometry placement below and does not modify solver state
    or any boundary-condition implementation.
    """
    return build_suboff_far_field_metadata(
        nx=nx, ny=ny, nz=nz, hull_length=hull_length, u_in=u_in,
        hull_center_x=float(nx) * 0.35, transient_steps=transient_steps,
    )


# Keep streaming direction order exactly aligned with d3q27.C.  The old
# lexicographic construction used a different population ordering, so it
# streamed populations in the wrong directions.
_C27_SHIFTS = [(int(C27[q, 0]), int(C27[q, 1]), int(C27[q, 2])) for q in range(27)]

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

def stream27_roll(f):
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _C27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def pressure_drag_x_27(pressure, solid, interior=None):
    """Integrate ``-p n_x`` over voxel faces, positive in the drag direction."""
    fluid = ~solid
    solid_at_plus_x = torch.roll(solid, 1, dims=2)
    solid_at_minus_x = torch.roll(solid, -1, dims=2)
    if interior is None:
        interior = torch.ones_like(solid)
    return (pressure * (solid_at_minus_x.to(pressure.dtype) -
                        solid_at_plus_x.to(pressure.dtype)) *
            fluid.to(pressure.dtype) * interior.to(pressure.dtype)).sum()


def apply_halfway_bounce_back_27(streamed, postcollision, solid):
    """Reflect populations whose pull-streaming source is a solid cell.

    The wall function supplies modeled shear, but it does not impose an
    impermeable wall.  This standard halfway bounce-back prevents the
    collision/streaming update from transporting distributions through the
    voxelized hull.  ``postcollision`` is used because the reflected value is
    the opposite population at the adjacent fluid node before streaming.
    """
    from tensorlbm.d3q27 import OPPOSITE

    fluid = ~solid
    opposite = OPPOSITE.to(streamed.device)
    out = streamed.clone()
    for q, (sx, sy, sz) in enumerate(_C27_SHIFTS):
        solid_source = torch.roll(solid, shifts=(sz, sy, sx), dims=(0, 1, 2))
        wall_link = fluid & solid_source
        out[q] = torch.where(wall_link, postcollision[opposite[q]], out[q])
    return out

def halo_exchange(f_local, rank, world_size):
    if world_size == 1:
        f_local[:, :, :, 0:1] = f_local[:, :, :, -2:-1]
        f_local[:, :, :, -1:] = f_local[:, :, :, 1:2]
        return
    left_interior = f_local[:, :, :, 1:2].contiguous()
    right_interior = f_local[:, :, :, -2:-1].contiguous()
    left_halo = torch.empty_like(left_interior)
    right_halo = torch.empty_like(right_interior)
    left_rank = (rank - 1) % world_size
    right_rank = (rank + 1) % world_size
    # TCCL point-to-point can deadlock for SDAA tensors.  The two ordered
    # all-gathers are collective and give the same one-cell periodic halos.
    right_gather = [torch.empty_like(right_interior) for _ in range(world_size)]
    dist.all_gather(right_gather, right_interior)
    left_halo = right_gather[left_rank]
    left_gather = [torch.empty_like(left_interior) for _ in range(world_size)]
    dist.all_gather(left_gather, left_interior)
    right_halo = left_gather[right_rank]
    f_local[:, :, :, 0:1] = left_halo
    f_local[:, :, :, -1:] = right_halo


_CHECKPOINT_FORMAT = "suboff-d3q27-cumulant-xslab-v1"


def suboff_checkpoint_metadata(*, nx, ny, nz, hull_length, re, u_in, y_val,
                               world_size, rank):
    if nx % world_size:
        raise ValueError("checkpoint decomposition requires nx divisible by world_size")
    return {"format": _CHECKPOINT_FORMAT, "nx": int(nx), "ny": int(ny), "nz": int(nz),
            "hull_length": float(hull_length), "re": float(re), "u_in": float(u_in),
            "y_val": float(y_val), "world_size": int(world_size), "rank": int(rank),
            "nx_local": int(nx // world_size), "q": 27}


def _checkpoint_file(path, rank):
    return os.path.join(os.fspath(path), f"rank{rank:04d}.pt")


def save_suboff_checkpoint(path, *, f, step, metadata, target_mass, mass_cadence,
                           friction_sum, pressure_sum, drag_samples, rank):
    """Atomically persist owned populations and time-averaging continuation state."""
    os.makedirs(os.fspath(path), exist_ok=True)
    payload = {"metadata": dict(metadata), "step": int(step),
               "owned_populations": f[:, :, :, 1:-1].detach().cpu().clone(),
               "target_mass": float(target_mass.item()), "mass_cadence": int(mass_cadence),
               "friction_sum": float(friction_sum), "pressure_sum": float(pressure_sum),
               "drag_samples": int(drag_samples)}
    filename = _checkpoint_file(path, rank)
    temporary = filename + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, filename)


def load_suboff_checkpoint(path, *, metadata, shape, rank, world_size, device):
    """Load rank-owned cells and regenerate communication ghosts."""
    payload = torch.load(_checkpoint_file(path, rank), map_location="cpu", weights_only=True)
    if payload.get("metadata") != dict(metadata):
        raise ValueError("incompatible SUBOFF checkpoint metadata")
    owned = payload.get("owned_populations")
    expected_owned_shape = (shape[0], shape[1], shape[2], shape[3] - 2)
    if not isinstance(owned, torch.Tensor) or tuple(owned.shape) != expected_owned_shape:
        raise ValueError("incompatible SUBOFF checkpoint owned-population shape")
    required = ("step", "target_mass", "mass_cadence", "friction_sum", "pressure_sum", "drag_samples")
    if any(key not in payload for key in required):
        raise ValueError("incompatible SUBOFF checkpoint continuation state")
    f = torch.empty(shape, dtype=owned.dtype, device=device)
    f[:, :, :, 1:-1] = owned.to(device=device)
    halo_exchange(f, rank, world_size)
    state = {key: payload[key] for key in required}
    state["target_mass"] = torch.tensor(state["target_mass"], dtype=f.dtype, device=device)
    return f, state

def run_multicard(nx=384, ny=160, nz=160, n_steps=1000, warmup=300,
                  re=2e6, hull_length=160.0, u_in=0.06, y_val=0.5,
                  checkpoint_path=None, checkpoint_every=None, resume_path=None):
    validate_suboff_voxel_resolution(hull_length)
    rank, world_size, device = _setup()
    is_main = rank == 0
    assert nx % world_size == 0
    nx_local = nx // world_size
    nx_halo = nx_local + 2
    checkpoint_metadata = suboff_checkpoint_metadata(
        nx=nx, ny=ny, nz=nz, hull_length=hull_length, re=re, u_in=u_in,
        y_val=y_val, world_size=world_size, rank=rank)
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    if is_main:
        print(f"D3Q27 Cumulant Multi-card: {world_size} cards")
        print(f"Grid: {nx}x{ny}x{nz} = {nx*ny*nz:,} cells ({nx*ny*nz/1e6:.1f}M)")
        print(f"Per card: {nx_local}x{ny}x{nz} = {nx_local*ny*nz:,} cells")
        print(f"Re={re:.0e} tau={tau:.5f} | Experimental AFF-8 Ct ~ 0.004\n")

    from tensorlbm.d3q27 import equilibrium27, macroscopic27
    from tensorlbm.cumulant import collide_cumulant_d3q27
    from tensorlbm.suboff_cad import build_suboff_mask
    from tensorlbm.suboff_resistance import voxel_wetted_area_x_slab

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

    S_local = voxel_wetted_area_x_slab(
        solid[:, :, 1:-1], 1.0,
        has_left_neighbor=rank > 0,
        has_right_neighbor=rank < world_size - 1,
    )
    S_tensor = torch.tensor([S_local], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(S_tensor, op=dist.ReduceOp.SUM)
    S = float(S_tensor.item())
    dyn_p_S = 0.5 * 1.0 * u_in**2 * S

    # D3Q27 constants
    c = C27.to(device).float()
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)
    w27 = torch.tensor([8/27]+[2/27]*6+[1/54]*12+[1/216]*8,
                       dtype=torch.float32, device=device).view(27, 1, 1, 1)
    cs2 = 1.0/3.0
    fluid = ~solid
    # Ghost planes are communication storage, not physical control volumes;
    # never include them in force integration.
    interior = torch.zeros_like(solid)
    interior[:, :, 1:-1] = True
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    near = nbrs

    # Init
    rho0 = torch.ones(nz, ny, nx_halo, device=device)
    ux0 = torch.full((nz, ny, nx_halo), u_in, device=device); ux0[solid] = 0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    # Ghost planes are communication storage. Correct only physical control
    # volumes and use one global factor; rank-local halo-inclusive correction
    # creates partition-dependent density and pressure bias.
    target_mass = torch.tensor(float(nx * ny * nz), device=device, dtype=f.dtype)

    def wall_fn_27(f, nu, y_val=0.5):
        rho, ux, uy, uz = macroscopic27(f)
        u_mag = torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
        u_tau = torch.sqrt(nu*u_mag/y_val).clamp(min=1e-12)
        y_plus = y_val*u_tau/nu; turb = (y_plus>11.6)&near
        # Fully vectorized: avoids SDAA host synchronization and advanced
        # indexing while a TCCL process group is active.
        ut = u_tau.clone()
        for _ in range(8):
            lyp = torch.log(y_val*ut/nu); fv = ut*(lyp/KAPPA+B_CONST)-u_mag
            fp = (lyp/KAPPA+B_CONST)+1.0/KAPPA; ut = (ut-fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau = torch.where(turb, ut, u_tau)
        force_cells = near & interior
        tau_w = u_tau*u_tau; inv_umag = 1.0/u_mag; coef = -(tau_w/y_val)*force_cells.to(f.dtype)
        fx = coef*(ux*inv_umag); fy = coef*(uy*inv_umag); fz = coef*(uz*inv_umag)
        cu = cx*ux + cy*uy + cz*uz
        forcing = w27 * (1.0 + cu/cs2) * (cx*fx + cy*fy + cz*fz) / cs2
        f = f + forcing
        df = (tau_w*(ux*inv_umag)*force_cells.to(f.dtype)).sum()
        p = (rho-1.0)/3.0
        dp = pressure_drag_x_27(p, solid, interior)
        return f, df, dp

    def far_field_27(f, u_in=0.06):
        nz, ny, nx_l = f.shape[1], f.shape[2], f.shape[3]
        rho1 = torch.ones(nz, ny, nx_l, dtype=f.dtype, device=f.device)
        feq = equilibrium27(rho1, torch.full_like(rho1, u_in),
                            torch.zeros_like(rho1), torch.zeros_like(rho1))
        f = f.clone()
        # The physical X faces are the first/last interior planes; halo planes
        # are only sources for the subsequent local streaming operation.
        if rank == 0:
            f[:, :, :, 1] = feq[:, :, :, 1]
        if rank == world_size - 1:
            f[:, :, :, -2] = f[:, :, :, -3]
        f[:, 0, :, :] = feq[:, 0, :, :]; f[:, -1, :, :] = feq[:, -1, :, :]
        f[:, :, 0, :] = feq[:, :, 0, :]; f[:, :, -1, :] = feq[:, :, -1, :]
        return f

    fric_sum = 0.0; pres_sum = 0.0; drag_samples = 0
    start_step = 0
    mass_cadence = 100
    if resume_path:
        f, state = load_suboff_checkpoint(resume_path, metadata=checkpoint_metadata,
                                          shape=f.shape, rank=rank, world_size=world_size,
                                          device=device)
        start_step = state["step"]
        target_mass = state["target_mass"]
        mass_cadence = state["mass_cadence"]
        fric_sum = state["friction_sum"]
        pres_sum = state["pressure_sum"]
        drag_samples = state["drag_samples"]
        if start_step > n_steps:
            raise ValueError("checkpoint step exceeds requested n_steps")
    t0 = time.time(); t_step_total = 0.0
    total_cells = nx * ny * nz

    for step in range(start_step + 1, n_steps + 1):
        ts = time.time()
        f = collide_cumulant_d3q27(f, tau=tau)
        # Exchange post-collision populations before streaming.  Receiving
        # into ghost planes makes stream27_roll use the neighboring rank as
        # its pull source at each partition boundary.
        halo_exchange(f, rank, world_size)
        f_postcollision = f
        f = stream27_roll(f_postcollision)
        f = apply_halfway_bounce_back_27(f, f_postcollision, solid)
        f, df_local, dp_local = wall_fn_27(f, nu_lat, y_val=y_val)
        f = far_field_27(f, u_in=u_in)
        if step % mass_cadence == 0:
            interior_mass = f[:, :, :, 1:-1].sum()
            if world_size > 1:
                dist.all_reduce(interior_mass, op=dist.ReduceOp.SUM)
            if interior_mass.abs() >= 1e-30:
                f = f * (target_mass / interior_mass)
        t_step_total += time.time() - ts

        if step > warmup:
            drag_tensor = torch.tensor([df_local, dp_local], device=device, dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(drag_tensor, op=dist.ReduceOp.SUM)
            fric_sum += float(drag_tensor[0].item())
            pres_sum += float(drag_tensor[1].item())
            drag_samples += 1

        if checkpoint_path and checkpoint_every and step % checkpoint_every == 0:
            save_suboff_checkpoint(checkpoint_path, f=f, step=step,
                                   metadata=checkpoint_metadata, target_mass=target_mass,
                                   mass_cadence=mass_cadence, friction_sum=fric_sum,
                                   pressure_sum=pres_sum, drag_samples=drag_samples, rank=rank)
        if step % 100 == 0 or step == n_steps:
            cf = fric_sum/max(drag_samples,1)/dyn_p_S
            cp = pres_sum/max(drag_samples,1)/dyn_p_S
            completed = max(step - start_step, 1)
            avg = t_step_total/completed; mlups = total_cells/avg/1e6
            if is_main:
                print(f"  step {step:4d}: Ct_f={cf:.4f} Ct_p={cp:.4f} Ct={cf+cp:.4f} "
                      f"{avg*1000:.0f}ms/step {mlups:.1f}MLUPS", flush=True)

    cf = fric_sum/max(drag_samples,1)/dyn_p_S
    cp = pres_sum/max(drag_samples,1)/dyn_p_S
    total = time.time()-t0; completed = max(n_steps - start_step, 1)
    avg = t_step_total/completed; mlups = total_cells/avg/1e6

    if is_main:
        print(f"\n{'='*60}")
        print(f"Final: Ct_fric={cf:.4f} Ct_pres={cp:.4f} Ct_total={cf+cp:.4f}")
        print(f"  (exp ~0.004, ratio {(cf+cp)/0.004:.2f}x)")
        print(f"Perf: {avg*1000:.0f}ms/step | {mlups:.1f}MLUPS | {total:.1f}s")
        print(f"Cards: {world_size} | D3Q27 Cumulant | Grid: {nx}x{ny}x{nz}")
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
    p.add_argument("--re", type=float, default=2e6)
    p.add_argument("--u-in", type=float, default=0.06)
    p.add_argument("--y-val", type=float, default=0.5)
    p.add_argument("--checkpoint", default=None, help="directory for rank-owned restart checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint directory to resume")
    p.add_argument("--report-far-field-plan", action="store_true",
                   help="print physical-domain/outlet planning metadata and exit")
    a = p.parse_args()
    if a.report_far_field_plan:
        print(suboff_far_field_plan(nx=a.nx, ny=a.ny, nz=a.nz, hull_length=a.hull,
                                    u_in=a.u_in, transient_steps=a.steps))
    else:
        run_multicard(nx=a.nx, ny=a.ny, nz=a.nz, n_steps=a.steps, warmup=a.warmup,
                      hull_length=a.hull, re=a.re, u_in=a.u_in, y_val=a.y_val,
                      checkpoint_path=a.checkpoint, checkpoint_every=a.checkpoint_every,
                      resume_path=a.resume)
