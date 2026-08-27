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

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist

from tensorlbm.d3q19 import C as C19

KAPPA = 0.41
B_CONST = 5.0
_SUBOFF_L_OVER_D = 8.57
_MIN_ABSOLUTE_CT_DIAMETER_CELLS = 24.0


def validate_suboff_diagnostic(
    *, nx: int, ny: int, nz: int, hull_length: float, n_steps: int, warmup: int, u_in: float
) -> None:
    """Fail closed for a smooth-body, time-windowed SUBOFF diagnostic.

    This is deliberately a diagnostic gate, not a stability condition.  A
    stair-stepped hull below 24 nodes over D cannot support an absolute Ct
    comparison.  Also, the averaging window must start only after the initial
    disturbance can travel from the hull stern to the outlet.
    """
    diameter = hull_length / _SUBOFF_L_OVER_D
    if diameter < _MIN_ABSOLUTE_CT_DIAMETER_CELLS:
        raise ValueError(
            f"SUBOFF absolute-Ct diagnostic requires D >= "
            f"{_MIN_ABSOLUTE_CT_DIAMETER_CELLS:g} cells; got {diameter:.2f}"
        )
    if not (0.0 < u_in < 0.15):
        raise ValueError("u_in must lie in (0, 0.15) for this low-Mach diagnostic")
    if nx < hull_length + 2.0 * diameter or min(ny, nz) < 6.0 * diameter:
        raise ValueError("domain is too tight for the resolved hull/wake diagnostic")
    stern = nx * 0.35 + hull_length / 2.0
    outlet_settle_steps = math.ceil((nx - stern) / u_in)
    if warmup < outlet_settle_steps:
        raise ValueError(
            "warmup is shorter than stern-to-outlet convection time: "
            f"need >= {outlet_settle_steps}, got {warmup}"
        )
    if n_steps <= warmup:
        raise ValueError("n_steps must exceed warmup")


def pressure_drag_x_19(pressure, solid, interior=None):
    """Voxel-face pressure force on body, with positive streamwise drag."""
    if interior is None:
        interior = torch.ones_like(solid)
    fluid = ~solid
    solid_at_plus_x = torch.roll(solid, 1, dims=2)
    solid_at_minus_x = torch.roll(solid, -1, dims=2)
    return (
        pressure
        * (solid_at_minus_x.to(pressure.dtype) - solid_at_plus_x.to(pressure.dtype))
        * fluid.to(pressure.dtype)
        * interior.to(pressure.dtype)
    ).sum()


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


def run_multicard(
    nx=416,
    ny=208,
    nz=208,
    n_steps=3600,
    warmup=2800,
    re=2e6,
    hull_length=206.0,
    u_in=0.06,
    y_val=0.5,
    report_path=None,
):
    validate_suboff_diagnostic(
        nx=nx, ny=ny, nz=nz, hull_length=hull_length, n_steps=n_steps, warmup=warmup, u_in=u_in
    )
    rank, world_size, device = _setup()
    is_main = rank == 0
    assert nx % world_size == 0
    nx_local = nx // world_size
    nx_halo = nx_local + 2
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5

    if is_main:
        print(f"D3Q19 MRT Multi-card: {world_size} cards")
        print(f"Grid: {nx}x{ny}x{nz} = {nx * ny * nz:,} cells ({nx * ny * nz / 1e6:.1f}M)")
        print(f"Per card: {nx_local}x{ny}x{nz} = {nx_local * ny * nz:,} cells")
        print(f"Re={re:.0e} tau={tau:.5f} | Experimental AFF-8 Ct ~ 0.004\n")

    from tensorlbm.d3q19 import C as C19
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_mrt3d
    from tensorlbm.suboff_cad import build_suboff_mask
    from tensorlbm.suboff_resistance import voxel_wetted_area_x_slab

    # Build mask on CPU, slice, transfer to SDAA
    cx_global = nx * 0.35
    x_start = rank * nx_local
    x_end = x_start + nx_local
    full_solid, _ = build_suboff_mask(
        hull_type="full",
        nx=nx,
        ny=ny,
        nz=nz,
        cx=cx_global,
        cy=ny / 2.0,
        cz=nz / 2.0,
        length=hull_length,
        device="cpu",
    )
    left_halo_idx = (x_start - 1) % nx
    right_halo_idx = x_end % nx
    solid = torch.zeros(nz, ny, nx_halo, dtype=torch.bool, device=device)
    solid[:, :, 1:-1] = full_solid[:, :, x_start:x_end].to(device)
    solid[:, :, 0] = full_solid[:, :, left_halo_idx].to(device)
    solid[:, :, -1] = full_solid[:, :, right_halo_idx].to(device)
    del full_solid

    S_local = voxel_wetted_area_x_slab(
        solid[:, :, 1:-1],
        1.0,
        has_left_neighbor=rank > 0,
        has_right_neighbor=rank < world_size - 1,
    )
    S_tensor = torch.tensor([S_local], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(S_tensor, op=dist.ReduceOp.SUM)
    S = float(S_tensor.item())
    dyn_p_S = 0.5 * 1.0 * u_in**2 * S

    # D3Q19 constants
    c = C19.to(device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    w19 = torch.tensor(
        [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=torch.float32, device=device
    ).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    fluid = ~solid
    interior = torch.zeros_like(solid)
    interior[:, :, 1:-1] = True
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        nbrs |= torch.roll(solid, sgn, dims=ax) & fluid
    near = nbrs

    # Init
    rho0 = torch.ones(nz, ny, nx_halo, device=device)
    ux0 = torch.full((nz, ny, nx_halo), u_in, device=device)
    ux0[solid] = 0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    target_mass = torch.tensor(float(nx * ny * nz), device=device, dtype=f.dtype)

    def wall_fn_19(f, nu, y_val=0.5):
        rho, ux, uy, uz = macroscopic3d(f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        u_tau = torch.sqrt(nu * u_mag / y_val).clamp(min=1e-12)
        y_plus = y_val * u_tau / nu
        turb = (y_plus > 11.6) & near
        # Vectorized Newton iteration (no advanced indexing — avoids SDAA→CPU sync deadlock under TCCL)
        ut = u_tau.clone()
        um = u_mag
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu)
            fv = ut * (lyp / KAPPA + B_CONST) - um
            fp = (lyp / KAPPA + B_CONST) + 1.0 / KAPPA
            ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau = torch.where(turb, ut, u_tau)
        force_cells = near & interior
        tau_w = u_tau * u_tau
        inv_umag = 1.0 / u_mag
        coef = -(tau_w / y_val) * force_cells.to(f.dtype)
        fx = coef * (ux * inv_umag)
        fy = coef * (uy * inv_umag)
        fz = coef * (uz * inv_umag)
        cu = cx * ux + cy * uy + cz * uz
        forcing = w19 * (1.0 + cu / cs2) * (cx * fx + cy * fy + cz * fz) / cs2
        f = f + forcing
        df = (tau_w * (ux * inv_umag) * force_cells.to(f.dtype)).sum()
        p = (rho - 1.0) / 3.0
        dp = pressure_drag_x_19(p, solid, interior)
        return f, df, dp

    def far_field_19(f, u_in=0.06):
        nz, ny, nx_l = f.shape[1], f.shape[2], f.shape[3]
        rho1 = torch.ones(nz, ny, nx_l, dtype=f.dtype, device=f.device)
        feq = equilibrium3d(
            rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )
        f = f.clone()
        if rank == 0:
            f[:, :, :, 1] = feq[:, :, :, 1]
        if rank == world_size - 1:
            f[:, :, :, -2] = f[:, :, :, -3]
        f[:, 0, :, :] = feq[:, 0, :, :]
        f[:, -1, :, :] = feq[:, -1, :, :]
        f[:, :, 0, :] = feq[:, :, 0, :]
        f[:, :, -1, :] = feq[:, :, -1, :]
        return f

    fric_list = []
    pres_list = []
    t0 = time.time()
    t_step_total = 0.0
    total_cells = nx * ny * nz

    for step in range(1, n_steps + 1):
        ts = time.time()
        f = collide_mrt3d(f, tau=tau)
        # Exchange post-collision populations before local periodic streaming.
        # Exchanging after torch.roll only repairs halo cells; it leaves the
        # first/last physical x planes contaminated by a same-rank wraparound.
        halo_exchange(f, rank, world_size)
        f = stream19_roll(f)
        f, df_local, dp_local = wall_fn_19(f, nu_lat, y_val=y_val)
        f = far_field_19(f, u_in=u_in)
        if step % 100 == 0:
            interior_mass = f[:, :, :, 1:-1].sum()
            if world_size > 1:
                dist.all_reduce(interior_mass, op=dist.ReduceOp.SUM)
            f = f * (target_mass / interior_mass)
        t_step_total += time.time() - ts

        if step > warmup:
            drag_tensor = torch.stack((df_local, dp_local)).to(dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(drag_tensor, op=dist.ReduceOp.SUM)
            fric_list.append(float(drag_tensor[0].item()))
            pres_list.append(float(drag_tensor[1].item()))

        if step % 100 == 0 or step == n_steps:
            cf = sum(fric_list) / max(len(fric_list), 1) / dyn_p_S
            cp = sum(pres_list) / max(len(pres_list), 1) / dyn_p_S
            avg = t_step_total / step
            mlups = total_cells / avg / 1e6
            if is_main:
                print(
                    f"  step {step:4d}: Ct_f={cf:.4f} Ct_p={cp:.4f} Ct={cf + cp:.4f} "
                    f"{avg * 1000:.0f}ms/step {mlups:.1f}MLUPS",
                    flush=True,
                )

    cf = sum(fric_list) / max(len(fric_list), 1) / dyn_p_S
    cp = sum(pres_list) / max(len(pres_list), 1) / dyn_p_S
    total = time.time() - t0
    avg = t_step_total / n_steps
    mlups = total_cells / avg / 1e6
    finite_local = torch.isfinite(f).all().to(torch.int32)
    mass_local = f[:, :, :, 1:-1].sum()
    if world_size > 1:
        dist.all_reduce(finite_local, op=dist.ReduceOp.MIN)
        dist.all_reduce(mass_local, op=dist.ReduceOp.SUM)

    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Final: Ct_fric={cf:.4f} Ct_pres={cp:.4f} Ct_total={cf + cp:.4f}")
        print(
            f"  finite={bool(finite_local.item())} mass={mass_local.item():.6g} target={target_mass.item():.6g}"
        )
        print(f"Perf: {avg * 1000:.0f}ms/step | {mlups:.1f}MLUPS | {total:.1f}s")
        print(f"Cards: {world_size} | D3Q19 MRT | Grid: {nx}x{ny}x{nz}")
        print(f"{'=' * 60}")
        if report_path:
            report = {
                "lattice": "D3Q19 MRT",
                "cards": world_size,
                "mesh": [nx, ny, nz],
                "hull_length": hull_length,
                "diameter_cells": hull_length / _SUBOFF_L_OVER_D,
                "steps": n_steps,
                "warmup": warmup,
                "samples": len(fric_list),
                "ct_friction": cf,
                "ct_pressure": cp,
                "ct_total": cf + cp,
                "fields_finite": bool(finite_local.item()),
                "mass": float(mass_local.item()),
                "target_mass": float(target_mass.item()),
                "mass_relative_error": float((mass_local / target_mass - 1).item()),
                "seconds_per_step": avg,
                "mlups": mlups,
            }
            with open(report_path, "w", encoding="utf-8") as out:
                json.dump(report, out, indent=2)

    _cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=416)
    p.add_argument("--ny", type=int, default=208)
    p.add_argument("--nz", type=int, default=208)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--warmup", type=int, default=2800)
    p.add_argument("--hull", type=float, default=206.0)
    p.add_argument("--re", type=float, default=2e6)
    p.add_argument("--u-in", type=float, default=0.06)
    p.add_argument("--report", default=None)
    a = p.parse_args()
    run_multicard(
        nx=a.nx,
        ny=a.ny,
        nz=a.nz,
        n_steps=a.steps,
        warmup=a.warmup,
        hull_length=a.hull,
        re=a.re,
        u_in=a.u_in,
        report_path=a.report,
    )
