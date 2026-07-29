"""Cylinder Re=200 Cumulant+Smag D3Q27 worker.

D3Q27 lattice needs different streaming (roll-based), equilibrium, and
mass correction.  Pressure-integration drag is computed inline since
drag_pressure_integration uses D3Q19 macroscopic3d.

Usage:
    python _cyl_collision_d3q27_worker.py <device_id> <output_path>
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

# D3Q27 velocity set and opposite-direction table (computed from d3q27.C at runtime)
_C27 = None
_OPP27 = None


def _ensure_c27(device):
    """Lazily build C27 and opposite-index table on the target device."""
    global _C27, _OPP27
    if _C27 is not None and _C27.device == device:
        return
    from tensorlbm.d3q27 import C as C27
    _C27 = C27.to(device).int()
    opp = []
    for q in range(27):
        target = -_C27[q]
        for q2 in range(27):
            if int(_C27[q2, 0]) == int(target[0]) and \
               int(_C27[q2, 1]) == int(target[1]) and \
               int(_C27[q2, 2]) == int(target[2]):
                opp.append(q2)
                break
    _OPP27 = torch.tensor(opp, device=device, dtype=torch.long)


def stream27_roll(f):
    """Stream D3Q27 populations via torch.roll (periodic), using d3q27.C ordering."""
    _ensure_c27(f.device)
    out = torch.empty_like(f)
    for q in range(27):
        sx = int(_C27[q, 0])
        sy = int(_C27[q, 1])
        sz = int(_C27[q, 2])
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def far_field_bc_27(f, u_in=0.08):
    """Far-field BC for D3Q27 (free-stream inlet, zero-gradient outlet)."""
    from tensorlbm.d3q27 import equilibrium27
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium27(
        rho1,
        torch.full_like(rho1, u_in),
        torch.zeros_like(rho1),
        torch.zeros_like(rho1),
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet
    f[:, :, :, -1] = f[:, :, :, -2]       # outlet (zero gradient)
    f[:, :, 0, :] = feq[:, :, 0, :]       # y- far-field
    f[:, :, -1, :] = feq[:, :, -1, :]     # y+ far-field
    # z± periodic for 2D extruded (nz=4 ≤ 4)
    return f


def bounce_back_27(f, solid):
    """Half-way bounce-back for D3Q27 (vectorized, like D3Q19 version)."""
    _ensure_c27(f.device)
    return torch.where(solid.unsqueeze(0), f[_OPP27], f)


def drag_pressure_27(f, mesh, dpS):
    """Pressure drag for D3Q27 (rho = sum over 27 directions)."""
    rho = f.sum(0)
    p = (rho - 1.0) / 3.0
    mask_float = mesh.near.float()
    n_near = mask_float.sum().clamp(min=1.0)
    p0 = (p * mask_float).sum() / n_near
    p_corr = p - p0
    mask = mask_float * mesh.dA
    fpx = -(p_corr * mesh.nx_n * mask).sum()
    fpy = -(p_corr * mesh.ny_n * mask).sum()
    fpz = -(p_corr * mesh.nz_n * mask).sum()
    return float(fpx.item() / dpS), float(fpy.item() / dpS), float(fpz.item() / dpS)


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def main():
    did = int(sys.argv[1])
    output_path = sys.argv[2]

    nx, ny, nz = 400, 160, 4
    D = 48.0
    R = D / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 3000
    win = 300
    ref_cd = 1.30
    Cs = 0.05

    A_frontal = D * nz
    dpS = 0.5 * 1.0 * u_in ** 2 * A_frontal

    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{did} Cumulant+Smag D3Q27]"
    print(
        f"{tag} nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} "
        f"nu={nu:.6e} tau={tau:.6f} Re={Re} Cs={Cs}",
        flush=True,
    )

    t0 = time.time()

    cx_cyl = nx * 0.25
    cy_cyl = ny * 0.5
    solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, R, device)

    from tensorlbm.drag_pressure import get_near_wall_2d, SurfaceMesh
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_cyl, cy_cyl, R, axis="z")
    sm = solid.unsqueeze(0).expand(27, nz, ny, nx)

    from tensorlbm.d3q27 import equilibrium27, macroscopic27, correct_mass27
    from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    cd_hist = []
    cl_hist = []
    lbm_time = 0.0
    all_finite = True
    final_step = 0

    for step in range(1, n_steps + 1):
        f_pre = f.clone()

        ts = time.time()
        f = collide_cumulant_smag_d3q27(f, tau, C_s=Cs)

        # NoDynamics: restore solid cells
        for q in range(27):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # Half-way bounce-back
        f = bounce_back_27(f, solid)

        # Streaming
        f = stream27_roll(f)

        # Far-field BC
        f = far_field_bc_27(f, u_in)

        # Mass correction
        if step % 200 == 0:
            f = correct_mass27(f, initial_mass)
        lbm_time += time.time() - ts

        # Drag / lift
        cd, cl, _ = drag_pressure_27(f, mesh, dpS)
        cd_hist.append(cd)
        cl_hist.append(cl)
        final_step = step

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            all_finite = False
            break

        if step % 500 == 0 or step == n_steps:
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd={cd:.4f} Cl={cl:.4f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    tail_cd = cd_hist[-win:] if len(cd_hist) >= win else cd_hist
    tail_cl = cl_hist[-win:] if len(cl_hist) >= win else cl_hist

    cd_mean = sum(tail_cd) / max(len(tail_cd), 1)
    cl_mean = sum(tail_cl) / max(len(tail_cl), 1)
    cl_max = max(tail_cl) if tail_cl else 0.0
    cl_min = min(tail_cl) if tail_cl else 0.0
    cl_amp = (cl_max - cl_min) / 2.0
    cl_rms = math.sqrt(sum(c * c for c in tail_cl) / max(len(tail_cl), 1))

    err_pct = abs(cd_mean - ref_cd) / ref_cd * 100 if ref_cd > 0 else float("nan")
    time_per_step = lbm_time / max(final_step, 1)

    result = {
        "collision": "Cumulant+Smag (D3Q27)",
        "device": f"sdaa:{did}",
        "lattice": "D3Q27",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D,
        "u_in": u_in,
        "Re": Re,
        "nu": nu,
        "tau": tau,
        "Cs": Cs,
        "n_steps": n_steps,
        "win": win,
        "Cd": cd_mean,
        "Cl_mean": cl_mean,
        "Cl_amp": cl_amp,
        "Cl_rms": cl_rms,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "time_per_step_s": time_per_step,
        "lbm_time_s": lbm_time,
        "total_elapsed_s": elapsed,
        "steps_run": final_step,
        "finite": all_finite,
    }
    print(
        f"{tag} DONE Cd={cd_mean:.4f} (ref={ref_cd}) err={err_pct:.1f}% "
        f"Cl_amp={cl_amp:.4f} t/step={time_per_step:.4f}s ({elapsed:.0f}s)",
        flush=True,
    )
    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
