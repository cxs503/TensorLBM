"""Runners for XVAL-LBM-CARTESIAN-MPI: monolithic and domain-decomposed Poiseuille flow.

Two runners:
1. ``run_monolithic``: standard single-domain TensorLBM D2Q9 BGK solver
2. ``run_decomposed``: domain-decomposed solver emulating pylbm's cartesian-MPI
   approach (halo exchange between subdomains, same collision/streaming kernels)

Both produce identical artefact schemas for fair comparison.

The simulation is initialized with the analytical Poiseuille profile and run
for a moderate number of steps to verify that the steady-state solution is
preserved (mass conservation, velocity profile maintenance).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .case_spec import PoiseuilleSpec
from .domain_decomp import CartesianDecomposition

HIST_HEADER = "step,umax_centreline,umax_computed,l2_error_pct"

# D2Q9 lattice velocities
C = torch.tensor(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=torch.int64,
)
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)


def _write_history(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HIST_HEADER + "\n")
        for r in rows:
            fh.write(",".join(repr(float(x)) for x in r) + "\n")


def _poiseuille_init_analytical(spec: PoiseuilleSpec, nx: int, ny: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Initialize D2Q9 distributions with the analytical Poiseuille profile."""
    from ..d2q9 import equilibrium

    jj, ii = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64, device=device),
        torch.arange(nx, dtype=torch.float64, device=device),
        indexing="ij",
    )
    y = jj + 0.5
    u = 4.0 * spec.u_max / (spec.H ** 2) * y * (spec.H - y)
    v = torch.zeros_like(u)
    rho = torch.ones_like(u)

    f = equilibrium(rho.to(dtype), u.to(dtype), v.to(dtype), device=device)
    return f


def _body_force_x(spec: PoiseuilleSpec) -> float:
    """Compute the body force F_x needed to drive Poiseuille flow.

    F_x = 8 * nu * U_max / H^2
    """
    return 8.0 * spec.nu * spec.u_max / (spec.H ** 2)


def _collide_bgk_with_force(
    f: torch.Tensor,
    tau: float,
    force_x: float,
) -> torch.Tensor:
    """BGK collision with Guo forcing scheme.

    Modified equilibrium velocity: u_eq = u + tau * F / rho
    """
    from ..d2q9 import equilibrium, macroscopic

    rho, ux, uy = macroscopic(f)
    ux_eq = ux + tau * force_x / rho
    uy_eq = uy
    feq = equilibrium(rho, ux_eq, uy_eq)
    return f - (f - feq) / tau


def _stream_and_bounce_back(f: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
    """Streaming with periodic x and half-way bounce-back at y=0 and y=ny-1 walls."""
    f_new = torch.zeros_like(f)

    for q in range(9):
        cx, cy = C[q, 0].item(), C[q, 1].item()
        if cy == 0:
            f_new[q] = torch.roll(f[q], shifts=cx, dims=1)
        elif cy > 0:
            src = f[q, :, :]
            if cx != 0:
                src = torch.roll(src, shifts=cx, dims=1)
            f_new[q, 1:, :] = src[:-1, :]
        else:
            src = f[q, :, :]
            if cx != 0:
                src = torch.roll(src, shifts=cx, dims=1)
            f_new[q, :-1, :] = src[1:, :]

    # Bounce-back at walls
    for q in range(9):
        cy = C[q, 1].item()
        if cy > 0:
            opp = OPPOSITE[q].item()
            f_new[q, 0, :] = f_new[opp, 0, :]
        if cy < 0:
            opp = OPPOSITE[q].item()
            f_new[q, ny - 1, :] = f_new[opp, ny - 1, :]

    return f_new


def _macroscopic(f: torch.Tensor) -> tuple:
    """Compute macroscopic quantities."""
    from ..d2q9 import macroscopic
    return macroscopic(f)


# ------------------------------------------------------------------ monolithic runner

def run_monolithic(
    spec: PoiseuilleSpec,
    out_dir: str | Path,
    *,
    dtype: str = "float64",
    threads: int = 8,
) -> dict[str, Any]:
    """Run Poiseuille flow with monolithic (single-domain) TensorLBM."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(threads)))
    dt = torch.float64 if dtype == "float64" else torch.float32
    device = torch.device("cpu")

    nx, ny = spec.nx, spec.ny
    tau = spec.tau_computed
    force_x = _body_force_x(spec)

    # Initialize with analytical profile
    f = _poiseuille_init_analytical(spec, nx, ny, dt, device)

    history = []
    t0 = time.time()
    status = "ok"

    for step in range(1, spec.steps + 1):
        f = _collide_bgk_with_force(f, tau, force_x)
        f = _stream_and_bounce_back(f, ny, nx)

        if step % spec.sample_every == 0 or step == spec.steps:
            rho, ux, uy = _macroscopic(f)
            centre_idx = ny // 2
            u_centreline = float(ux[centre_idx, nx // 2])
            u_max_computed = float(ux.max())

            jj = torch.arange(ny, dtype=torch.float64, device=device) + 0.5
            u_analytical = torch.tensor(
                [spec.analytical_velocity(float(y)) for y in jj.cpu().numpy()],
                dtype=torch.float64, device=device,
            )
            ux_avg = ux.mean(dim=1)
            l2_num = torch.sqrt(((ux_avg - u_analytical.to(dt)) ** 2).sum())
            l2_den = torch.sqrt((u_analytical.to(dt) ** 2).sum())
            l2_error = float(l2_num / l2_den * 100.0) if l2_den > 0 else 0.0

            history.append((step, u_centreline, u_max_computed, l2_error))

    wall_time = time.time() - t0

    _write_history(out / "k_history_monolithic.csv", history)

    rho, ux, uy = _macroscopic(f)
    field_path = out / "field_monolithic_final.csv"
    with open(field_path, "w") as fh:
        fh.write("i,j,ux,uy\n")
        ux_np = ux.cpu().numpy()
        uy_np = uy.cpu().numpy()
        for j in range(ny):
            for i in range(nx):
                fh.write(f"{i},{j},{ux_np[j, i]:.10e},{uy_np[j, i]:.10e}\n")

    run_info = {
        "tool": "tensorlbm_monolithic",
        "dtype": dtype,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "force_x": force_x,
        "re": spec.re,
        "u_max_target": spec.u_max,
        "steps": spec.steps,
        "wall_time_s": round(wall_time, 2),
        "status": status,
        "n_samples": len(history),
    }
    with open(out / "run_info_monolithic.json", "w") as fh:
        json.dump(run_info, fh, indent=2)

    return {
        "tool": "tensorlbm_monolithic",
        "history": {
            "step": [r[0] for r in history],
            "umax_centreline": [r[1] for r in history],
            "umax_computed": [r[2] for r in history],
            "l2_error_pct": [r[3] for r in history],
        },
        "run_info": run_info,
        "status": status,
    }


# ------------------------------------------------------------------ decomposed runner

def run_decomposed(
    spec: PoiseuilleSpec,
    out_dir: str | Path,
    *,
    dtype: str = "float64",
    threads: int = 8,
) -> dict[str, Any]:
    """Run Poiseuille flow with domain-decomposed TensorLBM (pylbm cartesian-MPI emulation)."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(threads)))
    dt = torch.float64 if dtype == "float64" else torch.float32
    device = torch.device("cpu")

    decomp = CartesianDecomposition(
        nx=spec.nx, ny=spec.ny,
        npx=spec.npx, npy=spec.npy,
        halo=spec.halo,
        periodic=(True, False),
    )
    tau = spec.tau_computed
    force_x = _body_force_x(spec)

    sub_f = []
    sub_is_wall_bottom = []
    sub_is_wall_top = []
    for sd in decomp.subdomains:
        from ..d2q9 import equilibrium

        jj, ii = torch.meshgrid(
            torch.arange(sd.ny_local, dtype=torch.float64, device=device),
            torch.arange(sd.nx_local, dtype=torch.float64, device=device),
            indexing="ij",
        )
        gy = jj - spec.halo + sd.y_start + 0.5
        u = 4.0 * spec.u_max / (spec.H ** 2) * gy * (spec.H - gy)
        v = torch.zeros_like(u)
        rho = torch.ones_like(u)
        f = equilibrium(rho.to(dt), u.to(dt), v.to(dt), device=device)
        sub_f.append(f)

        sub_is_wall_bottom.append(sd.y_start == 0)
        sub_is_wall_top.append(sd.y_end == spec.ny)

    history = []
    t0 = time.time()
    status = "ok"

    for step in range(1, spec.steps + 1):
        for r in range(decomp.nprocs):
            sub_f[r] = _collide_bgk_with_force(sub_f[r], tau, force_x)

        sub_f = _halo_exchange(decomp, sub_f)

        for r in range(decomp.nprocs):
            sd = decomp.subdomains[r]
            sub_f[r] = _stream_and_bounce_back_local(
                sub_f[r], sd.ny_local, sd.nx_local,
                is_wall_bottom=sub_is_wall_bottom[r],
                is_wall_top=sub_is_wall_top[r],
                halo=spec.halo,
            )

        if step % spec.sample_every == 0 or step == spec.steps:
            ux_global = torch.zeros(spec.ny, spec.nx, dtype=dt, device=device)
            for sd in decomp.subdomains:
                r = sd.rank
                rho, ux, uy = _macroscopic(sub_f[r])
                interior = decomp.interior_slice(r)
                y_slice = slice(sd.y_start, sd.y_end)
                x_slice = slice(sd.x_start, sd.x_end)
                ux_global[y_slice, x_slice] = ux[interior]

            centre_idx = spec.ny // 2
            u_centreline = float(ux_global[centre_idx, spec.nx // 2])
            u_max_computed = float(ux_global.max())

            jj = torch.arange(spec.ny, dtype=torch.float64, device=device) + 0.5
            u_analytical = torch.tensor(
                [spec.analytical_velocity(float(y)) for y in jj.cpu().numpy()],
                dtype=torch.float64, device=device,
            )
            ux_avg = ux_global.mean(dim=1)
            l2_num = torch.sqrt(((ux_avg - u_analytical.to(dt)) ** 2).sum())
            l2_den = torch.sqrt((u_analytical.to(dt) ** 2).sum())
            l2_error = float(l2_num / l2_den * 100.0) if l2_den > 0 else 0.0

            history.append((step, u_centreline, u_max_computed, l2_error))

    wall_time = time.time() - t0

    _write_history(out / "k_history_decomposed.csv", history)

    ux_global = torch.zeros(spec.ny, spec.nx, dtype=dt, device=device)
    uy_global = torch.zeros(spec.ny, spec.nx, dtype=dt, device=device)
    for sd in decomp.subdomains:
        r = sd.rank
        rho, ux, uy = _macroscopic(sub_f[r])
        interior = decomp.interior_slice(r)
        y_slice = slice(sd.y_start, sd.y_end)
        x_slice = slice(sd.x_start, sd.x_end)
        ux_global[y_slice, x_slice] = ux[interior]
        uy_global[y_slice, x_slice] = uy[interior]

    field_path = out / "field_decomposed_final.csv"
    with open(field_path, "w") as fh:
        fh.write("i,j,ux,uy\n")
        ux_np = ux_global.cpu().numpy()
        uy_np = uy_global.cpu().numpy()
        for j in range(spec.ny):
            for i in range(spec.nx):
                fh.write(f"{i},{j},{ux_np[j, i]:.10e},{uy_np[j, i]:.10e}\n")

    run_info = {
        "tool": "tensorlbm_decomposed",
        "dtype": dtype,
        "nx": spec.nx,
        "ny": spec.ny,
        "npx": spec.npx,
        "npy": spec.npy,
        "nprocs": decomp.nprocs,
        "tau": tau,
        "force_x": force_x,
        "re": spec.re,
        "u_max_target": spec.u_max,
        "steps": spec.steps,
        "wall_time_s": round(wall_time, 2),
        "status": status,
        "n_samples": len(history),
        "pylbm_approach": "cartesian-MPI emulation (halo exchange)",
    }
    with open(out / "run_info_decomposed.json", "w") as fh:
        json.dump(run_info, fh, indent=2)

    return {
        "tool": "tensorlbm_decomposed",
        "history": {
            "step": [r[0] for r in history],
            "umax_centreline": [r[1] for r in history],
            "umax_computed": [r[2] for r in history],
            "l2_error_pct": [r[3] for r in history],
        },
        "run_info": run_info,
        "status": status,
    }


def _stream_and_bounce_back_local(
    f: torch.Tensor,
    ny_local: int,
    nx_local: int,
    is_wall_bottom: bool,
    is_wall_top: bool,
    halo: int,
) -> torch.Tensor:
    """Streaming with periodic x and bounce-back at physical y walls."""
    f_new = torch.zeros_like(f)

    for q in range(9):
        cx, cy = C[q, 0].item(), C[q, 1].item()
        if cy == 0:
            f_new[q] = torch.roll(f[q], shifts=cx, dims=1)
        elif cy > 0:
            src = f[q, :, :]
            if cx != 0:
                src = torch.roll(src, shifts=cx, dims=1)
            f_new[q, 1:, :] = src[:-1, :]
        else:
            src = f[q, :, :]
            if cx != 0:
                src = torch.roll(src, shifts=cx, dims=1)
            f_new[q, :-1, :] = src[1:, :]

    if is_wall_bottom:
        wall_y = halo
        for q in range(9):
            cy = C[q, 1].item()
            if cy > 0:
                opp = OPPOSITE[q].item()
                f_new[q, wall_y, :] = f_new[opp, wall_y, :]

    if is_wall_top:
        wall_y = ny_local - halo - 1
        for q in range(9):
            cy = C[q, 1].item()
            if cy < 0:
                opp = OPPOSITE[q].item()
                f_new[q, wall_y, :] = f_new[opp, wall_y, :]

    return f_new


def _halo_exchange(decomp: CartesianDecomposition, sub_f: list[torch.Tensor]) -> list[torch.Tensor]:
    """Exchange halo values between neighbouring subdomains."""
    new_f = [f.clone() for f in sub_f]

    for sd in decomp.subdomains:
        r = sd.rank
        f = sub_f[r]
        h = decomp.halo

        left = sd.neighbours[0]
        if left is not None:
            new_f[left][:, :, -h:] = f[:, :, h:2 * h].clone()

        right = sd.neighbours[1]
        if right is not None:
            new_f[right][:, :, :h] = f[:, :, -2 * h:-h].clone()

        bottom = sd.neighbours[2]
        if bottom is not None:
            new_f[bottom][:, -h:, :] = f[:, h:2 * h, :].clone()

        top = sd.neighbours[3]
        if top is not None:
            new_f[top][:, :h, :] = f[:, -2 * h:-h, :].clone()

    return new_f
