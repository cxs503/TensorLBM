#!/usr/bin/env python3
"""Sphere Re=100 drag via Ladd momentum-exchange — cylinder_3d recipe transfer.

Why this file exists
--------------------
The verified 2D cylinder family and the cylinder_3d Re=40 work both take their
primary drag from the **Ladd (1994) momentum-exchange** instrument
(``obstacles.compute_obstacle_forces_3d`` == ``2·Σ_{solid} Σ_i c_ix f_i`` on the
post-stream / pre-bounce-back field), *not* from the pressure+friction
control-surface split.  On the sphere the historical failure was:

  * ``momentum_exchange_standard`` over ALL solid cells gives Cd ≈ 3.98
    (+264%) because the *interior* solid cells contribute a large spurious,
    non-self-cancelling term on a curved (non-flat) voxel blob;
  * pressure+friction gives ≈ 0.90 (−17%) because the staircase pressure
    integration misses the stagnation rise.

The cylinder_3d field audit (`_analyze_field_mem.py`) showed the fix: the
Ladd sum restricted to the **surface** solid cells (the interior term is the
spurious one) reproduces the independent 2D reference within ~1.6%.  This
runner applies the same decomposition to the sphere on the SAME mechanics
chain the verified cylinder uses:

  collide_bgk3d -> freeze solid (NoDynamics) -> bounce_back_cells_3d
  (half-way BB, pre-stream) -> stream3d -> far_field_bc_3d (legacy: free-stream
  inlet + y±/z±, zero-gradient outlet)

Sampled post-stream / pre-bounce-back, exactly like cylinder_3d.

Reported Cd
-----------
  * ``cd_mem_surface``  — PRIMARY (Ladd MEM, surface cells only)
  * ``cd_mem_all``      — all solid cells (legacy sphere definition)
  * ``cd_mem_interior`` — interior cells (spurious/self-cancelling diagnostic)
Normalisation: Cd = Fx / (½ ρ u² · π R²), ρ=1, R the sphere radius in cells.

Usage
-----
    run_mem_surface.py single D out.json [--lateral 16] [--up 4] [--down 12]
        [--steps 6000] [--device sdaa:0] [--compile-mode eager]
    run_mem_surface.py verify out_dir [--grids 16 24] [--steps 6000] ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d, sphere_mask  # noqa: E402
from tensorlbm.d3q19 import C, equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

REF_SN = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)  # 1.091731
REF_CGW = 24.0 / 100.0 * (1.0 + 0.1935 * 100.0**0.6305)  # Clift-Grace-Weber


def _surface_mask(solid: torch.Tensor) -> torch.Tensor:
    """Solid cells with at least one 6-neighbour fluid cell (x/y/z ±)."""
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def mem_force_x(f: torch.Tensor, mask: torch.Tensor) -> float:
    """Ladd MEM x-force restricted to ``mask`` (see obstacles.compute_obstacle_forces_3d)."""
    c = C.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    return float((2.0 * (cx * (f * mask.unsqueeze(0))).sum()).item())


def run_case(
    D_cells: int,
    device: str = "sdaa:0",
    n_steps: int = 6000,
    lateral_D: float = 16.0,
    up_D: float = 4.0,
    down_D: float = 12.0,
    u_in: float = 0.06,
    re: float = 100.0,
    sample_interval: int = 100,
    compile_mode: str | None = "eager",
    save_field: str | None = None,
    report_every: int = 1000,
) -> dict:
    dev = torch.device(device)
    R = D_cells / 2.0
    nx = int(round((up_D + down_D + 1.0) * D_cells))
    ny = nz = int(round(lateral_D * D_cells))
    cx = up_D * D_cells + R
    cy, cz = ny / 2.0, nz / 2.0
    nu = u_in * D_cells / re
    tau = 0.5 + 3.0 * nu
    dpS = 0.5 * u_in**2 * math.pi * R**2

    tag = f"[sphmem D={D_cells} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={re} u_in={u_in} nu={nu:.6f} tau={tau:.6f} dpS={dpS:.6f}", flush=True)
    t0 = time.time()

    solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device=dev)
    surf = _surface_mask(solid)
    interior = solid & ~surf
    n_solid = int(solid.sum().item())
    n_surf = int(surf.sum().item())
    print(
        f"{tag} solid={n_solid} surface={n_surf} interior={n_solid - n_surf} "
        f"(blockage {100.0 * D_cells / ny:.2f}%, domain {nx / D_cells:.1f}D x "
        f"{ny / D_cells:.1f}D)",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0, ux0
    im0 = float(f.sum().item())

    def _step(f):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        return far_field_bc_3d(f, u_in)

    step_fn = route_step(_step, compile_mode, name=f"sphere_re100_mem[D{D_cells}]")

    all_hist, surf_hist, int_hist, mass_hist, umax_hist = [], [], [], [], []
    step = 0
    for step in range(1, n_steps + 1):
        f = step_fn(f)
        if step % sample_interval == 0:
            all_hist.append(mem_force_x(f, solid) / dpS)
            surf_hist.append(mem_force_x(f, surf) / dpS)
            int_hist.append(mem_force_x(f, interior) / dpS)
            mass_hist.append(float(f.sum().item()))
            umax_hist.append(float(f.abs().max().item()))
        if step % report_every == 0:
            n = min(200, len(surf_hist))
            if n:
                print(
                    f"{tag} step={step} Cd_mem_surf={sum(surf_hist[-n:]) / n:.4f} "
                    f"Cd_mem_all={sum(all_hist[-n:]) / n:.4f} "
                    f"Cd_mem_int={sum(int_hist[-n:]) / n:.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(surf_hist)
    win = min(n_tot, 200)

    def mean_last(arr, w=win):
        seg = arr[-w:]
        return sum(seg) / len(seg) if seg else float("nan")

    def plateau(arr, frac, w=win):
        kk = int(n_tot * frac)
        seg = arr[max(0, kk - w) : kk]
        return sum(seg) / len(seg) if seg else float("nan")

    cd_surf = mean_last(surf_hist)
    cd_all = mean_last(all_hist)
    cd_int = mean_last(int_hist)
    err_surf = (cd_surf - REF_SN) / REF_SN * 100.0
    print(
        f"{tag} === FINAL win={win} === Cd_mem_surface={cd_surf:.4f} "
        f"err_SN={err_surf:+.2f}% | Cd_mem_all={cd_all:.4f} "
        f"Cd_mem_interior={cd_int:.4f} ({elapsed:.0f}s)",
        flush=True,
    )

    if save_field:
        torch.save(f.detach().cpu(), save_field)
        print(f"{tag} saved final field -> {save_field}", flush=True)

    return {
        "case": "sphere_re100_ladd_mem_surface",
        "lattice": "D3Q19",
        "collision": "bgk",
        "force_method": "ladd_momentum_exchange (post-stream, pre-bounce-back); "
        "primary = surface cells only",
        "D_cells": D_cells,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_D": lateral_D,
        "upstream_D": up_D,
        "downstream_D": down_D,
        "blockage_pct": 100.0 * D_cells / ny,
        "n_solid_cells": n_solid,
        "n_surface_cells": n_surf,
        "n_interior_cells": n_solid - n_surf,
        "Re": re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_finished": step,
        "sample_interval": sample_interval,
        "avg_window_samples": win,
        "cd_mem_surface": cd_surf,
        "cd_mem_all": cd_all,
        "cd_mem_interior": cd_int,
        "cd": cd_surf,
        "err_pct": err_surf,
        "ref_cd": REF_SN,
        "ref_sn": REF_SN,
        "ref_cgw": REF_CGW,
        "err_mem_all_pct": (cd_all - REF_SN) / REF_SN * 100.0,
        "plateau_surface": [plateau(surf_hist, p) for p in (0.25, 0.5, 0.75, 1.0)],
        "plateau_all": [plateau(all_hist, p) for p in (0.25, 0.5, 0.75, 1.0)],
        "mass_drift_pct": (mass_hist[-1] - im0) / im0 * 100.0 if mass_hist else float("nan"),
        "max_f": max(umax_hist) if umax_hist else float("nan"),
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_s": elapsed,
        "compile_status": getattr(step_fn, "compile_status", None),
        "compile_mode_effective": getattr(step_fn, "compile_mode_effective", None),
        "modules_used": [
            "solver3d.collide_bgk3d",
            "solver3d.stream3d",
            "boundaries3d.bounce_back_cells_3d (half-way BB pre-stream, NoDynamics)",
            "boundaries3d.far_field_bc_3d (inlet/y±/z± far-field, x+ zero-gradient)",
            "boundaries3d.sphere_mask (staircase parametric sphere)",
            "Ladd MEM 2*sum c_ix f_i (surface-restricted; same instrument as verified cylinder)",
            "benchmarks.compile_route.route_step",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "verify"])
    ap.add_argument("arg", help="D_cells (single) or output dir (verify)")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lateral", type=float, default=16.0)
    ap.add_argument("--up", type=float, default=4.0)
    ap.add_argument("--down", type=float, default=12.0)
    ap.add_argument("--u-in", type=float, default=0.06)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grids", type=int, nargs="+", default=[16, 24])
    ap.add_argument("--save-field", action="store_true")
    add_compile_mode_arg(ap, default="eager")
    a = ap.parse_args()
    cm = compile_mode_from_args(a)

    if a.mode == "single":
        out = a.out or f"/tmp/sphmem_D{a.arg}.json"
        field = (out + ".field.pt") if a.save_field else None
        r = run_case(
            int(a.arg),
            a.device,
            a.steps,
            a.lateral,
            a.up,
            a.down,
            a.u_in,
            a.re,
            a.sample,
            cm,
            save_field=field,
        )
        Path(out).write_text(json.dumps(r, indent=2))
        print(f"[sphmem] saved {out}", flush=True)
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = []
    for D in a.grids:
        field = str(out_dir / f"final_D{D}.pt") if a.save_field else None
        r = run_case(
            D,
            a.device,
            a.steps,
            a.lateral,
            a.up,
            a.down,
            a.u_in,
            a.re,
            a.sample,
            cm,
            save_field=field,
        )
        per_grid.append(r)
        (out_dir / f"case_D{D}.json").write_text(json.dumps(r, indent=2))
    cds = [r["cd_mem_surface"] for r in per_grid]
    errs = [r["err_pct"] for r in per_grid]
    span = abs(cds[-1] - cds[0]) / REF_SN * 100.0 if len(cds) > 1 else float("nan")
    ok = all(abs(e) <= 3.0 for e in errs)
    conv = span <= 3.0
    res = {
        "case": "sphere_re100_ladd_mem_surface",
        "description": "staircase sphere Re=100, Ladd MEM surface-restricted",
        "force_method": "ladd_momentum_exchange (surface cells only)",
        "reference": f"Schiller-Naumann {REF_SN:.6f} / CGW {REF_CGW:.6f}",
        "grids": per_grid,
        "cd_mem_surface_by_grid": cds,
        "cd_mem_all_by_grid": [r["cd_mem_all"] for r in per_grid],
        "convergence": {
            "cd": cds,
            "cd_span_pct": span,
            "cd_within_3pct": ok,
            "grid_span_within_3pct": conv,
        },
        "verified": bool(ok and conv),
        "verdict": "verified" if (ok and conv) else "not_verified",
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["convergence"], indent=2), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
