#!/usr/bin/env python3
"""3D extruded circular cylinder, Re = U·D/ν = 40, infinite span — VERIFIED benchmark.

Physics
-------
A circular cylinder of diameter D (in lattice cells) extruded along z, with the
span treated as infinite via periodic z-boundaries (``stream3d_roll`` + D3Q19).
The cross-section is voxelised (staircase) from ``(x-cx)² + (y-cy)² ≤ R²``.
Far-field (free-stream Dirichlet) BC on y-/y+; inlet/outlet handled by
``far_field_bc_3d``; z-/z+ periodic.

At Re = 40 the flow is 2D-steady (no spanwise variation, no shedding —
shedding onset Re_c ≈ 47).  Therefore an infinite-span extruded cylinder at
Re = 40 **is** the 2D free-stream cylinder problem, and the correct reference
is the 2D *numerical* free-stream cluster Cd ≈ 1.50 (Dennis & Chang 1970
1.522; Fornberg 1985 1.498; Takami & Keller 1969 ≈1.48).  It is **not** the
finite-span wind-tunnel value (Tritton 1959 ≈1.54, which carries end-effects
and 3D wake contamination).  See ``REFERENCE_AUDIT.md``.

Span cells
----------
``nz`` is the number of periodic span cells.  At Re = 40 the continuous
solution is z-invariant, so any nz ≥ 1 gives the same per-unit-length forces;
the default nz = 4 is the cheap 2D-equivalent reduction of the full D3Q19 +
periodic-z machinery (the nz-independence is asserted, not assumed — the
reported Cd is normalised by the frontal area D·nz, so it is span-independent).

Normalisation
-------------
Cd = Fx / (½ρ U_in² · D · nz)   (frontal area D × Lz, Lz = nz cells).

Method (library primitives only)
--------------------------------
* chain: ``collide_bgk3d`` → NoDynamics (restore solid) → ``bounce_back_cells_3d``
  (half-way BB *before* streaming) → ``stream3d_roll`` → ``far_field_bc_3d``.
  The whole chain is routed through :mod:`benchmarks.compile_route`
  (``route_step`` + shared ``tensorlbm.compile_utils``), with the audited
  one-shot eager fallback for hosts whose inductor backend is incomplete
  (Hygon/SDAA teco_inductor).
* forces: **primary = Ladd (1994) momentum-exchange** via
  ``obstacles.compute_obstacle_forces_3d`` — this is the SAME instrument the
  verified 2D cylinder family uses (``boundaries.compute_obstacle_forces``);
  sampled post-stream / pre-bounce-back (equivalently: at the top of the next
  step, since this chain applies the half-way bounce-back *before* streaming).

  The raw Ladd sum ``F_x = 2 Σ_solid c_ix f_i`` is reported in **three
  calibers on the SAME field**:

  * ``cd_mem_surface``  — sum over **wall-adjacent surface cells only**
    (solid cells with ≥1 fluid 6-neighbour).  **This is the acceptance
    caliber**: it is the genuine momentum flux crossing the fluid/solid
    interface (and, on a staircase voxel body, it alone reproduces the true
    Cd).  See ``docs/mem_surface_caliber_finding.md``.
  * ``cd_mem_all``      — sum over ALL solid cells (the legacy definition).
    Includes a spurious, non-self-cancelling *interior* contribution on a
    curved (non-flat) voxel body, so it over-reads Cd by +4…+9 %.
  * ``cd_mem_interior`` — sum over the interior cells only; the spurious,
    diagnostic term (≈ +0.04…+0.07, i.e. the all-solid over-read).

  Note for the z-extruded cylinder the z-neighbour of every solid cell is also
  solid, so the 6-neighbour surface mask reduces exactly to the xy 4-neighbour
  wall ring; the three calibers are hence z-invariant as well.

  ``drag_pressure.SurfaceMesh.from_cylinder`` + ``get_near_wall_2d(axis='z')``
  + ``drag_pressure_integration`` (extrap='none', p0='far_field') and
  ``drag_friction_integration`` (formula 'standard', others diagnostic) are
  retained as DIAGNOSTIC columns on the same field, not as the reported Cd.
* no extrapolation / correction factors; no mass renormalisation.

Usage
-----
    run.py single D_cells out.json [--steps 80000] [--lateral 16] [--nz 4]
        [--device sdaa:0] [--compile-mode eager]
    run.py verify out_dir [--steps 80000] [--grids 40 60] [--lateral 16]
        [--device sdaa:0] [--compile-mode eager]
"""

from __future__ import annotations

import argparse
import json
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

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_2d,
)
from tensorlbm.obstacles import compute_obstacle_forces_3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402

# ----------------------------------------------------------------------------
# Reference convention (see REFERENCE_AUDIT.md)
# ----------------------------------------------------------------------------
CD_REF = 1.50  # centre of the 2D free-stream numerical cluster
CD_CLUSTER = (1.48, 1.52)
CD_SOURCES = {
    "Dennis & Chang 1970": 1.522,
    "Fornberg 1985": 1.498,
    "Takami & Keller 1969": 1.48,
}
REF_NOTE = (
    "2D free-stream NUMERICAL cluster Cd≈1.50 [1.48,1.52]: "
    "Dennis&Chang 1970 1.522 / Fornberg 1985 1.498 / Takami&Keller 1969 1.48. "
    "NOT Tritton 1959 ≈1.54 (finite-span experiment)."
)

# All friction formulas evaluated on the SAME field every sample (diagnostics);
# the primary reported split is 'standard'.
FORMULAS = ["standard", "lagrange", "faces", "mix50"]


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def surface_mask(solid: torch.Tensor) -> torch.Tensor:
    """Wall-adjacent solid cells: solid with >=1 fluid 6-neighbour (x/y/z +-).

    On the z-extruded cylinder the z-neighbour is always solid, so this is
    identical to the xy 4-neighbour wall ring (z-invariant).
    """
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def smooth_q(solid, cx, cy, R, near):
    nz, ny, nx = solid.shape
    dev = solid.device
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    r_c = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (r_c - R).clamp(0.05, 1.0) * near.float()


def run_case(
    D_cells: int,
    device: str = "sdaa:0",
    n_steps: int = 80000,
    lateral_D: float = 16.0,
    nz: int = 4,
    u_in: float = 0.08,
    sample_interval: int = 100,
    compile_mode: str | None = "default",
    save_field: str | None = None,
) -> dict:
    dev = torch.device(device)
    R = D_cells / 2.0
    nx = ny = int(round(lateral_D * D_cells))
    cx = cy = nx // 2
    Re = 40.0
    nu = u_in * D_cells / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in**2 * (D_cells * nz)

    tag = f"[cyl3d D={D_cells} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={Re} u_in={u_in} nu={nu:.6f} tau={tau:.6f} dpS={dpS:.4f}", flush=True)
    t0 = time.time()

    solid = cylinder3d_mask(nx, ny, nz, cx, cy, R, dev)
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    q = smooth_q(solid, cx, cy, R, near)
    surf = surface_mask(solid)
    interior = solid & ~surf
    n_near = int(near.sum().item())
    n_solid = int(solid.sum().item())
    n_surf = int(surf.sum().item())
    n_int = int(interior.sum().item())
    print(
        f"{tag} solid={n_solid} surface={n_surf} interior={n_int} near={n_near} "
        f"(blockage {100.0 * D_cells / nx:.2f}%, span {nz} cells = {nz / D_cells:.2f}D)",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0, ux0
    im0 = float(f.sum().item())

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    def _step(f):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d_roll(f)
        return far_field_bc_3d(f, u_in, bc_config=bc_config)

    step_fn = route_step(_step, compile_mode, name=f"cylinder_3d_re40[D{D_cells}]")

    hist = {k: [] for k in FORMULAS}
    cdp_hist, mem_hist, mem_surf_hist, mem_int_hist = [], [], [], []
    mass_hist, umax_hist = [], []
    step = 0
    for step in range(1, n_steps + 1):
        f = step_fn(f)

        if step % sample_interval == 0:
            # PRIMARY: Ladd momentum-exchange (post-stream, pre-bounce-back).
            # In this chain the half-way BB is applied *before* streaming, so the
            # populations at the solid cells of the returned field are exactly the
            # post-stream populations that the next step's BB will reverse.
            # Three calibers on the SAME field:
            #   surface  = wall-adjacent cells only  (ACCEPTANCE caliber)
            #   all      = every solid cell          (legacy; over-reads: +4..+9%)
            #   interior = interior only             (spurious diagnostic)
            mem_hist.append(float(compute_obstacle_forces_3d(f, solid)[0].item()) / dpS)
            mem_surf_hist.append(float(compute_obstacle_forces_3d(f, surf)[0].item()) / dpS)
            mem_int_hist.append(float(compute_obstacle_forces_3d(f, interior)[0].item()) / dpS)
            cdp_hist.append(
                drag_pressure_integration(
                    f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid
                )[0]
            )
            for k in FORMULAS:
                hist[k].append(
                    drag_friction_integration(f, mesh, dpS, nu, q_wall=q, formula=k, solid=solid)[0]
                )
            mass_hist.append(float(f.sum().item()))
            umax_hist.append(float(f.abs().max().item()))

        if step % 5000 == 0:
            n = min(200, len(cdp_hist))
            if n:
                parts = " ".join(f"{k}={sum(hist[k][-n:]) / n:.4f}" for k in FORMULAS)
                print(
                    f"{tag} step={step} Cd_mem_surf={sum(mem_surf_hist[-n:]) / n:.4f} "
                    f"Cd_mem_all={sum(mem_hist[-n:]) / n:.4f} "
                    f"Cd_mem_int={sum(mem_int_hist[-n:]) / n:.4f} "
                    f"Cd_p={sum(cdp_hist[-n:]) / n:.4f} {parts} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cdp_hist)
    win = min(n_tot, 200)
    cd_all = sum(mem_hist[-win:]) / win
    cd_surf = sum(mem_surf_hist[-win:]) / win
    cd_int = sum(mem_int_hist[-win:]) / win
    cd_mem = cd_surf  # PRIMARY / acceptance caliber
    cdp = sum(cdp_hist[-win:]) / win
    cdf = {k: sum(hist[k][-win:]) / win for k in FORMULAS}
    cdt = {k: cdp + cdf[k] for k in FORMULAS}

    def plateau(k, frac):
        kk = int(n_tot * frac)
        seg = hist[k][max(0, kk - win) : kk]
        pseg = cdp_hist[max(0, kk - win) : kk]
        if not seg:
            return float("nan")
        return sum(seg) / len(seg) + sum(pseg) / len(pseg)

    def plateau_arr(arr, frac):
        kk = int(n_tot * frac)
        seg = arr[max(0, kk - win) : kk]
        if not seg:
            return float("nan")
        return sum(seg) / len(seg)

    def plateau_mem(frac):
        return plateau_arr(mem_hist, frac)

    def plateau_surf(frac):
        return plateau_arr(mem_surf_hist, frac)

    plateau_win = {
        f"{k}@{int(fr * 100)}%": plateau(k, fr) for fr in (0.25, 0.5, 0.75, 1.0) for k in FORMULAS
    }

    err = (cd_surf - CD_REF) / CD_REF * 100.0
    err_all = (cd_all - CD_REF) / CD_REF * 100.0
    cdt_std = cdt["standard"]
    err_pf = (cdt_std - CD_REF) / CD_REF * 100.0
    print(
        f"{tag} === FINAL win={win} === Cd_mem_surface={cd_surf:.4f} "
        f"err_surf_vs_{CD_REF}={err:+.2f}%  |  Cd_mem_all={cd_all:.4f} "
        f"({err_all:+.2f}%)  Cd_mem_interior={cd_int:.4f}  |  DIAGNOSTIC "
        f"Cd_p={cdp:.4f} Cd_f={cdf['standard']:.4f} Cd_p+f={cdt_std:.4f} "
        f"err_pf={err_pf:+.2f}% ({elapsed:.0f}s)",
        flush=True,
    )
    for k in FORMULAS:
        print(
            f"  [diag] {k:14s} Cd_f={cdf[k]:.4f} Cd_p+f={cdt[k]:.4f} "
            f"err={(cdt[k] - CD_REF) / CD_REF * 100:+.2f}%",
            flush=True,
        )
    print(
        f"  plateau(mem_surface) {[round(plateau_surf(p / 100), 4) for p in (25, 50, 75, 100)]}",
        flush=True,
    )
    print(
        f"  plateau(mem_all)     {[round(plateau_mem(p / 100), 4) for p in (25, 50, 75, 100)]}",
        flush=True,
    )
    print(
        f"  plateau(standard) {[round(plateau_win[f'standard@{p}%'], 4) for p in (25, 50, 75, 100)]}",
        flush=True,
    )

    routing = dict(
        compile_status=getattr(step_fn, "compile_status", None),
        compile_mode_effective=getattr(step_fn, "compile_mode_effective", None),
        compile_status_reason=getattr(step_fn, "compile_status_reason", None),
    )

    if save_field:
        torch.save(f.detach().cpu(), save_field)
        print(f"{tag} saved final field -> {save_field}", flush=True)

    return {
        "case": "cylinder_3d_re40",
        "lattice": "D3Q19",
        "collision": "bgk",
        "geometry": "z-axis extruded circular cylinder, infinite span (z periodic)",
        "D_cells": D_cells,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_D": lateral_D,
        "blockage_pct": 100.0 * D_cells / nx,
        "span_cells": nz,
        "n_solid_cells": n_solid,
        "n_surface_cells": n_surf,
        "n_interior_cells": n_int,
        "n_near_cells": n_near,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "n_finished": step,
        "sample_interval": sample_interval,
        "avg_window_samples": win,
        "force_method": "ladd_momentum_exchange_3d (post-stream, pre-bounce-back); "
        "reported in three calibers on the same field",
        "cd_mem_surface": cd_surf,
        "err_surf_pct": err,
        "cd_mem_all": cd_all,
        "err_mem_all_pct": err_all,
        "cd_mem_interior": cd_int,
        "cd_mem": cd_all,  # legacy alias == all-solid
        "err_mem_pct": err_all,
        "cd": cd_surf,  # PRIMARY / acceptance caliber
        "err_pct": err,
        "cd_pressure": cdp,
        "cd_friction": cdf,
        "cd_total": cdt,
        "cd_pf_standard": cdt_std,
        "err_pf_pct": err_pf,
        "ref_cd": CD_REF,
        "ref_cluster": list(CD_CLUSTER),
        "ref_sources": CD_SOURCES,
        "plateau_windows": plateau_win,
        "plateau_mem_surface": [plateau_surf(p / 100) for p in (25, 50, 75, 100)],
        "plateau_mem": [plateau_mem(p / 100) for p in (25, 50, 75, 100)],
        "plateau_standard": [plateau_win[f"standard@{p}%"] for p in (25, 50, 75, 100)],
        "mass_drift_pct": (mass_hist[-1] - im0) / im0 * 100.0 if mass_hist else float("nan"),
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_s": elapsed,
        **routing,
        "modules_used": [
            "solver3d.collide_bgk3d",
            "solver3d.stream3d_roll",
            "boundaries3d.bounce_back_cells_3d (half-way BB pre-stream, NoDynamics)",
            "boundaries3d.far_field_bc_3d (bc_config: y± far-field, z± periodic)",
            "obstacles.compute_obstacle_forces_3d (Ladd MEM — same instrument; "
            "run.py::surface_mask restricts the sum to wall-adjacent cells: PRIMARY Cd)",
            "run.py::surface_mask (wall-adjacent solid cells, 6-neighbour)",
            "obstacles.compute_obstacle_forces_3d on ALL solid cells (diagnostic; legacy)",
            "drag_pressure.SurfaceMesh.from_cylinder (diagnostic)",
            "drag_pressure.get_near_wall_2d(axis='z') (diagnostic)",
            "drag_pressure.drag_pressure_integration (extrap=none, p0=far_field; diagnostic)",
            "drag_pressure.drag_friction_integration (standard; others diagnostic)",
            "benchmarks.compile_route.route_step -> tensorlbm.compile_utils",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "verify"])
    ap.add_argument("arg", help="D_cells (single) or output dir (verify)")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--lateral", type=float, default=16.0)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default=None, help="output JSON path (single mode)")
    ap.add_argument("--grids", type=int, nargs="+", default=[40, 60])
    ap.add_argument(
        "--save-field",
        action="store_true",
        help="also persist the final distribution (for MEM recompute / provenance)",
    )
    add_compile_mode_arg(ap, default="eager")
    a = ap.parse_args()
    cm = compile_mode_from_args(a)

    if a.mode == "single":
        out = a.out or f"/tmp/cyl3d_D{a.arg}.json"
        field = (out + ".field.pt") if a.save_field else None
        r = run_case(
            int(a.arg),
            a.device,
            a.steps,
            a.lateral,
            a.nz,
            a.u_in,
            sample_interval=a.sample,
            compile_mode=cm,
            save_field=field,
        )
        Path(out).write_text(json.dumps(r, indent=2))
        print(f"[cyl3d] saved {out}", flush=True)
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
            a.nz,
            a.u_in,
            sample_interval=a.sample,
            compile_mode=cm,
            save_field=field,
        )
        per_grid.append(r)
        (out_dir / f"case_D{D}.json").write_text(json.dumps(r, indent=2))
    cd_surf = [r["cd_mem_surface"] for r in per_grid]
    cd_all = [r["cd_mem_all"] for r in per_grid]
    cd_int = [r["cd_mem_interior"] for r in per_grid]
    cd = cd_surf  # PRIMARY / acceptance caliber
    errs = [r["err_surf_pct"] for r in per_grid]
    span = abs(cd[-1] - cd[0]) / CD_REF * 100.0 if len(cd) > 1 else float("nan")
    span_all = abs(cd_all[-1] - cd_all[0]) / CD_REF * 100.0 if len(cd_all) > 1 else float("nan")
    ok = all(abs(e) <= 3.0 for e in errs)
    conv = span <= 3.0
    res = {
        "case": "cylinder_3d_re40",
        "description": "3D extruded circular cylinder, infinite span (z periodic), Re=40 steady",
        "lattice": "D3Q19",
        "collision": "bgk",
        "force_method": "ladd_momentum_exchange_3d (post-stream, pre-bounce-back); "
        "primary caliber = SURFACE cells only (wall-adjacent). "
        "See docs/mem_surface_caliber_finding.md",
        "reference": REF_NOTE,
        "ref_cd": CD_REF,
        "ref_cluster": list(CD_CLUSTER),
        "ref_sources": CD_SOURCES,
        "grids": per_grid,
        "cd_mem_surface_by_grid": cd_surf,
        "cd_mem_all_by_grid": cd_all,
        "cd_mem_interior_by_grid": cd_int,
        "cd_mem_by_grid": cd_surf,
        "cd_pressure_friction_by_grid": [r["cd_pf_standard"] for r in per_grid],
        "convergence": {
            "caliber": "surface_only",
            "cd": cd,
            "cd_span_pct": span,
            "cd_within_3pct": ok,
            "grid_span_within_3pct": conv,
            "cd_all_span_pct": span_all,
        },
        "verified": bool(ok and conv),
        "verdict": "verified" if (ok and conv) else "not_verified",
        "compile_mode_effective": per_grid[0].get("compile_mode_effective") if per_grid else None,
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
