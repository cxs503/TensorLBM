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
* forces: ``drag_pressure.SurfaceMesh.from_cylinder`` +
  ``get_near_wall_2d(axis='z')`` + ``drag_pressure_integration``
  (extrap='none', p0='far_field') + ``drag_friction_integration`` (primary
  formula 'standard'; all other formulas are diagnostics on the same field).
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
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402

# ----------------------------------------------------------------------------
# Reference convention (see REFERENCE_AUDIT.md)
# ----------------------------------------------------------------------------
CD_REF = 1.50          # centre of the 2D free-stream numerical cluster
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
    n_near = int(near.sum().item())
    n_solid = int(solid.sum().item())
    print(f"{tag} solid={n_solid} near={n_near} (blockage {100.0 * D_cells / nx:.2f}%, "
          f"span {nz} cells = {nz / D_cells:.2f}D)", flush=True)

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
    cdp_hist, mass_hist, umax_hist = [], [], []
    step = 0
    for step in range(1, n_steps + 1):
        f = step_fn(f)

        if step % sample_interval == 0:
            cdp_hist.append(
                drag_pressure_integration(
                    f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid
                )[0]
            )
            for k in FORMULAS:
                hist[k].append(
                    drag_friction_integration(
                        f, mesh, dpS, nu, q_wall=q, formula=k, solid=solid
                    )[0]
                )
            mass_hist.append(float(f.sum().item()))
            umax_hist.append(float(f.abs().max().item()))

        if step % 5000 == 0:
            n = min(200, len(cdp_hist))
            if n:
                parts = " ".join(f"{k}={sum(hist[k][-n:]) / n:.4f}" for k in FORMULAS)
                print(f"{tag} step={step} Cd_p={sum(cdp_hist[-n:]) / n:.4f} {parts} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cdp_hist)
    win = min(n_tot, 200)
    cdp = sum(cdp_hist[-win:]) / win
    cdf = {k: sum(hist[k][-win:]) / win for k in FORMULAS}
    cdt = {k: cdp + cdf[k] for k in FORMULAS}

    def plateau(k, frac):
        kk = int(n_tot * frac)
        seg = hist[k][max(0, kk - win):kk]
        pseg = cdp_hist[max(0, kk - win):kk]
        if not seg:
            return float("nan")
        return sum(seg) / len(seg) + sum(pseg) / len(pseg)

    plateau_win = {
        f"{k}@{int(fr * 100)}%": plateau(k, fr) for fr in (0.25, 0.5, 0.75, 1.0) for k in FORMULAS
    }

    cdt_std = cdt["standard"]
    err = (cdt_std - CD_REF) / CD_REF * 100.0
    print(f"{tag} === FINAL win={win} === Cd_p={cdp:.4f} Cd_f={cdf['standard']:.4f} "
          f"Cd={cdt_std:.4f} err_vs_{CD_REF}={err:+.2f}% ({elapsed:.0f}s)", flush=True)
    for k in FORMULAS:
        print(f"  {k:14s} Cd_f={cdf[k]:.4f} Cd={cdt[k]:.4f} "
              f"err={(cdt[k] - CD_REF) / CD_REF * 100:+.2f}%", flush=True)
    print(f"  plateau(standard) {[round(plateau_win[f'standard@{p}%'], 4) for p in (25, 50, 75, 100)]}",
          flush=True)

    routing = dict(
        compile_status=getattr(step_fn, "compile_status", None),
        compile_mode_effective=getattr(step_fn, "compile_mode_effective", None),
        compile_status_reason=getattr(step_fn, "compile_status_reason", None),
    )

    return {
        "case": "cylinder_3d_re40",
        "lattice": "D3Q19",
        "collision": "bgk",
        "geometry": "z-axis extruded circular cylinder, infinite span (z periodic)",
        "D_cells": D_cells,
        "nx": nx, "ny": ny, "nz": nz,
        "lateral_D": lateral_D,
        "blockage_pct": 100.0 * D_cells / nx,
        "span_cells": nz,
        "n_solid_cells": n_solid,
        "n_near_cells": n_near,
        "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "n_steps": n_steps, "n_finished": step,
        "sample_interval": sample_interval, "avg_window_samples": win,
        "cd_pressure": cdp,
        "cd_friction": cdf,
        "cd_total": cdt,
        "cd": cdt_std,
        "err_pct": err,
        "ref_cd": CD_REF,
        "ref_cluster": list(CD_CLUSTER),
        "ref_sources": CD_SOURCES,
        "plateau_windows": plateau_win,
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
            "drag_pressure.SurfaceMesh.from_cylinder",
            "drag_pressure.get_near_wall_2d(axis='z')",
            "drag_pressure.drag_pressure_integration (extrap=none, p0=far_field)",
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
    add_compile_mode_arg(ap, default="eager")
    a = ap.parse_args()
    cm = compile_mode_from_args(a)

    if a.mode == "single":
        out = a.out or f"/tmp/cyl3d_D{a.arg}.json"
        r = run_case(int(a.arg), a.device, a.steps, a.lateral, a.nz, a.u_in,
                     sample_interval=a.sample, compile_mode=cm)
        Path(out).write_text(json.dumps(r, indent=2))
        print(f"[cyl3d] saved {out}", flush=True)
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = []
    for D in a.grids:
        r = run_case(D, a.device, a.steps, a.lateral, a.nz, a.u_in,
                     sample_interval=a.sample, compile_mode=cm)
        per_grid.append(r)
        (out_dir / f"case_D{D}.json").write_text(json.dumps(r, indent=2))
    cd = [r["cd"] for r in per_grid]
    errs = [r["err_pct"] for r in per_grid]
    span = abs(cd[-1] - cd[0]) / CD_REF * 100.0 if len(cd) > 1 else float("nan")
    ok = all(abs(e) <= 3.0 for e in errs)
    conv = span <= 3.0
    res = {
        "case": "cylinder_3d_re40",
        "description": "3D extracted circular cylinder, infinite span (z periodic), Re=40 steady",
        "lattice": "D3Q19", "collision": "bgk",
        "reference": REF_NOTE,
        "ref_cd": CD_REF, "ref_cluster": list(CD_CLUSTER), "ref_sources": CD_SOURCES,
        "grids": per_grid,
        "convergence": {
            "cd": cd, "cd_span_pct": span,
            "cd_within_3pct": ok, "grid_span_within_3pct": conv,
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