#!/usr/bin/env python3
"""Cylinder Re=40, 2D free-stream steady state — 2D infinite-span reference cluster.

Physics
-------
Viscous incompressible flow past a circular cylinder, Re = U·D/ν = 40.
Below the shedding threshold (Re_c ≈ 47) the wake is STEADY and symmetric,
so the quantity of interest is the steady drag coefficient
    Cd = Fx / (½ ρ U² D).

Configuration stack — same caliber as `benchmarks/verified/cylinder`
(the verified 2D free-stream family) and its Re=40 sibling `cylinder_3d`:
  * domain 40D × 40D, cylinder centre 10D from the inlet, blockage D/40 = 2.5 %
  * D2Q9; collision = BGK (uniform τ) — identical choice to the verified
    `cylinder_3d` Re=40 case; a downstream absorbing sponge is applied with the
    library ``sponge_bc.apply_viscous_sponge_2d`` (valid for any collision,
    unlike the MRT tau_field sponge) so no MRT-only path is required.
  * chain: collide → restore solid (NoDynamics) → viscous sponge →
    half-way bounce-back (pre-streaming) → stream → far_field_bc_2d.
  * boundaries.far_field_bc_2d  (inlet + both lateral faces Dirichlet free
    stream, zero-gradient outlet) — no obstacle argument, since the obstacle
    half-way bounce-back is applied pre-streaming.
  * boundaries.cylinder_mask, boundaries.bounce_back_cells, solver.stream,
    solver.collide_bgk, d2q9.equilibrium/macroscopic — library primitives only.
  * Ladd (1994) momentum exchange `boundaries.compute_obstacle_forces`,
    sampled post-streaming / pre-(next-step)-bounce-back, i.e. on the returned
    field at the solid cells exactly as `cylinder_3d` does.

Force caliber (★ decisive point, see
`benchmarks/verified/cylinder_3d/REFERENCE_AUDIT.md` and
`docs/mem_surface_caliber_finding.md`):
  the Ladd MEM sum Σ_solid 2 c_ix f_i must be restricted to the **surface**
  solid cells (solid cells with ≥1 fluid 4-neighbour).  Summing over *all*
  solid cells adds a spurious non-self-cancelling interior pseudo-force that
  inflates Cd.  Both calibers are recorded; **surface-only is the acceptance
  caliber**.

  Note (why the chain must freeze the solid): if the collision is applied to
  the interior solid cells (no NoDynamics restore) the interior cells evolve
  and the surface-shell Ladd sum is corrupted (measured: even negative for a
  low-resolution circle).  Freezing the solid at every collision — exactly the
  `cylinder_3d` / `cylinder/re20_st` verified convention — makes the interior
  contribution cancel and the surface sum physical.

Reference
---------
Re=40 lies below vortex shedding, so the flow is 2D steady: the correct
comparison is the **2D free-stream numerical cluster**, NOT a finite-span
wind-tunnel experiment:

    Dennis & Chang 1970 (JFM 42,471, spectral)   Cd = 1.522
    Fornberg 1985 (JCP 61,297, pseudospectral)   Cd = 1.498
    Takami & Keller 1969 (Phys. Fluids 12)        Cd ≈ 1.48
    adopted centre   Cd_ref = 1.50, cluster [1.48, 1.52]

Tritton 1959 (≈1.54) is finite-span experiment and is explicitly excluded.

Acceptance: BOTH grids |err| ≤ 3 % AND grid span ≤ 3 %.

Usage
-----
    run.py --D 20 40 --device sdaa:0 [--steps 30000 40000]
           [--u-in 0.08] [--out DIR] [--compile-mode eager]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in _HERE.parents:
    if (_p / "compile_route.py").is_file():
        sys.path.insert(0, str(_p))  # <repo>/benchmarks
        break

import torch  # noqa: E402
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    compile_status_of,
    route_step,
)

from tensorlbm.boundaries import (  # noqa: E402
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.sponge_bc import apply_viscous_sponge_2d, sponge_profile  # noqa: E402

# --- adopted 2D free-stream numerical cluster (Dennis&Chang / Fornberg / Takami) ---
REF_CD = 1.50
REF_CD_CLUSTER = (1.48, 1.52)
REF_CD_SOURCES = {
    "Dennis & Chang 1970 (2D spectral)": 1.522,
    "Fornberg 1985 (2D pseudospectral)": 1.498,
    "Takami & Keller 1969 (2D FD)": 1.48,
}

# --- domain / sponge layout (same caliber as verified/cylinder) ---
DOMAIN_D = 40.0  # square domain side = 40 D  (blockage 2.5 %)
CYL_X_D = 10.0  # cylinder centre 10 D downstream of the inlet
SPONGE_D = 10.0  # downstream sponge thickness
SPONGE_AMPL = 0.5
SPONGE_EXP = 3.0


def _surface_layer_2d(solid: torch.Tensor) -> torch.Tensor:
    """Solid cells that touch at least one fluid cell (4-neighbourhood)."""
    fluid = ~solid
    return solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )


def run_case(
    D: int,
    re: float,
    u_in: float,
    steps: int,
    device: torch.device,
    compile_mode: str | None = "default",
    sample_interval: int = 100,
    warmup_frac: float = 0.6,
    out_path: str | None = None,
) -> dict:
    nx = ny = int(DOMAIN_D * D)
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    mask = cylinder_mask(nx, ny, CYL_X_D * D, ny / 2.0, D / 2.0, device)
    surf = _surface_layer_2d(mask)
    interior = mask & ~surf
    n_solid = int(mask.sum().item())
    n_surface = int(surf.sum().item())
    n_interior = int(interior.sum().item())

    sponge = sponge_profile(
        nx, int(nx - SPONGE_D * D), nx - 1, amplitude=SPONGE_AMPL, exponent=SPONGE_EXP, device=device
    )

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))
    im0 = float(f.sum())

    def _step(f):
        f_pre = f[:, mask].clone()  # NoDynamics: freeze solid
        f = collide_bgk(f, tau)
        f[:, mask] = f_pre
        rho, ux, uy = macroscopic(f)
        f = apply_viscous_sponge_2d(f, rho, ux, uy, tau, sponge)
        f = bounce_back_cells(f, mask)  # half-way BB *pre*-streaming
        f = stream(f)
        return far_field_bc_2d(f, u_in, None)

    step = route_step(_step, compile_mode, name=f"cylinder_re40_st[D{D}]")

    q_dyn = 0.5 * 1.0 * u_in * u_in * D  # ½ρU²D (rho=1, D in cells)

    mem_s_hist: list[torch.Tensor] = []
    mem_a_hist: list[torch.Tensor] = []
    mem_i_hist: list[torch.Tensor] = []
    t0 = time.time()
    for i in range(1, steps + 1):
        f = step(f)
        if i % sample_interval == 0:
            # Ladd MEM on the returned field (post-stream / pre-next-BB)
            mem_s_hist.append(compute_obstacle_forces(f, surf)[0])
            mem_a_hist.append(compute_obstacle_forces(f, mask)[0])
            mem_i_hist.append(compute_obstacle_forces(f, interior)[0])
        if i % 5000 == 0:
            k = min(500, len(mem_s_hist))
            if k:
                cs = sum(float(x) for x in mem_s_hist[-k:]) / k / q_dyn
                ca = sum(float(x) for x in mem_a_hist[-k:]) / k / q_dyn
                print(
                    f"  [D={D}] step {i}: Cd_surf={cs:.4f} Cd_all={ca:.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"D={D}: non-finite populations after {steps} steps")

    ms = torch.stack(mem_s_hist).detach().cpu().numpy().astype("float64") / q_dyn
    ma = torch.stack(mem_a_hist).detach().cpu().numpy().astype("float64") / q_dyn
    mi = torch.stack(mem_i_hist).detach().cpu().numpy().astype("float64") / q_dyn
    n_samp = ms.size

    w0 = int(warmup_frac * n_samp)
    cd_surf = float(ms[w0:].mean())
    cd_all = float(ma[w0:].mean())
    cd_int = float(mi[w0:].mean())

    # plateau windows (25/50/75/100 %) of the sample series
    win = max(1, n_samp // 20)
    plateau_surf = {}
    plateau_all = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        end = int(frac * n_samp)
        seg_s = ms[max(0, end - win) : end]
        seg_a = ma[max(0, end - win) : end]
        plateau_surf[f"{int(frac * 100)}%"] = float(seg_s.mean())
        plateau_all[f"{int(frac * 100)}%"] = float(seg_a.mean())

    err_cd = (cd_surf - REF_CD) / REF_CD * 100.0
    routing = compile_status_of(step)

    result = {
        "case": "cylinder_re40_st_2d_free_stream",
        "D": D,
        "nx": nx,
        "ny": ny,
        "domain_D": DOMAIN_D,
        "blockage_pct": round(100.0 * D / nx, 3),
        "re": re,
        "u_in": u_in,
        "nu_lb": round(nu, 6),
        "tau": round(tau, 6),
        "lattice": "D2Q9",
        "collision": "bgk (NoDynamics solid restore)",
        "sponge": {
            "module": "sponge_bc.apply_viscous_sponge_2d",
            "x0": int(nx - SPONGE_D * D),
            "x1": nx - 1,
            "amplitude": SPONGE_AMPL,
            "exponent": SPONGE_EXP,
        },
        "n_solid_cells": n_solid,
        "n_surface_cells": n_surface,
        "n_interior_cells": n_interior,
        "steps": steps,
        "sample_interval": sample_interval,
        "n_samples": n_samp,
        "analyze_from": w0,
        "warmup_frac": warmup_frac,
        "cd_mem_surface": round(cd_surf, 4),
        "cd_mem_all": round(cd_all, 4),
        "cd_mem_interior": round(cd_int, 4),
        "cd_ref": REF_CD,
        "ref_cluster": list(REF_CD_CLUSTER),
        "err_pct": round(err_cd, 2),
        "plateau_mem_surface": plateau_surf,
        "plateau_mem_all": plateau_all,
        "ma": round(u_in / (1.0 / math.sqrt(3.0)), 4),
        "finite": True,
        "wall_s": round(elapsed, 1),
        **routing,
    }
    print(
        f"[cylinder_re40_st D={D}] steps={steps} t={elapsed:.0f}s "
        f"Cd_surf={cd_surf:.4f} ({err_cd:+.2f}%) Cd_all={cd_all:.4f} Cd_int={cd_int:.4f}",
        flush=True,
    )
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--D", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--steps", type=int, nargs="+", default=None)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--warmup-frac", type=float, default=0.6)
    ap.add_argument("--out", default="")
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    device = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    default_steps = {20: 30000, 40: 40000, 80: 80000}
    steps_list = args.steps or [default_steps.get(D, 40000) for D in args.D]

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    grids = {}
    for D, steps in zip(args.D, steps_list):
        grids[f"D{D}"] = run_case(
            D,
            args.re,
            args.u_in,
            steps,
            device,
            compile_mode=compile_mode,
            sample_interval=args.sample,
            warmup_frac=args.warmup_frac,
            out_path=str(out / f"case_D{D}.json") if out else None,
        )

    cds = [g["cd_mem_surface"] for g in grids.values()]
    span_pct = (max(cds) - min(cds)) / REF_CD * 100.0 if len(cds) > 1 else 0.0
    within = all(abs(g["err_pct"]) <= 3.0 for g in grids.values())
    ok = within and span_pct <= 3.0 and len(cds) >= 2
    summary = {
        "case": "cylinder_re40_st_2d_free_stream",
        "description": "2D free-stream circular cylinder, Re=40, steady wake (below shedding threshold)",
        "lattice": "D2Q9",
        "collision": "BGK (NoDynamics solid restore) + viscous sponge",
        "boundary": "far_field_bc_2d + downstream 10D viscous sponge + obstacle half-way BB (pre-stream)",
        "force": "Ladd momentum-exchange; PRIMARY = SURFACE solid cells only (post-stream/pre-BB). "
        "See docs/mem_surface_caliber_finding.md",
        "reference": {
            "cd_ref": REF_CD,
            "cluster": list(REF_CD_CLUSTER),
            "sources": REF_CD_SOURCES,
            "convention": "Re=40 < shedding threshold: 2D steady; correct comparison is the 2D "
            "free-stream numerical cluster, not finite-span experiment (Tritton 1.54 excluded).",
        },
        "grids": grids,
        "cd_mem_surface_by_grid": cds,
        "convergence": {
            "cd_mem_surface": cds,
            "cd_span_pct": round(span_pct, 2),
            "cd_within_3pct": within,
            "grid_span_within_3pct": span_pct <= 3.0,
        },
        "verified": ok,
        "verdict": "verified" if ok else "not_verified",
    }
    if out:
        (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["convergence"], indent=2), flush=True)
    print(f"VERDICT: {summary['verdict']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()