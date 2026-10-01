#!/usr/bin/env python3
"""NACA 0012 2-D airfoil at low Reynolds number (steady) -- benchmark run.

Physics
-------
Two-dimensional incompressible flow past a NACA 0012 aerofoil at chord-based
Reynolds number Re = U_in * C / nu.  At Re = 1000 the flow stays attached and
**steady** for angles of attack below the shedding onset (alpha < ~8 deg;
Kurtulus 2015, Di Ilio et al. 2020).  We therefore pick the steady window
alpha = 0 deg (symmetric: Cl == 0, Cd is the observable) as the primary
verification point, and optionally alpha = 5 deg (Cl, Cd) as a cross-check.

Reference convention
--------------------
The lowest-Re NACA 0012 reference is a **2-D numerical** cluster
(Kurtulus 2015 Fluent laminar; Di Ilio et al. 2020 HLBM).  High-Re wind-tunnel
experiments (Ladson/NASA TM-4074, Re ~ 3-6e6) are NOT applicable.
See REFERENCE_AUDIT.md.

Force caliber (the key point)
-----------------------------
Primary Cd = Ladd (1994) momentum-exchange force summed over the **surface**
solid cells only (wall-adjacent), following
``docs/mem_surface_caliber_finding.md``.  The all-solid sum carries a spurious
interior pseudo-force on the curved staircase aerofoil; the interior-only sum
is reported as a diagnostic.

Sampling note
-------------
The half-way bounce-back is applied *last* in the step (inside
``far_field_bc_2d``), so the returned field holds ``f_i(solid) = f_opp(solid)``
post-stream.  Hence the post-step MEM sum equals ``-F`` and the physical force
is ``F = -compute_obstacle_forces(f_step, mask)`` (verified algebraically;
identical to sampling pre-bounce-back).

Usage
-----
    run.py single C [--alpha 0] [--steps 60000] ...
    run.py verify OUTDIR [--grids 64 96] [--alpha 0] ...
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

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

# ---------------------------------------------------------------------------
# Reference convention (filled by REFERENCE_AUDIT.md) -- see constants below
# ---------------------------------------------------------------------------
# Re=1000, alpha=0 (2-D numerical cluster).  Placeholders are overwritten by
# the audited values; run.py never tunes the reference.
CD_REF = 0.105
CD_CLUSTER = (0.09, 0.12)
REF_NOTE = "NACA0012 Re=1000 alpha=0 2-D NUMERICAL cluster (see REFERENCE_AUDIT.md)"

# geometry / solver defaults
RE_DEFAULT = 1000.0
U_IN_DEFAULT = 0.1
UPSTREAM_C = 6.0
DOWNSTREAM_C = 16.0
HALFHEIGHT_C = 8.0
SPONGE_C = 6.0  # downstream sponge width (chords)
SPONGE_ALPHA = 10.0  # tau_eff = tau * (1 + alpha*sigma)


def naca0012_half_thickness(xi: torch.Tensor, t: float = 0.12) -> torch.Tensor:
    """NACA 4-digit symmetric half-thickness, normalised by chord."""
    a = xi.clamp(min=1e-12)
    return (t / 0.2) * (
        0.2969 * torch.sqrt(a)
        - 0.1260 * a
        - 0.3516 * a * a
        + 0.2843 * a * a * a
        - 0.1015 * a * a * a * a
    )


def airfoil_mask(
    nx: int,
    ny: int,
    chord: float,
    alpha_deg: float,
    x_le: float,
    y_c: float,
    device: torch.device,
) -> torch.Tensor:
    """Voxelised NACA 0012 solid mask (True = solid).

    The chord midpoint sits at ``(x_le + chord/2, y_c)``; the chord line is
    rotated by ``alpha_deg`` (positive = trailing edge raised, positive lift).
    """
    x_mid = x_le + 0.5 * chord
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - x_mid
    dy = yy - y_c
    sp = dx * ca + dy * sa  # along chord from mid-chord
    nn = -dx * sa + dy * ca  # normal to chord
    xi = (sp + 0.5 * chord) / chord  # 0 at LE, 1 at TE
    half = chord * naca0012_half_thickness(xi)
    return (xi >= 0.0) & (xi <= 1.0) & (nn.abs() <= half)


def surface_mask_2d(solid: torch.Tensor) -> torch.Tensor:
    """Wall-adjacent solid cells: solid with >=1 fluid 4-neighbour."""
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def run_case(
    C: int,
    alpha_deg: float = 0.0,
    device: str = "sdaa:0",
    n_steps: int = 60000,
    u_in: float = U_IN_DEFAULT,
    re: float = RE_DEFAULT,
    sample_interval: int = 100,
    warmup_frac: float = 0.4,
    compile_mode: str | None = "default",
    save_field: str | None = None,
) -> dict:
    dev = torch.device(device)
    nx = int(round((UPSTREAM_C + 1.0 + DOWNSTREAM_C) * C))
    ny = int(round(2.0 * HALFHEIGHT_C * C))
    x_le = UPSTREAM_C * C
    y_c = ny / 2.0
    nu = u_in * C / re
    tau = 0.5 + 3.0 * nu
    dpS = 0.5 * u_in**2 * C  # dynamic pressure * chord (2-D, unit depth)

    tag = f"[naca0012 C={C} a={alpha_deg:g} {nx}x{ny}]"
    print(f"{tag} Re={re} u_in={u_in} nu={nu:.6f} tau={tau:.6f} dpS={dpS:.6f}", flush=True)
    t0 = time.time()

    solid = airfoil_mask(nx, ny, float(C), alpha_deg, x_le, y_c, dev)
    surf = surface_mask_2d(solid)
    interior = solid & ~surf
    n_solid = int(solid.sum().item())
    n_surf = int(surf.sum().item())
    n_int = int(interior.sum().item())
    # blockage = max thickness / domain height
    blockage = 100.0 * (0.12 * C) / ny
    print(
        f"{tag} solid={n_solid} surface={n_surf} interior={n_int} "
        f"(blockage {blockage:.3f}%, upstream {UPSTREAM_C}c, downstream {DOWNSTREAM_C}c, "
        f"halfheight {HALFHEIGHT_C}c)",
        flush=True,
    )

    # downstream sponge (absorbing layer); MEM force is unaffected
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_C * C), int(SPONGE_C * C), power=2.0, device=dev
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)
    del rho0, ux0, uy0
    im0 = float(f.sum().item())

    def _step(f):
        # collide -> stream -> far-field (applies half-way BB last)
        return far_field_bc_2d(stream(collide_mrt(f, tau, tau_field=tau_field)), u_in, solid)

    step_fn = route_step(_step, compile_mode, name=f"naca0012_re{int(re)}[C{C}]")

    # post-step field holds post-BB solid populations -> MEM = -F, so F = -MEM
    cds: list[float] = []
    cls: list[float] = []
    cds_all: list[float] = []
    cls_all: list[float] = []
    cds_int: list[float] = []
    umax_hist: list[float] = []
    step = 0
    for step in range(1, n_steps + 1):
        f = step_fn(f)
        if step % sample_interval == 0:
            fx_s, fy_s = compute_obstacle_forces(f, surf)
            fx_a, fy_a = compute_obstacle_forces(f, solid)
            fx_i, _ = compute_obstacle_forces(f, interior)
            cds.append(-float(fx_s.item()) / dpS)
            cls.append(-float(fy_s.item()) / dpS)
            cds_all.append(-float(fx_a.item()) / dpS)
            cls_all.append(-float(fy_a.item()) / dpS)
            cds_int.append(-float(fx_i.item()) / dpS)
        if step % 5000 == 0:
            n = min(200, len(cds))
            if n:
                print(
                    f"{tag} step={step} Cd_surf={sum(cds[-n:]) / n:.5f} "
                    f"Cl_surf={sum(cls[-n:]) / n:+.5f} Cd_all={sum(cds_all[-n:]) / n:.5f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cds)
    win = min(n_tot, 300)
    cd = sum(cds[-win:]) / win
    cl = sum(cls[-win:]) / win
    cd_all = sum(cds_all[-win:]) / win
    cd_int = sum(cds_int[-win:]) / win

    def plateau(arr, frac):
        kk = int(n_tot * frac)
        seg = arr[max(0, kk - win) : kk]
        return sum(seg) / len(seg) if seg else float("nan")

    cd_pl = [plateau(cds, fr) for fr in (0.25, 0.5, 0.75, 1.0)]
    cd_all_pl = [plateau(cds_all, fr) for fr in (0.25, 0.5, 0.75, 1.0)]
    cl_pl = [plateau(cls, fr) for fr in (0.25, 0.5, 0.75, 1.0)]
    err = (cd - CD_REF) / CD_REF * 100.0

    print(
        f"{tag} === FINAL win={win} Cd_surf={cd:.5f} err={err:+.2f}% "
        f"Cl_surf={cl:+.5f} | Cd_all={cd_all:.5f} Cd_int={cd_int:.5f} "
        f"(plateau Cd {[round(v, 4) for v in cd_pl]}) ({elapsed:.0f}s)",
        flush=True,
    )
    print(f"{tag} plateau Cl_surf {[round(v, 4) for v in cl_pl]}", flush=True)

    routing = dict(
        compile_status=getattr(step_fn, "compile_status", None),
        compile_mode_effective=getattr(step_fn, "compile_mode_effective", None),
        compile_status_reason=getattr(step_fn, "compile_status_reason", None),
    )

    if save_field:
        torch.save(f.detach().cpu(), save_field)
        print(f"{tag} saved final field -> {save_field}", flush=True)

    return {
        "C": C,
        "alpha_deg": alpha_deg,
        "nx": nx,
        "ny": ny,
        "x_le": x_le,
        "y_c": y_c,
        "upstream_c": UPSTREAM_C,
        "downstream_c": DOWNSTREAM_C,
        "halfheight_c": HALFHEIGHT_C,
        "blockage_pct": blockage,
        "Re": re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_solid_cells": n_solid,
        "n_surface_cells": n_surf,
        "n_interior_cells": n_int,
        "n_steps": n_steps,
        "n_finished": step,
        "sample_interval": sample_interval,
        "avg_window_samples": win,
        "cd_mem_surface": cd,
        "cl_mem_surface": cl,
        "cd_mem_all": cd_all,
        "cd_mem_interior": cd_int,
        "err_cd_pct": err,
        "cd": cd,
        "cl": cl,
        "plateau_cd_surface": cd_pl,
        "plateau_cd_all": cd_all_pl,
        "plateau_cl_surface": cl_pl,
        "ref_cd": CD_REF,
        "ref_cluster": list(CD_CLUSTER),
        "ref_note": REF_NOTE,
        "mass_drift_pct": (float(f.sum().item()) - im0) / im0 * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_s": elapsed,
        **routing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "verify"])
    ap.add_argument("arg", help="C cells (single) or output dir (verify)")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--u-in", type=float, default=U_IN_DEFAULT)
    ap.add_argument("--re", type=float, default=RE_DEFAULT)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--save-field", action="store_true")
    add_compile_mode_arg(ap, default="eager")
    a = ap.parse_args()
    cm = compile_mode_from_args(a)

    if a.mode == "single":
        out = a.out or f"/tmp/naca0012_C{a.arg}.json"
        field = (out + ".field.pt") if a.save_field else None
        r = run_case(
            int(a.arg),
            a.alpha,
            a.device,
            a.steps,
            a.u_in,
            a.re,
            sample_interval=a.sample,
            warmup_frac=a.warmup_frac,
            compile_mode=cm,
            save_field=field,
        )
        Path(out).write_text(json.dumps(r, indent=2))
        print(f"[naca0012] saved {out}", flush=True)
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = []
    for C in a.grids:
        field = str(out_dir / f"final_C{C}.pt") if a.save_field else None
        r = run_case(
            C,
            a.alpha,
            a.device,
            a.steps,
            a.u_in,
            a.re,
            sample_interval=a.sample,
            warmup_frac=a.warmup_frac,
            compile_mode=cm,
            save_field=field,
        )
        per_grid.append(r)
        (out_dir / f"case_C{C}.json").write_text(json.dumps(r, indent=2))
    cd = [r["cd_mem_surface"] for r in per_grid]
    errs = [r["err_cd_pct"] for r in per_grid]
    span = abs(cd[-1] - cd[0]) / CD_REF * 100.0 if len(cd) > 1 else float("nan")
    ok = all(abs(e) <= 3.0 for e in errs)
    conv = span <= 3.0
    res = {
        "case": "naca0012_lowre",
        "alpha_deg": a.alpha,
        "Re": a.re,
        "force_method": "Ladd MEM, surface-only (post-step field, negated)",
        "reference": REF_NOTE,
        "ref_cd": CD_REF,
        "ref_cluster": list(CD_CLUSTER),
        "grids": per_grid,
        "convergence": {
            "cd": cd,
            "cd_span_pct": span,
            "cd_within_3pct": ok,
            "grid_span_within_3pct": conv,
        },
        "verified": bool(ok and conv),
        "verdict": "verified" if (ok and conv) else "not_verified",
        "compile_mode_effective": per_grid[0].get("compile_mode_effective") if per_grid else None,
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["convergence"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
