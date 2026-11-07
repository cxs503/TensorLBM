#!/usr/bin/env python3
"""IN25-T6 seat C -- mature-code alignment seat: *fresh* three-grid convergence
ladder on the DARPA SUBOFF bare hull at Re = 2000 (a physical point that is
**not** in ``benchmarks/verified``; the verified SUBOFF case sits at Re = 1000).

Purpose
-------
The survey compares the grid-convergence *practice* reported by mature
external-flow codes (three grids, refinement ratio ~1.3-1.5, Richardson
observed order, GCI) with what our own solver actually delivers on the same
class of geometry.  This script produces the "ours" column from our own runs,
never from a number copied out of another group's paper:

  * three resolutions of the same physical case (Re and domain padding fixed),
  * Richardson extrapolation -> observed order ``p`` and GCI.

Everything is driven by library entry points (GeneralSimEngine /
lbm_step_correct / drag_*_integration), reusing the verified-benchmark
time-stepping route so the ladder is comparable with the verified Re=1000 case.

Usage
-----
    python scripts/in25_t6/seat_c_grid_convergence.py --resolutions 20 28 40 --steps 6000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

OUT = REPO / "artifacts" / "in25_t6"

SUBOFF_LENGTH_M = 4.356
SUBOFF_RADIUS_M = 0.254
U_PHYS = 1.0e-3


def load_b6_module():
    """Import the verified SUBOFF runner (shared step route + area helpers)."""
    path = REPO / "benchmarks" / "verified" / "suboff_re1000" / "run.py"
    spec = importlib.util.spec_from_file_location("b6_suboff_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_config(resolution: int, steps: int, re: float, device: str, out_dir: Path):
    from tensorlbm.general_sim import (
        CollisionModel,
        ForceMethod,
        GeneralSimConfig,
        GeometryConfig,
        GeometrySource,
        LatticeModel,
        OutputConfig,
        PhysicsConfig,
        SolverConfig,
        WallTreatment,
    )

    viscosity = U_PHYS * SUBOFF_LENGTH_M / re
    return GeneralSimConfig(
        name=f"in25_t6_suboff_re{int(re)}_L{resolution}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=resolution,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="mix50",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(out_dir),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )


def tail_mean(values: list[float], fraction: float = 0.4) -> float:
    if not values:
        raise RuntimeError("no force samples logged (run too short or diverged)")
    n = max(1, int(round(len(values) * fraction)))
    tail = values[-n:]
    return sum(tail) / len(tail)


def richardson(h: list[float], f: list[float]) -> dict:
    """Observed order + GCI for a three-grid sequence (coarse -> fine).

    Standard triple-grid estimator with measured refinement ratio r = h1/h3:
        s = (f3-f2)/(f2-f1),  p = ln s / ln r,  f_exact = f3 + (f3-f2)/(s-1)
        GCI_fine = |(f3-f2)/(f2-f1)| / (r^p - 1) * 100 %
    """
    r = h[0] / h[2]
    s = (f[2] - f[1]) / (f[1] - f[0])
    out = {"refinement_ratio_r": r, "convergence_ratio_s": s}
    if not (s > 0.0) or s == 1.0:
        out.update(monotone=False, verdict="non-monotone: no Richardson estimate")
        return out
    p = math.log(s) / math.log(r)
    plateau = (f[2] - f[1]) / (s - 1.0)
    f_ext = f[2] + plateau
    out.update(
        monotone=True,
        observed_order_p=p,
        f_extrapolated=f_ext,
        rel_change_fine_pct=100.0 * abs(f[2] - f[1]) / abs(f[1]),
        gci_fine_pct=100.0 * abs((f[2] - f[1]) / (f[1] * (r**p - 1.0))),
        verdict="converged" if abs((f[2] - f[1]) / f[1]) < 0.02 else "grid-sensitive",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[20, 28, 40])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--re", type=float, default=2000.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", default=str(OUT / "seat_c_grid_ladder.json"))
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    b6 = load_b6_module()
    prev = {}
    prev_path = Path(args.out)
    if prev_path.exists():
        try:
            prev = {r["resolution_cells_per_L"]: r
                    for r in json.loads(prev_path.read_text()).get("ladder", [])}
        except Exception:
            prev = {}
    from tensorlbm.general_sim import GeneralSimEngine

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in args.resolutions:
        cfg = build_config(L, args.steps, args.re, args.device, OUT / f"seat_c_L{L}")
        engine = GeneralSimEngine(cfg)
        info = engine.setup()
        t0 = time.time()
        run_info, elapsed = b6.run_engine_routed(engine, compile_mode="eager")
        hn = float(L)  # cells per hull length; spacing h = 1/L
        R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L)
        dpS_wet = b6.wetted_dpS(engine.uc.u_lb, R_lb, float(L))
        dpS_front = b6.frontal_dpS(engine.uc.u_lb, R_lb)
        rescale = dpS_front / dpS_wet
        log = engine.forces_log
        print(f"[diag] L={L} run_info={run_info} n_force_samples={len(log)}", flush=True)
        cd_p = tail_mean([e["cd_pressure"] for e in log]) * rescale
        cd_f = tail_mean([e["cd_friction"] for e in log]) * rescale
        cd_tot = tail_mean([e["cd_total"] for e in log]) * rescale
        cd_tot_full = sum(e["cd_total"] for e in log) / len(log) * rescale
        row = {
            "resolution_cells_per_L": L,
            "grid_spacing_h": 1.0 / hn,
            "re": info.get("Re"),
            "tau": info.get("tau"),
            "nu_lb": info.get("nu_lb"),
            "u_lb": info.get("u_lb"),
            "total_cells": info.get("total_cells"),
            "obstacle_cells": info.get("obstacle_cells"),
            "near_wall_cells": info.get("near_wall_cells"),
            "domain_lu": info.get("domain_lu"),
            "steps": run_info["steps"],
            "diverged": run_info["diverged"],
            "seconds": elapsed,
            "ms_per_step": 1000.0 * elapsed / max(run_info["steps"], 1),
            "cd_pressure": cd_p,
            "cd_friction": cd_f,
            "cd_total": cd_tot,
            "cd_total_window_all_samples": cd_tot_full,
            "cd_total_series_last20": [e["cd_total"] * rescale for e in log[-20:]],
            "wall_seconds": time.time() - t0,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("resolution_cells_per_L", "cd_total",
                                             "ms_per_step", "diverged")}), flush=True)
        del engine
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    merged = {**prev, **{r["resolution_cells_per_L"]: r for r in rows}}
    rows = [merged[k] for k in sorted(merged)]
    h = [r["grid_spacing_h"] for r in rows]
    f = [r["cd_total"] for r in rows]
    rich = richardson(h, f) if len(rows) == 3 else {"note": "need exactly 3 grids"}
    payload = {
        "seat": "C 成熟代码对标楼 / 网格收敛",
        "case": {
            "geometry": "DARPA SUBOFF bare hull (AFF-1), parametric",
            "re": args.re,
            "note": "Re=2000 is NOT in benchmarks/verified (verified case = Re=1000)",
            "domain_padding_streamwise": "6L (1L upstream + 4L downstream)",
            "collision": "D3Q19 MRT",
            "wall": "half-way bounce-back (WallTreatment.AUTO)",
            "friction_formula": "mix50",
            "steps": args.steps,
            "normalisation": "S_wet = pi*D*L (repo Re-family convention)",
        },
        "ladder": rows,
        "richardson": rich,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(rich, indent=1))
    print(f"[seat C] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())