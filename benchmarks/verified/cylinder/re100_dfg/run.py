#!/usr/bin/env python3
"""Schäfer & Turek (1996) DFG benchmark, case 2D-2 — cylinder at Re=100 in a channel.

Task anchor: CASE-CYLINDER-DRAG-RE100 (mission m15776).

This script is the *entry point* of the verification case: it runs the grid
series through :mod:`tensorlbm.validation.cylinder_dfg` (the implementation),
writes one artifact per grid level plus the aggregated ``result.json`` that the
validation-matrix case ``cylinder_re100_dfg`` is judged from.

Reference values: ``REFERENCE.md`` next to this file (frozen register, re-read
by ``benchmarks/validation_matrix/reference_audit.py``).

Usage
-----
Whole series (default D = 20 / 28 / 40, 60k steps each, ~1 h on 4 CPU threads
per level)::

    python benchmarks/verified/cylinder/re100_dfg/run.py --threads 4

One level only (parallel friendly; the aggregator reuses existing level files)::

    python benchmarks/verified/cylinder/re100_dfg/run.py --levels 40 --steps 60000
    python benchmarks/verified/cylinder/re100_dfg/run.py --aggregate-only

Mach/τ sensitivity point (same Re, lower mean velocity → different τ)::

    python benchmarks/verified/cylinder/re100_dfg/run.py --levels 40 --u-mean 0.05 --tag u0050
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

LEVELS = (20, 28, 40)


def level_path(out_dir: pathlib.Path, D: int, tag: str) -> pathlib.Path:
    return out_dir / f"level_D{D}_{tag}.json"


def run_level(D: int, steps: int, u_mean: float, threads: int, device: str, out_dir: pathlib.Path,
              tag: str, avg_periods: float, field: bool = False) -> dict:
    cfg = cdfg.CaseConfig(
        D_cells=D, u_mean_lb=u_mean, n_steps=steps, avg_periods=avg_periods
    )
    res = cdfg.run_case(cfg, device=device, threads=threads, keep_series=True,
                        keep_field=field)
    series = res.pop("series", None)
    fld = res.pop("field", None)
    if fld:
        np.savez_compressed(level_path(out_dir, D, tag).with_suffix(".field.npz"), **fld)
    path = level_path(out_dir, D, tag)
    path.write_text(json.dumps(res, indent=2, default=cdfg._json_default))
    if series:
        import numpy as np

        np.savez(
            path.with_suffix(".series.npz"),
            **{k: np.asarray(v, dtype=np.float32) for k, v in series.items()
               if k != "window_start_step"},
        )
    print(f"[level D={D} tag={tag}] cd={res['cd']:.4f} ({res['err_pct']['cd']:+.2f}%) "
          f"st={res['st']:.4f} cl_amp={res['cl_amp']:.4f} dp*={res['dp']:.4f} "
          f"({res['wall_s']:.0f}s)", flush=True)
    return res


def aggregate(out_dir: pathlib.Path, tag: str, levels: tuple[int, ...], steps: int,
              u_mean: float, tol_pct: float) -> dict:
    grids: dict[str, dict] = {}
    bands = cdfg.REFERENCE_BANDS["2D-2"]
    for D in levels:
        path = level_path(out_dir, D, tag)
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        # Errors are re-derived from values + bands here (never trusted from the
        # level file): a level artifact written before a band/unit fix must not
        # be able to carry a stale percentage into the aggregate.
        res["err_pct"] = {
            "cd": cdfg.band_error(res["cd"], bands["cd"])["err_pct_vs_mid"],
            "cl_amp": cdfg.band_error(res["cl_amp"], bands["cl_amp"])["err_pct_vs_mid"],
            "st": cdfg.band_error(res["st"], bands["st"])["err_pct_vs_mid"],
            "dp": cdfg.band_error(res["dp"], bands["dp_half_rho_u2"])["err_pct_vs_mid"],
        }
        grids[f"D{D}"] = {
            "nx": res["nx"],
            "ny": res["ny"],
            "cells": res["cells"],
            "u_mean_lb": res["u_mean_lb"],
            "tau": res["tau"],
            "n_steps": res["n_steps"],
            "avg_window": res["avg_window"],
            "artifact": str(path.relative_to(REPO_ROOT)),
            "cd": res["cd"],
            "cd_stderr": res.get("cd_stderr", float("nan")),
            "cd_first_half": res.get("cd_first_half", float("nan")),
            "cd_second_half": res.get("cd_second_half", float("nan")),
            "cl_mean": res["cl_mean"],
            "cl_amp": res["cl_amp"],
            "st": res["st"],
            "st_acf": res.get("st_acf", float("nan")),
            "st_fft": res.get("st_fft", float("nan")),
            "st_ambiguous": res.get("st_ambiguous", float("nan")),
            "dp": res["dp"],  # Δp/(0.5ρU²)
            "dp_over_rho_u2": res.get("dp_over_rho_u2", res["dp"] / 2.0),  # paper units
            "dp_nearest_fluid_columns": res.get("dp_nearest_fluid_columns", float("nan")),
            "bench": res["bench"],
            # flat paths consumed by the validation matrix
            "err_pct": res["err_pct"],
            "err_pct_cd": res["err_pct"]["cd"],
            "err_pct_st": res["err_pct"]["st"],
            "err_pct_cl_amp": res["err_pct"]["cl_amp"],
            "err_pct_dp": res["err_pct"]["dp"],
            "drift_pct": res.get("drift_pct", float("nan")),
        }

    ref = cdfg.REFERENCE_BANDS["2D-2"]
    names = sorted(grids, key=lambda k: int(k[1:]))
    finest = grids[names[-1]] if names else None

    def _ok(err: float) -> bool:
        return abs(err) <= tol_pct

    metrics = ("cd", "st", "cl_amp", "dp")
    per_metric = {
        m: {
            "err_pct": {k: grids[k]["err_pct"][m] for k in names},
            "all_levels_within_tol": all(_ok(grids[k]["err_pct"][m]) for k in names),
            "finest_within_tol": _ok(finest["err_pct"][m]) if finest else False,
        }
        for m in metrics
    }
    convergence = {}
    if len(names) >= 2:
        a, b = grids[names[-2]], grids[names[-1]]
        convergence["cd_span_pct"] = abs(b["cd"] - a["cd"]) / max(abs(a["cd"]), 1e-12) * 100.0
        convergence["st_span_pct"] = abs(b["st"] - a["st"]) / max(abs(a["st"]), 1e-12) * 100.0
    converged = all(v <= 3.0 for v in convergence.values()) if convergence else False

    machine_pass = bool(
        finest
        and all(_ok(finest["err_pct"][m]) for m in metrics)
        and converged
    )
    result = {
        "case": "CASE-CYLINDER-DRAG-RE100",
        "benchmark": "Schäfer & Turek 1996, NNFM 48, case 2D-2 (Re=100, confined)",
        "lattice": "D2Q9",
        "collision": "MRT (tensorlbm.solver.collide_mrt)",
        "boundary": "zou_he_inlet_velocity(parabolic) + zou_he_outlet_pressure(rho=1) + "
        "half-way bounce-back on walls and cylinder",
        "force": "ladd_momentum_exchange (post-stream, pre-bounce-back)",
        "geometry": {
            "channel": [cdfg.LX, cdfg.LY],
            "cylinder_centre": [cdfg.CYL_CX, cdfg.CYL_CY],
            "cylinder_diameter": cdfg.CYL_D,
            "blockage_D_over_H": cdfg.CYL_D / cdfg.LY,
            "re": cdfg.REFERENCE_BANDS["2D-2"]["re"],
        },
        "reference": {
            "source": ref["source"],
            "register": "benchmarks/verified/cylinder/re100_dfg/REFERENCE.md",
            "cd": list(ref["cd"]),
            "cl_amp": list(ref["cl_amp"]),
            "st": list(ref["st"]),
            "dp_raw": list(ref["dp"]),  # Δp/ρU², as printed in the paper
            "dp_half_rho_u2": list(ref["dp_half_rho_u2"]),  # comparison band (0.5ρU² units)
        },
        "tolerance": {"err_pct": tol_pct, "convergence_span_pct": 3.0,
                      "error_convention": "value vs band midpoint"},
        "config": {"levels": list(names), "steps": steps, "u_mean_lb": u_mean, "tag": tag},
        "grids": grids,
        "per_metric": per_metric,
        "convergence": convergence,
        "machine_pass": machine_pass,
        "verdict": "verified" if machine_pass else "not_verified",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    ap.add_argument("--steps", type=int, nargs="+", default=[60_000],
                    help="steps per level (one value = same for all levels)")
    ap.add_argument("--avg-periods", type=float, default=20.0)
    ap.add_argument("--u-mean", type=float, default=0.0833)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None, help="artifact tag (default from u_mean)")
    ap.add_argument("--tol", type=float, default=5.0, help="acceptance error band, percent")
    ap.add_argument("--out-dir", default=str(HERE / "artifacts"))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--field", action="store_true",
                    help="also dump the final rho/u/vorticity field (report figures)")
    a = ap.parse_args(argv)

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or ("u0833" if abs(a.u_mean - 0.0833) < 1e-9 else f"u{a.u_mean:.4f}".replace(".", ""))

    if not a.aggregate_only:
        steps_for = (a.steps if len(a.steps) == len(a.levels)
                     else [a.steps[0]] * len(a.levels))
        for D, n_steps in zip(a.levels, steps_for):
            run_level(D, n_steps, a.u_mean, a.threads, a.device, out_dir, tag, a.avg_periods,
                      field=a.field)

    levels = tuple(sorted({*LEVELS, *a.levels}))
    result = aggregate(out_dir, tag, levels, a.steps, a.u_mean, a.tol)
    print(json.dumps({k: result[k] for k in ("per_metric", "convergence", "machine_pass",
                                             "verdict")}, indent=2))
    return 0 if result["machine_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())