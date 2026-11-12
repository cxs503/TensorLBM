#!/usr/bin/env python3
"""CASE-CYLINDER-DRAG-RE20 — case-pool entry for the Schäfer–Turek 2D-1 cylinder, Re = 20.

One command reproduces the whole benchmark: it runs the library case
(:mod:`tensorlbm.cylinder_st2d1`) on one or more grid resolutions, compares
Cd/Cl against the pinned reference (:file:`refs.json`), and writes the run
artifacts (``results_D<k>.json``, ``series_D<k>.csv``, ``fields_D<k>.npz``,
``summary.json``) into ``--out``.

The solver chain itself lives in the library (no physics in this file): this
entry only owns the *benchmark protocol* — grid list, tolerances, artifacts,
provenance and the pass/fail verdict.

    PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20/run_case.py \
        --grids 20 40 --out runs/cylinder_drag_re20

    # fast smoke check (not a validation claim: not converged)
    PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20/run_case.py \
        --grids 8 --steps 300 --avg-window 100 --out /tmp/cyl_quick
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.cylinder_st2d1 import (  # noqa: E402
    GEOMETRY,
    REFERENCE,
    compare_to_reference,
    result_manifest,
    run_cylinder_st2d1,
)


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _environment(n_threads: int) -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": n_threads,
        "git_rev": _git_rev(),
        "device": "cpu",
    }


def _write_series(path: Path, series: dict) -> None:
    with path.open("w") as fh:
        fh.write("step,cd,cl\n")
        for step, cd, cl in zip(series["step"], series["cd"], series["cl"]):
            fh.write(f"{step},{cd:.10g},{cl:.10g}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=int, nargs="+", default=[20, 40], help="cylinder diameters in cells")
    ap.add_argument("--steps", type=int, default=200_000, help="step cap per grid")
    ap.add_argument("--min-steps", type=int, default=None, help="earliest early-stop step")
    ap.add_argument("--tau", type=float, default=0.8, help="MRT relaxation (u_mean derived from Re)")
    ap.add_argument("--u-lb", type=float, default=None, help="fix u_mean instead of tau")
    ap.add_argument("--avg-window", type=int, default=20_000)
    ap.add_argument("--plateau-tol", type=float, default=1.0e-3)
    ap.add_argument("--stop-on-plateau", action="store_true")
    ap.add_argument("--tolerance", type=float, default=None, help="gate on |err_Cd| in %%")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    threads = a.threads or min(32, os.cpu_count() or 1)
    torch.set_num_threads(threads)

    refs = json.loads((HERE / "refs.json").read_text())
    tol = refs["tolerance_pct"] if a.tolerance is None else a.tolerance
    if abs(REFERENCE["cd"] - refs["ref"]["cd"]) > 1e-9:  # guard against drift
        raise SystemExit(
            f"refs.json Cd {refs['ref']['cd']} != module REFERENCE {REFERENCE['cd']}"
        )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = _environment(threads)
    rows: list[dict] = []
    verdicts: list[dict] = []
    t_all = time.time()

    for D in a.grids:
        print(f"=== D={D} cells ===", flush=True)
        res = run_cylinder_st2d1(
            D_cells=D,
            steps=a.steps,
            tau=a.tau,
            u_lb=a.u_lb,
            device="cpu",
            avg_window=min(a.avg_window, max(a.steps // 2, 100)),
            plateau_tol=a.plateau_tol,
            stop_on_plateau=a.stop_on_plateau,
            min_steps=a.min_steps,
        )
        verdict = compare_to_reference(res, tol)
        ver = dict(verdict)
        ver["passed"] = bool(ver["passed"])
        (out / f"results_D{D}.json").write_text(
            json.dumps({**result_manifest(res), "verdict": ver, "environment": env}, indent=2, ensure_ascii=False)
        )
        _write_series(out / f"series_D{D}.csv", res["series"])
        fields = res.get("fields")
        if fields is not None:
            np.savez_compressed(out / f"fields_D{D}.npz", **fields)
        rows.append(
            {
                "D_cells": D,
                "nx": res["geometry"]["nx"],
                "ny": res["geometry"]["ny"],
                "u_mean_lb": res["lattice_units"]["u_mean"],
                "tau": res["lattice_units"]["tau"],
                "ma": res["lattice_units"]["ma"],
                "n_steps": res["n_steps"],
                "stop_reason": res["stop_reason"],
                "cd": res["cd"],
                "err_cd_pct": res["err_cd_pct"],
                "cl": res["cl"],
                "err_cl_pct": res["err_cl_pct"],
                "cd_plateau_drift": res["cd_plateau_drift"],
                "stationary": res["stationary"],
                "wall_s": res["wall_s"],
                "pass": bool(ver["passed"]),
            }
        )
        verdicts.append(ver)
        print(json.dumps(rows[-1], indent=2, ensure_ascii=False), flush=True)

    summary = {
        "anchor": refs["anchor"],
        "case": refs["case"],
        "geometry_physical": GEOMETRY,
        "reference": refs["ref"],
        "provenance": refs["provenance"],
        "tolerance_pct": tol,
        "gate": "every grid: |err_Cd| <= tolerance AND mean taken over a stationary window",
        "rows": rows,
        "per_grid_verdicts": verdicts,
        "passed": bool(all(r["pass"] for r in rows)),
        "repro_cmd": refs["repro_cmd"],
        "entry_cmd": "PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20/run_case.py "
        + " ".join(sys.argv[1:] if argv is None else argv),
        "environment": env,
        "wall_s_total": time.time() - t_all,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n| grid | nx x ny | tau | Ma | steps | stop | Cd | ref | err% | drift | stationary | pass |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| D={r['D_cells']} | {r['nx']}x{r['ny']} | {r['tau']:.3f} | {r['ma']:.3f} | "
            f"{r['n_steps']} | {r['stop_reason']} | {r['cd']:.6f} | {refs['ref']['cd']:.6f} | "
            f"{r['err_cd_pct']:+.3f}% | {r['cd_plateau_drift']:.2e} | {r['stationary']} | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    print(f"\n-> {out}/summary.json  ({'PASS' if summary['passed'] else 'FAIL'})", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())