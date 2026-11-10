#!/usr/bin/env python
"""Campaign driver for the TensorLBM invariance certificate (P493-N02).

Sub-commands (``--case``):

``periodic``   C1/C6/C8 — force-driven periodic channel, four wall resolutions,
               walls at the lattice midpoint (C1) and off-midpoint with the
               interpolated vs the half-way wall closure (C6 negative control)
``poiseuille`` C1b/C8/C9 — inclined-channel Poiseuille at theta = 0/15/30/45
``cylinder``   C2/C3 — Cd / St orientation spread at Re = 100 and 200
``galilean``   C4/C5 — boosted Taylor-Green superposition defect and Ma^2 law
``bulk``       C10 — bulk orientation invariance of the Taylor-Green decay rate
``repro``      C7 — bitwise reproducibility of two identical runs
``report``     evaluate C1..C10 and write report.json (no simulation)

Every simulation result is written immediately as JSON under
``artifacts/invariance/P493-N02/`` together with a provenance block (commit,
host, timings) and the SHA-256 of the artefact itself.

Usage
-----
    python scripts/run_invariance_campaign_p493.py --case all --workers 16
    python scripts/run_invariance_campaign_p493.py --case report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.invariance import (  # noqa: E402
    bulk_orientation_case,
    cylinder_case,
    evaluate_criteria,
    galilean_superposition_residual,
    observed_order,
    periodic_channel_case,
    poiseuille_case,
    taylor_green_case,
)
from tensorlbm.invariance.cases import _PeriodicBox, tg_exact  # noqa: E402
from tensorlbm.invariance.solver import D2Q9Solver  # noqa: E402

OUT = ROOT / "artifacts" / "invariance" / "P493-N02"

ANGLES = (0.0, 15.0, 30.0, 45.0)
WIDTHS = (8, 12, 16, 24)
OFFSET = 0.25


# --------------------------------------------------------------------- utils
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def provenance(extra: dict | None = None) -> dict:
    prov = {
        "commit": _git_commit(),
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": str(ROOT),
        "criterion_owner": "P493-N02",
    }
    if extra:
        prov.update(extra)
    return prov


def write_json(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    blob = json.dumps(payload, indent=2, sort_keys=False, default=float)
    path.write_text(blob)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    (OUT / (name + ".sha256")).write_text(digest + "\n")
    print(f"[write] {path.relative_to(ROOT)}  sha256={digest[:16]}…", flush=True)
    return path


def _wrap(job: tuple[str, dict]) -> dict:
    kind, kwargs = job
    t0 = time.time()
    if kind == "periodic":
        res = periodic_channel_case(**kwargs)
    elif kind == "poiseuille":
        res = poiseuille_case(**kwargs)
    elif kind == "cylinder":
        res = cylinder_case(**kwargs)
    elif kind == "tg":
        res = taylor_green_case(**kwargs)
    elif kind == "galilean":
        res = galilean_superposition_residual(**kwargs)
    elif kind == "bulk":
        res = bulk_orientation_case(**kwargs)
    else:  # pragma: no cover
        raise ValueError(kind)
    res["wall_seconds"] = round(time.time() - t0, 2)
    res["kind"] = kind
    print(f"[{kind}] {kwargs} -> th={res.get('theta_deg')} w={res.get('width')} "
          f"ma={res.get('ma')} {res['wall_seconds']}s", flush=True)
    return res


def run_jobs(jobs: list[tuple[str, dict]], workers: int) -> list[dict]:
    if workers <= 1:
        return [_wrap(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(workers, len(jobs)), maxtasksperchild=1) as pool:
        return pool.map(_wrap, jobs, chunksize=1)


# --------------------------------------------------------------------- cases
def case_periodic(workers: int, widths=WIDTHS, offset: float = 0.0,
                  bounce_back: str = "interpolated", label: str = "interpolated") -> dict:
    jobs = [("periodic", dict(width=h, offset=offset, bounce_back=bounce_back,
                              u_mean=0.01, nu=0.1, steps_factor=20.0)) for h in widths]
    results = sorted(run_jobs(jobs, workers), key=lambda r: r["width"])
    errors = [r["l2_error"] for r in results]
    summary = {
        "widths": [r["width"] for r in results],
        "errors": errors,
        "linf": [r["linf_error"] for r in results],
        "wall_slip": [r["wall_slip"] for r in results],
        "wall_offset_cells": [r["wall_offset_cells"] for r in results],
        "drho": [r["density"]["max_abs"] for r in results],
        "orders": observed_order(errors, [r["width"] for r in results]),
        "runs": results,
        "offset": offset,
        "bounce_back": bounce_back,
        "label": label,
    }
    return {label: summary}


def case_poiseuille(workers: int, widths=WIDTHS, angles=ANGLES,
                    bounce_back: str = "interpolated",
                    drive: str = "body_force") -> dict:
    jobs = []
    for theta in angles:
        for h in widths:
            jobs.append(("poiseuille", dict(theta_deg=theta, width=h,
                                            bounce_back=bounce_back, u_mean=0.01,
                                            nu=0.1, steps_factor=30.0, drive=drive)))
    results = run_jobs(jobs, workers)
    summary: dict[str, dict] = {}
    for res in results:
        key = f"{res['theta_deg']:g}"
        entry = summary.setdefault(key, {"widths": [], "errors": [], "linf": [],
                                        "flux": [], "wall_slip": [],
                                        "wall_offset_cells": [], "runs": []})
        entry["widths"].append(res["width"])
        entry["errors"].append(res["l2_error"])
        entry["linf"].append(res["linf_error"])
        entry["flux"].append(res["flux_error_rel"])
        entry["wall_slip"].append(res["wall_slip"])
        entry["wall_offset_cells"].append(res["wall_offset_cells"])
        entry["runs"].append(res)
    for key, entry in summary.items():
        order = np.argsort(entry["widths"])
        for name in ("widths", "errors", "linf", "flux", "wall_slip",
                     "wall_offset_cells", "runs"):
            entry[name] = [entry[name][i] for i in order]
        entry["orders"] = observed_order(entry["errors"], entry["widths"])
    return summary


def case_cylinder(workers: int, angles=ANGLES, re_values=(100.0, 200.0),
                  diameter: float = 16.0, domain_per_diameter: float = 8.0,
                  spinup_periods: float = 20.0, sample_periods: float = 15.0) -> dict:
    jobs = []
    for re in re_values:
        for theta in angles:
            jobs.append(("cylinder", dict(theta_deg=theta, re=re, diameter=diameter,
                                          domain_per_diameter=domain_per_diameter,
                                          spinup_periods=spinup_periods,
                                          sample_periods=sample_periods)))
    results = run_jobs(jobs, workers)
    summary: dict[str, list] = {}
    for res in results:
        summary.setdefault(f"{res['re']:g}", []).append(res)
    for key in summary:
        summary[key].sort(key=lambda r: r["theta_deg"])
    return summary


def case_galilean(workers: int, periods=(1, 2, 3, 4, 5), n: int = 64,
                  steps: int = 12000, angle_deg: float = 0.0) -> list:
    jobs = [("galilean", dict(ma=1.0, periods=p, n=n, steps=steps, u0=0.02,
                              nu=0.01, angle_deg=angle_deg,
                              sample_every=20, fit_start=500)) for p in periods]
    jobs.append(("galilean", dict(ma=0.0, periods=0, n=n, steps=steps, u0=0.02,
                                  nu=0.01, angle_deg=angle_deg,
                                  sample_every=20, fit_start=500)))
    return sorted(run_jobs(jobs, workers), key=lambda r: r["ma"])


def case_bulk(workers: int, modes=((1, 0), (3, 1), (2, 1), (1, 1), (1, 3)),
              n: int = 64, steps: int = 6000) -> dict:
    """Bulk (wall-free) orientation sweep with commensurate wave vectors."""
    jobs = [("bulk", dict(m1=m1, m2=m2, n=n, steps=steps, u0=0.02, nu=0.01,
                          sample_every=20, fit_start=500)) for m1, m2 in modes]
    results = sorted(run_jobs(jobs, workers), key=lambda r: r["angle_deg"])
    ratios = {f"{r['angle_deg']:.2f}": r["decay_ratio"] for r in results}
    dev = {f"{r['angle_deg']:.2f}": r["rel_deviation"] for r in results}
    ref = ratios.get("0.00")
    spread = max((abs(v - ref) / ref for k, v in ratios.items() if ref), default=float("nan"))
    return {
        "runs": results,
        "decay_ratio": ratios,
        "rel_deviation": dev,
        "max_rel_deviation": max(dev.values()) if dev else float("nan"),
        "max_spread_vs_zero": spread,
        "modes": [list(m) for m in modes],
    }


def case_repro(workers: int) -> list[str]:
    """Two bitwise-identical runs of the same boosted Taylor-Green case."""
    hashes = []
    for _ in range(2):
        box = _PeriodicBox(48)
        solver = D2Q9Solver(box, tau=0.8)
        ux, uy = tg_exact(48, 0.02, 0.01, 0.0, 0.01)
        solver.initialise(ux, uy)
        for _ in range(120):
            solver.step()
        hashes.append(solver.state_hash())
    return hashes


# ---------------------------------------------------------------- aggregation
def aggregate() -> dict:
    """Read every artefact written by the sub-commands into one results dict."""
    results: dict = {"provenance": provenance({"stage": "aggregate"})}
    files = {
        "periodic": "periodic_channel.json",
        "periodic_offset": "periodic_channel_offset.json",
        "poiseuille": "poiseuille.json",
        "cylinder": "cylinder.json",
        "galilean": "galilean.json",
        "bulk": "bulk_orientation.json",
        "repro": "reproducibility.json",
        "legacy": "poiseuille_legacy.json",
    }
    loaded = {}
    for key, fname in files.items():
        path = OUT / fname
        if path.exists():
            loaded[key] = json.loads(path.read_text())
    if "periodic" in loaded:
        results["periodic_channel"] = {k: v for k, v in loaded["periodic"].items()
                                       if k != "provenance"}
    if "poiseuille" in loaded:
        results["poiseuille"] = {k: v for k, v in loaded["poiseuille"]["poiseuille"].items()}
    if "cylinder" in loaded:
        results["cylinder"] = loaded["cylinder"]["by_re"]
    if "galilean" in loaded:
        results["galilean"] = loaded["galilean"]["galilean"]
    if "bulk" in loaded:
        results["bulk"] = loaded["bulk"]["bulk"]
    if "repro" in loaded:
        results["reproducibility"] = loaded["repro"]["hashes"]
    if "legacy" in loaded:
        results["legacy"] = loaded["legacy"]["comparison"]
    if "periodic_offset" in loaded and "periodic" in loaded:
        results["negative_control"] = {
            "interpolated": loaded["periodic"]["interpolated"],
            "halfway": loaded["periodic_offset"]["halfway"],
        }
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="all",
                    choices=["periodic", "poiseuille", "cylinder", "galilean",
                             "bulk", "repro", "report", "all"])
    ap.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    ap.add_argument("--angles", type=float, nargs="*", default=list(ANGLES))
    ap.add_argument("--widths", type=float, nargs="*", default=list(WIDTHS))
    ap.add_argument("--diameter", type=float, default=16.0)
    ap.add_argument("--domain", type=float, default=8.0)
    ap.add_argument("--spinup-periods", type=float, default=20.0)
    ap.add_argument("--sample-periods", type=float, default=15.0)
    args = ap.parse_args(argv)

    t_start = time.time()
    if args.case in ("periodic", "all"):
        write_json("periodic_channel.json", {
            "provenance": provenance({"case": "periodic"}),
            **case_periodic(args.workers, args.widths),
        })
        write_json("periodic_channel_offset.json", {
            "provenance": provenance({"case": "periodic_offset"}),
            **case_periodic(args.workers, args.widths, offset=OFFSET,
                            bounce_back="halfway", label="halfway"),
        })
    if args.case in ("poiseuille", "all"):
        write_json("poiseuille.json", {
            "provenance": provenance({"case": "poiseuille"}),
            "poiseuille": case_poiseuille(args.workers, args.widths, args.angles),
        })
    if args.case in ("cylinder", "all"):
        write_json("cylinder.json", {
            "provenance": provenance({"case": "cylinder"}),
            "by_re": case_cylinder(args.workers, args.angles,
                                   diameter=args.diameter,
                                   domain_per_diameter=args.domain,
                                   spinup_periods=args.spinup_periods,
                                   sample_periods=args.sample_periods),
        })
    if args.case in ("galilean", "all"):
        write_json("galilean.json", {
            "provenance": provenance({"case": "galilean"}),
            "galilean": case_galilean(args.workers),
        })
    if args.case in ("bulk", "all"):
        write_json("bulk_orientation.json", {
            "provenance": provenance({"case": "bulk"}),
            "bulk": case_bulk(args.workers),
        })
    if args.case in ("repro", "all"):
        write_json("reproducibility.json", {
            "provenance": provenance({"case": "repro"}),
            "hashes": case_repro(args.workers),
        })
    if args.case in ("report", "all"):
        results = aggregate()
        report = evaluate_criteria(results)
        report["measured"] = {k: v for k, v in results.items()
                              if k not in ("provenance",)}
        report["provenance"]["wall_seconds_total"] = round(time.time() - t_start, 1)
        write_json("report.json", report)
        print(f"[report] {report['n_pass']}/{report['n_total']} criteria pass, "
              f"all_pass={report['all_pass']}", flush=True)
        for c in report["criteria"]:
            print(f"   {c['id']:4s} {'PASS' if c['pass'] else 'FAIL'}  {c['name']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())