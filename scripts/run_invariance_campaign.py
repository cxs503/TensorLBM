#!/usr/bin/env python
"""Campaign driver for the TensorLBM invariance certificate (P170W2-N08).

Sub-commands (``--case``):

``poiseuille``  C1 — inclined-channel Poiseuille, 4 wall resolutions x 4 angles
``cylinder``    C2/C3 — Cd / St orientation spread at Re = 100 and 200
``negative``    C6 — same sweep with raw BGK and tau -> 0.5+ (must break C2)
``galilean``    C4/C5 — boosted Taylor-Green decay + superposition residual
``repro``       C7 — bitwise reproducibility of two identical runs
``report``      evaluate C1..C7 and write report.json (no simulation)

Every simulation result is written immediately as JSON under
``artifacts/invariance/P170W2-N08/`` together with a provenance block
(commit, host, timings, SHA-256 of the artefact itself).

Usage
-----
    python scripts/run_invariance_campaign.py --case all --workers 8
    python scripts/run_invariance_campaign.py --case report
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
    cylinder_case,
    evaluate_criteria,
    galilean_superposition_residual,
    observed_order,
    poiseuille_case,
    taylor_green_case,
)
from tensorlbm.invariance.solver import D2Q9Solver  # noqa: E402
from tensorlbm.invariance.cases import _PeriodicBox, tg_exact  # noqa: E402

OUT = ROOT / "artifacts" / "invariance" / "P170W2-N08"

ANGLES = (0.0, 15.0, 30.0, 45.0)
WIDTHS = (8, 12, 16, 24)


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
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": str(ROOT),
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


def _wrap(job: tuple[str, dict]) -> tuple[str, dict, float]:
    kind, kwargs = job
    t0 = time.time()
    if kind == "poiseuille":
        res = poiseuille_case(**kwargs)
    elif kind == "cylinder":
        res = cylinder_case(**kwargs)
    elif kind == "tg":
        res = taylor_green_case(**kwargs)
    elif kind == "superposition":
        res = galilean_superposition_residual(**kwargs)
    else:  # pragma: no cover
        raise ValueError(kind)
    res["wall_seconds"] = round(time.time() - t0, 2)
    res["kind"] = kind
    print(f"[{kind}] {kwargs} -> {res.get('theta_deg', '')} "
          f"{res.get('ma', '')} {res['wall_seconds']}s", flush=True)
    return kind, res, res["wall_seconds"]


def run_jobs(jobs: list[tuple[str, dict]], workers: int) -> list[tuple[str, dict, float]]:
    if workers <= 1:
        return [_wrap(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        return pool.map(_wrap, jobs, chunksize=1)


# --------------------------------------------------------------------- cases
def case_poiseuille(workers: int, widths=WIDTHS, angles=ANGLES,
                    bounce_back: str = "interpolated", label: str = "poiseuille") -> dict:
    jobs = []
    for theta in angles:
        for h in widths:
            jobs.append(("poiseuille", dict(theta_deg=theta, width=h,
                                            bounce_back=bounce_back, u_mean=0.01,
                                            nu=0.1, steps_factor=30.0)))
    t0 = time.time()
    results = run_jobs(jobs, workers)
    summary = {}
    for _, res, _ in results:
        key = f"{res['theta_deg']:g}"
        entry = summary.setdefault(key, {"widths": [], "errors": [], "linf": [],
                                         "flux": [], "rows": []})
        entry["widths"].append(res["width"])
        entry["errors"].append(res["l2_error"])
        entry["linf"].append(res["linf_error"])
        entry["flux"].append(res["flux_error_rel"])
        entry["rows"].append(res)
    for key, entry in summary.items():
        order = np.argsort(entry["widths"])
        entry["widths"] = [entry["widths"][i] for i in order]
        entry["errors"] = [entry["errors"][i] for i in order]
        entry["linf"] = [entry["linf"][i] for i in order]
        entry["flux"] = [entry["flux"][i] for i in order]
        entry["rows"] = [entry["rows"][i] for i in order]
        entry["orders"] = observed_order(entry["errors"], entry["widths"])
        entry["mean_order"] = float(np.mean(entry["orders"]))
    payload = {
        "case": label,
        "bounce_back": bounce_back,
        "angles_deg": list(angles),
        "widths": list(widths),
        "parameters": {"u_mean": 0.01, "nu": 0.1, "tau": 0.8,
                       "steps_factor": 30.0, "trim": 0.25, "dtype": "float64"},
        "results": summary,
        "wall_seconds_total": round(time.time() - t0, 1),
    }
    payload["provenance"] = provenance()
    write_json(f"{label}.json", payload)
    return payload


def case_cylinder(workers: int, re_list=(100.0, 200.0), label: str = "cylinder",
                  collision: str = "trt", bounce_back: str = "interpolated",
                  diameter: float = 12.0, transient_d: float = 30.0,
                  sample_d: float = 18.0) -> dict:
    jobs = []
    for re in re_list:
        for theta in ANGLES:
            jobs.append(("cylinder", dict(theta_deg=theta, re=re, diameter=diameter,
                                          collision=collision,
                                          bounce_back=bounce_back,
                                          transient_d=transient_d,
                                          sample_d=sample_d)))
    t0 = time.time()
    results = run_jobs(jobs, workers)
    grouped: dict[str, list] = {}
    for _, res, _ in results:
        grouped.setdefault(f"{res['re']:g}", []).append(res)
    for k in grouped:
        grouped[k].sort(key=lambda r: r["theta_deg"])
    # keep the JSON compact: drop the raw probe signal from the summary file
    compact = {
        re: [{kk: vv for kk, vv in r.items() if kk != "probe_signal"} for r in runs]
        for re, runs in grouped.items()
    }
    payload = {
        "case": "cylinder",
        "label": label,
        "angles_deg": list(ANGLES),
        "parameters": {"diameter": diameter, "domain_per_diameter": 16.0,
                       "u_target": 0.05, "collision": collision,
                       "bounce_back": bounce_back, "transient_d": transient_d,
                       "sample_d": sample_d, "dtype": "float32",
                       "drive": "dead-beat bulk-velocity controller "
                                "(f_n = rho*(u_target - u_mean))"},
        "results": grouped,
        "results_compact": compact,
        "wall_seconds_total": round(time.time() - t0, 1),
    }
    payload["provenance"] = provenance()
    write_json(f"{label}.json", payload)
    return payload


def case_negative(workers: int, label: str = "negative_control") -> dict:
    """Raw BGK with tau -> 0.5+ (Re far above the resolved range)."""
    jobs = []
    for theta in ANGLES:
        jobs.append(("cylinder", dict(theta_deg=theta, re=20000.0, diameter=12.0,
                                      domain_per_diameter=16.0,
                                      collision="bgk", bounce_back="interpolated",
                                      transient_d=15.0, sample_d=10.0)))
    t0 = time.time()
    results = run_jobs(jobs, workers)
    grouped: dict[str, list] = {}
    for _, res, _ in results:
        grouped.setdefault(f"{res['re']:g}", []).append(res)
    for k in grouped:
        grouped[k].sort(key=lambda r: r["theta_deg"])
    payload = {
        "case": "cylinder",
        "label": label,
        "note": "raw BGK, tau -> 0.5+ (Re_effective ~ 2e4), same resolution as C2",
        "results": grouped,
        "wall_seconds_total": round(time.time() - t0, 1),
    }
    payload["provenance"] = provenance()
    write_json(f"{label}.json", payload)
    return payload


def case_galilean(workers: int, periods=(1, 2, 3, 4, 5, 6, 8, 10),
                  n: int = 64, steps: int = 12000, nu: float = 0.01,
                  u0: float = 0.02, label: str = "galilean") -> dict:
    jobs = [("tg", dict(periods=0, ma=0.0, n=n, steps=steps, nu=nu, u0=u0))]
    jobs += [("tg", dict(periods=p, ma=1.0, n=n, steps=steps, nu=nu, u0=u0))
             for p in periods]
    jobs += [("superposition", dict(periods=p, ma=1.0, n=n, steps=steps, nu=nu,
                                    u0=u0)) for p in periods if p <= 5]
    t0 = time.time()
    results = run_jobs(jobs, workers)
    tg = [r for k, r, _ in results if k == "tg"]
    sup = [r for k, r, _ in results if k == "superposition"]
    sup_by_ma = {round(r["ma"], 10): r for r in sup}
    for r in tg:
        s = sup_by_ma.get(round(r["ma"], 10))
        r["superposition_residual_relU"] = (
            s["superposition_residual_relU"] if s else None)
    tg.sort(key=lambda r: r["ma"])
    payload = {
        "case": "taylor_green",
        "label": label,
        "parameters": {"n": n, "steps": steps, "nu": nu, "u0": u0,
                       "periods": list(periods), "collision": "bgk",
                       "dtype": "float64",
                       "note": "boost U = periods*n/steps so the vortex shifts by "
                               "an integer number of cells"},
        "results": tg,
        "superposition": sup,
        "wall_seconds_total": round(time.time() - t0, 1),
    }
    payload["provenance"] = provenance()
    write_json(f"{label}.json", payload)
    return payload


def case_repro(label: str = "reproducibility") -> dict:
    """Same seed/parameters twice -> identical SHA-256 state hash (C7)."""
    box = _PeriodicBox(64)
    hashes = []
    for _ in range(2):
        solver = D2Q9Solver(box, tau=0.8, dtype=np.float64)
        ux, uy = tg_exact(64, 0.02, 0.01, 0.0, 0.02)
        solver.initialise(ux, uy)
        for _ in range(400):
            solver.step()
        hashes.append(solver.state_hash())
    payload = {
        "case": "reproducibility",
        "parameters": {"n": 64, "steps": 400, "tau": 0.8, "dtype": "float64"},
        "hashes": hashes,
        "states": ["taylor_green_boosted_64x64_400steps"] * 2,
        "provenance": provenance(),
    }
    write_json(f"{label}.json", payload)
    return payload


def case_report() -> dict:
    """Evaluate C1..C7 from the on-disk artefacts and write report.json."""
    def load(name):
        p = OUT / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    pois = load("poiseuille")
    cyl = load("cylinder")
    neg = load("negative_control")
    gal = load("galilean")
    rep = load("reproducibility")
    results = {
        "poiseuille": (pois or {}).get("results"),
        "cylinder": (cyl or {}).get("results"),
        "negative_control": (neg or {}).get("results"),
        "galilean": (gal or {}).get("results"),
        "reproducibility": (rep or {}).get("hashes"),
        "provenance": provenance({"artefacts": sorted(
            p.name for p in OUT.glob("*.json"))}),
    }
    report = evaluate_criteria(results)
    report["artefact_sha256"] = {
        p.name: (OUT / (p.name + ".sha256")).read_text().strip()
        for p in sorted(OUT.glob("*.json"))
        if (OUT / (p.name + ".sha256")).exists()
    }
    write_json("report.json", report)
    print(json.dumps({"n_pass": report["n_pass"], "n_total": report["n_total"],
                      "all_pass": report["all_pass"],
                      "criteria": [(c["id"], c.get("pass")) for c in report["criteria"]]},
                     indent=2))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="all",
                    choices=["poiseuille", "cylinder", "negative", "galilean",
                             "repro", "report", "all"])
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true",
                    help="reduced grids/steps for a fast smoke run")
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.case in ("poiseuille", "all"):
        case_poiseuille(args.workers,
                        widths=(8, 12, 16) if args.quick else WIDTHS)
    if args.case in ("cylinder", "all"):
        case_cylinder(args.workers,
                      transient_d=10.0 if args.quick else 30.0,
                      sample_d=6.0 if args.quick else 18.0)
    if args.case in ("negative", "all"):
        case_negative(args.workers)
    if args.case in ("galilean", "all"):
        case_galilean(args.workers,
                      steps=3000 if args.quick else 12000,
                      periods=(1, 2, 4) if args.quick else (1, 2, 3, 4, 5, 6, 8, 10))
    if args.case in ("repro", "all"):
        case_repro()
    if args.case in ("report", "all"):
        case_report()
    print(f"[done] {args.case} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())