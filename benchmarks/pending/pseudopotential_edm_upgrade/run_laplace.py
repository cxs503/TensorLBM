"""Static-droplet Laplace law + spurious-current benchmark for the SC force formats.

One run = one (force format, initial radius) pair; each run writes its own JSON
under ``runs/`` (so a campaign is resumable and partial results survive), then
the driver fits the 2-D Young–Laplace law ``Δp = σ/R_eq`` and reports

* per-radius surface tension σ_i = Δp·R_eq and its deviation from the fit
  (the Laplace-law criterion, target ≤ 3 %),
* fit quality R² and the intercept-over-slope test,
* the spurious-current floor ``max|u|`` (target ≤ 1e-4), whole domain and
  interface band,
* mass drift and boundedness (stability).

Usage:
    python run_laplace.py --radii 16,20,24 --L 128 --jobs 6 --threads 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FORMATS, DropletSetup, laplace_fit, run_droplet, save_json  # noqa: E402

HERE = Path(__file__).resolve().parent

#: force format ladder. Label -> (stencil, form, forcing scheme)
LADDER = {
    "F0_sc_classic_vshift": ("d2q9", "product", "velocity_shift"),
    "F1_sc_classic_edmforce": ("d2q9", "product", "kupershtokh"),
    "F2_edm_d2q9": ("d2q9", "difference", "kupershtokh"),
    "F3_edm_iso6": ("iso6", "difference", "kupershtokh"),
    "F4_pt_iso6": ("iso6", "psi2", "kupershtokh"),
}


def _run_path(fmt: str, R: float, L: int) -> Path:
    return HERE / "runs" / f"laplace_{fmt}_L{L}_R{R:g}.json"


def _task(job: dict) -> dict:
    setup = DropletSetup(**job["setup"])
    t0 = time.time()
    try:
        res = run_droplet(
            setup, fmt=job["fmt"], forcing=job["forcing"],
            max_steps=job["max_steps"], check_every=job.get("check_every", 200),
            steady_tol=job.get("steady_tol", 2e-4), steady_window=job.get("steady_window", 3),
            thread_count=job.get("threads", 4),
        )
        res["wall_s"] = time.time() - t0
    except Exception:                                     # pragma: no cover
        res = {"format": job["fmt"], "setup": job["setup"], "error": traceback.format_exc(),
               "wall_s": time.time() - t0}
    save_json(_run_path(job["fmt"], setup.R, setup.L), res)
    return {"ok": "error" not in res, "path": str(_run_path(job["fmt"], setup.R, setup.L))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--radii", default="16,20,24")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--min-steps", type=int, default=4000)
    ap.add_argument("--formats", default=",".join(LADDER))
    ap.add_argument("--out", default="result_laplace.json")
    args = ap.parse_args()

    radii = [float(a) for a in args.radii.split(",") if a]
    fmts = [f for f in args.formats.split(",") if f]

    jobs = []
    for fmt in fmts:
        for R in radii:
            path = _run_path(fmt, R, args.L)
            if path.exists():
                print(f"[driver] skip (done) {path.name}")
                continue
            jobs.append({
                "fmt": fmt, "forcing": LADDER[fmt][2],
                "setup": asdict(DropletSetup(L=args.L, R=R, tau=args.tau)),
                "max_steps": args.max_steps, "threads": args.threads,
            })

    t0 = time.time()
    if jobs:
        ctx = mp.get_context("spawn")   # forking after torch init can deadlock OpenMP
        with ctx.Pool(args.jobs) as pool:
            for i, out in enumerate(pool.imap_unordered(_task, jobs), 1):
                print(f"[driver] {i}/{len(jobs)} {out['path']} ok={out['ok']}", flush=True)
    print(f"[driver] {len(jobs)} runs in {time.time() - t0:.1f} s", flush=True)

    results = []
    for fmt in fmts:
        for R in radii:
            p = _run_path(fmt, R, args.L)
            if p.exists():
                results.append(json.loads(p.read_text()))
    bad = [r for r in results if "error" in r]
    if bad:
        print(f"[driver] WARNING {len(bad)} failed runs")

    fits = {}
    for fmt in fmts:
        rs = [r for r in results if r.get("format") == fmt and "error" not in r]
        if len(rs) < 2:
            continue
        rs.sort(key=lambda r: r["setup"]["R"])
        fits[fmt] = laplace_fit(rs)
        fits[fmt]["min_steps_used"] = args.min_steps
        fits[fmt]["spurious_u_max"] = float(np.mean([r["u_max"] for r in rs]))
        fits[fmt]["spurious_u_max_iface"] = float(np.nanmean([r["u_max_iface"] for r in rs]))
        fits[fmt]["mass_drift_rel"] = float(np.max([r["mass_drift_rel"] for r in rs]))
        fits[fmt]["all_converged"] = all(r["converged"] for r in rs)
        fits[fmt]["all_finite"] = all(r["finite"] for r in rs)
        fits[fmt]["steps"] = [r["steps"] for r in rs]

    payload = {
        "config": {"L": args.L, "radii": radii, "tau": args.tau,
                   "max_steps": args.max_steps, "min_steps": args.min_steps},
        "ladder": {k: list(v) for k, v in LADDER.items()},
        "fits": fits,
        "runs": [{k: v for k, v in r.items() if k != "history"} for r in results
                 if "error" not in r],
        "histories": {f"{r['format']}_R{r['setup']['R']:g}": r["history"] for r in results
                      if "error" not in r},
        "errors": {f"{r['format']}_R{r['setup']['R']:g}": r["error"] for r in bad},
    }
    save_json(HERE / args.out, payload)

    print(f"\n{'format':26s} {'sigma_fit':>10s} {'R2':>9s} {'max_dev%':>9s} "
          f"{'u_max':>10s} {'u_iface':>10s} {'mass':>9s}")
    for fmt, f in fits.items():
        print(f"{fmt:26s} {f['sigma_fit']:10.5f} {f['r2']:9.6f} {f['max_sigma_dev_pct']:9.3f} "
              f"{f['spurious_u_max']:10.3e} {f['spurious_u_max_iface']:10.3e} "
              f"{f['mass_drift_rel']:9.2e}")
    print("\nper-radius σ_i deviation (%):")
    for fmt, f in fits.items():
        print(f"  {fmt:26s} " + "  ".join(
            f"R={p['R_init']:g}: σ={p['sigma_i']:.5f} ({p['dev_pct']:+.2f}%)"
            for p in f["per_radius"]))


if __name__ == "__main__":
    main()