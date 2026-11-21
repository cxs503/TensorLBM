"""Command line entry for the Re = 3200 cavity reproduction.

Exit codes (so a job script can tell "ran" from "passed"):

``0``  every requested grid produced a finite solution (a scored comparison may
       still fail its tolerance -- that is reported, not hidden)
``2``  at least one grid diverged, produced non-finite fields or hit a config
       error

A manifest (``run_manifest.json``) is always written next to the per-grid
records: it carries the exact ``repro_cmd`` per grid, the environment, the
reference hash and the sha256 of every artefact, so the numbers can be traced
back to one invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as _platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import reference as refmod
from . import runner


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd or Path.cwd()),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tensorlbm.verification.cavity_re3200",
        description="Re = 3200 lid-driven cavity vs Ghia (1982), anchor REPRO-LBM-HOU1995")
    ap.add_argument("--n", type=int, nargs="+", default=[128])
    ap.add_argument("--re", type=float, default=runner.RE)
    ap.add_argument("--u-lid", type=float, default=runner.U_LID)
    ap.add_argument("--steps", type=int, default=2000, help="maximum lattice steps")
    ap.add_argument("--monitor-every", type=int, default=500,
                    help="residual/error monitor interval (0 disables monitoring)")
    ap.add_argument("--resid-tol", type=float, default=None,
                    help="stop when max|du| over one interval drops below this; "
                         "without it the step count is a truncation, not a steady state")
    ap.add_argument("--ramp-steps", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-center-average", action="store_true")
    ap.add_argument("--coord-convention", choices=("wall", "node"), default="wall")
    ap.add_argument("--ref-json", default=None,
                    help="reference table (default: the admitted Ghia Re=3200 column)")
    ap.add_argument("--no-ref", action="store_true",
                    help="run without a reference -> comparison WITHHELD")
    ap.add_argument("--init-from", default=None,
                    help="npz of a coarse ux/uy field, prolongated as initial condition")
    ap.add_argument("--save-fields", action="store_true",
                    help="save the final velocity field as <out-dir>/re3200_n<n>_fields.npz")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    here = Path(__file__).resolve()
    default_out = here.parents[5] / "benchmarks/verified/cavity/re3200/out" \
        if len(here.parents) > 5 else Path.cwd() / "out"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    ref: Optional[Dict[str, Any]] = None
    if not args.no_ref:
        try:
            ref = refmod.load_reference(args.ref_json)
        except refmod.ReferenceError as exc:
            print(f"[fatal] reference unusable: {exc}", file=sys.stderr)
            return 2

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records: List[Dict[str, Any]] = []
    failed: List[Any] = []
    for n in args.n:
        record = runner.run_grid(
            n,
            re=args.re,
            u_lid=args.u_lid,
            max_steps=args.steps,
            monitor_every=args.monitor_every,
            resid_tol=args.resid_tol,
            ramp_steps=args.ramp_steps,
            threads=args.threads,
            center_average=not args.no_center_average,
            coord_convention=args.coord_convention,
            ref=ref,
            init_from=args.init_from,
            save_fields=(out_dir / f"re3200_n{n}_fields.npz") if args.save_fields else None,
            verbose=args.verbose,
        )
        # the exact command that produced this record, replayable verbatim
        replay = [sys.executable, "-m", "tensorlbm.verification.cavity_re3200",
                  "--n", str(n), "--re", f"{args.re:g}", "--u-lid", f"{args.u_lid:g}",
                  "--steps", str(args.steps),
                  "--monitor-every", str(args.monitor_every),
                  "--ramp-steps", str(args.ramp_steps),
                  "--coord-convention", args.coord_convention,
                  "--out-dir", str(out_dir)]
        if args.resid_tol is not None:
            replay += ["--resid-tol", f"{args.resid_tol:g}"]
        if args.init_from:
            replay += ["--init-from", str(args.init_from)]
        if args.no_center_average:
            replay.append("--no-center-average")
        if args.no_ref:
            replay.append("--no-ref")
        record["repro_cmd"] = " ".join(replay)
        path = runner.write_record(record, out_dir)
        record["_path"] = str(path)
        print(f"[ok] n={n} status={record['run_status']} steps={record['steps']} "
              f"-> {path}", flush=True)
        cmp_ = record.get("comparison") or {}
        if "tol_pct" in cmp_:
            print(f"     max|err| wall={cmp_['wall']['max_abs_err_pct']:.2f}% "
                  f"node={cmp_['node']['max_abs_err_pct']:.2f}% "
                  f"(tol {cmp_['tol_pct']:.1f}%) pass={cmp_['wall']['pass']}",
                  flush=True)
        elif cmp_.get("status"):
            print(f"     comparison {cmp_['status']}: {cmp_.get('reason', '')}", flush=True)
        if record["run_status"] not in ("ok", "not_converged"):
            failed.append(n)
        records.append(record)

    manifest = {
        "case": runner.CASE,
        "anchor": runner.ANCHOR,
        "description": runner.DESCRIPTION,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv[1:],
        "out_dir": str(out_dir),
        "git_sha": _git_sha(),
        "environment": {
            "python": sys.version.split()[0],
            "host": _platform.node(),
            "machine": _platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch": records[0]["environment"].get("torch") if records else None,
            "torch_threads": records[0]["environment"].get("torch_threads") if records else None,
        },
        "reference": None if ref is None else {
            "file": ref.get("_path"),
            "sha256": ref.get("_sha256_content"),
            "file_sha256": ref.get("_sha256_file"),
            "hash_verified": ref.get("_hash_verified"),
            "source": ref.get("source"), "table": ref.get("table"),
        },
        "tolerance_pct": runner.TOL_PCT,
        "grids": [
            {
                "n": r["n"],
                "run_status": r["run_status"],
                "steps": r["steps"],
                "converged": r["converged"],
                "tau": r.get("tau"),
                "final_resid": r.get("final_resid"),
                "elapsed_s": r.get("elapsed_s"),
                "repro_cmd": r.get("repro_cmd"),
                "file": Path(r["_path"]).name,
                "sha256": hashlib.sha256(Path(r["_path"]).read_bytes()).hexdigest(),
                "max_abs_err_pct": (None if "tol_pct" not in (r.get("comparison") or {})
                                    else r["comparison"][r["coord_convention"]]["max_abs_err_pct"]),
                "pass": (None if "tol_pct" not in (r.get("comparison") or {})
                         else r["comparison"][r["coord_convention"]]["pass"]),
            }
            for r in records
        ],
        "failed_grids": failed,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1),
                                               encoding="utf-8")
    print(f"[ok] manifest -> {out_dir / 'run_manifest.json'}", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())