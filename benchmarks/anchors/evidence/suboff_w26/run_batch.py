#!/usr/bin/env python3
"""W26 / mission m17569 — batch driver for the SUBOFF case matrix (local, bounded concurrency).

Runs ``run_case.py`` for a declared list of (N, tau) / (N, Re) cases with a fixed
number of concurrent processes and a fixed thread budget per process, so the
compute footprint on the shared 99 control box stays bounded and auditable.  The
case list is written before the run (preregistration) and every case record lands
in its own JSON (the raw time series in its own npz), so a partially finished
batch is still fully evidential.

Usage:
    python3 run_batch.py --plan plan.json --out <abs dir> --concurrency 3 --threads 6
plan.json = {"cases": [{"tag": "...", "N": 48, "tau": 0.5072 | "re_lattice": 1000,
                        "steps": 5760, "snapshot": false}, ...]}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True, help="absolute output root (inside the repo)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--mission", default="m17569")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    cases = plan["cases"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "_logs"
    log_root.mkdir(exist_ok=True)

    manifest = {
        "schema": "w26_batch_manifest/v1",
        "mission": args.mission,
        "plan_path": args.plan,
        "plan_sha256": __import__("hashlib").sha256(Path(args.plan).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "concurrency": args.concurrency,
        "threads_per_case": args.threads,
        "cases": cases,
        "results": [],
    }

    running: list[tuple[dict, subprocess.Popen, Path]] = []
    queue = list(cases)
    t0 = time.time()

    while queue or running:
        while queue and len(running) < args.concurrency:
            case = queue.pop(0)
            tag = case["tag"]
            cmd = [sys.executable, str(_HERE / "run_case.py"),
                   "--resolution", str(case["N"]), "--steps", str(case["steps"]),
                   "--threads", str(args.threads),
                   "--out", str(out_root / tag), "--tag", tag,
                   "--mission", args.mission]
            if "re_lattice" in case and case["re_lattice"] is not None:
                cmd += ["--re-lattice", str(case["re_lattice"])]
            else:
                cmd += ["--tau", repr(case["tau"])]
            if case.get("snapshot"):
                cmd.append("--snapshot")
            logp = log_root / f"{tag}.log"
            fh = open(logp, "w")
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                       MKL_NUM_THREADS=str(args.threads))
            t_start = time.time()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((dict(case, t_start=t_start, log=str(logp)), proc, logp))
            print(f"[launch] {tag} (queue left {len(queue)})", flush=True)

        time.sleep(5)
        still = []
        for case, proc, logp in running:
            rc = proc.poll()
            if rc is None:
                still.append((case, proc, logp))
                continue
            rec = {"tag": case["tag"], "N": case["N"], "rc": rc,
                   "wall_s": time.time() - case["t_start"], "log": str(logp)}
            cj = out_root / case["tag"] / f"case_{case['tag']}.json"
            rec["case_json"] = str(cj) if cj.exists() else None
            if cj.exists():
                d = json.loads(cj.read_text())
                rec.update({"C_T": d["C_T"], "C_F": d["C_F"], "tau": d["tau_achieved"],
                            "Re_lattice": d["Re_lattice_measured"], "cells": d["cells"],
                            "steps": d["steps_done"], "finite": d["field_finite_at_end"],
                            "drift": d["C_T_window_drift"]})
            manifest["results"].append(rec)
            print(f"[done] {case['tag']} rc={rc} {rec.get('C_T')} ({rec['wall_s']:.0f}s)", flush=True)
        running = still

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["wall_s"] = time.time() - t0
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[batch] {len(manifest['results'])} cases in {manifest['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()