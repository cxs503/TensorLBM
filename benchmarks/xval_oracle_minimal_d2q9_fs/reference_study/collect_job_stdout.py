#!/usr/bin/env python3
"""Recover the per-run R3 artifacts from dispatched job logs.

Why: dispatched jobs run in a node-side checkout and ``fetch`` returns the
*staged input* tree, not the files the job wrote, so the only dependable return
channel is stdout (``/srv/tensor/agent/cluster/jobs/cjN.log``, tail-truncated).
``resolution_study.py --run`` therefore prints, besides the per-run maximum
line ``R2RUN:``, one compact ``R2SER:`` line per run carrying the **full**
T-series ([T, Z_cell, Z_cell0, Z_sub.02, Z_sub.05, Z_sub.10]).  This helper
turns those lines back into the same per-run JSON files a local run would have
written (``artifacts/runs_r3/<label>.json``), so ``--analyze`` is
source-agnostic and sharded jobs compose.

Usage:
    python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/collect_job_stdout.py cj3120 [cj3121 ...]
    python benchmarks/xval_oracle_minimal_d2q9_fs/reference_study/resolution_study.py --analyze
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
RUNS = ART / "runs_r3"
LOG_DIR = Path("/srv/tensor/agent/cluster/jobs")
sys.path.insert(0, str(HERE))
import resolution_study as _R  # noqa: E402


def _log_text(jid: str) -> str:
    log = LOG_DIR / f"{jid}.log"
    if not log.exists():
        raise SystemExit(f"missing log {log}")
    text = log.read_text(errors="ignore")
    # the cluster wraps job stdout as a JSON-string field; prefer that blob
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        blob = rec.get("output_tail") or rec.get("output") or ""
        if blob:
            return blob
    return text


def parse_job(jid: str) -> dict:
    """Return {"head": {...}, "runs": {label: run_dict}, "complete": bool}."""
    head, runs, complete = {}, {}, False
    for line in _log_text(jid).splitlines():
        if line.startswith("R2HEAD:"):
            head = json.loads(line[len("R2HEAD:") :])
        elif line.startswith("R2SER:"):
            rec = json.loads(line[len("R2SER:") :])
            lab = rec["l"]
            run = {
                "label": lab,
                "arm": rec.get("k", "arm"),
                "kind": rec.get("k", "arm"),
                "a": rec.get("a"),
                "tau": rec.get("tau"),
                "status": rec.get("st") or "ok",
                "steps": rec.get("steps"),
                "wall_seconds": rec.get("wall"),
                "u_max": rec.get("u"),
                "recovered_from_job": jid,
            }
            if rec.get("S"):
                run["series"] = _R.series_from_rows(rec["S"])
            if rec.get("inv") is not None:
                # only the maximum drift survives in the compact line; keep the
                # scalar so eval_run can still judge the ledger
                run["inventory"] = [{"step": None, "drift_pct": rec["inv"]}]
            # controls report u_max_lattice under a separate key
            if rec.get("k") == "control":
                run["kind"] = "control"
                run["u_max_lattice"] = rec.get("u")
            runs[lab] = run
        elif line.startswith("R2END:"):
            complete = True
    return {"head": head, "runs": runs, "complete": complete}


def main() -> int:
    jids = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not jids:
        raise SystemExit(__doc__)
    RUNS.mkdir(parents=True, exist_ok=True)
    total, merged = 0, {}
    for jid in jids:
        got = parse_job(jid)
        if not got["runs"]:
            print(f"[collect] {jid}: no R2SER payload (queued/crashed?)", flush=True)
            continue
        for lab, run in got["runs"].items():
            (RUNS / f"{lab}.json").write_text(json.dumps(run, indent=1))
            merged[lab] = jid
            total += 1
        print(f"[collect] {jid}: {len(got['runs'])} runs, complete={got['complete']}", flush=True)
    if not total:
        raise SystemExit("nothing recovered")
    (ART / "r3_recovery.json").write_text(
        json.dumps({"source_jobs": jids, "runs": sorted(merged), "per_job": merged}, indent=1)
    )
    print(f"[collect] {total} per-run artifacts -> artifacts/runs_r3/*.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
