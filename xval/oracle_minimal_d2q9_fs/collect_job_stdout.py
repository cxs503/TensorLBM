#!/usr/bin/env python3
"""Collect the R2 payload from a dispatched job's stdout log and analyse it.

Why: dispatched jobs run in a node-side checkout; ``fetch`` returns the *staged
input* tree, not the files the job wrote.  ``resolution_study.py --run`` therefore
also prints a one-line compact payload (``R2JSON:{...}``) which this helper
scrapes out of ``/srv/tensor/agent/cluster/jobs/cjN.log`` and turns back into
``artifacts/resolution_ladder.json`` -> ``--analyze`` -> figures.

Usage:  python xval/oracle_minimal_d2q9_fs/collect_job_stdout.py cj3062
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
LOG_DIR = Path("/srv/tensor/agent/cluster/jobs")
sys.path.insert(0, str(HERE))
import resolution_study as _R  # noqa: E402

R_DESIGN = _R.DESIGN


def find_payload(jid: str) -> dict:
    log = LOG_DIR / f"{jid}.log"
    text = log.read_text()
    # the cluster logs the job stdout inline as a JSON string field "output_tail"
    blob = None
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
            break
    if blob is None:
        blob = text
    head, runs, done = {}, [], False
    for line in blob.splitlines():
        if line.startswith("R2HEAD:"):
            head = json.loads(line[len("R2HEAD:"):])
        elif line.startswith("R2RUN:"):
            runs.append(json.loads(line[len("R2RUN:"):]))
        elif line.startswith("R2END:"):
            done = True
    if not runs:
        raise SystemExit(f"no R2RUN payload in {log} (job may still be running or crashed)")
    for r in runs:  # controls carry no tau/series
        if "series" not in r:
            r["kind"] = "control"
    return {"design": R_DESIGN, "env": head.get("env", {}),
            "wall_seconds": head.get("wall_seconds"), "runs": runs, "complete": done}


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    jid = sys.argv[1]
    payload = find_payload(jid)
    payload["source_job"] = jid
    (ART / "resolution_ladder.json").write_text(json.dumps(payload, indent=1))
    print(f"[collect] {len(payload['runs'])} runs -> artifacts/resolution_ladder.json")
    import resolution_study as R
    analysis = R.analyze(payload)
    analysis["source_job"] = jid
    (ART / "resolution_analysis.json").write_text(json.dumps(analysis, indent=1))
    (ART / "resolution_numbers.md").write_text(R.render_numbers_md(analysis, payload))
    for ref in ("mm", "ritter"):
        for k, f in analysis["fits"][ref].items():
            if f.get("ok"):
                print(f"  fit[{ref}] {k}: p={f['p']} E_inf={f['E_inf_pct']}% "
                      f"monotone={f['monotone_falling']} H1={f['hypothesis_H1_supported']}")
            else:
                print(f"  fit[{ref}] {k}: FAILED {f.get('reason')}")
        for k, r in analysis["richardson"][ref].items():
            if r.get("ok"):
                print(f"  rich[{ref}] {k}: Z_inf max {r['Z_inf_max_abs_rel_err_pct']}% "
                      f"p_obs {r['p_obs_median']} pass3%={r['pass_3pct']}")
    for c in analysis["controls"]:
        print(f"  control {c.get('arm')}: u_max={c.get('u_max_lattice')} pass={c.get('pass')}")
    print(f"[collect] wrote artifacts/resolution_analysis.json + resolution_numbers.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())