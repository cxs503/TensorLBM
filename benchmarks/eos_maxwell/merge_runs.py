#!/usr/bin/env python
"""Merge split-job campaign artefacts into one ``results.json`` (provenance kept).

The campaign is deliberately split across cluster jobs (the core arms take ~1 h,
the lever/control arms ~25 min), and each job runs in its own checkout, so each
job writes its own ``results.json``.  This merger is the only supported way to
combine them:

* it refuses to merge artefacts whose ``prereg`` block, ``eos`` or code hashes
  differ (a merged table must describe one code state and one predeclaration);
* it refuses duplicate tags (arms are append-only, one run == one record);
* it keeps the per-job provenance (job wall time, generated stamp, code hashes,
  which arms came from which artefact) instead of pretending one job made all of
  it, and re-derives the verdict block with the *driver's own* rule
  (``run._verdict``) so the merged verdict cannot drift from the code.

Usage::

    python merge_runs.py --out out/results.json stage_a/results.json stage_b/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run as R  # noqa: E402

ARMS = ("tiers", "controls", "boundary", "widths", "longhold", "schemes", "repeats", "threads", "robust", "tau")
IDENTITY_KEYS = ("benchmark", "eos", "prereg")


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _mw_core(mw: dict) -> dict:
    """Maxwell construction without the volatile timing field.

    Each job times its own bisection, so ``wall_s`` differs between checkouts
    while the construction itself is deterministic; comparing the timing would
    make every merge impossible (and it is not physics).
    """
    return {k: v for k, v in mw.items() if k != "wall_s"}


def merge(paths: list[Path]) -> dict:
    docs = [(p, _load(p)) for p in paths]
    p0, base = docs[0]
    for p, d in docs[1:]:
        for k in IDENTITY_KEYS:
            if d.get(k) != base.get(k):
                raise SystemExit(f"refusing to merge {p}: {k} differs from {p0}")
        if d.get("environment", {}).get("probe_lib_sha256") != base["environment"].get("probe_lib_sha256"):
            raise SystemExit(
                f"refusing to merge {p}: probe_lib sha "
                f"{d['environment'].get('probe_lib_sha256')} != "
                f"{base['environment'].get('probe_lib_sha256')} (different library states)"
            )
        if d.get("driver_sha256") != base.get("driver_sha256"):
            raise SystemExit(f"refusing to merge {p}: driver sha differs (different driver revision)")
        if {k: _mw_core(v) for k, v in d["maxwell_tiers"].items()} != {
            k: _mw_core(v) for k, v in base["maxwell_tiers"].items()
        }:
            raise SystemExit(f"refusing to merge {p}: Maxwell construction differs from the other artefacts")

    out = {k: base.get(k) for k in IDENTITY_KEYS}
    out["environment"] = base["environment"]
    out["driver_sha256"] = base["driver_sha256"]
    out["maxwell_tiers"] = base["maxwell_tiers"]
    sources = []
    for p, d in docs:
        counts = {}
        for arm in ARMS:
            rows = d.get(arm, [])
            counts[arm] = len(rows)
            out.setdefault(arm, [])
            known = {r["tag"] for r in out[arm]}
            for r in rows:
                if r["tag"] in known:
                    raise SystemExit(f"refusing to merge {p}: duplicate tag {r['tag']}")
                out[arm].append(r)
        sources.append(
            dict(
                artefact=str(p),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                generated_utc=d.get("generated_utc"),
                wall_s_total=d.get("wall_s_total"),
                arms={k: v for k, v in counts.items() if v},
                machine=d.get("environment", {}).get("machine"),
                threads=d.get("environment", {}).get("threads"),
            )
        )
    out["verdict"] = R._verdict(out)
    out["wall_s_total"] = round(sum(float(s["wall_s_total"] or 0.0) for s in sources), 1)
    out["generated_utc"] = datetime.now(timezone.utc).isoformat()
    out["provenance"] = {
        "merged_utc": out["generated_utc"],
        "merge_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "note": (
            "each source artefact is one driver process (the campaign is split so that a "
            "small box can run it in parallel); per-run code hashes are kept in every run "
            "record under 'code', so a run can be traced to the exact library state.  The "
            "stage artefacts themselves are not archived: 'sha256' pins the exact bytes "
            "that were merged, and the merged records below are reproduced verbatim."
        ),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=str(_HERE / "out" / "results.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    merged = merge([Path(x) for x in args.inputs])
    print(json.dumps({k: (len(v) if isinstance(v, list) else "…") for k, v in merged.items()}, indent=1))
    if args.dry_run:
        return 0
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())