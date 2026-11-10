"""Merge raw campaign files into one record set, keeping provenance.

The campaign was executed in two groups (a drag ladder, then the shear sweep)
because the shear sweep had to be re-designed once the measurement noise floor
was measured (AMENDMENT-3/4 in ``P46W3_N03_PREDECLARED_DESIGN.md``).  Merging is
a pure bookkeeping step: no value is modified, and every input file is recorded
in the output's ``provenance`` block so a reader can retrace which run produced
which record.

Usage::

    PYTHONPATH=src python benchmarks/suspension/merge_raw.py \
        --in benchmarks/suspension/artifacts/raw_cases.json \
             benchmarks/suspension/artifacts/raw_shear_am4.json \
        --out benchmarks/suspension/artifacts/raw_merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def merge(paths: list[str]) -> dict:
    cases, designs, prov = [], [], []
    for path in paths:
        with open(path) as fh:
            raw = json.load(fh)
        n_shear = sum(1 for c in raw.get("cases", []) if c.get("mode") == "shear")
        prov.append({
            "file": os.path.relpath(os.path.abspath(path), REPO),
            "n_cases": len(raw.get("cases", [])),
            "n_shear_cases": n_shear,
            "elapsed_s": raw.get("elapsed_s"),
        })
        # the *shear* file carries the design that governs the sweep, so it wins
        # when the files disagree (the drag group may have been run separately)
        designs.append((n_shear, raw.get("design", {})))
        for rec in raw.get("cases", []):
            rec = dict(rec)
            rec["source_file"] = os.path.basename(path)
            cases.append(rec)
    tags = [c.get("tag") for c in cases]
    if len(tags) != len(set(tags)):
        dup = sorted({t for t in tags if tags.count(t) > 1})
        raise SystemExit(f"duplicate tags across inputs: {dup}")
    modes = sorted({c.get("mode") for c in cases})
    design = {}
    ordered = [d for _, d in sorted(designs, key=lambda t: -t[0])]
    for key in ("cell", "steps", "dilute_phi", "dense_phi", "seeds"):
        for d in ordered:
            if key in d:
                design[key] = d[key]
                break
    radii = sorted({float(c["radius"]) for c in cases if "radius" in c})
    design["radii"] = radii
    design["dense_radius"] = next(
        (d.get("dense_radius") for d in ordered if d.get("dense_radius")), None)
    return {"design": design, "modes": modes, "provenance": prov, "cases": cases}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "raw_merged.json"))
    args = ap.parse_args(argv)
    out = merge(args.inputs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"{len(out['cases'])} records ({', '.join(out['modes'])}) -> {args.out}")
    errors = [c["tag"] for c in out["cases"] if "error" in c]
    if errors:
        print(f"failed cases: {errors}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())