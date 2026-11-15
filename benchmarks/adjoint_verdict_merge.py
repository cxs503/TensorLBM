#!/usr/bin/env python3
"""Merge sharded GAP-DIFFLBM-ADJOINT-VERDICT campaign outputs into one artifact.

The paper-resolution campaign (``PAPER_CONFIGS``: two resolutions x two
parameter points) is long enough that it is run as several cluster jobs, each
producing its own JSON with ``--only <configs>``.  This tool assembles the
shards into the single canonical artifact that
``benchmarks/adjoint_verdict_report.py`` consumes, **re-using the benchmark's
own** :func:`summarise` so a sharded campaign is judged by exactly the same code
as a serial run (no second implementation of the verdict logic)::

    PYTHONPATH=src python benchmarks/adjoint_verdict_merge.py \
        --out artifacts/adjoint_verdict_diff_lbm.json \
        artifacts/adjoint_verdict_diff_lbm_R1.json \
        artifacts/adjoint_verdict_diff_lbm_R2.json

Consistency guards (a merge that silently mixes incomparable runs would be
worse than no merge): every shard must carry the same ``schema``, ``mode`` and
``predeclared`` block, and configs must be disjoint.  Field snapshots
(``fields_<config>.npz``) from the per-shard artefact directories are copied
next to the merged JSON so the report generator finds them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adjoint_verdict_diff_lbm import summarise  # noqa: E402


def load_shards(paths: list[str]) -> dict:
    shards = []
    for path in paths:
        with open(path) as fh:
            shards.append((path, json.load(fh)))
    if not shards:
        raise SystemExit("merge: no shard JSONs given")

    ref_path, ref = shards[0]
    merged = {
        "anchor": ref["anchor"],
        "schema": ref["schema"],
        "mode": ref["mode"],
        "generated_at": ref["generated_at"],
        "environment": ref["environment"],
        "predeclared": ref["predeclared"],
        "configs": {},
        "shards": [],
    }
    for path, payload in shards:
        for key in ("schema", "mode", "predeclared"):
            if payload.get(key) != ref.get(key):
                raise SystemExit(
                    f"merge: shard {path} disagrees with {ref_path} on '{key}' "
                    f"({payload.get(key)!r} != {ref.get(key)!r}); refusing to mix runs"
                )
        overlap = set(payload["configs"]) & set(merged["configs"])
        if overlap:
            raise SystemExit(f"merge: duplicate configs {sorted(overlap)} (from {path})")
        merged["configs"].update(payload["configs"])
        merged["shards"].append(
            {
                "path": path,
                "configs": sorted(payload["configs"]),
                "generated_at": payload.get("generated_at"),
                "wall_seconds": payload.get("wall_seconds"),
            }
        )
        merged["generated_at"] = max(merged["generated_at"], payload.get("generated_at") or "")
    merged["wall_seconds"] = sum(float(p.get("wall_seconds") or 0.0) for _, p in shards)
    summarise(merged, ad_tol=0.05)
    return merged


def collect_fields(merged: dict, out_path: str) -> list[str]:
    """Copy per-shard ``fields_<config>.npz`` next to the merged JSON."""
    out_dir = os.path.dirname(out_path) or "."
    needle_dir = os.path.join(out_dir, os.path.basename(out_path)[:-5])
    os.makedirs(needle_dir, exist_ok=True)
    copied = []
    for name in merged["configs"]:
        for npz in glob.glob(f"artifacts/**/fields_{name}.npz", recursive=True) + glob.glob(
            f"{out_dir}/**/fields_{name}.npz", recursive=True
        ):
            dst = os.path.join(needle_dir, os.path.basename(npz))
            if os.path.abspath(npz) != os.path.abspath(dst):
                shutil.copy2(npz, dst)
            copied.append(dst)
            break
    return copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", default="artifacts/adjoint_verdict_diff_lbm.json")
    ap.add_argument("--no-fields", action="store_true")
    args = ap.parse_args()

    merged = load_shards(args.shards)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(json.dumps(merged["verdict"], indent=2))
    if not args.no_fields:
        for path in collect_fields(merged, args.out):
            print("fields:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())