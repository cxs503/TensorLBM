#!/usr/bin/env python3
"""Diff the evidence payload of two ``tensorlbm_module_maturity`` SQL dumps.

Motivation (m15617 / PR#1995 round-3): the audit deliverable existed as two
``docs/`` copies generated from one string.  A schema-only pytest guard stayed
green while the older copy carried evidence from a previous run (27/45 rows,
e.g. ``status_row`` pointing at a superseded ``benchmarks/STATUS.md`` line), so
the drift survived into main.  This tool answers "do these two copies still say
the same thing?" at the *value* level, per module and per field, and emits a
machine-readable report plus a figure for the mission write-up.

Usage::

    python tools/audit/docs_sql_drift.py --old /tmp/a.sql --new docs/tensorlbm_maturity_m15912.sql \\
        --out artifacts/m18321_round3

``--old`` may be a git blob spec via ``--old-git HEAD:path``::

    python tools/audit/docs_sql_drift.py --old-git HEAD:docs/tensorlbm_maturity_m15912.sql \\
        --new docs/tensorlbm_maturity_m15912.sql --out artifacts/m18321_round3

Exit code 0 when the two payloads are identical, 1 when drift was found (so the
tool doubles as a CI check).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

TABLE = "tensorlbm_module_maturity"
QUOTED = re.compile(r"'((?:[^']|'')*)'")


def evidence_rows(text: str) -> dict:
    """``{module_path: evidence dict}`` parsed out of the INSERT statements."""
    out = {}
    for line in text.splitlines():
        if not line.startswith(f"INSERT INTO {TABLE} "):
            continue
        fields = [f.replace("''", "'") for f in QUOTED.findall(line)]
        if len(fields) < 4:
            raise ValueError(f"unparsable INSERT: {line[:120]}")
        out[fields[0]] = json.loads(fields[3])
    return out


def diff(old: dict, new: dict) -> dict:
    per_field, per_module = {}, {}
    for key in sorted(old):
        a, b = old[key], new.get(key) or {}
        changed = [f for f in a if a[f] != b.get(f)]
        changed += [f for f in b if f not in a]
        if changed:
            per_module[key] = {"fields": sorted(changed),
                               "old": {f: a.get(f) for f in changed},
                               "new": {f: b.get(f) for f in changed}}
            for f in changed:
                per_field[f] = per_field.get(f, 0) + 1
    return {"n_modules_old": len(old), "n_modules_new": len(new),
            "n_modules_drift": len(per_module),
            "field_counts": dict(sorted(per_field.items(), key=lambda kv: -kv[1])),
            "modules": per_module,
            "identical": bool(old) and old == new}


def figure(rep: dict, path: str) -> str:
    """Two-panel drift figure: per-module field drift and field histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    mods = list(rep["modules"])
    if mods:
        counts = [len(rep["modules"][m]["fields"]) for m in mods]
        ax1.barh(range(len(mods)), counts, color="#c33")
        ax1.set_yticks(range(len(mods)))
        ax1.set_yticklabels(mods, fontsize=4.5)
        ax1.invert_yaxis()
        ax1.set_xlabel("stale fields in this module")
        ax1.set_title(f"evidence drift: {rep['n_modules_drift']}/{rep['n_modules_old']} modules", fontsize=9)
    fields = list(rep["field_counts"])
    if fields:
        ax2.bar(fields, [rep["field_counts"][f] for f in fields], color="#08c")
        for i, f in enumerate(fields):
            ax2.text(i, rep["field_counts"][f] + 0.4, str(rep["field_counts"][f]), ha="center", fontsize=8)
        ax2.set_ylabel("modules affected")
        ax2.set_title("drift by evidence field", fontsize=9)
        ax2.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evidence-payload drift check for the m15617 SQL deliverable")
    ap.add_argument("--old", default=None, help="path to the reference SQL dump")
    ap.add_argument("--old-git", default=None, help="git blob spec, e.g. HEAD:docs/tensorlbm_maturity_m15912.sql")
    ap.add_argument("--new", required=True, help="path to the SQL dump under test")
    ap.add_argument("--out", default=None, help="directory for drift.json / drift.png")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    if not a.old and not a.old_git:
        ap.error("one of --old / --old-git is required")
    old_text = (open(a.old, encoding="utf-8").read() if a.old
                else subprocess.run(["git", "show", a.old_git], capture_output=True, text=True,
                                    check=True).stdout)
    rep = diff(evidence_rows(old_text), evidence_rows(open(a.new, encoding="utf-8").read()))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        if not a.no_figure:
            rep["figure"] = figure(rep, os.path.join(a.out, "drift.png"))
        with open(os.path.join(a.out, "drift.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("n_modules_old", "n_modules_new", "n_modules_drift",
                                          "field_counts", "identical", "figure") if k in rep},
                     ensure_ascii=False))
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())