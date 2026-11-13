#!/usr/bin/env python3
"""Diff the evidence payload of two ``tensorlbm_module_maturity`` SQL dumps.

Motivation (TensorLBM #1995 / m15617 round-3): the audit deliverable is one
generated SQL snapshot that ships under two ``docs/`` names.  A schema-only
pytest guard stays green while one copy carries the evidence of an earlier run,
so the drift reaches main and a reader of ``docs/`` is handed a grading the
generator no longer produces.  On the pre-fix tree the m15617-named copy was
stale for 45/45 modules (``status_row`` 24, ``test_paths`` 10, ``tier_why`` 45,
``downgrade_reason`` 6, ``evidence_shared_with`` 5, ``pending_status_row`` 1,
``impl_paths`` 1) and its headline level distribution said verified=7 where the
current grading says verified=4 (three modules downgraded for evidence reuse --
``downgrade_reason`` is exactly the field a stale copy silently drops).

This tool answers "do these two dumps still say the same thing?" at the *value*
level, per module and per field.  Two row layouts are understood:

* ``text``  -- current generator: ``(module_path, area, tier, evidence, ...)``
  with the evidence dict as a single-quoted JSON literal (4th literal);
* ``jsonb`` -- the superseded m15617 layout: ``(audit_id, module_key, family,
  level, evidence_tier, tier_why, $json$...$json$::jsonb, mission_id)`` with the
  evidence dict in a dollar-quoted JSON block (2nd literal = module key).  The
  two layouts are also an incompatibility finding in their own right: a dump in
  one layout cannot be loaded into the table the other one creates.

Usage::

    python tools/audit/docs_sql_drift.py --old /tmp/a.sql --new docs/tensorlbm_maturity_m15617.sql \\
        --out artifacts/m15617

``--old`` may be a git blob spec via ``--old-git``::

    python tools/audit/docs_sql_drift.py --old-git HEAD:docs/tensorlbm_maturity_m15617.sql \\
        --new docs/tensorlbm_maturity_m15617.sql --out artifacts/m15617

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
DOLLAR_JSON = re.compile(r"\$json\$(.*?)\$json\$", re.S)


def _level(dump: str) -> str:
    """Row layout tag: ``jsonb`` (superseded m15617 dump) or ``text`` (generator)."""
    return "jsonb" if DOLLAR_JSON.search(dump) else "text"


def evidence_rows(text: str) -> dict:
    """``{module: evidence dict}`` parsed out of the INSERT statements."""
    jsonb = _level(text) == "jsonb"
    out = {}
    for line in text.splitlines():
        if not line.startswith(f"INSERT INTO {TABLE} "):
            continue
        fields = [f.replace("''", "'") for f in QUOTED.findall(line)]
        if jsonb:
            blob = DOLLAR_JSON.search(line)
            if len(fields) < 4 or blob is None:
                raise ValueError(f"unparsable jsonb INSERT: {line[:120]}")
            out[fields[1]] = json.loads(blob.group(1))
            continue
        if len(fields) < 4:
            raise ValueError(f"unparsable INSERT: {line[:120]}")
        out[fields[0]] = json.loads(fields[3])
    return out


def level_rows(text: str) -> dict:
    """``{module: level}`` -- ``verified|partial|implemented|failed`` per row."""
    jsonb = _level(text) == "jsonb"
    out = {}
    for line in text.splitlines():
        if not line.startswith(f"INSERT INTO {TABLE} "):
            continue
        fields = [f.replace("''", "'") for f in QUOTED.findall(line)]
        if jsonb:
            out[fields[1]] = fields[3]
        else:
            out[fields[0]] = fields[2]
    return out


def diff(old: dict, new: dict, old_levels: dict | None = None,
         new_levels: dict | None = None) -> dict:
    """Per-module, per-field value diff.  ``identical`` is the CI answer."""
    per_field, per_module = {}, {}
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key) or {}, new.get(key) or {}
        changed = [f for f in a if a[f] != b.get(f)]
        changed += [f for f in b if f not in a]
        if old_levels is not None and new_levels is not None:
            if old_levels.get(key) != new_levels.get(key):
                changed = sorted(changed + ["level"])
        if changed:
            per_module[key] = {"fields": sorted(changed),
                               "old": ({f: a.get(f) for f in changed}
                                       if old_levels is None else
                                       {**{f: a.get(f) for f in changed if f != "level"},
                                        **({"level": old_levels.get(key)} if "level" in changed else {})}),
                               "new": ({f: b.get(f) for f in changed}
                                       if new_levels is None else
                                       {**{f: b.get(f) for f in changed if f != "level"},
                                        **({"level": new_levels.get(key)} if "level" in changed else {})})}
            for f in changed:
                per_field[f] = per_field.get(f, 0) + 1
    return {"n_modules_old": len(old), "n_modules_new": len(new),
            "modules_only_in_old": sorted(set(old) - set(new)),
            "modules_only_in_new": sorted(set(new) - set(old)),
            "n_modules_drift": len(per_module),
            "field_counts": dict(sorted(per_field.items(), key=lambda kv: -kv[1])),
            "modules": per_module,
            "identical": bool(old) and old == new and (
                old_levels is None or old_levels == new_levels)}


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
        ax1.set_title(f"evidence drift: {rep['n_modules_drift']}/{rep['n_modules_old']} modules",
                      fontsize=9)
    fields = list(rep["field_counts"])
    if fields:
        ax2.bar(fields, [rep["field_counts"][f] for f in fields], color="#08c")
        for i, f in enumerate(fields):
            ax2.text(i, rep["field_counts"][f] + 0.4, str(rep["field_counts"][f]),
                     ha="center", fontsize=8)
        ax2.set_ylabel("modules affected")
        ax2.set_title("drift by evidence field", fontsize=9)
        ax2.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="evidence-payload drift check for the m15617 SQL deliverable")
    ap.add_argument("--old", default=None, help="path to the reference SQL dump")
    ap.add_argument("--old-git", default=None,
                    help="git blob spec, e.g. HEAD:docs/tensorlbm_maturity_m15617.sql")
    ap.add_argument("--new", required=True, help="path to the SQL dump under test")
    ap.add_argument("--out", default=None, help="directory for drift.json / drift.png")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    if not a.old and not a.old_git:
        ap.error("one of --old / --old-git is required")
    old_text = (open(a.old, encoding="utf-8").read() if a.old
                else subprocess.run(["git", "show", a.old_git], capture_output=True,
                                    text=True, check=True).stdout)
    new_text = open(a.new, encoding="utf-8").read()
    rep = diff(evidence_rows(old_text), evidence_rows(new_text),
               level_rows(old_text), level_rows(new_text))
    rep["layouts"] = {"old": _level(old_text), "new": _level(new_text)}
    rep["old_sha256"] = __import__("hashlib").sha256(old_text.encode()).hexdigest()
    rep["new_sha256"] = __import__("hashlib").sha256(new_text.encode()).hexdigest()
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        if not a.no_figure:
            rep["figure"] = figure(rep, os.path.join(a.out, "docs_sql_drift.png"))
        with open(os.path.join(a.out, "docs_sql_drift.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("n_modules_old", "n_modules_new", "n_modules_drift",
                                          "field_counts", "layouts", "identical", "figure")
                      if k in rep}, ensure_ascii=False))
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())