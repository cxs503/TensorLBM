#!/usr/bin/env python3
"""Horizontal OSS survey instrument (ported practice, oss_solutions id=386).

Practice being cross-validated (``repo='oss-inventory/tensorplatform'``,
``feature_slug='GAP-DEM-CLUMP'``, ``tag='horizontal-survey-empty'``):

    "Open-source horizontal survey: SELECT repo, license, methods FROM
     oss_inventory WHERE methods|summary|focus ILIKE any of {...} returns 0 rows
     (218 rows / 140 feature slugs in the table) ... There is therefore no
     mature third-party implementation to lift code structure from; the design
     is built from the literature entries instead."

That practice has three machine-checkable parts, and this script reproduces all
three inside TensorLBM so the verdict is auditable rather than asserted:

  S1 census scan      -- keyword families against the oss_inventory dump
                         (``data/oss_inventory_dump.csv``, dumped by the SQL in
                         ``data/PROVENANCE.json``).
  S2 solution ledger  -- same keyword families against the oss_solutions
                         ledger dump (method/conventions/transfer text: the
                         columns that record *what an implementation does*).
  S3 source scan      -- same keyword families against any locally mirrored
                         OSS source tree (``--oss-cache``, default
                         ``~/oss-cache``); reported as ``unavailable`` when the
                         mirror is missing instead of silently counting as 0.

Nothing here is hand-copied: every hit carries the row id and the matched
snippet, and the verdict of a family is a pure function of the hit count.

Usage:
  python xval/horizontal_survey_empty/survey.py            # full table
  python xval/horizontal_survey_empty/survey.py --focus added_mass_pmm
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Capability families.  `src` is the TensorLBM-side presence probe (glob or
# regex over src/tensorlbm) so the survey can say, for every family, whether
# *we* claim the capability; the census side then says whether a third-party
# implementation exists to cross-validate against.
FAMILIES: dict[str, dict] = {
    "added_mass_pmm": {
        "label": "oscillatory-flow inertia coefficient / added mass / PMM derivatives",
        "patterns": [
            r"added[ -]?mass", r"\bpmm\b", r"planar motion mechanism",
            r"oscillat\w* (?:flow|cylinder|free[ -]?stream)",
            r"morison", r"inertia coefficient", r"keulegan", r"\bkc\b.*\btheta\b",
            r"force oscillation test", r"captive (?:model )?test",
        ],
        "src_probe": r"added|pmm|manoeu|maneu|oscillat",
    },
    "taylor_couette": {
        "label": "Taylor-Couette / circular Couette concentric rotating cylinders",
        "patterns": [r"taylor[ -]?couette", r"circular couette", r"concentric cylinder"],
        "src_probe": r"taylor.?couette",
    },
    "stagnation_hiemenz": {
        "label": "Hiemenz stagnation-point flow",
        "patterns": [r"hiemenz", r"stagnation[ -]point"],
        "src_probe": r"hiemenz|stagnation",
    },
    "electrokinetic": {
        "label": "electro-osmotic / Poisson-Boltzmann electrokinetic LBM",
        "patterns": [r"electro[ -]?osmo", r"poisson[ -]?boltzmann", r"electrokinet", r"debye"],
        "src_probe": r"electro.?osmo|poisson.?boltzmann|debye",
    },
    "non_newtonian_powerlaw": {
        "label": "non-Newtonian power-law / Herschel-Bulkley rheology",
        "patterns": [r"power[ -]?law", r"herschel", r"bingham", r"non[ -]?newton", r"viscoplastic"],
        "src_probe": r"power.?law|herschel|bingham|non.?newton",
    },
    "mhd_lorentz": {
        "label": "low-Rm MHD (Hartmann/Shercliff) Lorentz force",
        "patterns": [r"hartmann", r"magnetohydro|\bmhd\b", r"lorentz"],
        "src_probe": r"hartmann|\bmhd\b|lorentz",
    },
    "immersed_boundary": {
        "label": "immersed-boundary method coupling",
        "patterns": [r"immersed[ -]boundary", r"\bibm\b"],
        "src_probe": r"immersed|\bibm\b",
    },
    "aeroacoustics": {
        "label": "aeroacoustics (FWH / acoustic analogy)",
        "patterns": [r"aeroacoust", r"fowcs", r"acoustic analog"],
        "src_probe": r"acoustic",
    },
}

CENSUS_FIELDS = ("focus", "methods", "multiphase", "boundary", "gpu", "summary", "lang")
LEDGER_FIELDS = ("feature_slug", "tag", "method", "files", "conventions", "transfer", "evidence")


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _scan_rows(rows: list[dict], fields: tuple[str, ...], patterns: list[str]) -> list[dict]:
    rx = re.compile("|".join(f"(?:{q})" for q in patterns), re.IGNORECASE)
    hits: list[dict] = []
    for row in rows:
        blob = " \n ".join(str(row.get(k) or "") for k in fields)
        m = rx.search(blob)
        if not m:
            continue
        lo = max(0, m.start() - 60)
        hits.append(
            {
                "id": row.get("id"),
                "repo": row.get("repo"),
                "field": next(
                    (k for k in fields if re.search(rx, str(row.get(k) or ""))), "?"
                ),
                "matched": m.group(0),
                "snippet": re.sub(r"\s+", " ", blob[lo : m.end() + 60]).strip(),
            }
        )
    return hits


def _scan_sources(root: Path, patterns: list[str]) -> dict:
    if not root.is_dir():
        return {"available": False, "root": str(root), "reason": "mirror absent"}
    rx = "|".join(f"(?:{p})" for p in patterns)
    try:
        out = subprocess.run(
            ["grep", "-rIl", "-i", "-E", rx, str(root)],
            capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        return {"available": False, "root": str(root), "reason": str(exc)}
    if out.returncode not in (0, 1):
        return {"available": False, "root": str(root), "reason": out.stderr[:200]}
    files = [str(Path(p).relative_to(root)) for p in out.stdout.split("\n") if p.strip()]
    repos = sorted({f.split(os.sep)[0] for f in files})
    return {"available": True, "root": str(root), "n_files": len(files), "repos": repos,
            "files": files[:40]}


def _src_probe(pattern: str) -> dict:
    root = HERE.parents[1] / "src" / "tensorlbm"
    if not root.is_dir():
        return {"available": False}
    out = subprocess.run(["grep", "-rIl", "-i", "-E", pattern, str(root)],
                         capture_output=True, text=True, timeout=180)
    files = [str(Path(p).relative_to(root)) for p in out.stdout.split("\n") if p.strip()]
    return {"available": True, "n_files": len(files), "files": files[:25]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oss-cache", default=os.path.expanduser("~/oss-cache"))
    ap.add_argument("--focus", default="added_mass_pmm")
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    inv = _csv_rows(DATA / "oss_inventory_dump.csv")
    led = _csv_rows(DATA / "oss_solutions_dump.csv")

    result = {
        "sources": {
            "oss_inventory": {"path": "data/oss_inventory_dump.csv", "n_rows": len(inv),
                              "sha256": hashlib.sha256(
                                  (DATA / "oss_inventory_dump.csv").read_bytes()).hexdigest()},
            "oss_solutions": {"path": "data/oss_solutions_dump.csv", "n_rows": len(led),
                              "sha256": hashlib.sha256(
                                  (DATA / "oss_solutions_dump.csv").read_bytes()).hexdigest()},
        },
        "families": {},
    }
    table = []
    for key, fam in FAMILIES.items():
        inv_hits = _scan_rows(inv, CENSUS_FIELDS, fam["patterns"])
        led_hits = _scan_rows(led, LEDGER_FIELDS, fam["patterns"])
        src = _scan_sources(Path(args.oss_cache), fam["patterns"])
        own = _src_probe(fam["src_probe"])
        verdict = "empty" if (not inv_hits and not led_hits and not src.get("n_files")) else "covered"
        result["families"][key] = {
            "label": fam["label"], "patterns": fam["patterns"], "verdict": verdict,
            "census_hits": inv_hits, "ledger_hits": led_hits, "source_scan": src,
            "tensorlbm_src": own,
        }
        table.append({
            "family": key, "label": fam["label"], "verdict": verdict,
            "census_hits": len(inv_hits), "census_repos": ";".join(sorted({h["repo"] for h in inv_hits})),
            "ledger_hits": len(led_hits), "ledger_slugs": ";".join(sorted({h["repo"] + "/" + str(h["id"]) for h in led_hits})),
            "oss_cache_hits": src.get("n_files", "unavailable"),
            "tensorlbm_src_files": own.get("n_files", 0),
        })

    out_dir = Path(args.out_dir)
    with (out_dir / "survey_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (out_dir / "survey_summary.json").write_text(json.dumps(result, indent=2) + "\n")

    focus = result["families"][args.focus]
    print(json.dumps({
        "focus": args.focus, "label": focus["label"], "verdict": focus["verdict"],
        "census_hits": len(focus["census_hits"]), "ledger_hits": len(focus["ledger_hits"]),
        "source_scan": {k: focus["source_scan"].get(k) for k in ("available", "n_files", "root")},
        "tensorlbm_src_n_files": focus["tensorlbm_src"].get("n_files"),
        "table": table,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())