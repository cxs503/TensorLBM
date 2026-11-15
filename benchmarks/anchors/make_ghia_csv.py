#!/usr/bin/env python3
"""Mechanically generate the Ghia (1982) reference CSVs from the provenance file.

Why this script exists
----------------------
Downstream codes must not carry hand-typed copies of the Ghia (1982) tables:
three separate hand-typed copies of ``GHIA_RE400`` had drifted from the
sha256-bound OCR witness (see ``suspect_entries`` in
``ghia1982_provenance.json``).  The CSVs produced here are the single source of
truth for the Re=100 / Re=400 centreline anchors; :mod:`run_anchors` verifies
their sha256 before any consumer is allowed to use them, and cross-checks the
values back against the provenance tables.

Chain of custody
----------------
OCR witness text (sha256 pinned in the provenance)
    -> ``reynolds_numbers.<Re>`` tables in ``ghia1982_provenance.json``
    -> CSV files written by this script (sha256 pinned in ``csv_anchors``)

Usage::

    python benchmarks/anchors/make_ghia_csv.py            # write / refresh
    python benchmarks/anchors/make_ghia_csv.py --check    # fail if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ANCHOR_DIR = Path(__file__).resolve().parent
REPO = ANCHOR_DIR.parents[1]
PROVENANCE_FILE = ANCHOR_DIR / "ghia1982_provenance.json"
DATA_DIR = ANCHOR_DIR / "data"

CITATION = (
    "Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for "
    "incompressible flow using the Navier-Stokes equations and a multigrid "
    "method. Journal of Computational Physics 48(3), 387-411."
)

#: (Re, table label, coordinate column, profile column, comment)
TABLES = (
    (100, "I", "y", "u", "u-velocity along the vertical centreline x/L=0.5"),
    (100, "II", "x", "v", "v-velocity along the horizontal centreline y/L=0.5"),
    (400, "I", "y", "u", "u-velocity along the vertical centreline x/L=0.5"),
    (400, "II", "x", "v", "v-velocity along the horizontal centreline y/L=0.5"),
)


def csv_name(re: int, table: str) -> str:
    return f"ghia1982_re{re}_table{table}.csv"


def render_csv(prov: dict, re: int, table: str, coord: str, prof: str) -> str:
    block = prov["reynolds_numbers"][str(re)]
    ref = block[coord]
    val = block[prof]
    if len(ref) != len(val):
        raise ValueError(f"Re={re} table {table}: {coord}/{prof} length mismatch")
    witness = prov["primary_witness"]
    head = [
        f"# {CITATION}",
        f"# Table {table} (p. 398-399): Re={re}, {prof} vs {coord}, "
        "129x129 grid results.",
        "# Provenance: BOUND. Transcribed from the OCR text layer of the published PDF",
        f"#   witness: {Path(witness['file']).name}",
        f"#   witness sha256: {witness['sha256']}",
        f"#   pdf sha256:     {witness['pdf_sha256']}",
        "# Generated mechanically by benchmarks/anchors/make_ghia_csv.py from",
        "#   benchmarks/anchors/ghia1982_provenance.json -- do not edit by hand.",
        f"{coord},{prof}",
    ]
    body = [f"{c:.4f},{v:.5f}" for c, v in zip(ref, val)]
    return "\n".join(head + body) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(prov: dict) -> dict[str, str]:
    """Return {relative path: file content} for every anchor CSV."""
    out: dict[str, str] = {}
    for re, table, coord, prof, _ in TABLES:
        path = DATA_DIR / csv_name(re, table)
        rel = path.relative_to(REPO).as_posix()
        out[rel] = render_csv(prov, re, table, coord, prof)
    return out


def csv_anchor_records(files: dict[str, str]) -> list[dict]:
    """csv_anchors block for the provenance file."""
    meta = {f"re{re}_t{table}": (re, table, coord, prof, comment)
            for re, table, coord, prof, comment in TABLES}
    records = []
    for rel, text in sorted(files.items()):
        key = next(k for k in meta if rel.endswith(csv_name(meta[k][0], meta[k][1])))
        re, table, coord, prof, comment = meta[key]
        records.append(
            {
                "path": rel,
                "sha256": sha256_text(text),
                "bytes": len(text.encode("utf-8")),
                "re": re,
                "table": table,
                "columns": [coord, prof],
                "description": comment,
            }
        )
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if CSVs or their sha256 pins are stale")
    args = ap.parse_args()

    prov = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    files = build(prov)
    recorded = {e["path"]: e for e in prov.get("csv_anchors", [])}

    stale = []
    for rel, text in files.items():
        disk = REPO / rel
        if not disk.exists() or disk.read_text(encoding="utf-8") != text:
            stale.append(f"content stale: {rel}")
        rec = recorded.get(rel)
        if rec is None or rec["sha256"] != sha256_text(text):
            stale.append(f"sha256 pin stale: {rel}")

    if args.check:
        for s in stale:
            print(f"STALE {s}")
        print("RESULT:", "FAIL" if stale else "PASS (CSVs match provenance)")
        return 1 if stale else 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        (REPO / rel).write_text(text, encoding="utf-8")
        print(f"wrote {rel} ({len(text.encode('utf-8'))} bytes)")

    prov["csv_anchors"] = csv_anchor_records(files)
    PROVENANCE_FILE.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"updated {PROVENANCE_FILE.relative_to(REPO)}: "
          f"{len(prov['csv_anchors'])} csv_anchors pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())