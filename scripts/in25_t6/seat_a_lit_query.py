#!/usr/bin/env python3
"""IN25-T6 seat A -- five-measure literature verification（五措文献席）.

Every candidate anchor is *queried live* against Crossref (bibliographic
metadata: title/authors/year/venue/DOI) and, where the record exposes it, the
abstract.  The raw API payloads are archived so the anchor table in
:mod:`tensorlbm.external_survey.lit_anchors` is traceable to an on-disk
evidence file (retrieved_at timestamp + query string + returned DOI).

Nothing here is typed by hand: the anchor metadata that this script prints is
what the report and the module carry.

Usage
-----
    python scripts/in25_t6/seat_a_lit_query.py            # live queries
    python scripts/in25_t6/seat_a_lit_query.py --offline   # replay archive
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "in25_t6"
RAW = OUT / "seat_a_crossref_raw.json"
TABLE = OUT / "seat_a_lit_anchors.json"

CROSSREF = "https://api.crossref.org/works"
UA = "TensorLBM-IN25T6-survey/1.0 (mailto:survey@tensorlbm.invalid)"

# (measure_key, probe_id, bibliographic query) -- one to three probes per measure.
PROBES: tuple[tuple[str, str, str], ...] = (
    ("large_grid", "LG-1",
     "large eddy simulation submarine hull SUBOFF resistance grid convergence"),
    ("large_grid", "LG-2",
     "Posa Balaras large eddy simulation appended submarine hull"),
    ("parallel_lbm", "PL-1",
     "waLBerla block-structured high-performance framework multiphysics simulations"),
    ("parallel_lbm", "PL-2", "Palabos parallel lattice Boltzmann solver"),
    ("parallel_lbm", "PL-3",
     "framework hybrid parallel flow simulations trillion cells complex geometries"),
    ("amr", "AM-1", "grid refinement lattice Boltzmann Filippova Hanel"),
    ("amr", "AM-2", "multi-domain lattice Boltzmann grid refinement Lagrava"),
    ("amr", "AM-3", "grid refinement entropic lattice Boltzmann models Dorschner"),
    ("les", "LS-1", "lattice Boltzmann subgrid model for high Reynolds number flows"),
    ("les", "LS-2",
     "theory of the lattice Boltzmann method dispersion dissipation isotropy Lallemand Luo"),
    ("les", "LS-3", "Smagorinsky constant lattice Boltzmann large eddy simulation assessment"),
    ("wall_model", "WM-1",
     "power-law wall model lattice Boltzmann airfoil turbulent boundary layer"),
    ("wall_model", "WM-2", "wall function lattice Boltzmann method turbulent boundary layer"),
    ("wall_model", "WM-3",
     "explicit expression smooth wall velocity distribution turbulent boundary layer Musker"),
)

SELECT = "DOI,title,author,issued,container-title,type,abstract,publisher,volume,page"


def query_crossref(text: str, rows: int = 3, timeout: float = 25.0) -> dict:
    params = {
        "query.bibliographic": text,
        "rows": str(rows),
        "select": SELECT,
        "mailto": "survey@tensorlbm.invalid",
    }
    url = f"{CROSSREF}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def summarise(item: dict) -> dict:
    authors = []
    for a in item.get("author", []) or []:
        family = a.get("family") or a.get("name") or "?"
        given = a.get("given", "")
        authors.append(f"{family}, {given}".strip(", "))
    issued = item.get("issued", {}).get("date-parts", [[None]])[0]
    title = (item.get("title") or ["?"])[0]
    venue = (item.get("container-title") or [None])[0]
    abstract = item.get("abstract") or ""
    return {
        "doi": item.get("DOI"),
        "title": " ".join(title.split()),
        "authors": authors,
        "year": issued[0] if issued else None,
        "venue": venue,
        "type": item.get("type"),
        "publisher": item.get("publisher"),
        "volume": item.get("volume"),
        "page": item.get("page"),
        "abstract_present": bool(abstract),
        "abstract": " ".join(abstract.split())[:1800],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="replay archived payloads")
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    archive: dict[str, dict] = {}
    if args.offline:
        if not RAW.exists():
            raise SystemExit(f"no archive at {RAW}; run online first")
        archive = json.loads(RAW.read_text())

    records = []
    for measure, probe, text in PROBES:
        if args.offline:
            if probe not in archive:
                raise SystemExit(f"probe {probe} missing from archive")
            hits = archive[probe]["hits"]
        else:
            payload = query_crossref(text, rows=args.rows)
            hits = [summarise(it) for it in payload["message"]["items"]]
            archive[probe] = {
                "measure": measure,
                "query": text,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": CROSSREF,
                "total_results": payload["message"].get("total-results"),
                "hits": hits,
            }
            time.sleep(0.4)  # be polite to the public API
        records.append({"measure": measure, "probe": probe, "query": text, "hits": hits})

    if not args.offline:
        RAW.write_text(json.dumps(archive, indent=2, ensure_ascii=False))
    TABLE.write_text(json.dumps({"probes": records}, indent=2, ensure_ascii=False))

    print(f"{'probe':6s} {'year':5s} {'doi':32s} title")
    for rec in records:
        for hit in rec["hits"][:3]:
            print(f"{rec['probe']:6s} {str(hit['year']):5s} {str(hit['doi'])[:32]:32s} "
                  f"{hit['title'][:88]}")
    print(f"\n[seat A] {len(records)} probes -> {RAW.relative_to(ROOT)}, "
          f"{TABLE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())