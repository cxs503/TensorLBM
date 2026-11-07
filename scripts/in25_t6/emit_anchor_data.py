#!/usr/bin/env python3
"""IN25-T6 -- generate the in-package bibliographic record store.

Bibliographic fields are *copied programmatically* from the archived API
payloads (Crossref DOI records + arXiv Atom records in ``artifacts/in25_t6/``)
into ``src/tensorlbm/external_survey/data/lit_anchor_records.json``.  Nothing is
typed by hand, so the anchor table cannot drift from what the registries say.

Curated content (claims, applicability conditions, verification status) lives in
:mod:`tensorlbm.external_survey.lit_anchors`; the two are joined by ``id``
(DOI or arXiv id) at import time.

Usage
-----
    python scripts/in25_t6/emit_anchor_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "in25_t6"
DEST = ROOT / "src" / "tensorlbm" / "external_survey" / "data" / "lit_anchor_records.json"


def main() -> int:
    # seat-A probe payloads are the provenance for the DOI selection; the DOI
    # records below carry the strings that actually land in the store.
    if not (ART / "seat_a_crossref_raw.json").exists():
        raise SystemExit("seat-A probe archive missing: run scripts/in25_t6/seat_a_lit_query.py first")
    doi_records = json.loads((ART / "seat_bc_doi_records.json").read_text())
    arxiv = json.loads((ART / "seat_bc_arxiv_probes.json").read_text())

    records: dict[str, dict] = {}
    for doi, rec in doi_records["records"].items():
        records[doi] = {
            "id": doi,
            "id_kind": "doi",
            "measure": rec["measure"],
            "probe": rec["probe"],
            "title": rec["title"],
            "authors": rec["authors"],
            "year": rec["year"],
            "venue": rec["venue"],
            "volume": rec["volume"],
            "page": rec["page"],
            "type": rec["type"],
            "publisher": rec["publisher"],
            "abstract": rec["abstract"],
            "retrieved_at": doi_records["retrieved_at"],
            "endpoint": "https://api.crossref.org/works/{doi}",
        }

    for query, hits in arxiv["queries"].items():
        for hit in hits:
            aid = hit["id"].rsplit("/", 1)[-1]
            base = aid.split("v")[0]
            records[base] = {
                "id": base,
                "id_kind": "arxiv",
                "measure": None,  # assigned by the curated table
                "probe": "OA",
                "title": hit["title"].rstrip("."),
                "authors": hit["authors"],
                "year": int(hit["published"][:4]) if hit["published"] else None,
                "venue": "arXiv preprint",
                "volume": None,
                "page": None,
                "type": "preprint",
                "publisher": "arXiv",
                "abstract": hit["abstract"],
                "arxiv_id": aid,
                "retrieved_at": arxiv["retrieved_at"],
                "endpoint": arxiv["endpoint"],
            }

    DEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": "scripts/in25_t6/emit_anchor_data.py",
        "note": "verbatim bibliographic records from Crossref / arXiv; no hand typing",
        "sources": {
            "crossref_probe": str((ART / "seat_a_crossref_raw.json").relative_to(ROOT)),
            "crossref_doi": str((ART / "seat_bc_doi_records.json").relative_to(ROOT)),
            "arxiv": str((ART / "seat_bc_arxiv_probes.json").relative_to(ROOT)),
        },
        "records": records,
    }
    DEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {DEST.relative_to(ROOT)} ({len(records)} records)")
    for rid in sorted(records):
        r = records[rid]
        print(f"  {rid:34s} {str(r['year']):5s} {r['title'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())