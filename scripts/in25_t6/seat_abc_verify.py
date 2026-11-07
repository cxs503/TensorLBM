#!/usr/bin/env python3
"""IN25-T6 seat A/B/C shared -- DOI-level verification for the selected anchors
plus arXiv (open-access) probes whose *abstracts* carry usable numbers.

Crossref record fetch gives title/authors/year/venue/volume/page verbatim from
the registry, so the anchor table in :mod:`tensorlbm.external_survey` never
carries hand-typed bibliographic data.  arXiv abstracts are the only
open-access text available offline for numeric claims (Elsevier/JSTOR records
ship no abstract through Crossref), so every numeric literature claim in the
survey is either (a) from an arXiv abstract archived here, or (b) marked
``status='absent'``.

Usage
-----
    python scripts/in25_t6/seat_abc_verify.py
    python scripts/in25_t6/seat_abc_verify.py --offline
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "in25_t6"
RAW = OUT / "seat_bc_doi_records.json"
AXIV = OUT / "seat_bc_arxiv_probes.json"

UA = "TensorLBM-IN25T6-survey/1.0 (mailto:survey@tensorlbm.invalid)"

# Anchors selected from the seat-A probe result table (five measures).
SELECTED: tuple[tuple[str, str, str], ...] = (
    ("large_grid", "LG-2", "10.1016/j.compfluid.2018.01.013"),
    ("large_grid", "LG-1", "10.1080/17445302.2026.2731435"),
    ("parallel_lbm", "PL-1", "10.1016/j.camwa.2020.01.007"),
    ("parallel_lbm", "PL-2", "10.1016/j.camwa.2020.03.022"),
    ("parallel_lbm", "PL-3", "10.1145/2503210.2503273"),
    ("amr", "AM-1", "10.1006/jcph.1998.6089"),
    ("amr", "AM-2", "10.1016/j.jcp.2012.03.015"),
    ("amr", "AM-3", "10.1103/physreve.94.053311"),
    ("les", "LS-1", "10.1090/fic/006/12"),
    ("les", "LS-2", "10.1103/physreve.61.6546"),
    ("wall_model", "WM-3", "10.2514/3.61193"),
    ("wall_model", "WM-1", "10.1063/5.0283930"),
)

# Open-access records whose abstracts carry usable numbers (arXiv listing
# probes run 2026-11-06; ids pinned after relevance screening).
ARXIV_IDS: tuple[str, ...] = (
    "2503.19674",  # appendage effects on submarine turbulence/noise (SUBOFF family)
    "2407.10475",  # LES of hydrodynamic noise past an axisymmetric hull
    "1608.06915",  # grid refinement for entropic LBM (journal version = AM-3)
    "2611.07137",  # momentum-exchange force accuracy in LBM
    "2506.03905",  # wall boundary conditions + wall functions in LBM
    "2601.18443",  # enhanced wall boundary modelling in LBM
)

ATOM = "{http://www.w3.org/2005/Atom}"


def fetch_doi(doi: str, timeout: float = 25.0) -> dict:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]


def arxiv_search(query: str, max_results: int = 4, timeout: float = 30.0) -> list[dict]:
    key = "id_list" if query.startswith("id_list:") else "search_query"
    val = query.split(":", 1)[1] if key == "id_list" else query
    url = (
        "https://export.arxiv.org/api/query?"
        + urllib.parse.urlencode({key: val, "max_results": str(max_results)})
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        root = ET.fromstring(resp.read())
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        out.append(
            {
                "id": (entry.findtext(f"{ATOM}id") or "").strip(),
                "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
                "published": (entry.findtext(f"{ATOM}published") or "").strip(),
                "authors": [
                    (a.findtext(f"{ATOM}name") or "").strip()
                    for a in entry.findall(f"{ATOM}author")
                ][:6],
                "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split())[:2500],
                "doi": (entry.findtext(f"{ATOM}{'doi'}", "") or "").strip()
                if entry.find(f"{ATOM}{'doi'}") is not None
                else "",
            }
        )
    return out


def summarise(rec: dict) -> dict:
    issued = (rec.get("issued") or {}).get("date-parts", [[None]])[0]
    return {
        "doi": rec.get("DOI"),
        "title": " ".join((rec.get("title") or ["?"])[0].split()),
        "authors": [
            f"{(a.get('family') or a.get('name') or '?')}, {a.get('given', '')}".strip(", ")
            for a in rec.get("author", []) or []
        ],
        "year": issued[0] if issued else None,
        "venue": (rec.get("container-title") or [None])[0],
        "volume": rec.get("volume"),
        "page": rec.get("page"),
        "type": rec.get("type"),
        "publisher": rec.get("publisher"),
        "abstract": " ".join((rec.get("abstract") or "").split())[:2500],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.offline:
        payload = json.loads(RAW.read_text())
    else:
        payload = {
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": "https://api.crossref.org/works/{doi}",
            "records": {},
        }
        for measure, probe, doi in SELECTED:
            rec = summarise(fetch_doi(doi))
            rec.update(measure=measure, probe=probe)
            payload["records"][doi] = rec
            time.sleep(0.35)
        RAW.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    for doi, rec in payload["records"].items():
        print(f"{rec['probe']:6s} {str(rec['year']):5s} {doi:34s} "
              f"abs={'Y' if rec['abstract'] else '-'} {rec['title'][:66]}")

    if args.offline and AXIV.exists():
        probes = json.loads(AXIV.read_text())
    else:
        probes = {
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": "https://export.arxiv.org/api/query",
            "queries": {},
        }
        for q in (f"id_list:{','.join(ARXIV_IDS)}",):
            probes["queries"][q] = arxiv_search(q, max_results=len(ARXIV_IDS))
            time.sleep(3.0)  # arXiv API courtesy delay
        AXIV.write_text(json.dumps(probes, indent=2, ensure_ascii=False))

    print()
    for q, hits in probes["queries"].items():
        print(f"ARXIV {q} -> {len(hits)} hits")
        for h in hits:
            print(f"\n### {h['published'][:7]} {h['id'].rsplit('/', 1)[-1]} | {h['title'][:100]}")
            print(f"    authors: {', '.join(h['authors'])}")
            print(f"    abstract: {h['abstract'][:700]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())