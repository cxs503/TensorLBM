"""Targeted arXiv grep: experimental St / Cd for a SQUARE cylinder at Re = 200
(Norberg 1993, JWEIA 49:187-196 — paywalled, so we mine open-access papers that
tabulate Norberg's numbers, and record page + context for provenance).

Usage:  python3 tools/norberg1993_grep.py ["query1" "query2" ...]
Output: runs/norberg1993_research/arxiv_grep.log  (contexts, page numbers)
        runs/norberg1993_research/hits.json       (structured hits)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("runs/norberg1993_research")
OUT.mkdir(parents=True, exist_ok=True)
PDFS = OUT / "pdfs"
PDFS.mkdir(exist_ok=True)


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(query: str, n: int = 10):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": n, "sortBy": "relevance"}
    )
    xml = fetch(url).decode("utf-8", "replace")
    ids = re.findall(r"<id>https?://arxiv.org/abs/([^<]+)</id>", xml)
    titles = [t.strip() for t in re.findall(r"<title>([^<]*)</title>", xml)[1:]]
    return list(zip(ids, titles))


def pages_of(aid: str):
    pdf = PDFS / f"{aid.replace('/', '_')}.pdf"
    if not pdf.exists():
        pdf.write_bytes(fetch(f"https://arxiv.org/pdf/{aid}", timeout=120))
    import pypdf

    r = pypdf.PdfReader(str(pdf))
    return [(i, re.sub(r"\s+", " ", pg.extract_text() or "")) for i, pg in enumerate(r.pages, 1)]


QUERIES = sys.argv[1:] or [
    'abs:"square cylinder" AND abs:"Strouhal"',
    'all:"square cylinder" AND all:"Norberg"',
    'abs:"rectangular cylinder" AND abs:"Re = 200"',
    'abs:"square cylinder" AND abs:"direct numerical simulation"',
]

cands: dict[str, str] = {}
for q in QUERIES:
    try:
        for aid, t in search(q):
            cands[aid] = t
    except Exception as e:  # network flake -> keep going
        print(f"[search-fail] {q}: {e}", flush=True)

log = open(OUT / "arxiv_grep.log", "w")
hits = []
for aid, title in cands.items():
    try:
        pgs = pages_of(aid)
    except Exception as e:
        print(f"[pdf-fail] {aid}: {e}", flush=True)
        continue
    for pno, flat in pgs:
        for m in re.finditer(r"Norberg", flat):
            ctx = flat[max(0, m.start() - 300) : m.start() + 400]
            if re.search(r"0\.1[2-7]\d", ctx) or re.search(r"Re\s*[=≈]?\s*200", ctx):
                hits.append({"arxiv": aid, "title": title, "page": pno, "ctx": ctx})
                log.write(f"\n=== {aid} p{pno} | {title}\n{ctx}\n")
    print(f"[done] {aid} hits={len(hits)}", flush=True)

log.close()
(OUT / "hits.json").write_text(json.dumps(hits, indent=1))
print(f"TOTAL hits={len(hits)} papers={len(cands)}")