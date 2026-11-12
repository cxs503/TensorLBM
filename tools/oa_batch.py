"""Fetch a small batch of open-access PDFs and grep for St values near Re=100."""

import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pypdf

OUT = Path("artifacts/karman_re100/refs")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, timeout=35):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Accept": "application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def candidates(search, per=20):
    q = urllib.parse.urlencode({"search": search, "per-page": per, "filter": "is_oa:true"})
    data = json.loads(fetch("https://api.openalex.org/works?" + q).decode())
    out = []
    for w in data.get("results", []):
        loc = w.get("best_oa_location") or {}
        if loc.get("pdf_url"):
            out.append((w.get("publication_year"), w.get("title"), loc["pdf_url"]))
    return out


def scan(pdf):
    r = pypdf.PdfReader(str(pdf))
    hits = []
    for i, pg in enumerate(r.pages, 1):
        flat = re.sub(r"\s+", " ", pg.extract_text() or "")
        for m in re.finditer(r"0\.1[5-8]\d", flat):
            ctx = flat[max(0, m.start() - 170): m.start() + 130]
            if re.search(r"\bRe\b|Reynolds", ctx):
                hits.append((i, m.group(0), ctx))
    return hits


search = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cands = candidates(search)
print(f"search={search!r} candidates={len(cands)}")
for y, t, u in cands[:limit]:
    name = re.sub(r"[^A-Za-z0-9]+", "_", f"{y}_{t}")[:60]
    pdf = OUT / f"oa_{name}.pdf"
    try:
        if not pdf.exists():
            pdf.write_bytes(fetch(u))
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        hits = scan(pdf)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {y} {str(t)[:55]}: {str(e)[:70]}")
        continue
    print(f"\n== {y} {str(t)[:70]} sha={sha[:12]} hits={len(hits)}\n   {u}")
    for pno, v, ctx in hits[:5]:
        print(f"   p{pno} {v} :: {ctx[:250]}")