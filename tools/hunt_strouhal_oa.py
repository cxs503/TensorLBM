"""Systematic hunt: open-access PDFs that tabulate St at Re=100 (cylinder).

Zero hand-copy: numbers are read out of fetched PDFs; each PDF is archived with
its sha256 so the quoted table/page can be re-verified offline.
"""

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pypdf

OUT = Path("artifacts/karman_re100/refs")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def openalex(search, per=25):
    q = urllib.parse.urlencode(
        {"search": search, "per-page": per, "filter": "is_oa:true", "mailto": "eco@example.org"}
    )
    return json.loads(fetch("https://api.openalex.org/works?" + q)).get("results", [])


def scan(pdf_path):
    reader = pypdf.PdfReader(str(pdf_path))
    found = []
    for i, page in enumerate(reader.pages, 1):
        try:
            flat = re.sub(r"\s+", " ", page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
        # a Strouhal-looking value within ~180 chars of a "100" Reynolds mention
        for m in re.finditer(r"0\.1[5-8]\d", flat):
            ctx = flat[max(0, m.start() - 180): m.start() + 120]
            if re.search(r"\bRe\b|Reynolds|\bRe\s*=", ctx):
                found.append((i, m.group(0), ctx))
    return found


SEARCHES = [
    "Strouhal number circular cylinder Reynolds number 100 experiments table",
    "vortex shedding circular cylinder low Reynolds number soap film experiment Strouhal",
    "benchmark flow past circular cylinder Re 100 drag coefficient Strouhal experiments",
]

seen, hits = {}, []
for s in SEARCHES:
    try:
        rows = openalex(s)
    except Exception as e:  # noqa: BLE001
        print("search fail", str(e)[:70])
        continue
    for w in rows:
        loc = w.get("best_oa_location") or {}
        pdf = loc.get("pdf_url")
        if not pdf or pdf in seen:
            continue
        seen[pdf] = (w.get("title"), w.get("publication_year"))
    print(f"[{s[:45]}] cumulative candidates {len(seen)}", flush=True)

for url, (title, year) in list(seen.items())[:18]:
    name = re.sub(r"[^A-Za-z0-9]+", "_", f"{year}_{title}")[:60]
    pdf = OUT / f"oa_{name}.pdf"
    try:
        if not pdf.exists():
            pdf.write_bytes(fetch(url, timeout=45))
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        f = scan(pdf)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {year} {str(title)[:50]}: {str(e)[:70]}")
        continue
    print(f"\n== {year} {str(title)[:70]} sha={sha[:12]} hits={len(f)}")
    print(f"   src {url}")
    for pno, val, ctx in f[:4]:
        print(f"   p{pno} {val} :: {ctx[:260]}")
    if f:
        hits.append({"title": title, "year": year, "url": url, "sha256": sha,
                     "file": str(pdf), "hits": [{"page": p, "value": v, "context": c} for p, v, c in f]})

(OUT / "oa_scan.json").write_text(json.dumps(hits, indent=2))
print("\nPDFs with hits:", len(hits))