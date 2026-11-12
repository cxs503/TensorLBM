"""OpenAlex lookup for open-access PDFs of the canonical St(Re) experimental literature."""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("artifacts/karman_re100/refs")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0 (mailto:eco@example.org)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def oa(search, per=5):
    q = urllib.parse.urlencode({"search": search, "per-page": per, "mailto": "eco@example.org"})
    data = json.loads(fetch("https://api.openalex.org/works?" + q))
    rows = []
    for w in data.get("results", [])[:per]:
        locs = w.get("best_oa_location") or {}
        rows.append(
            {
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "doi": w.get("doi"),
                "venue": (w.get("primary_location") or {}).get("source", {}) and
                          ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "oa_pdf": locs.get("pdf_url"),
                "oa_url": locs.get("landing_page_url"),
                "is_oa": (w.get("open_access") or {}).get("is_oa"),
            }
        )
    return rows


SEARCHES = [
    "A series of new experiments on the wake dynamics of a circular cylinder",
    "A new Strouhal Reynolds number relationship for the circular cylinder",
    "Fluctuating lift on a circular cylinder review and new measurements",
    "Strouhal number circular cylinder Reynolds number measurements 47 2x10^5",
    "On the development of turbulent wakes from vortex streets Roshko",
]

allrows = []
for s in SEARCHES:
    try:
        rows = oa(s)
    except Exception as e:  # noqa: BLE001
        print("FAIL", s[:40], str(e)[:80])
        continue
    print(f"\n== query: {s[:60]}")
    for r in rows:
        print(f"  {r['year']} | {str(r['title'])[:70]} | {r['venue']} | pdf={r['oa_pdf']}")
    allrows += rows

(OUT / "openalex_lookup.json").write_text(json.dumps(allrows, indent=2))
print("\nrows", len(allrows))