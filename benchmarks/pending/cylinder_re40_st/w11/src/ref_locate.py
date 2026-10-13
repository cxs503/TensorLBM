"""Locate candidate OA PDFs for the reference cluster via OpenAlex title search."""

import json
import time
import urllib.parse
import urllib.request

QUERIES = [
    "Numerical solutions for steady flow past a circular cylinder for Reynolds numbers up to 100",
    "Numerical simulation of laminar flow past a circular cylinder",
    "Steady separated flow past a circular cylinder at low Reynolds numbers",
    "Experimental determination of the main features of the viscous flow in the wake of a circular cylinder in uniform translation",
]

BASE = "https://api.openalex.org/works?filter=title.search:{}&per-page=5&mailto=phase0@bm_w11.local"

for q in QUERIES:
    url = BASE.format(urllib.parse.quote(q))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        print("QUERY FAILED", q[:40], e)
        continue
    print("=" * 80)
    print("QUERY:", q[:70])
    for w in d.get("results", [])[:5]:
        print("  ", w.get("publication_year"), "|", (w.get("title") or "")[:70])
        print("    doi:", w.get("doi"))
        for loc in w.get("locations", []) or []:
            pdf = (loc or {}).get("pdf_url")
            landing = (loc or {}).get("landing_page_url")
            if pdf or landing:
                print("    loc:", pdf, "|", landing)
    time.sleep(1)
