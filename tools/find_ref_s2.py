"""Locate an open-access primary/literature source for St(Re=100) with a real PDF.

Phase A: Semantic Scholar openAccessPdf lookup for the canonical experimental papers.
Phase B: download whatever PDFs are reachable and print verbatim St/Re lines.
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

TITLES = [
    "A series of new experiments on the wake dynamics of a circular cylinder",
    "A new Strouhal-Reynolds-number relationship for the circular cylinder in the range 47",
    "Fluctuating lift on a circular cylinder: review and new measurements",
    "Measurements of base pressure in the wake of a circular cylinder",
    "On the development of turbulent wakes from vortex streets",
    "Experiments on the flow past a circular cylinder at low Reynolds numbers",
]


def fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def s2(title):
    q = urllib.parse.urlencode(
        {"query": title, "fields": "title,year,externalIds,openAccessPdf,venue", "limit": 3}
    )
    try:
        data = json.loads(fetch("https://api.semanticscholar.org/graph/v1/paper/search?" + q))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:100]}
    return data.get("data", [])


def pdf_lines(pdf_path, patterns, half=200):
    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    reader = pypdf.PdfReader(str(pdf_path))
    out = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            continue
        flat = re.sub(r"\s+", " ", text)
        for pat in patterns:
            for m in re.finditer(pat, flat):
                out.append({"page": i, "match": m.group(0),
                            "context": flat[max(0, m.start() - half): m.start() + half + 80]})
    return sha, out


summary = []
for t in TITLES:
    rec = s2(t)
    if isinstance(rec, dict):
        print("S2 FAIL", t[:40], rec)
        continue
    for p in rec[:1]:
        oa = p.get("openAccessPdf") or {}
        print(f"\n== {p.get('title','')[:80]} ({p.get('year')}) venue={p.get('venue')}")
        print("   ids:", p.get("externalIds"), "oa:", oa.get("url"))
        summary.append({"query": t, "title": p.get("title"), "year": p.get("year"),
                        "ids": p.get("externalIds"), "oa": oa.get("url")})
        if oa.get("url"):
            name = re.sub(r"[^A-Za-z0-9]+", "_", (p.get("title") or t))[:50]
            pdf = OUT / f"{name}_{p.get('year')}.pdf"
            try:
                pdf.write_bytes(fetch(oa["url"]))
                sha, hits = pdf_lines(
                    pdf,
                    [r"0\.16\d", r"St\s*=\s*0\.1\d+"],
                )
                print("   saved", pdf, "sha256", sha[:16], "hits", len(hits))
                for h in hits[:8]:
                    print(f"   p{h['page']} {h['match']}: {h['context'][:300]}")
            except Exception as e:  # noqa: BLE001
                print("   download/parse fail", str(e)[:90])

(OUT / "s2_lookup.json").write_text(json.dumps(summary, indent=2))
print("\nDONE")