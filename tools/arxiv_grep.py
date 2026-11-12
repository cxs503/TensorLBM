"""Targeted arXiv grep: find a table/text giving an EXPERIMENTAL St at Re=100."""

import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pypdf

OUT = Path("artifacts/karman_re100/refs")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(query, n=12):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "max_results": n, "sortBy": "relevance"}
    )
    xml = fetch(url).decode("utf-8", "replace")
    ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)
    titles = [t.strip() for t in re.findall(r"<title>([^<]*)</title>", xml)[1:]]
    return list(zip(ids, titles))


def text_of(aid):
    pdf = OUT / f"{aid.replace('/', '_')}.pdf"
    if not pdf.exists():
        pdf.write_bytes(fetch(f"https://arxiv.org/pdf/{aid}", timeout=45))
    r = pypdf.PdfReader(str(pdf))
    return [(i, re.sub(r"\s+", " ", pg.extract_text() or "")) for i, pg in enumerate(r.pages, 1)]


def grep(aid, pages):
    out = []
    for pno, flat in pages:
        for pat in (r"0\.1[5-8]\d",):
            for m in re.finditer(pat, flat):
                ctx = flat[max(0, m.start() - 220): m.start() + 140]
                if re.search(r"Re\s*[=≈]?\s*100|\b100\b", ctx) and re.search(
                    r"Williamson|Norberg|Roshko|experiment|EXP|measur", ctx, re.I
                ):
                    out.append((pno, m.group(0), ctx))
    return out


QUERIES = sys.argv[1:] or [
    'all:"circular cylinder" AND all:"Strouhal" AND all:benchmark',
    'all:"vortex shedding" AND all:"Reynolds number" AND all:"100"',
]
cands = {}
for q in QUERIES:
    try:
        for aid, t in search(q):
            cands[aid] = t
    except Exception as e:  # noqa: BLE001
        print("fail", q, str(e)[:60])
print("candidates", len(cands))
for aid in list(cands)[:7]:
    try:
        pages = text_of(aid)
    except Exception as e:  # noqa: BLE001
        print("pdf fail", aid, str(e)[:50])
        continue
    hits = grep(aid, pages)
    print(f"\n== {aid} {cands[aid][:70]} hits={len(hits)}")
    for pno, val, ctx in hits[:3]:
        print(f"   p{pno} {val} :: {ctx[:280]}")