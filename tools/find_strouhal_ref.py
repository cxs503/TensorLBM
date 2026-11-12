"""Find an open-access reference that tabulates the cylinder Strouhal number at Re=100.

Strategy (zero hand-copy): search arXiv, download PDFs, extract text, and report
verbatim lines that contain both a Reynolds number of 100 and a Strouhal value.
Nothing here is typed from memory; the numbers come out of the fetched PDFs.
"""

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
API = "http://export.arxiv.org/api/query?"

QUERIES = [
    'abs:"circular cylinder" AND abs:Strouhal AND abs:Reynolds',
    'all:"flow past a circular cylinder" AND all:"Re = 100" AND all:Strouhal',
]


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(query, n=20):
    url = API + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": n, "sortBy": "relevance"}
    )
    xml = fetch(url).decode("utf-8", "replace")
    ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)
    titles = re.findall(r"<title>([^<]*)</title>", xml)[1:]
    return list(zip(ids, [t.strip().replace("\n", " ") for t in titles]))


def pdf_text(aid):
    pdf = OUT / f"{aid.replace('/', '_')}.pdf"
    if not pdf.exists():
        pdf.write_bytes(fetch(f"https://arxiv.org/pdf/{aid}"))
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    reader = pypdf.PdfReader(str(pdf))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            pages.append((i, page.extract_text() or ""))
        except Exception:  # noqa: BLE001
            pages.append((i, ""))
    return sha, pages


def main():
    seen = {}
    for q in QUERIES:
        try:
            for aid, title in search(q):
                seen[aid] = title
        except Exception as e:  # noqa: BLE001
            print("search fail", q, str(e)[:80])
    print(f"candidates: {len(seen)}")
    hits = []
    for aid in list(seen)[:14]:
        try:
            sha, pages = pdf_text(aid)
        except Exception as e:  # noqa: BLE001
            print("pdf fail", aid, str(e)[:60])
            continue
        for pno, text in pages:
            flat = re.sub(r"\s+", " ", text)
            for m in re.finditer(r"\b0\.1[5-8]\d\b", flat):
                ctx = flat[max(0, m.start() - 260) : m.start() + 160]
                if re.search(r"\bRe\b|Reynolds|\b100\b", ctx):
                    hits.append(
                        {"arxiv": aid, "sha256": sha, "page": pno,
                         "value": m.group(0), "context": ctx}
                    )
        print(aid, seen[aid][:70], "hits", sum(1 for h in hits if h["arxiv"] == aid), flush=True)
    (OUT / "hits.json").write_text(json.dumps(hits, indent=2))
    print(f"total hits {len(hits)}")
    for h in hits[:25]:
        print(f"\n--- {h['arxiv']} p{h['page']} val={h['value']}\n{h['context']}")


if __name__ == "__main__":
    sys.exit(main())