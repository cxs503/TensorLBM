"""Fetch two open theses/reports that compile experimental St(Re) data; grep Re=100."""

import hashlib
import re
import urllib.request
from pathlib import Path

import pypdf

OUT = Path("artifacts/karman_re100/refs")
OUT.mkdir(parents=True, exist_ok=True)
UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/pdf,text/html,*/*",
}


def fetch(url, timeout=45):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


URLS = [
    "https://eprints.soton.ac.uk/393668/1/Final%2520e-thesis%2520for%2520e-prints%2520-NGUYEN%2520ID%252025216309.pdf",
    "https://www.jstage.jst.go.jp/article/jfst/6/6/6_6_1036/_pdf",
    "http://depositonce.tu-berlin.de/handle/11303/2691",
]

for u in URLS:
    name = re.sub(r"[^A-Za-z0-9]+", "_", u.split("/")[-1])[:50]
    try:
        raw = fetch(u)
    except Exception as e:  # noqa: BLE001
        print("FAIL", str(e)[:60], u[:80])
        continue
    if raw[:4] != b"%PDF":
        links = re.findall(r'href="([^"]+\.pdf[^"]*)"', raw.decode("utf-8", "replace"))
        print("HTML landing page, pdf links:", links[:5])
        continue
    pdf = OUT / f"th_{name}.pdf"
    pdf.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    print(f"\n== {u[:90]}\n   {pdf} sha={sha[:16]} bytes={len(raw)}")
    try:
        r = pypdf.PdfReader(str(pdf))
    except Exception as e:  # noqa: BLE001
        print("   parse fail", str(e)[:60])
        continue
    n = 0
    for i, pg in enumerate(r.pages, 1):
        flat = re.sub(r"\s+", " ", pg.extract_text() or "")
        for m in re.finditer(r"0\.1[5-8]\d", flat):
            ctx = flat[max(0, m.start() - 190): m.start() + 130]
            if re.search(r"\bRe\b|Reynolds", ctx):
                n += 1
                if n <= 6:
                    print(f"   p{i} {m.group(0)} :: {ctx[:300]}")
    print("   hits:", n, "pages:", len(r.pages))