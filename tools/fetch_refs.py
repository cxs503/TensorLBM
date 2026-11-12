"""Download candidate reference PDFs and print verbatim St(Re) context lines."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pypdf

OUT = Path("artifacts/karman_re100/refs")
OUT.mkdir(parents=True, exist_ok=True)

CANDS = {
    "roshko1953": "https://authors.library.caltech.edu/records/xk7z3-0vz37/files/19930092271.pdf",
    "roshko1953_alt": "https://authors.library.caltech.edu/428/1/ROSkondot.pdf",
    "giacomello2015": "https://arxiv.org/pdf/1708.04333",
    "jfm2024_wakes": "https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E0640790BFBFEC9619A20A7BECD4D2B3/S0022112024006153a.pdf/div-class-title-on-the-absence-of-a-secondary-vortex-street-in-three-dimensional-and-turbulent-cylinder-wakes-div.pdf",
}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "TensorLBM-ref-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


PAT = re.compile(r"(0\.1[4-8]\d?)|(St\b)|(Strouhal)", re.I)

for name, url in CANDS.items():
    pdf = OUT / f"{name}.pdf"
    try:
        if not pdf.exists():
            pdf.write_bytes(fetch(url))
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {str(e)[:90]}")
        continue
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    try:
        reader = pypdf.PdfReader(str(pdf))
    except Exception as e:  # noqa: BLE001
        print(f"PDFPARSE FAIL {name}: {str(e)[:80]}")
        continue
    print(f"\n===== {name} sha256={sha[:16]} pages={len(reader.pages)} size={pdf.stat().st_size}")
    hits = 0
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            continue
        flat = re.sub(r"\s+", " ", text)
        for m in re.finditer(r"0\.1[4-8]\d", flat):
            ctx = flat[max(0, m.start() - 200): m.start() + 150]
            if re.search(r"\b(100|Re)\b", ctx):
                hits += 1
                if hits <= 6:
                    print(f"  p{i} {m.group(0)} :: {ctx[:330]}")
    print(f"  total 0.14-0.18 hits with Re context: {hits}")