import sys

import pdfplumber

path = sys.argv[1]
with pdfplumber.open(path) as pdf:
    n = len(pdf.pages)
    print("pages:", n)
    for i in range(min(3, n)):
        t = pdf.pages[i].extract_text() or ""
        print(f"--- page {i}: {len(t)} chars")
        print(t[:400])
    # find pages mentioning 'Table' with numbers
    hits = []
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ""
        if "drag" in t.lower() and ("Table" in t or "C D" in t or "CD" in t):
            hits.append(i)
    print("table-ish pages:", hits[:30])
