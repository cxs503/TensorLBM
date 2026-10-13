import re

import pdfplumber

path = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/pdf/fornberg1980.pdf"
out = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/ref/fornberg1980_fulltext.txt"
with pdfplumber.open(path) as pdf:
    with open(out, "w") as fh:
        for i, p in enumerate(pdf.pages):
            t = p.extract_text() or ""
            fh.write(f"\n===== PAGE {i} =====\n{t}\n")
print("written", out)

txt = open(out).read()
# locate candidate numeric rows: lines containing several decimals like 1.49xx or 1.7xxx
for m in re.finditer(r"^.*1\.[4-8]\d\d.*$", txt, re.M):
    print(m.group(0)[:160])
