"""Machine extraction of reference values from downloaded primary sources.

1) Fornberg 1980 (author-hosted scan, OCR text layer) - inline Cd statements.
2) Gautier/Biau/Lamballais 2013 (arXiv:1310.6641 LaTeX source) - literature
   comparison table + their own reference solution + domain sensitivity table.

Outputs: ref/extracted_primary.json (consumed by ref_lock builder).
No hand-typed literature numbers: every value below is parsed from files.
"""

from __future__ import annotations

import json
import re

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"


def norm_num(s: str) -> str:
    return s.replace("'", ".").replace("’", ".").replace(" ", "")


def extract_fornberg() -> dict:
    txt = open(f"{BASE}/ref/fornberg1980_fulltext.txt").read()
    # inline statements: 'at Re = 20, Cn ~ 2.0001 :t 0.0002,'
    pats = re.findall(
        r"at Re\s*=\s*(\d+),?\s*Cn\s*[~=]\s*([\d.'’ ]+?)\s*:t\s*([\d.'’ ]+?),",
        txt,
    )
    cd = {}
    for re_s, v, e in pats:
        cd[int(re_s)] = {"cd": float(norm_num(v)), "pm": float(norm_num(e))}
    # Dennis (1976) quote: 'Cn 1.998 and Cn = 1.494'
    m = re.search(r"Dennis \(1976\).*?\(Cn\s*(\d\.\d+)[^\d]*and\s*Cn\s*=?\s*(\d\.\d+)", txt, re.S)
    dennis = None
    if m:
        dennis = {"re20_cd": float(m.group(1)), "re40_cd": float(m.group(2))}
    # normalization statement (drag definition): D = rho U^2 a C_D
    m2 = re.search(r"D\s*=\s*pU[\"a]?a?CD|D =pU[\"a]*aCD", txt)
    return {
        "source": "Fornberg 1980, JFM 98(4):819-855, doi:10.1017/S0022112080000419",
        "file": "pdf/fornberg1980.pdf (author-hosted: colorado.edu/amath attached-files jcp_80_fl_p_cyl.pdf)",
        "text_dump": "ref/fornberg1980_fulltext.txt",
        "cd_inline": cd,
        "dennis1976_quote": dennis,
        "norm_statement_found": bool(m2),
    }


def extract_gautier() -> dict:
    tex = open(f"{BASE}/ref/refsol_src/CF2012.tex").read()
    # ---- Table (tab_2): literature comparison at Re=40 ----
    i0 = tex.index("Physical parameters of the flow pattern")
    seg = tex[i0 : tex.index("\\end{table}", i0)]
    rows = {}
    for line in seg.splitlines():
        s = line.strip()
        s = re.sub(r"\\cite\{[^}]*\}", "", s).replace("\\&", "@AMP@")
        if not s.endswith("\\\\") or "&" not in s:
            continue
        cells = []
        for c in s.split("&"):
            c = c.replace("$^\\circ$", "").replace("^\\circ", "").replace("@AMP@", "&")
            c = re.sub(r"[{}\\]", "", c).strip()
            cells.append(c)
        if len(cells) < 4 or not cells[0] or not cells[1]:
            continue

        def fnum(c):
            c = c.strip().rstrip("°")
            return float(c) if re.fullmatch(r"[\d.]+", c) else None

        if fnum(cells[1]) is None:
            continue
        rows[re.sub(r"\s+", " ", cells[0])] = {
            "cd": fnum(cells[1]),
            "theta_s": fnum(cells[2]) if len(cells) > 2 else None,
            "lw": fnum(cells[3]) if len(cells) > 3 else None,
            "a": fnum(cells[4]) if len(cells) > 4 else None,
            "b": fnum(cells[5]) if len(cells) > 5 else None,
        }
    # ---- Table (tab_1): own domain/resolution sensitivity ----
    i1 = tex.index("for various grid resolutions and domain sizes")
    seg1 = tex[i1 : tex.index("\\end{table}", i1)]
    nums = re.findall(r"[\d.]+", seg1)
    # rows: r_inf 30 40 50 30 40 50 ; CD ..; Lw ..; theta_s ..
    m = re.search(
        r"\$C_D\$\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
        seg1,
    )
    cdtab = [float(x) for x in m.groups()] if m else None
    m = re.search(
        r"\$L_w/D\$\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
        seg1,
    )
    lwtab = [float(x) for x in m.groups()] if m else None
    m = re.search(
        r"\$\\theta_s\$\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
        seg1,
    )
    tstab = [float(x) for x in m.groups()] if m else None
    return {
        "source": "Gautier, Biau & Lamballais 2013, arXiv:1310.6641 (submitted to Comput. Fluids)",
        "file": "pdf/refsol1310.6641.pdf + LaTeX source ref/refsol_src/CF2012.tex",
        "comparison_table_re40": rows,
        "own_sensitivity": {
            "grid": ["coarse Nr100 Nth512", "fine Nr200 Nth1024"],
            "r_inf_values": [30, 40, 50],
            "cd": cdtab,
            "lw": lwtab,
            "theta_s": tstab,
        },
    }


def main() -> None:
    out = {"fornberg1980": extract_fornberg(), "gautier2013": extract_gautier()}
    path = f"{BASE}/ref/extracted_primary.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False)[:3000])
    print("written", path)


if __name__ == "__main__":
    main()
