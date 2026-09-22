#!/usr/bin/env python3
"""W6-B reference lock: vector-object extraction of Fig. 6 (top) from
Forslund et al. 2023, Transp. Porous Med. 148:545-569,
DOI 10.1007/s11242-023-01966-w (open access).

Extracts the (Re_Darcy, y) marker centroids of both data series
  series A (black x)   = Koch & Ladd 1997 as transcribed by Forslund et al.
  series B (red x)     = Forslund et al. FDM computation
from the vector layer of the OA PDF.  Axis calibration uses the vector
tick marks; tick label values were verified character-by-character on a
10x render (x: 0/50/100/150/200; y: 225/250/275/300/325/350/375).

Each x-marker glyph is two exact diagonals of a square; the data point is
the diagonal intersection == bounding-box centre (verified pointwise).

Usage:  python3 extract_ref_fig6.py  <tipm2023.pdf>  <out.json>
Requires PyMuPDF (local machine only; the server venv has no fitz).
"""

import json
import sys

import fitz  # PyMuPDF


def main() -> None:
    pdf, out = sys.argv[1], sys.argv[2]
    doc = fitz.open(pdf)
    page = doc[12]  # printed page 563; contains Fig. 6

    # --- axis calibration (vector tick marks, values verified at 10x) -----
    XT = {220.00: 0.0, 260.73: 50.0, 301.45: 100.0, 342.17: 150.0, 382.90: 200.0}
    YT = {
        74.08: 375.0,
        91.87: 350.0,
        109.69: 325.0,
        127.49: 300.0,
        145.31: 275.0,
        163.10: 250.0,
        180.93: 225.0,
    }

    def lsq_scale(ticks: dict) -> tuple:
        pts = sorted(ticks)
        n = len(pts)
        sx = sum(pts) / n
        sy = sum(ticks[p] for p in pts) / n
        sxy = sum((p - sx) * (ticks[p] - sy) for p in pts)
        sxx = sum((p - sx) ** 2 for p in pts)
        m = sxy / sxx
        return m, sy - m * sx  # value = m * pt + c

    mx, cx = lsq_scale(XT)
    my, cy = lsq_scale(YT)
    print(f"x calibration: value = {mx:.6f}*pt + {cx:.4f}")
    print(f"y calibration: value = {my:.6f}*pt + {cy:.4f}")

    # --- marker extraction ------------------------------------------------
    markers = {"A_koch_ladd_1997": [], "B_fdm_2023": []}
    for dr in page.get_drawings():
        r = dr["rect"]
        if not (4.5 < r.width < 5.7 and 4.5 < r.height < 5.7):
            continue
        if r.x0 < 212 or r.y1 > 196:  # keep plot frame interior only
            continue
        col = dr.get("color") or dr.get("fill")
        key = None
        if col == (0.0, 0.0, 0.0):
            key = "A_koch_ladd_1997"
        elif col == (1.0, 0.0, 0.0):
            key = "B_fdm_2023"
        if key:
            cxp, cyp = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
            markers[key].append((round(cxp, 3), round(cyp, 3)))
    for k in markers:
        uniq = []
        for m in sorted(markers[k]):
            if not any(abs(m[0] - u[0]) < 0.5 and abs(m[1] - u[1]) < 0.5 for u in uniq):
                uniq.append(m)
        markers[k] = uniq

    # legend glyphs: the two markers sharing x = 227.72 pt (stacked at the
    # legend text column) are legend samples, not data.
    for k in markers:
        xs = [m[0] for m in markers[k]]
        if xs.count(round(sorted(xs)[0], 2)) or True:
            from collections import Counter
        cnt = Counter(round(m[0], 1) for m in markers[k])
        legend_x = {x for x, c in cnt.items() if c >= 1 and abs(x - 227.7) < 0.2}
        markers[k] = [m for m in markers[k] if round(m[0], 1) not in legend_x]

    def convert(pts):
        tab = []
        for px, py in pts:
            tab.append(
                {"pt": [px, py], "re_darcy": round(mx * px + cx, 3), "y": round(my * py + cy, 3)}
            )
        return tab

    tabA = convert(markers["A_koch_ladd_1997"])
    tabB = convert(markers["B_fdm_ladd" if False else "B_fdm_2023"])

    # self-normalisation: R = y / y0 of the same series (Re->0 marker)
    y0A = min(tabA, key=lambda r: abs(r["re_darcy"]))["y"]
    y0B = min(tabB, key=lambda r: abs(r["re_darcy"]))["y"]
    for t in tabA:
        t["R"] = round(t["y"] / y0A, 6)
    for t in tabB:
        t["R"] = round(t["y"] / y0B, 6)

    result = {
        "source": {
            "paper": "Forslund, Larsson, Hellstrom, Lundstrom 2023, "
            "Steady-State Transitions in Ordered Porous Media, "
            "Transp Porous Med 148:545-569",
            "doi": "10.1007/s11242-023-01966-w",
            "figure": "Fig. 6 (top)",
            "series_A": "Koch & Ladd 1997 JFM 349:31-66 as transcribed by Forslund et al.",
            "series_B": "Forslund et al. 2023 FDM (artificial compressibility, staircase)",
            "geometry": "square array of circular cylinders, single periodic cell, "
            "phi_s = 0.40 (adjudicated: y-intercept matches Sangani-Acrivos "
            "f(0.40) = 217.89 to 0.26%; see NOTES section 2)",
            "y_axis_semantics": "numerically the f-form G*L^2/(mu*U_int) per the "
            "Stokes cross-check; the axis label reads F_delta/(mu*U_Darcy)",
            "x_axis_semantics": "Re_Darcy = U_Darcy*D_p/nu",
        },
        "calibration": {
            "x_ticks_pt": XT,
            "y_ticks_pt": YT,
            "x_units_per_pt": round(mx, 6),
            "x_offset": round(cx, 4),
            "y_units_per_pt": round(my, 6),
            "y_offset": round(cy, 4),
        },
        "A_koch_ladd_1997": {"y0_stokes": y0A, "points": tabA},
        "B_fdm_2023": {"y0_stokes": y0B, "points": tabB},
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nseries A (K&L97 transcribed): {len(tabA)} points, y0 = {y0A:.2f}")
    for t in tabA:
        print(f"  Re={t['re_darcy']:8.2f}  y={t['y']:7.2f}  R={t['R']:.6f}")
    print(f"\nseries B (FDM): {len(tabB)} points, y0 = {y0B:.2f}")
    for t in tabB:
        print(f"  Re={t['re_darcy']:8.2f}  y={t['y']:7.2f}  R={t['R']:.6f}")


if __name__ == "__main__":
    main()
