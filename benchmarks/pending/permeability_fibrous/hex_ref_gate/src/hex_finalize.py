#!/usr/bin/env python3
"""Hex finalization: aspect-correct the 97/56 extrapolation to ideal sqrt(3),
normalize to ideal r^2, compare with Gebart, FREEZE hex_final.json.

Protocol v2 (prereg §11.4, appended 2026-09-30): estimator upgraded from the
v1 free-alpha best-SSE scan (indefensible on slow tails per controller sq
verdict) to three estimators — fixed-alpha LSQ (1.0 / 1.5, windows = finest
4/5/6, point = window MEDIAN) + Aitken3 (last three rungs). Final value =
median of the three estimator points. Bands: headline = sum of fractional
components (three-point spread + max estimator band + aspect-slope
uncertainty) per §11.4; band_max = component max (sq-side §6 convention),
both reported.

Run only after the sq gate passes and hex ladders are complete (prereg §7/§11).
"""

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "out")

from extrapolate import aitken3, fit_fixed, fit_free  # noqa: E402

SQ3 = math.sqrt(3.0)
A_LAT = 97.0 / 56.0
A_PRIME = 26.0 / 15.0
GEBART = {0.70: 1.641231e-3, 0.75: 7.239255e-4}


def load(pattern, qkey):
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, "cases", pattern))):
        d = json.load(open(p))
        out.append((d["nx"], d[qkey]))
    return sorted(out)


def three_est_points(pairs):
    """{name: q_inf} + {name: band} for fixed10/fixed15/aitken3 (no truth)."""
    Ns = [float(n) for n, _ in pairs]
    qs = [float(q) for _, q in pairs]
    pts, bands, detail = {}, {}, {}
    for tag, alpha in (("fixed10", 1.0), ("fixed15", 1.5)):
        wins = {}
        for k in (4, 5, 6):
            if len(pairs) >= k:
                f = fit_fixed(Ns, qs, alpha, k)
                wins[f"win{k}"] = f["q_inf"]
        if wins:
            qinfs = list(wins.values())
            pts[tag] = float(np.median(qinfs))
            spread = float(max(qinfs) - min(qinfs))
            f4 = fit_fixed(Ns, qs, alpha, min(4, len(pairs)))
            lead = abs(f4["C"] * Ns[-min(4, len(pairs))] ** (-alpha))
            bands[tag] = float(max(spread, 0.25 * lead))
            detail[tag] = dict(windows=wins, median=pts[tag], band=bands[tag])
    a3 = aitken3(qs)
    if a3:
        pts["aitken3"] = a3["q_inf"]
        bands["aitken3"] = a3["correction"]
        detail["aitken3"] = dict(q_inf=a3["q_inf"], band=a3["correction"])
    free = fit_free(Ns, qs, min(4, len(pairs)))
    return pts, bands, detail, free


def interp_ln_at(ladder, nx_target):
    """Interpolate ln q vs 1/nx at nx_target (between rungs)."""
    x = np.array([1.0 / n for n, _ in ladder])
    y = np.array([math.log(q) for _, q in ladder])
    order = np.argsort(x)
    return float(np.interp(1.0 / nx_target, x[order], y[order]))


def main():
    res = {}
    for vf in (0.70, 0.75):
        A_lat = load(f"hexA_vf{vf:g}_*_schur.json", "K_s_over_R2")
        B_lat = load(f"hexB_vf{vf:g}_*_schur.json", "K_s_over_R2")
        pts, bands, detail, free = three_est_points(A_lat)
        q_vals = sorted(pts.values())
        q_inf = float(np.median(q_vals))  # median of the three estimator points
        three_spread = float(max(q_vals) - min(q_vals))
        band_lattice_max = float(max([three_spread] + list(bands.values())))

        # aspect slope from B vs A at matched nx (lnK ratio over dA)
        slopes = []
        for nxB, qB in B_lat:
            lnA = interp_ln_at(A_lat, nxB)
            slopes.append((math.log(qB) - lnA) / (A_PRIME - A_LAT))
        slope = float(np.mean(slopes))
        slope_resid = float(max(abs(s - slope) for s in slopes)) if len(slopes) > 1 else 0.0
        dA = SQ3 - A_LAT  # -9.2049e-5
        ln_final = math.log(q_inf) + slope * dA
        norm = A_LAT / SQ3  # r_latt^2/r_ideal^2 (exact)
        K_ideal = math.exp(ln_final) * norm

        band_frac_sum = (
            three_spread / abs(q_inf) + band_lattice_max / abs(q_inf) + abs(slope_resid * dA)
        )
        band_frac_max = max(three_spread, band_lattice_max) / abs(q_inf)
        band_sum = K_ideal * band_frac_sum
        band_max = K_ideal * max(band_frac_max, abs(slope_resid * dA))

        res[f"vf{vf:g}"] = dict(
            hexA_ladder=[{"nx": n, "K": q} for n, q in A_lat],
            hexB_ladder=[{"nx": n, "K": q} for n, q in B_lat],
            estimators=detail,
            free_alpha_diag=dict(alpha=free["alpha"], q_inf=free["q_inf"]),
            q_inf_lattice=q_inf,
            band_lattice_max=band_lattice_max,
            slope_dlnK_dA=slope,
            slope_rung_spread=slope_resid,
            dA_to_sqrt3=dA,
            norm_factor=norm,
            K_s_over_rideal2=K_ideal,
            band=band_sum,
            band_convention="sum per prereg §11.4",
            band_max=band_max,
            K_i_over_rideal2=K_ideal / (1.0 - vf),
            gebart_K_over_r2=GEBART[vf],
            dev_vs_gebart=K_ideal / GEBART[vf] - 1.0,
        )
    dest = os.path.join(OUT, "hex_final.json")
    with open(dest, "w") as f:
        json.dump(res, f, indent=1)
    print(
        json.dumps(
            {
                k: {
                    "K_s/r2_ideal": v["K_s_over_rideal2"],
                    "band_sum": v["band"],
                    "band_max": v["band_max"],
                    "dev_vs_gebart": v["dev_vs_gebart"],
                }
                for k, v in res.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
