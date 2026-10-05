#!/usr/bin/env python3
"""Richardson extrapolation + gate evaluation — protocol v2 (2026-09-29/30).

Controller verdict on v1: best-SSE FREE-alpha scan is not defensible on a slow
convergence tail (alpha pinned at the 0.30 scan floor on 5/6 points; vf0.7
q_inf=+7.7% boundary artifact). Protocol v2 (estimator bug fix, gate 0.5%
UNCHANGED): three estimators reported side by side, gate passes only if ALL
THREE satisfy |err| <= 0.5%:
  - fixed10: q_N = q_inf + C N^-1.0, 2-param LSQ, windows {finest 4,5,6},
    point = WORST window (max |err| across windows; kills window shopping)
  - fixed15: same with alpha = 1.5
  - aitken3: classic Aitken d^2 on the three finest rungs
Free-alpha fit is kept as a DIAGNOSTIC ONLY (with alpha_at_scan_boundary
flag); it never enters the gate. Bands reported (SSE-doubling for free-alpha;
window spread + subleading for fixed; correction size for Aitken) but do not
gate. The fit machinery never sees reference values.

Usage: extrapolate.py sq <out.json>   (Sangani gate)
       extrapolate.py hex <out.json>  (no truth; q = K_s_over_R2)
Reads out/cases/*_schur.json written by run_case.py.
"""

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

SANGANI = {
    0.05: 15.56,
    0.10: 24.83,
    0.20: 51.53,
    0.30: 102.90,
    0.40: 217.89,
    0.50: 532.55,
    0.60: 1.763e3,
    0.70: 1.352e4,
    0.75: 1.263e5,
}

ALPHA_GRID = np.arange(0.30, 4.001, 0.005)
WINDOWS = (4, 5, 6)


def load_ladder(pattern):
    cases = []
    for p in sorted(glob.glob(os.path.join(OUT, "cases", pattern))):
        m = re.search(r"vf([\d.]+)_nx(\d+)_ny(\d+)_", os.path.basename(p))
        d = json.load(open(p))
        cases.append(
            dict(
                vf=float(m.group(1)),
                nx=int(m.group(2)),
                ny=int(m.group(3)),
                f=d["f_press"],
                K=d["K_s_over_R2"],
                phi_a=d["phi_actual"],
                div=d["div_max"],
                cg=d["cg_res"],
            )
        )
    return cases


def fit_fixed(Ns, qs, alpha, k):
    """2-param fixed-alpha LSQ on the finest k rungs."""
    N = np.asarray(Ns[-k:], float)
    q = np.asarray(qs[-k:], float)
    x = N ** (-alpha)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    sse = float(np.sum((A @ coef - q) ** 2))
    return dict(q_inf=float(coef[0]), C=float(coef[1]), sse=sse)


def fit_free(Ns, qs, k):
    """Diagnostic 3-param free-alpha best-SSE fit (NOT gate-eligible)."""
    best = None
    for alpha in ALPHA_GRID:
        f = fit_fixed(Ns[-k:], qs[-k:], float(alpha), k)
        if best is None or f["sse"] < best["sse"]:
            best = f
            best["alpha"] = float(alpha)
    return best


def aitken3(qs):
    """Classic Aitken delta^2 on the three finest rungs."""
    q1, q2, q3 = qs[-3], qs[-2], qs[-1]
    d1, d2 = q2 - q1, q3 - q2
    denom = d2 - d1
    if denom == 0:
        return None
    corr = d2 * d2 / denom
    return dict(q_inf=float(q3 - corr), correction=float(abs(corr)))


def extrapolate(Ns, qs):
    """Three-estimator protocol v2. Truth-blind; returns estimator table."""
    out = {"rungs": [dict(N=int(n), q=float(q)) for n, q in zip(Ns, qs)]}

    for tag, alpha in (("fixed10", 1.0), ("fixed15", 1.5)):
        wins = {}
        for k in WINDOWS:
            if len(Ns) >= k:
                f = fit_fixed(Ns, qs, alpha, k)
                wins[f"win{k}"] = dict(q_inf=f["q_inf"], C=f["C"], sse=f["sse"])
        if wins:
            qinfs = [w["q_inf"] for w in wins.values()]
            e = out.setdefault(tag, {})
            e.update(wins)
            e["q_inf_median"] = float(np.median(qinfs))
            e["spread"] = float(max(qinfs) - min(qinfs))
            k4 = wins.get("win4")
            if k4:
                lead = abs(k4["C"] * (float(Ns[-4:][0] if len(Ns) >= 4 else Ns[-1])) ** (-alpha))
                e["band"] = float(max(e["spread"], 0.25 * lead))

    a3 = aitken3(qs)
    if a3:
        out["aitken3"] = dict(q_inf=a3["q_inf"], band=a3["correction"])

    free = fit_free(Ns, qs, 4)
    out["free_alpha_diag"] = dict(
        alpha=free["alpha"],
        q_inf=free["q_inf"],
        alpha_at_scan_boundary=bool(
            free["alpha"] <= ALPHA_GRID[0] + 0.005 or free["alpha"] >= ALPHA_GRID[-1] - 0.005
        ),
    )
    # SSE-doubling band on the free fit (reported only)
    qinfs_a = []
    for alpha in ALPHA_GRID:
        f = fit_fixed(Ns[-4:], qs[-4:], float(alpha), 4)
        if f["sse"] <= 2.0 * max(free["sse"], 1e-300):
            qinfs_a.append(f["q_inf"])
    out["free_alpha_diag"]["band_sse_doubling"] = (
        float(max(qinfs_a) - min(qinfs_a)) if qinfs_a else 0.0
    )
    return out


def est_points(rec):
    """(name, q_inf, band) triples for the three gate-eligible estimators."""
    pts = []
    for tag in ("fixed10", "fixed15"):
        if tag in rec:
            # worst-window point: the window q_inf FARTHEST from the window median
            wins = [v["q_inf"] for k, v in rec[tag].items() if k.startswith("win")]
            med = float(np.median(wins))
            worst = max(wins, key=lambda z: abs(z - med))
            pts.append((tag, worst, rec[tag]["band"]))
    if "aitken3" in rec:
        pts.append(("aitken3", rec["aitken3"]["q_inf"], rec["aitken3"]["band"]))
    return pts


def main():
    mode = sys.argv[1]
    result = {}
    if mode == "sq":
        cases = load_ladder("sq_vf*_schur.json")
        by_vf = {}
        for c in cases:
            by_vf.setdefault(c["vf"], []).append(c)
        for vf in sorted(by_vf):
            cs = sorted(by_vf[vf], key=lambda c: c["nx"])
            Ns = [c["nx"] for c in cs]
            qs = [c["f"] for c in cs]
            r = extrapolate(Ns, qs)
            if vf in SANGANI:
                ref = SANGANI[vf]
                r["ref_f"] = ref
                r["raw_err_pct"] = [100.0 * (c["f"] / ref - 1.0) for c in cs]
                errs = {}
                for name, q, band in est_points(r):
                    errs[name] = dict(
                        q_inf=q, err_pct=100.0 * (q / ref - 1.0), band_pct=100.0 * band / ref
                    )
                r["est_errs"] = errs
                r["gate_pass_3est"] = bool(
                    errs and all(abs(e["err_pct"]) <= 0.5 for e in errs.values())
                )
                r["free_err_pct"] = 100.0 * (r["free_alpha_diag"]["q_inf"] / ref - 1.0)
            result[f"vf{vf:g}"] = r
    elif mode == "hex":
        for pat, key in (
            ("hexA_vf0.7_*_schur.json", "hexA_vf0.70"),
            ("hexA_vf0.75_*_schur.json", "hexA_vf0.75"),
            ("hexB_vf0.7_*_schur.json", "hexB_vf0.70"),
            ("hexB_vf0.75_*_schur.json", "hexB_vf0.75"),
        ):
            cases = load_ladder(pat)
            if not cases:
                continue
            cs = sorted(cases, key=lambda c: c["nx"])
            r = extrapolate([c["nx"] for c in cs], [c["K"] for c in cs])
            r["phi_actual_rungs"] = [c["phi_a"] for c in cs]
            result[key] = r
    dest = sys.argv[2]
    with open(dest, "w") as f:
        json.dump(result, f, indent=1)
    # compact stdout table
    for k, v in result.items():
        if "est_errs" in v:
            row = " ".join(f"{n}={e['err_pct']:+.3f}%" for n, e in v["est_errs"].items())
            print(
                f"{k}: {row} free(a={v['free_alpha_diag']['alpha']:.2f})"
                f"={v.get('free_err_pct', float('nan')):+.3f}%"
                f" -> {'PASS' if v['gate_pass_3est'] else 'FAIL'}"
            )
        else:
            print(f"{k}: q_win4={v.get('fixed10', {}).get('win4', {}).get('q_inf', float('nan'))}")


if __name__ == "__main__":
    main()
