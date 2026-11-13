#!/usr/bin/env python3
"""W26 / mission m17569 — three-grid GCI + observed order (Celik et al. 2008) and tau x N sign surface.

What is computed here
---------------------
1. **GCI / observed order** for the fixed-Reynolds ladder: the three grids run at
   *identical* lattice Reynolds number (so tau is not a free parameter and does not
   confound the grid study).  The procedure is the ASME/Celik three-grid
   procedure, applied to C_T, C_F and C_P:

       R      = (f2 - f1) / (f3 - f2)                     monotonicity ratio
       p_obs  = ln|(f3 - f2)/(f2 - f1)| / ln(r)           observed order (r = h1/h2 = N2/N1)
       f_ex   = (r^p f1 - f2) / (r^p - 1)                 Richardson extrapolation
       e_a21  = |(f1 - f2) / f1|                          approximate rel. error (medium->fine)
       e_ext  = |(f_ex - f1) / f_ex|                      extrapolated rel. error
       GCI_1  = 1.25 * e_a21 / (r^p - 1)                  fine-grid GCI (Fs = 1.25, three grids)

   f1 = finest, f2 = medium, f3 = coarsest.  Everything is read from the case
   JSONs and re-derived from the raw time series where the case record exposes it,
   so a transcription error in the summary JSON cannot silently propagate.

2. **Negative-drag / negative-force (tau, N) surface** from the scan cases, with the
   definition and the *measured* time-window drift attached to every cell.  A cell
   whose drift exceeds the declared steady-state tolerance is reported as
   ``UNSTEADY`` rather than being quietly counted as a sign determination.

3. All the numbers this mission quotes live in ``gci_sign_surface.json``; the
   artifact builder only copies from it.

Usage:
    python3 build_gci.py --cases <dir with case_*.json> [--scan <dir>] --out gci_sign_surface.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[3]

DRIFT_TOL = 0.02  # 2 % window drift = "statistically steady" for sign purposes
GCI_TARGET = 0.05  # criterion 3
P_TARGET = 1.8  # criterion 3
REPORTED_P = 2.0  # order used when the observed order is unusable (declared fallback)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_case(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    ts = Path(d["raw_npz"])
    with np.load(ts) as z:
        step = z["step"]
        cd_t = z["cd_total_raw"]
        cd_f = z["cd_friction_raw"]
        cd_p = z["cd_pressure_raw"]
        scale = float(z["dpS_front"]) / float(z["dpS_wet"])
    d["_ts"] = {"step": step, "cd_t": cd_t, "cd_f": cd_f, "cd_p": cd_p, "scale": scale,
                "n_samples": int(len(step))}
    return d


def find_log(tag: str) -> Path | None:
    """Locate the detached-run log of a case (two layouts are possible)."""
    cands = list(REPO.glob(f".scratch/**/_logs/{tag}.log")) + list(REPO.glob(f".scratch/**/{tag}.log"))
    return cands[0] if cands else None


def solver_route(tag: str) -> str:
    p = find_log(tag)
    if p is None:
        return "UNKNOWN"
    txt = p.read_text(errors="replace")
    return "wall_function" if "wall-function path" in txt else "standard_compiled_step"


def window_mean(arr: np.ndarray, n_win: int) -> float:
    return float(arr[-n_win:].mean())


def gci_triplet(f1: float, f2: float, f3: float, r: float) -> dict:
    """ASME three-grid GCI (Celik et al. 2008). f1 finest, f3 coarsest."""
    out = {"f1_fine": f1, "f2_medium": f2, "f3_coarse": f3, "r": r}
    d21 = f2 - f1
    d32 = f3 - f2
    if d21 == 0 or d32 == 0:
        out.update({"observed_order": None, "status": "DEGENERATE_ZERO_DIFFERENCE"})
        return out
    ratio = d32 / d21
    out["convergence_ratio_R"] = ratio
    if ratio <= 0:
        out["status"] = "OSCILLATORY_DIVERGENT"  # non-monotone: GCI undefined
        out["observed_order"] = None
        return out
    p = abs(math.log(abs(ratio)) / math.log(r))
    out["observed_order"] = p
    if p <= 0.05:
        out["status"] = "ORDER_TOO_SMALL"
        return out
    rp = r**p
    f_ex = (rp * f1 - f2) / (rp - 1.0)
    e_a21 = abs((f1 - f2) / f1) if f1 != 0 else float("inf")
    e_ext = abs((f_ex - f1) / f_ex) if f_ex != 0 else float("inf")
    gci_fine = 1.25 * e_a21 / (rp - 1.0)
    out.update({
        "f_exact_richardson": f_ex,
        "e_a21_rel": e_a21,
        "e_ext_rel": e_ext,
        "gci_fine_pct": 100.0 * gci_fine,
        "gci_medium_pct": 100.0 * 1.25 * abs((f2 - f1) / f2) / (rp - 1.0) * r**p,
        "status": "CONVERGED_MONOTONE",
        "asymptotic_range_ok": bool(0.5 <= p <= 3.5),
    })
    # declared fallback: fixed observed order (first-order / second-order) reported alongside
    for p_fix in (1.0, 2.0):
        rpf = r**p_fix
        out[f"gci_fine_pct_p{p_fix:g}"] = 100.0 * 1.25 * abs((f1 - f2) / f1) / (rpf - 1.0)
        out[f"f_exact_p{p_fix:g}"] = (rpf * f1 - f2) / (rpf - 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="dir with ladder case_*.json")
    ap.add_argument("--scan", default=None, help="dir with scan case_*.json")
    ap.add_argument("--out", default=str(_HERE / "gci_sign_surface.json"))
    args = ap.parse_args()

    # ---------- 1. ladder -------------------------------------------------
    ladder_paths = sorted(glob.glob(str(Path(args.cases) / "case_*.json")))
    ladder = [load_case(Path(p)) for p in ladder_paths]
    if len(ladder) < 2:
        raise SystemExit(f"need >=2 ladder cases, found {len(ladder)}")
    two_grid_only = len(ladder) == 2

    # order by resolution; use the *measured* evaluation window of each case
    ladder.sort(key=lambda d: d["N"])
    per_case = []
    for d in ladder:
        n_win = min(40, d["_ts"]["n_samples"])
        sc = d["_ts"]["scale"]
        ct = window_mean(d["_ts"]["cd_t"], n_win) * sc
        cf = window_mean(d["_ts"]["cd_f"], n_win) * sc
        cp = window_mean(d["_ts"]["cd_p"], n_win) * sc
        per_case.append({
            "tag": d["tag"], "N": d["N"], "cells": d["cells"], "steps": d["steps_done"],
            "passes": d["passes"], "Re_lattice": d["Re_lattice_measured"],
            "tau": d["tau_achieved"], "nu_lb": d["nu_lb"],
            "C_T": ct, "C_F": cf, "C_P": cp,
            "C_T_sum_check_rel_dev": abs((cf + cp) - ct) / abs(ct),
            "wind_mean_recomputed_vs_case_json_rel_dev": abs(ct - d["C_T"]) / abs(d["C_T"]),
            "C_T_window_drift": d["C_T_window_drift"],
            "C_F_window_drift": d["C_F_window_drift"],
            "C_T_block_means": d["C_T_block_means"],
            "field_finite": d["field_finite_at_end"],
            "solver_route": solver_route(d["tag"]),
            "wall_treatment_effective": d.get("wall_treatment_effective"),
            "case_json": str(Path(d["raw_npz"]).parent / f"case_{d['tag']}.json"),
            "case_json_sha256": sha256_file(Path(d["raw_npz"]).parent / f"case_{d['tag']}.json"),
            "timeseries_sha256": d["raw_npz_sha256"],
            "wall_clock": d["wall_clock"],
        })

    r_ratio = [(per_case[i + 1]["N"] / per_case[i]["N"]) for i in range(len(per_case) - 1)]
    gci = {
        "procedure": "ASME V&V 20 / Celik et al. (2008) three-grid GCI, fine-grid value with Fs=1.25",
        "refinement_ratios": r_ratio,
        "r_is_uniform": all(abs(x - r_ratio[0]) <= 1e-12 for x in r_ratio),
        "quantities": {},
    }
    for q in ("C_T", "C_F", "C_P"):
        f1 = per_case[-1][q]  # finest
        f2 = per_case[-2][q]
        if two_grid_only:
            gci["quantities"][q] = {
                "status": "TWO_GRID_ONLY",
                "f1_fine": f1, "f2_medium": f2,
                "pair_rel_change": abs((f1 - f2) / f1),
                "note": "observed order and GCI need three grids; only the fine/medium pair change is "
                        "reported here so nothing is extrapolated from two points",
            }
            continue
        f3 = per_case[-3][q]
        gci["quantities"][q] = gci_triplet(f1, f2, f3, r_ratio[-1])
    gci["steady_state_check"] = {
        "declared_tolerance_pct": 100.0 * DRIFT_TOL,
        "per_case_drift_pct": {c["tag"]: 100.0 * c["C_T_window_drift"] for c in per_case},
        "all_cases_below_tolerance": all(c["C_T_window_drift"] <= DRIFT_TOL for c in per_case),
        "note": "GCI is only meaningful on a statistically steady solution; the drift column is "
                "reported so the reader can discount the number if it is not",
    }
    gci["cases"] = per_case

    gci["two_grid_only"] = two_grid_only
    p_obs = gci["quantities"]["C_T"].get("observed_order")
    gci["criteria"] = {
        "GCI_below_5pct": bool(gci["quantities"]["C_T"].get("gci_fine_pct", 1e9) < 100 * GCI_TARGET),
        "observed_order_ge_1p8": bool(p_obs is not None and p_obs >= P_TARGET),
        "observed_order": p_obs,
        "gci_fine_pct": gci["quantities"]["C_T"].get("gci_fine_pct"),
        "gci_basis": "C_T (total resistance coefficient of the bare hull)",
        "verdict_basis": "both sub-criteria must hold on C_T",
        "pass": bool(gci["quantities"]["C_T"].get("gci_fine_pct", 1e9) < 100 * GCI_TARGET
                     and p_obs is not None and p_obs >= P_TARGET),
    }

    # ---------- 2. tau x N sign surface -----------------------------------
    surface = None
    if args.scan:
        scan_paths = sorted(glob.glob(str(Path(args.scan) / "case_*.json")))
        scan = [load_case(Path(p)) for p in scan_paths]
        cells = []
        for d in scan:
            n_win = min(40, d["_ts"]["n_samples"])
            sc = d["_ts"]["scale"]
            ct = window_mean(d["_ts"]["cd_t"], n_win) * sc
            cf = window_mean(d["_ts"]["cd_f"], n_win) * sc
            steady = (d["C_T_window_drift"] <= DRIFT_TOL) and d["field_finite_at_end"]
            if not d["field_finite_at_end"]:
                status = "DIVERGED"
            elif ct < 0.0:
                status = "NEGATIVE" if steady else "NEGATIVE_UNSTEADY"
            else:
                status = "POSITIVE" if steady else "POSITIVE_UNSTEADY"
            cells.append({
                "tag": d["tag"], "tau": d["tau_achieved"], "N": d["N"],
                "Re_lattice": d["Re_lattice_measured"], "nu_lb": d["nu_lb"],
                "steps": d["steps_done"], "passes": d["passes"],
                "C_T": ct, "C_F": cf, "C_P": window_mean(d["_ts"]["cd_p"], n_win) * sc,
                "C_T_min_window": float(d["_ts"]["cd_t"][-n_win:].min() * sc),
                "C_T_max_window": float(d["_ts"]["cd_t"][-n_win:].max() * sc),
                "C_T_window_drift": d["C_T_window_drift"],
                "C_T_series_min": float(d["_ts"]["cd_t"].min() * sc),
                "C_T_series_min_step": int(d["_ts"]["step"][int(np.argmin(d["_ts"]["cd_t"]))]),
                "n_samples": int(len(d["_ts"]["cd_t"])),
                "n_samples_negative": int((d["_ts"]["cd_t"] < 0).sum()),
                "frac_samples_negative": float((d["_ts"]["cd_t"] < 0).mean()),
                "window_mean_definition": "last 200 steps (40 samples at interval 5)",
                "excursion_definition": "any instantaneous sample of the run < 0 -> NEG_EXCURSION",
                "steady": bool(steady), "status": status,
                "solver_route": solver_route(d["tag"]),
                "case_json": str(Path(d["raw_npz"]).parent / f"case_{d['tag']}.json"),
                "timeseries_sha256": d["raw_npz_sha256"],
                "wall_clock": d["wall_clock"],
            })
        taus = sorted({c["tau"] for c in cells})
        Ns = sorted({c["N"] for c in cells})
        grid = {}
        for c in cells:
            grid.setdefault(f"{c['N']}", {})[f"{c['tau']:.6g}"] = c["status"]
        surface = {
            "definition": "cell = sign of the window-mean C_T at the given (tau, N); "
                          "UNSTEADY suffix = last-window drift above the declared "
                          f"{100 * DRIFT_TOL:.0f}% steady-state tolerance",
            "tau_values": taus, "N_values": Ns,
            "sign_grid": grid, "cells": cells,
            "boundary": _boundary(cells, taus, Ns),
            "route_audit": {
                "note": "the frozen engine selects the wall treatment itself (WallTreatment.AUTO); at the "
                        "high-Reynolds end of the scan it switches to the wall-function route, so the "
                        "surface is NOT a single numerical setup. Route per cell is recorded and the "
                        "route boundary is reported next to the sign boundary.",
                "by_route": {r: sorted({f"tau={c['tau']:.5g}/N={c['N']}" for c in cells
                                        if c["solver_route"] == r})
                             for r in sorted({c["solver_route"] for c in cells})},
            },
            "secondary_definition": {
                "rule": "a cell is NEG_EXCURSION if any instantaneous force sample of the run is "
                        "negative; this is registered separately because the window-mean sign is "
                        "unstable at the high-Reynolds end of the scan",
                "cells": [{"tau": c["tau"], "N": c["N"], "frac_samples_negative": c["frac_samples_negative"],
                           "C_T_series_min": c["C_T_series_min"],
                           "status": "NEG_EXCURSION" if c["frac_samples_negative"] > 0 else "NO_EXCURSION"}
                          for c in cells],
            },
        }

    out = {
        "schema": "w26_gci_sign_surface/v1",
        "mission": "m17569",
        "generated_by": "papers/artifacts/w26/lbm-suboff-tau/build_gci.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tolerances": {"gci_pct": 100 * GCI_TARGET, "observed_order": P_TARGET,
                       "steady_drift_pct": 100 * DRIFT_TOL},
        "ladder": gci,
        "sign_surface": surface,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[gci] wrote {args.out}")
    q = gci["quantities"]["C_T"]
    print(f"  C_T: f=[{q['f1_fine']:.6g},{q['f2_medium']:.6g},{q['f3_coarse']:.6g}] "
          f"p={q.get('observed_order')} GCI={q.get('gci_fine_pct')} status={q['status']}")
    if surface:
        print(f"  scan cells={len(surface['cells'])} statuses={sorted({c['status'] for c in surface['cells']})}")


def _boundary(cells, taus, Ns) -> dict:
    """Locate, per N, the largest tau that still gives a negative window-mean C_T."""
    rows = []
    for N in Ns:
        sub = sorted([c for c in cells if c["N"] == N], key=lambda c: c["tau"])
        neg = [c for c in sub if c["status"].startswith("NEGATIVE")]
        pos = [c for c in sub if c["status"].startswith("POSITIVE")]
        row = {"N": N, "n_negative": len(neg), "n_positive": len(pos),
               "tau_max_negative": max((c["tau"] for c in neg), default=None),
               "tau_min_positive": min((c["tau"] for c in pos), default=None)}
        if row["tau_max_negative"] is not None and row["tau_min_positive"] is not None:
            row["boundary_bracket_tau"] = [row["tau_max_negative"], row["tau_min_positive"]]
        rows.append(row)
    return {"per_N": rows,
            "note": "bracketed by the scan resolution in tau; the true boundary lies inside "
                    "the bracket, no interpolation is claimed"}


if __name__ == "__main__":
    main()