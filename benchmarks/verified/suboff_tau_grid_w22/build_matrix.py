#!/usr/bin/env python3
"""W22 /**/ build_matrix.py — assemble the tau x grid matrix (path A) and the
negative-drag surface from the raw per-case artifacts.

Inputs (written by the runner, never hand-edited):
  raw/case_<tag>.json  per-case scalars (path A values, np.mean over the window)
  raw/case_<tag>.npz   raw force series + the two dpS calibers
  prereg.json          frozen protocol / criteria
  ../suboff_anchor/anchor.json   AN-SUBOFF + AN-ITTC57 anchors (read, not copied)

Outputs:
  tau_grid_matrix.json      full matrix + criterion verdicts + iso-Re analysis
  negative_drag_surface.json  (tau,N) sign map with 4-neighbour support test
  recompute_check.json      produced by recompute_matrix.py (path B); read here
                            only to record the C4 verdict

Nothing is transcribed by hand: every number below is computed from the raw
artifacts.  Missing cases stay missing and are reported as such.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / os.environ.get("W22_RAW_DIR", "raw")
ANCHOR = HERE.parent / "suboff_anchor" / "anchor.json"
TAU_VALUES = [0.5005, 0.505, 0.510, 0.515, 0.520, 0.530]
GRIDS = [int(x) for x in os.environ.get("W22_GRIDS", "96,144,192").split(",")]
PREFIX = os.environ.get("W22_PREFIX", "")
U_LB = 0.05
EVAL_SAMPLES = 40


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def six(x: float) -> str:
    """6 significant digits (the mission's numeric band)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.5e}"


def ittc1957(re: float) -> float:
    den = math.log10(re) - 2.0
    return 0.075 / den**2


def load_anchors() -> list[dict]:
    a = json.loads(ANCHOR.read_text())
    rows = []
    for key in ("A_AFF1_bare_DTRC_T14", "A2_AFF8_full_DTRC_T14", "B_AFF8_full_CFDletters_T3"):
        for r in a["anchors"][key]["rows"]:
            rows.append(
                {
                    "anchor": key,
                    "row_id": r["row_id"],
                    "Re": float(r["Re"]),
                    "CT": float(r["CT"]),
                }
            )
    rows.sort(key=lambda r: r["Re"])
    return rows


def main() -> None:
    prereg = json.loads((HERE / "prereg.json").read_text())
    anchors = load_anchors()
    anchor_low = anchors[0]

    cases = []
    for N in GRIDS:
        for tau in TAU_VALUES:
            tag = f"tau{tau:.4f}_N{N}"
            jp = RAW / f"case_{tag}.json"
            npz = RAW / f"case_{tag}.npz"
            if not jp.exists():
                cases.append({"tag": tag, "tau": tau, "N": N, "status": "MISSING"})
                continue
            c = json.loads(jp.read_text())
            # path A: recompute from the raw series with the stored dpS calibers
            d = np.load(npz)
            rescale = float(d["dpS_front"]) / float(d["dpS_wet"])
            n = min(EVAL_SAMPLES, len(d["step"]))
            C_P = float(np.mean(d["cd_pressure_frontal"][-n:]) * rescale)
            C_F = float(np.mean(d["cd_friction_frontal"][-n:]) * rescale)
            C_T = float(np.mean(d["cd_total_frontal"][-n:]) * rescale)
            row = {k: c[k] for k in c}
            row.update(
                {
                    "status": "OK",
                    "tau": tau,
                    "N": N,
                    "C_T": C_T,
                    "C_F": C_F,
                    "C_P": C_P,
                    "C_P_plus_C_F": C_P + C_F,
                    "negative_drag": bool(C_T < 0.0),
                    "Re_sim": float(c["Re_sim"]),
                    "n_samples_used": int(n),
                }
            )
            # transient shape: three consecutive windows of the sampled series
            def wmean(a, lo, hi):
                seg = a[lo:hi]
                return float(np.mean(seg) * rescale) if len(seg) else float("nan")

            m = len(d["step"])
            row["C_T_windows"] = {
                "first_third": wmean(d["cd_total_frontal"], 0, m // 3),
                "mid_third": wmean(d["cd_total_frontal"], m // 3, 2 * m // 3),
                "last_window": C_T,
            }
            # anchor deviation (nearest Re anchor = lowest-Re anchor row for every case)
            dev = (C_T - anchor_low["CT"]) / anchor_low["CT"] * 100.0
            row["anchor"] = {
                "row": f"{anchor_low['anchor']}:{anchor_low['row_id']}",
                "Re_anchor": anchor_low["Re"],
                "CT_anchor": anchor_low["CT"],
                "Re_ratio_sim_over_anchor": row["Re_sim"] / anchor_low["Re"],
                "dev_pct": dev,
            }
            row["CF_ittc1957_at_Re_sim"] = ittc1957(row["Re_sim"]) if row["Re_sim"] > 100 else float("nan")
            row["CF_minus_ittc1957"] = row["C_F"] - row["CF_ittc1957_at_Re_sim"]
            row["six_sig"] = {
                "C_T": six(C_T),
                "C_F": six(C_F),
                "C_P": six(C_P),
                "dev_pct_vs_anchor": six(dev),
            }
            cases.append(row)

    ok = [c for c in cases if c.get("status") == "OK"]
    n_ok = len(ok)
    n_expected = len(GRIDS) * len(TAU_VALUES)

    # ---------------- criterion C1 -----------------------------------------
    complete = [c for c in ok if all(k in c for k in ("C_T", "C_F", "C_P", "anchor"))]
    C1_ok = n_ok >= 18 and len(complete) == n_ok
    C1 = {
        "criterion": ">=18 cases with C_T/C_F/C_P and anchor deviation tabulated",
        "n_expected": n_expected,
        "n_completed": n_ok,
        "n_with_all_fields": len(complete),
        "verdict": "PASS" if C1_ok else "FAIL",
        "missing_tags": [c["tag"] for c in cases if c.get("status") != "OK"],
    }

    # ---------------- criterion C2: negative drag surface -------------------
    idx = {(c["tau"], c["N"]): c for c in ok}
    sign_map = []
    for N in GRIDS:
        for tau in TAU_VALUES:
            c = idx.get((tau, N))
            if c is None:
                sign_map.append({"tau": tau, "N": N, "status": "MISSING"})
                continue
            neighbours = []
            for dtau, dN in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                ti = TAU_VALUES.index(tau) + dtau
                Ni = GRIDS.index(N) + dN
                if 0 <= ti < len(TAU_VALUES) and 0 <= Ni < len(GRIDS):
                    nb = idx.get((TAU_VALUES[ti], GRIDS[Ni]))
                    if nb is not None:
                        neighbours.append(
                            {
                                "tau": TAU_VALUES[ti],
                                "N": GRIDS[Ni],
                                "C_T": nb["C_T"],
                                "negative": nb["negative_drag"],
                            }
                        )
            same = [x for x in neighbours if x["negative"] == c["negative_drag"]]
            sign_map.append(
                {
                    "tau": tau,
                    "N": N,
                    "status": "OK",
                    "C_T": c["C_T"],
                    "negative": c["negative_drag"],
                    "neighbours": neighbours,
                    "n_neighbours": len(neighbours),
                    "n_same_sign_neighbours": len(same),
                    "boundary_supported": (len(same) >= 1) if neighbours else False,
                }
            )
    neg_cells = [s for s in sign_map if s.get("status") == "OK" and s["negative"]]
    unsupported = [s for s in neg_cells if not s["boundary_supported"]]
    C2_ok = n_ok == n_expected and len(unsupported) == 0
    C2 = {
        "criterion": "negative-drag region registered on the (tau,N) matrix; every negative cell has >=1 same-sign 4-neighbour",
        "n_negative_cells": len(neg_cells),
        "negative_cells": [{"tau": s["tau"], "N": s["N"], "C_T": s["C_T"]} for s in neg_cells],
        "n_unsupported_cells": len(unsupported),
        "unsupported_cells": [{"tau": s["tau"], "N": s["N"]} for s in unsupported],
        "verdict": "PASS" if C2_ok else "FAIL",
    }

    # ---------------- criterion C3a: same tau, across N ---------------------
    drift_rows = []
    for tau in TAU_VALUES:
        vals = {N: idx[(tau, N)]["C_T"] for N in GRIDS if (tau, N) in idx}
        if len(vals) < 2:
            drift_rows.append({"tau": tau, "status": "INCOMPLETE", "values": vals})
            continue
        vmax = max(vals.values())
        vmin = min(vals.values())
        ref = vals.get(GRIDS[-1], max(vals.values(), key=abs))
        d = (vmax - vmin) / abs(ref) * 100.0 if ref else float("inf")
        drift_rows.append(
            {
                "tau": tau,
                "status": "OK",
                "C_T_by_N": {str(k): v for k, v in vals.items()},
                "spread_abs": vmax - vmin,
                "drift_pct": d,
                "pass": bool(d < 3.0),
            }
        )
    dr_ok = [r for r in drift_rows if r["status"] == "OK"]
    C3a_ok = len(dr_ok) == len(TAU_VALUES) and all(r["pass"] for r in dr_ok)
    C3a = {
        "criterion": "same tau across N: relative C_T drift < 3%",
        "definition": "(max - min) / |C_T at N=192| * 100",
        "worst_drift_pct": max((r["drift_pct"] for r in dr_ok), default=float("nan")),
        "rows": drift_rows,
        "verdict": "PASS" if C3a_ok else "FAIL",
    }

    # ---------------- criterion C3b: same N, across tau --------------------
    mono_rows = []
    for N in GRIDS:
        seq = [(tau, idx[(tau, N)]["C_T"]) for tau in TAU_VALUES if (tau, N) in idx]
        if len(seq) < 3:
            mono_rows.append({"N": N, "status": "INCOMPLETE", "n_points": len(seq)})
            continue
        diffs = [seq[i + 1][1] - seq[i][1] for i in range(len(seq) - 1)]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        direction = "monotone_increasing" if n_neg == 0 else ("monotone_decreasing" if n_pos == 0 else "non_monotone")
        mono_rows.append(
            {
                "N": N,
                "status": "OK",
                "tau_sequence": [t for t, _ in seq],
                "C_T_sequence": [v for _, v in seq],
                "successive_diffs": diffs,
                "n_increasing": n_pos,
                "n_decreasing": n_neg,
                "direction": direction,
            }
        )
    mono_ok = [r for r in mono_rows if r["status"] == "OK"]
    directions = sorted({r["direction"] for r in mono_ok})
    C3b = {
        "criterion": "same N across tau: monotonicity decided by machine check (yes/no + evidence rows)",
        "rows": mono_rows,
        "directions": directions,
        "monotone_everywhere": bool(len(mono_ok) == len(GRIDS) and all(r["direction"].startswith("monotone") for r in mono_ok)),
        "consistent_direction": bool(len(directions) == 1 and directions[0].startswith("monotone")),
        "verdict": (
            "PASS (monotone, consistent direction)"
            if len(mono_ok) == len(GRIDS) and len(directions) == 1 and directions[0].startswith("monotone")
            else "FAIL (non-monotone or direction inconsistent between grids)"
        ),
    }

    # ---------------- iso-Re groups (decoupling core) ----------------------
    # Re_sim = 3*u_lb*N/(tau-0.5)  =>  N/(tau-0.5) is the composite that sets Re
    groups: dict[float, list] = {}
    for c in ok:
        key = round(c["N"] / (c["tau"] - 0.5), 6)
        groups.setdefault(key, []).append(
            {"tau": c["tau"], "N": c["N"], "C_T": c["C_T"], "Re_sim": c["Re_sim"]}
        )
    iso = []
    for key, members in sorted(groups.items(), reverse=True):
        Re = 3.0 * U_LB * key
        if len(members) < 2:
            iso.append({"N_over_dnu": key, "Re_sim": Re, "n_members": 1, "members": members})
            continue
        vals = [m["C_T"] for m in members]
        spread = (max(vals) - min(vals)) / abs(np.mean(vals)) * 100.0
        iso.append(
            {
                "N_over_dnu": key,
                "Re_sim": Re,
                "n_members": len(members),
                "members": members,
                "spread_pct": float(spread),
                "pass_3pct": bool(spread < 3.0),
            }
        )
    iso_multi = [g for g in iso if g["n_members"] >= 2]

    # ---------------- criterion C4 (path B cross-check) --------------------
    rc_path = HERE / os.environ.get("W22_RECOMPUTE", "recompute_check.json")
    if rc_path.exists():
        rc = json.loads(rc_path.read_text())
        C4 = {
            "criterion": "6 significant digits; every number recomputed by 2 independent scripts, agreement 1e-6",
            "path_A": "build_matrix.py (stored dpS + np.mean)",
            "path_B": "recompute_matrix.py (analytic dpS + cumsum mean)",
            "n_cases_path_B": rc.get("n_cases_recomputed"),
            "max_abs_rel_delta": rc.get("max_abs_rel_delta"),
            "tolerance": 1e-6,
            "six_sig_band_present": True,
            "verdict": "PASS" if rc.get("pass") and rc.get("n_cases_recomputed") == n_ok else "FAIL",
        }
    else:
        C4 = {
            "criterion": "6 significant digits; every number recomputed by 2 independent scripts, agreement 1e-6",
            "verdict": "NOT_RUN",
            "note": "recompute_check.json absent",
        }

    matrix = {
        "schema": "w22_tau_grid_matrix_v1",
        "mission": "m14505",
        "mission_key": "lbm-suboff-tau",
        "repo": "TensorLBM",
        "discipline": "resistance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": sha256(HERE / "prereg.json"),
        "generated_by": "benchmarks/verified/suboff_tau_grid_w22/build_matrix.py",
        "protocol": {
            "collision_model": "MRT (D3Q19)",
            "wall": "half-way bounce-back",
            "force_method": "PRESSURE_FRICTION / friction_formula=standard / p0=near_wall / extrap=none",
            "domain_padding_in_L": prereg["fixed_protocol"]["domain_padding_in_L"],
            "u_lb": U_LB,
            "steps": prereg["fixed_protocol"]["steps"],
            "force_sample_interval": prereg["fixed_protocol"]["force_sample_interval"],
            "eval_window_steps": prereg["fixed_protocol"]["eval_window_steps"],
            "normalisation": prereg["fixed_protocol"]["normalisation"],
            "initial_field": prereg["fixed_protocol"]["initial_field"],
        },
        "anchor": {
            "artifact": "benchmarks/verified/suboff_anchor/anchor.json",
            "artifact_sha256": sha256(ANCHOR),
            "anchor_used": f"{anchor_low['anchor']}:{anchor_low['row_id']}",
            "Re_anchor": anchor_low["Re"],
            "CT_anchor": anchor_low["CT"],
            "all_anchor_rows": anchors,
            "ittc1957_line": "C_F = 0.075/(log10 Re - 2)^2",
        },
        "cases": cases,
        "summary": {
            "n_cases_completed": n_ok,
            "n_cases_expected": n_expected,
            "n_negative_drag": len(neg_cells),
            "Re_sim_range": [
                min((c["Re_sim"] for c in ok), default=None),
                max((c["Re_sim"] for c in ok), default=None),
            ],
            "C_T_range": [
                min((c["C_T"] for c in ok), default=None),
                max((c["C_T"] for c in ok), default=None),
            ],
        },
        "criteria": {"C1": C1, "C2": C2, "C3a": C3a, "C3b": C3b, "C4": C4},
        "decoupling": {
            "identity": prereg["decoupling_identity"]["statement"],
            "iso_Re_groups": iso,
            "n_iso_groups_with_multiple_members": len(iso_multi),
            "max_spread_within_iso_Re_group_pct": max(
                (g["spread_pct"] for g in iso_multi), default=float("nan")
            ),
            "n_iso_groups_pass_3pct": sum(1 for g in iso_multi if g["pass_3pct"]),
        },
        "negative_drag_surface": {
            "tau_values": TAU_VALUES,
            "N_values": GRIDS,
            "sign_map": sign_map,
        },
    }

    (HERE / f"{PREFIX}tau_grid_matrix.json").write_text(json.dumps(matrix, indent=1))
    nds = {
        "schema": "w22_negative_drag_surface_v1",
        "mission": "m14505",
        "generated_by": "benchmarks/verified/suboff_tau_grid_w22/build_matrix.py",
        "generated_at_utc": matrix["generated_at_utc"],
        "source_matrix": "benchmarks/verified/suboff_tau_grid_w22/tau_grid_matrix.json",
        "tau_values": TAU_VALUES,
        "N_values": GRIDS,
        "definition": "cell is NEGATIVE when the window-mean C_T = (F_x/(0.5 rho u^2 pi D L)) < 0",
        "sign_grid": {
            str(N): [
                next(
                    (
                        ("NEG" if s["negative"] else "POS") if s.get("status") == "OK" else "MISSING"
                        for s in sign_map
                        if s["tau"] == tau and s["N"] == N
                    ),
                    "MISSING",
                )
                for tau in TAU_VALUES
            ]
            for N in GRIDS
        },
        "C_T_grid": {
            str(N): [
                next(
                    (s["C_T"] if s.get("status") == "OK" else None
                     for s in sign_map if s["tau"] == tau and s["N"] == N),
                    None,
                )
                for tau in TAU_VALUES
            ]
            for N in GRIDS
        },
        "cells": sign_map,
        "boundary_support": {
            "rule": "every negative cell must have >=1 same-sign 4-neighbour in (tau,N) index space",
            "n_negative": len(neg_cells),
            "n_unsupported": len(unsupported),
            "pass": len(unsupported) == 0,
        },
    }
    (HERE / f"{PREFIX}negative_drag_surface.json").write_text(json.dumps(nds, indent=1))

    print(f"[matrix] cases={n_ok}/{n_expected} negative={len(neg_cells)}")
    for k, v in matrix["criteria"].items():
        print(f"[matrix] {k}: {v['verdict']}")
    print(f"[matrix] C3a worst drift = {C3a['worst_drift_pct']}%")
    print(f"[matrix] iso-Re groups with >=2 members: {len(iso_multi)} (max spread "
          f"{matrix['decoupling']['max_spread_within_iso_Re_group_pct']}%)")


if __name__ == "__main__":
    main()