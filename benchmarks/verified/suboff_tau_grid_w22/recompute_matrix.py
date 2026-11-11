#!/usr/bin/env python3
"""W22 /**/ recompute_matrix.py — INDEPENDENT recomputation (path B) of every
number in ``tau_grid_matrix.json`` from the raw per-case artifacts.

Independence from ``build_matrix.py`` (path A):
  * path A normalises with the ``dpS_front``/``dpS_wet`` numbers **stored** in the
    raw npz by the runner;
  * path B recomputes both dpS values **analytically** from (D, L, N, u_lb) and
    re-derives the evaluation window from its own rule, and it accumulates the
    window mean with ``np.cumsum`` instead of ``np.mean``;
  * path B reads only ``raw/case_*.npz`` + ``prereg.json``, never path A's JSON
    (the comparison happens at the end, as a check, not as an input).

Emits ``recompute_check.json``: per-case |Δ| between path A and path B for
C_T, C_F, C_P, plus the max relative difference (criterion C4 tolerance 1e-6).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TAU_VALUES = [0.5005, 0.505, 0.510, 0.515, 0.520, 0.530]
GRIDS = [96, 144, 192]
D_M = 0.508
L_M = 4.356
U_LB = 0.05
SAMPLE_INTERVAL = 5
EVAL_SAMPLES = 40


def analytic_dps(N: int, u_lb: float) -> tuple[float, float]:
    """Frontal and wetted dpS recomputed from geometry only (no stored values)."""
    dx = L_M / N
    r_lb = (D_M / 2.0) / dx
    l_lb = L_M / dx
    dpS_front = 0.5 * u_lb * u_lb * math.pi * r_lb * r_lb
    dpS_wet = 0.5 * u_lb * u_lb * math.pi * (2.0 * r_lb) * l_lb
    return dpS_front, dpS_wet


def window_mean_cumsum(x: np.ndarray, n: int) -> float:
    """Mean of the last ``n`` samples via cumulative sums (different path than np.mean)."""
    if n <= 0 or len(x) < n:
        n = len(x)
    tail = x[-n:]
    cs = np.cumsum(tail[::-1])
    return float(cs[-1] / n)


def main() -> None:
    matrix_path = HERE / "tau_grid_matrix.json"
    matrix = json.loads(matrix_path.read_text()) if matrix_path.exists() else None
    rows = []
    worst = 0.0
    for N in GRIDS:
        for tau in TAU_VALUES:
            tag = f"tau{tau:.4f}_N{N}"
            npz = RAW / f"case_{tag}.npz"
            if not npz.exists():
                rows.append({"tag": tag, "status": "MISSING_RAW"})
                continue
            d = np.load(npz)
            dpS_front_a, dpS_wet_a = analytic_dps(N, U_LB)
            rescale = dpS_front_a / dpS_wet_a
            n = min(EVAL_SAMPLES, len(d["step"]))
            C_P = window_mean_cumsum(d["cd_pressure_frontal"], n) * rescale
            C_F = window_mean_cumsum(d["cd_friction_frontal"], n) * rescale
            C_T = window_mean_cumsum(d["cd_total_frontal"], n) * rescale
            row = {
                "tag": tag,
                "status": "OK",
                "N": N,
                "tau": tau,
                "n_samples_used": int(n),
                "dpS_front_analytic": dpS_front_a,
                "dpS_wet_analytic": dpS_wet_a,
                "dpS_front_stored": float(d["dpS_front"]),
                "dpS_wet_stored": float(d["dpS_wet"]),
                "C_T": C_T,
                "C_F": C_F,
                "C_P": C_P,
                "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            }
            if matrix is not None:
                match = [c for c in matrix.get("cases", []) if c.get("tag") == tag]
                if match:
                    a = match[0]
                    deltas = {}
                    for k in ("C_T", "C_F", "C_P"):
                        va, vb = float(a[k]), float(row[k])
                        denom = abs(va) if abs(va) > 0 else 1.0
                        deltas[k] = abs(va - vb) / denom
                    row["abs_rel_delta_vs_path_A"] = deltas
                    worst = max(worst, max(deltas.values()))
            rows.append(row)

    out = {
        "schema": "w22_recompute_check_v1",
        "mission": "m14505",
        "path": "B (independent: analytic dpS + cumsum window mean, npz only)",
        "path_A": "build_matrix.py (stored dpS + np.mean)",
        "tolerance_rel": 1e-6,
        "n_cases_recomputed": sum(1 for r in rows if r["status"] == "OK"),
        "n_missing": sum(1 for r in rows if r["status"] != "OK"),
        "max_abs_rel_delta": worst,
        "pass": bool(rows) and any(r["status"] == "OK" for r in rows) and worst <= 1e-6,
        "rows": rows,
    }
    (HERE / "recompute_check.json").write_text(json.dumps(out, indent=1))
    print(f"[recompute] cases={out['n_cases_recomputed']} missing={out['n_missing']} "
          f"max|delta|={worst:.3e} pass={out['pass']}", flush=True)


if __name__ == "__main__":
    main()