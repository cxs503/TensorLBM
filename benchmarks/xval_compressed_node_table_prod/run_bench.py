#!/usr/bin/env python3
"""Benchmark: compressed fluid-node table as a production backend of porous_media.

Runs the single-phase periodic-slab Darcy channel
(``tensorlbm.porous_media.run_single_phase_darcy``) in three layouts and writes a
reference-vs-got table with the artefacts of record:

* ``compressed``      -- fluid-node compaction + neighbour table (the new backend)
* ``dense_halfway``   -- main-solver primitives with the same half-way wall rule
* ``dense``           -- the shipped main-solver update (``stream`` +
  ``bounce_back_cells``, wall on the solid-node centre)

Reference: analytic intrinsic permeability of the periodic slab,
``k_super = (H / ny) * H**2 / 12`` (half-way no-slip planes).

Usage
-----
``PYTHONPATH=src python3 benchmarks/xval_compressed_node_table_prod/run_bench.py``
writes ``results.json`` next to this file (and ``--figures DIR`` for the plots).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.porous_media import SinglePhaseDarcyConfig, run_single_phase_darcy

HERE = Path(__file__).resolve().parent
G = 1.0e-4
TAU = 0.8
# (ny, h_fluid) pairs: porosity H/ny = 0.667 for both so only H changes
CASES = ((24, 16), (40, 32))
LAYOUTS = ("compressed", "dense_halfway", "dense")
TOL_PCT = 3.0


def run_case(ny: int, h: int, layout: str, steps_cap: int) -> dict:
    cfg = SinglePhaseDarcyConfig(
        ny=ny,
        nx=ny,
        h_fluid=h,
        g=G,
        tau=TAU,
        steps_cap=steps_cap,
        layout=layout,  # type: ignore[arg-type]
        output_root=HERE / "runs",
        overwrite=True,
    )
    t0 = time.perf_counter()
    meta = run_single_phase_darcy(cfg)
    wall = time.perf_counter() - t0
    meta.pop("config", None)
    meta["wall_seconds"] = wall
    return meta


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--figures", default=str(HERE / "figures"))
    ap.add_argument(
        "--cases",
        default=",".join(f"{ny}x{h}" for ny, h in CASES),
        help="comma separated ny x h_fluid pairs (default: full ladder)",
    )
    args = ap.parse_args()
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cases = tuple(tuple(int(v) for v in c.split("x")) for c in args.cases.split(","))
    results: dict[str, dict] = {}
    rows: list[dict] = []
    for ny, h in cases:
        steps_cap = 4000 if h <= 8 else (10000 if h <= 16 else 30000)
        for layout in LAYOUTS:
            key = f"{layout}_ny{ny}_h{h}"
            meta = run_case(ny, h, layout, steps_cap)
            results[key] = meta
            rows.append(
                {
                    "case": key,
                    "layout": layout,
                    "ny": ny,
                    "h_fluid": h,
                    "steps": meta["steps"],
                    "converged": meta["converged"],
                    "k_ref_analytic": meta["k_ref_analytic"],
                    "k_got": meta["k_darcy_super"],
                    "err_pct": meta["err_pct"],
                    "u_max": meta["u_max"],
                    "ma_max": meta["final_state"]["ma_max"],
                    "wall_seconds": meta["wall_seconds"],
                }
            )

    # G1: representation identity (compressed vs dense_halfway)
    g1 = {}
    for ny, h in cases:
        a = results[f"compressed_ny{ny}_h{h}"]
        b = results[f"dense_halfway_ny{ny}_h{h}"]
        prof_a, prof_b = a["ux_profile_mean_x"], b["ux_profile_mean_x"]
        scale = max(abs(v) for v in prof_a)
        g1[f"H{h}"] = {
            "k_rel_diff": abs(a["k_darcy_super"] - b["k_darcy_super"]) / abs(a["k_darcy_super"]),
            "profile_max_rel": max(abs(x - y) for x, y in zip(prof_a, prof_b)) / scale,
            "steps_equal": a["steps"] == b["steps"],
        }
    # G2/G3: physics error against the analytic half-way reference
    g2 = {
        row["case"]: {"err_pct": row["err_pct"], "pass_3pct": abs(row["err_pct"]) < TOL_PCT}
        for row in rows
    }
    # wall-convention study: the shipped dense update places the no-slip plane on
    # the solid-node centre -> effective channel width h+1 instead of h.
    conv = {}
    for ny, h in cases:
        row = next(r for r in rows if r["case"] == f"dense_ny{ny}_h{h}")
        conv[f"H{h}"] = {
            "err_pct": row["err_pct"],
            "k_eff_width": (row["k_got"] * ny * 12.0) ** (1.0 / 3.0),
            "predicted_width_h_plus_1": h + 1,
            "k_at_h_plus_1": ((h + 1) / ny) * (h + 1) ** 2 / 12.0,
        }
    keys = sorted(conv, key=lambda k: int(k[1:]))
    lo, hi = keys[0], keys[-1]
    err_lo, err_hi = conv[lo]["err_pct"], conv[hi]["err_pct"]

    payload = {
        "case_family": "periodic-slab single-phase Darcy channel, body force g",
        "params": {"g": G, "tau": TAU, "nu": (TAU - 0.5) / 3.0, "nx": "= ny", "porosity": 2.0 / 3.0},
        "reference": "k_super = (H/ny) * H**2 / 12 (half-way no-slip planes)",
        "rows": rows,
        "gates": {
            "G1_identity_lt_1e-8": g1,
            "G2_physics_lt_3pct": g2,
            "G2_all_compressed_pass": bool(
                all(abs(results[f"compressed_ny{ny}_h{h}"]["err_pct"]) < TOL_PCT for ny, h in cases)
            ),
            "G3_wall_convention": {
                "detail": conv,
                "err_pct_dense": {lo: err_lo, hi: err_hi},
                "ratio_lowH_over_highH": err_lo / err_hi,
                "expected_ratio_for_half_cell_offset": float(hi[1:]) / float(lo[1:]),
            },
        },
        "artifacts": {
            "profile_figure": str(fig_dir / "profiles.png"),
            "convention_figure": str(fig_dir / "wall_convention.png"),
        },
    }

    # --- figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, len(cases), figsize=(4.8 * len(cases), 3.4), dpi=140, sharey=False)
    for ax, (ny, h) in zip(axes, cases):
        y0 = (ny - h) // 2
        ys = [y - (ny - 1) / 2.0 for y in range(y0, y0 + h)]
        for layout, style in (("compressed", "o"), ("dense", "s")):
            prof = results[f"{layout}_ny{ny}_h{h}"]["ux_profile_mean_x"][y0 : y0 + h]
            ax.plot([v * 1e3 for v in prof], ys, style, ms=3, label=layout)
        nu = (TAU - 0.5) / 3.0
        u_ana = [G / (2 * nu) * ((h * h) / 4.0 - (y - y0 - (h - 1) / 2.0) ** 2) for y in range(h)]
        ax.plot([v * 1e3 for v in u_ana], ys, "k--", lw=1.0, label="analytic")
        ax.set_title(f"H={h}, ny={ny}")
        ax.set_xlabel(r"$\langle u_x\rangle_x$ [$10^{-3}$]")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$y-(n_y-1)/2$")
    fig.tight_layout()
    fig.savefig(fig_dir / "profiles.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=140)
    for layout, style in (("compressed", "o-"), ("dense", "s-")):
        xs = [1.0 / row["h_fluid"] for row in rows if row["layout"] == layout]
        ys = [row["err_pct"] for row in rows if row["layout"] == layout]
        ax.plot(xs, ys, style, label=layout)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-TOL_PCT, TOL_PCT, color="g", alpha=0.12, label="+/-3 % band")
    ax.set_xlabel("1 / H")
    ax.set_ylabel("permeability error vs analytic [%]")
    ax.set_title("wall-convention bias vs resolution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "wall_convention.png")
    plt.close(fig)

    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- report table ------------------------------------------------------
    print(f"{'case':28s} {'steps':>6s} {'k_ref':>11s} {'k_got':>11s} {'err%':>9s} {'conv':>5s}")
    for row in rows:
        print(
            f"{row['case']:28s} {row['steps']:6d} {row['k_ref_analytic']:11.5e} "
            f"{row['k_got']:11.5e} {row['err_pct']:9.4f} {str(row['converged']):>5s}"
        )
    print(json.dumps(payload["gates"], indent=2))
    print(f"results -> {args.out}")
    return payload


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
