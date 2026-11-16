"""Round-3 evidence: the boundary-condition axis of the model matrix.

Runs the new ``wall_poiseuille`` case over a resolution sweep for one
single-rate (BGK) and one two-relaxation-time (TRT, Lambda = 3/16) model,
fits the observed convergence order, and writes the machine-generated table
used in ``docs/MODEL_MATRIX_CI_PLUS_COVERAGE_R3.md``.

    PYTHONPATH=src python xval/model_matrix_wall_axis/run_study.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorlbm.model_matrix.cases import Grid, run_wall_poiseuille
from tensorlbm.model_matrix.registry import WALL_CASE, get_model, registry

OUT = Path("artifacts/model_matrix_wall_axis")
TAU = 0.8
PLAN = [("d3q19/bgk", 10), ("d3q19/bgk", 18), ("d3q19/bgk", 34), ("d3q19/trt", 10), ("d3q19/trt", 18)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    profiles = {}
    for model, ny in PLAN:
        res = run_wall_poiseuille(get_model(model), Grid(nx=4, ny=ny, nz=1), TAU)
        m = res["metrics"]
        rows.append(
            {
                "model": model,
                "ny": ny,
                "H": int(ny - 2),
                "status": res["status"],
                "rel_l2_pct": m["rel_l2_pct"],
                "wall_offset_lattice": m["wall_offset_lattice"],
                "u_center_rel_err_pct": m["u_center_rel_err_pct"],
                "symmetry_rel": m["symmetry_rel"],
                "mass_drift_rel": m["mass_drift_rel"],
                "steps": int(m["steps"]),
                "converged": bool(m["converged"]),
            }
        )
        profiles[f"{model}@{ny}"] = res["curves"]
        print(f"[wall] {model:10s} H={ny-2:3d} {res['status']:4s} l2={m['rel_l2_pct']:.3f}% "
              f"offset={m['wall_offset_lattice']:+.4f} uc_err={m['u_center_rel_err_pct']:+.3f}%", flush=True)

    orders = []
    for model in ("d3q19/bgk", "d3q19/trt"):
        sel = [r for r in rows if r["model"] == model]
        sel.sort(key=lambda r: r["H"])
        for a, b in zip(sel, sel[1:]):
            p = math.log(a["rel_l2_pct"] / b["rel_l2_pct"]) / math.log(b["H"] / a["H"])
            orders.append({"model": model, "H_from": a["H"], "H_to": b["H"], "observed_order": p})
            print(f"[order] {model} H {a['H']}->{b['H']}: p={p:.2f}", flush=True)

    declared = sum(1 for s in registry().values() if WALL_CASE in s.cases)
    payload = {
        "tau": TAU,
        "models": len(registry()),
        "models_with_wall_case": declared,
        "rows": rows,
        "orders": orders,
        "note": (
            "reference = a-priori parabola g/(2nu)(Y-1/2)(ny-3/2-Y); H = ny-2; "
            "wall_offset_lattice is the wall position implied by a parabola fitted to the "
            "interior nodes minus the ideal half-lattice position"
        ),
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Wall axis (round 3) — reference vs measured",
        "",
        "Reference: `u(Y) = g/(2 nu) (Y - 1/2)(ny - 3/2 - Y)`, `nu = (tau-1/2)/3`, `tau = %.2f`," % TAU,
        "`g = 1e-5` lattice units, half-way bounce-back walls, `H = ny - 2` fluid rows.",
        "",
        "| model | H | status | rel L2 err % | implied wall offset (lattice) | centre-vel err % | symmetry | mass drift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['model']}` | {r['H']} | {r['status']} | {r['rel_l2_pct']:.3f} | "
            f"{r['wall_offset_lattice']:+.4f} | {r['u_center_rel_err_pct']:+.3f} | "
            f"{r['symmetry_rel']:.1e} | {r['mass_drift_rel']:.1e} |"
        )
    lines += ["", "## Observed convergence order", "",
              "| model | H from | H to | p |", "|---|---|---|---|"]
    for o in orders:
        lines.append(f"| `{o['model']}` | {o['H_from']} | {o['H_to']} | {o['observed_order']:.2f} |")
    lines += ["", f"Registry: {declared}/{len(registry())} models declare the `{WALL_CASE}` case."]
    (OUT / "BENCHMARK_REF_VS_GOT.md").write_text("\n".join(lines) + "\n")

    try:  # figure: profile at the middle resolution + convergence
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        key = "d3q19/bgk@18"
        cur = profiles[key]
        ax1.plot(cur["y"], cur["u_exact"], "k-", lw=1.6, label="analytic (walls at $\\pm 1/2$)")
        ax1.plot(cur["y"], cur["u"], "o", ms=3.5, label="TensorLBM BGK, halfway BB")
        ax1.set_xlabel("Y (lattice units)")
        ax1.set_ylabel("$u_x$")
        ax1.set_title("steady plane Poiseuille, H = 16")
        ax1.legend()
        ax1.grid(alpha=0.3)
        for model, style in (("d3q19/bgk", "o-"), ("d3q19/trt", "s--")):
            sel = sorted([r for r in rows if r["model"] == model], key=lambda r: r["H"])
            ax2.loglog([r["H"] for r in sel], [r["rel_l2_pct"] for r in sel], style, label=model)
        hs = [r["H"] for r in rows if r["model"] == "d3q19/bgk"]
        ref = [rows[0]["rel_l2_pct"] * (hs[0] / h) ** 2 for h in hs]
        ax2.loglog(hs, ref, "k:", label="$O(H^{-2})$")
        ax2.set_xlabel("channel height H")
        ax2.set_ylabel("relative L2 error %")
        ax2.set_title("wall convergence")
        ax2.legend()
        ax2.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(OUT / "fig1_wall_axis.png", dpi=140)
        print("[fig] wrote", OUT / "fig1_wall_axis.png")
    except Exception as exc:  # noqa: BLE001
        print("[fig] skipped:", exc)

    worst = max(r["rel_l2_pct"] for r in rows)
    print(f"[verdict] worst rel L2 = {worst:.3f}% (owner criterion: <15%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
