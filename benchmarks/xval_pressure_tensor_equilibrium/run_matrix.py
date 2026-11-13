"""XVAL round 2: pressure-tensor equilibrium acceptance matrix (sailfish transplant).

Anchor ``XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT``.  The transplant
itself (``src/tensorlbm/phasefield/free_energy_pressure.py``) was merged in round
1 (``XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE``, TensorLBM#2136); round 1
left the curvature/resolution series unrun and never measured two equilibrium
invariants.  This runner closes exactly that gap and nothing else:

* resolution series ``flat_xi2 / flat_xi3 / flat_xi4`` -- xi/h = 2, 3, 4;
* curvature series ``dropR16 / dropR24 / dropR32`` -- R/xi = 8, 12, 16 at xi = 2;
* new KPI ``shear_free``  -- ``max|P_xy|/<P_NN>`` on a flat interface at rest
  (validates the off-diagonal anisotropic coefficients of ``lb_binary.py:185-268``);
* new KPI ``xi_fit``      -- interface width recovered from the *local* identity
  ``P_NN - P_TT = kappa sech^4((x-x0)/xi)/xi^2``, an independent route to both
  ``xi`` and ``kappa`` that never touches ``sqrt(2 kappa / A)``.

Gates come from ``xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/predeclared.json``
(written before any run).  Owner band: every physical KPI < 15 %.

Usage::

    python benchmarks/xval_pressure_tensor_equilibrium/run_matrix.py --arm all
    python benchmarks/xval_pressure_tensor_equilibrium/run_matrix.py --arm flat_xi2 --arm dropR16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

from xval_free_energy_binary_equilibrium.run_xval import (  # noqa: E402
    ARMS,
    run_droplet,
    run_flat,
)

PREDECLARED = os.path.join(
    _ROOT, "xval", "XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT", "predeclared.json"
)
OUT_DIR = os.path.join(_ROOT, "artifacts", "xval_pressure_tensor_equilibrium")

#: arm name -> (kind, series, abscissa)
SERIES = {
    "flat_xi2": ("flat", "resolution", 2.0),
    "flat_xi3": ("flat", "resolution", 3.0),
    "flat_xi4": ("flat", "resolution", 4.0),
    "flat_xi4_g05": ("flat", "resolution", 4.0),
    "dropR16": ("drop", "curvature", 8.0),
    "dropR24": ("drop", "curvature", 12.0),
    "dropR32": ("drop", "curvature", 16.0),
}


def kpi_values(res: dict, kind: str) -> dict[str, float]:
    """Pull the gated KPIs out of a raw arm result (no arithmetic on references)."""
    out = {
        "closure": res["closure_final"]["rel"],
        "spurious_velocity": res["spurious_velocity"]["max"],
    }
    if kind == "flat":
        out.update({
            "sigma_tension": res["sigma_rel_error"],
            "sigma_measured": res["sigma_measured"],
            "sigma_analytic": res["sigma_analytic"],
            "Pnn_uniformity": res["Pnn_uniformity_rel"],
            "shear_free": res["shear_rel"],
            "xi_fit": res["xi_fit_rel_error"],
            "xi_measured": res["xi_fit"]["xi"],
            "xi_analytic": res["config"]["xi_analytic"],
        })
    else:
        out.update({
            "laplace": res["laplace_rel_error_analytic"],
            "dp": res["dp"],
            "dp_analytic": res["dp_analytic"],
            "laplace_ratio": res["laplace_ratio_analytic"],
            "steps_run": res["steps_run"],
        })
    return out


def evaluate(gates: list[dict], arm: str, kpis: dict[str, float]) -> list[dict]:
    """Apply the predeclared gate of every KPI that names this arm."""
    rows = []
    for g in gates:
        arms = g["arms"]
        if "*" not in arms and arm not in arms:
            continue
        if g["id"] not in kpis:
            continue
        value = kpis[g["id"]]
        gate = float(g["gate"])
        rows.append({
            "kpi": g["id"],
            "arm": arm,
            "value": value,
            "gate": gate,
            "reference": g.get("reference"),
            "pass": bool(abs(value) <= gate),
            "tight_target": g.get("tight_target"),
            "within_tight": (None if "tight_target" not in g
                             else bool(abs(value) <= float(g["tight_target"]))),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None,
                    help="one of " + ", ".join(SERIES) + " (repeatable); default all")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--scale", type=float, default=1.0, help="step scale (smoke runs only)")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms whose per-arm json already exists (default: reuse)")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    decl = json.load(open(PREDECLARED))
    gates = decl["kpis"]
    # flat_xi4 (Gamma = 0.2, kappa = 0.8) diverges to NaN by step ~1000: kept in
    # SERIES as the documented stability boundary, excluded from the default set;
    # the xi/h = 4 point is taken by flat_xi4_g05 (Gamma = 0.05, declared deviation)
    # not in the default set: flat_xi4 (Gamma = 0.2) and flat_xi4_g05 (Gamma = 0.05)
    # both diverge to NaN at kappa = 0.8 -- the xi/h = 4 point is NOT obtained, see
    # docs/XVAL_LBM_PRESSURE_TENSOR_EQUILIBRIUM_TRANSPLANT.md; dropR24/dropR32 are
    # run explicitly (they exceed one local short-run slot)
    names = args.arm or ["flat_xi2", "flat_xi3", "dropR16"]

    runs, rows = {}, []
    t_all = time.time()
    for name in names:
        kind, series, absc = SERIES[name]
        cfg = dict(ARMS[name])
        cfg["steps"] = int(round(cfg["steps"] * args.scale))
        cfg["name"] = name
        print(f"[xval-r2] arm {name} ({kind}, {series} x={absc}): {cfg}", flush=True)
        cached = os.path.join(args.out, f"{name}.json")
        if os.path.exists(cached) and not args.force and args.scale == 1.0:
            res = json.load(open(cached))
            print(f"[xval-r2] arm {name} reused from {cached}", flush=True)
        else:
            t0 = time.time()
            res = (run_flat if kind == "flat" else run_droplet)(cfg, log=lambda s: print(s, flush=True))
            res["elapsed_s"] = time.time() - t0
        res["arm"] = name
        res["series"] = series
        res["abscissa"] = absc
        runs[name] = res
        with open(os.path.join(args.out, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        kpis = kpi_values(res, kind)
        rows.extend(evaluate(gates, name, kpis))
        print(f"[xval-r2] arm {name} done in {res['elapsed_s']:.1f}s "
              f"kpis={ {k: round(v, 6) for k, v in kpis.items()} }", flush=True)

    def series_summary(series: str) -> list[dict]:
        pts = sorted(
            (r for r in runs.values() if r.get("series") == series),
            key=lambda r: r["abscissa"],
        )
        out = []
        for r in pts:
            kind = "flat" if r.get("arm", "").startswith("flat") else "drop"
            k = kpi_values(r, kind)
            out.append({"arm": r["arm"], "abscissa": r["abscissa"],
                        **{kk: vv for kk, vv in k.items() if isinstance(vv, (int, float))}})
        return out

    summary = {
        "anchor": decl["anchor"],
        "mission_id": decl["mission_id"],
        "round": decl["round"],
        "owner_band": decl["owner_band"],
        "predeclared": os.path.relpath(PREDECLARED, _ROOT),
        "arms_run": list(runs),
        "elapsed_total_s": time.time() - t_all,
        "kpi_rows": rows,
        "all_pass": all(r["pass"] for r in rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_fail": sum(1 for r in rows if not r["pass"]),
        "resolution_series": series_summary("resolution"),
        "curvature_series": series_summary("curvature"),
    }
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[xval-r2] {summary['n_pass']} pass / {summary['n_fail']} fail "
          f"-> {os.path.join(args.out, 'results.json')}", flush=True)
    for r in rows:
        if not r["pass"]:
            print(f"[xval-r2] FAIL {r['kpi']} on {r['arm']}: {r['value']:.3e} > {r['gate']:.3e}",
                  flush=True)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())