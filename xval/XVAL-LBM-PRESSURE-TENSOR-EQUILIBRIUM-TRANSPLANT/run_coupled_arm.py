#!/usr/bin/env python3
"""Coupled free-energy droplet arm: the Laplace-law benchmark (ref vs got).

Stage C of ``run_xval.py`` split out so it can run on its own (the full driver
re-runs the frozen-phi stages too).  Writes
``artifacts/xval_pressure_tensor_equilibrium/coupled_laplace.json`` and the
ref-vs-got table ``benchmark_table.csv``.

    python xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/run_coupled_arm.py
"""
from __future__ import annotations
import csv, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from tensorlbm.phasefield.pressure_tensor_equilibrium import run_coupled_droplet  # noqa: E402

ART = os.path.join(ROOT, "artifacts", "xval_pressure_tensor_equilibrium")
A, XI, GAMMA, TAU_F, TAU_PHI, TOL = 0.1, 2.0, 0.2, 1.0, 1.0, 0.15
KAPPA = A * XI**2 / 2.0


def run(R, steps, *, xi=XI, tau_phi=TAU_PHI, gamma=GAMMA):
    kappa = A * xi**2 / 2.0
    t0 = time.time()
    try:
        r = run_coupled_droplet(A=A, kappa=kappa, radius=R, steps=steps, tau_f=TAU_F,
                               tau_phi=tau_phi, Gamma=gamma, record_every=max(1, steps // 20))
        return dict(r.__dict__, wall_s=round(time.time() - t0, 1), nan=False)
    except RuntimeError as exc:
        return dict(radius=R, xi=xi, steps=steps, nan=True, error=str(exc)[:140],
                    wall_s=round(time.time() - t0, 1))


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    out = {"anchor": "XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT",
           "tolerance_pct": 100 * TOL,
           "params": {"A": A, "xi": XI, "kappa": KAPPA, "Gamma": GAMMA,
                      "tau_f": TAU_F, "tau_phi": TAU_PHI,
                      "criterion": "|dp_mean - sigma/R| / (sigma/R) < 15% (owner standard)"},
           "environment": {"torch": torch.__version__, "dtype": "float64", "device": "cpu"}}
    sweep = [run(R, 2000 if R <= 16 else 2000) for R in (10.0, 12.0, 16.0, 24.0)]
    out["laplace_sweep"] = sweep
    out["series"] = (sweep[2].get("series") if not sweep[2]["nan"] else [])
    out["sensitivity"] = [
        dict(label="tau_phi=1.5", **run(16.0, 1500, tau_phi=1.5)),
        dict(label="Gamma=0.1", **run(16.0, 1500, gamma=0.1)),
        dict(label="xi=3", **run(16.0, 1500, xi=3.0)),
        dict(label="tau_phi=0.8", **run(10.0, 400, tau_phi=0.8)),
    ]
    rows, ok = [], True
    for r in sweep:
        if r["nan"]:
            rows.append(dict(case=f"coupled droplet R={r['radius']:g}", quantity="dp/(sigma/R)",
                             ref=1.0, got="", err_pct="", tolerance_pct=15.0, verdict="FAIL",
                             note="non-finite run (honest FAIL)"))
            ok = False
            continue
        rows.append(dict(case=f"coupled droplet R={r['radius']:g} xi={r['xi']:g}",
                         quantity="dp_mean/(sigma/R)", ref=1.0,
                         got=round(1 + r["laplace_tail_rel_error"], 6),
                         err_pct=round(100 * r["laplace_tail_rel_error"], 3),
                         tolerance_pct=15.0,
                         verdict="PASS" if abs(r["laplace_tail_rel_error"]) < TOL else "FAIL", note=""))
        rows.append(dict(case=f"coupled droplet R={r['radius']:g} xi={r['xi']:g}",
                         quantity="sigma(int(P_rr-P_tt)dr)", ref=round(r["sigma_analytic"], 6),
                         got=round(r["sigma_measured"], 6),
                         err_pct=round(100 * r["sigma_rel_error"], 3), tolerance_pct=15.0,
                         verdict="PASS" if abs(r["sigma_rel_error"]) < TOL else "FAIL", note=""))
        ok = ok and rows[-1]["verdict"] == "PASS" and rows[-2]["verdict"] == "PASS"
    out["benchmark_rows"] = rows
    out["verdict"] = "PASS" if ok else "FAIL"
    json.dump(out, open(os.path.join(ART, "coupled_laplace.json"), "w"), indent=2)
    with open(os.path.join(ART, "benchmark_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "quantity", "ref", "got", "err_pct", "tolerance_pct", "verdict", "note"])
        w.writeheader()
        w.writerows(rows)
    json.dump({"rows": len(rows), "passed": sum(1 for r in rows if r["verdict"] == "PASS"),
               "verdict": out["verdict"],
               "worst_laplace_pct": max(abs(r["err_pct"]) for r in rows if r["err_pct"] != "" and "dp" in r["quantity"]),
               "worst_sigma_pct": max(abs(r["err_pct"]) for r in rows if r["err_pct"] != "" and "sigma" in r["quantity"])},
              open(os.path.join(ART, "benchmark_summary.json"), "w"), indent=2)
    print(json.dumps({"verdict": out["verdict"],
                      "laplace": [(r["radius"], round(100 * r["laplace_tail_rel_error"], 2)) for r in sweep if not r["nan"]],
                      "sigma_pct": [(r["radius"], round(100 * r["sigma_rel_error"], 2)) for r in sweep if not r["nan"]],
                      "sens": [(s["label"], s.get("nan"), None if s["nan"] else round(100 * s["laplace_tail_rel_error"], 2)) for s in out["sensitivity"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
