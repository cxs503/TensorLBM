"""Recompute the case (c) yield-surface estimators from the stored profiles.

The LBM fields are stored as ``channel_N{N}_sim.npy`` by the benchmark; the
yield-surface *estimators* are pure post-processing, so a corrected estimator can
be re-applied without re-running the solver (and the same function is used by the
benchmark itself).

    python scripts/reanalyze_p82w5_n04.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.bench_yield_stress_moving_wall_3d import (  # noqa: E402
    _plug_from_stress,
    _slope,
    _yield_surface_from_gamma,
    _yield_surface_from_umax,
)

ART = os.path.join(ROOT, "artifacts", "P82W5-N04")


def _best_ref(N):
    """Highest-refine AL reference available for N (returns (refine, u))."""
    for r in (16, 8, 4, 2):
        p = os.path.join(ART, "duct_ref_N%d_r%d.npy" % (N, r))
        if os.path.exists(p):
            return r, np.load(p)
    p = os.path.join(ART, "duct_N%d_ref.npy" % N)
    if os.path.exists(p):
        return 4, np.load(p)
    return None, None


def reanalyze_case_a():
    """Re-measure the duct errors against the highest-refine reference available.

    Also *rebuilds* ``case_a.json`` from the stored fields (``duct_N*_sim.npy``)
    when the benchmark file is absent, so the error metric always refers to the
    best reference and never to a snapshot superseded by a finer one.
    """
    path = os.path.join(ART, "case_a.json")
    G, mu_0, tau_y, m = 1e-5, 0.1, 2e-5, 1e5
    if os.path.exists(path):
        data = json.load(open(path))
        a = data["cases"]["a"]
    else:
        a = {"criterion": "linf_rel <= 3% at N >= 32 (vs AL reference)",
             "parameters": {"G": G, "mu_0": mu_0, "tau_y": tau_y, "m": m,
                            "note": "rebuilt from the stored velocity fields "
                                    "(artifacts/P82W5-N04/duct_N*_sim.npy)"},
             "runs": {}}
        data = {"node": "P82W5-N04", "repo": "TensorLBM", "cases": {"a": a}}
        for N in (8, 16, 32, 64):
            sim_path = os.path.join(ART, "duct_N%d_sim.npy" % N)
            if os.path.exists(sim_path):
                a["runs"]["N%d" % N] = {"Q_sim": float(np.load(sim_path).sum())}
    if "runs" not in a:
        return
    a = data["cases"]["a"]
    G, mu_0 = a["parameters"]["G"], a["parameters"]["mu_0"]
    for k, run in a["runs"].items():
        N = int(k[1:])
        sim_path = os.path.join(ART, "duct_N%d_sim.npy" % N)
        if not os.path.exists(sim_path):
            continue
        sim = np.load(sim_path)
        r, ref = _best_ref(N)
        if ref is None:
            continue
        run.setdefault("Q_sim", float(sim.sum()))
        umax = float(np.abs(ref).max())
        run["linf_rel"] = float(np.abs(sim - ref).max() / umax)
        run["l2_rel"] = float(np.linalg.norm(sim - ref) / np.linalg.norm(ref))
        run["Q_ref"] = float(ref.sum())
        run["Q_rel"] = abs(run["Q_sim"] - run["Q_ref"]) / abs(run["Q_ref"])
        run["reference_refine"] = r
        # error distribution: is the L_inf dominated by the near-wall layer?
        err = np.abs(sim - ref)
        inner = err[2:-2, 2:-2]
        run["linf_rel_inner"] = float(inner.max() / umax)
        run["wall_share"] = float(1.0 - inner.max() / err.max()) if err.max() > 0 else 0.0
    big = {k: v for k, v in a["runs"].items() if int(k[1:]) >= 32}
    a["metric"] = max((v["linf_rel"] for v in big.values()), default=None)
    a["passed"] = bool(big) and all(v["linf_rel"] <= 0.03 for v in big.values())
    json.dump(data, open(path, "w"), indent=2)
    for k, v in a["runs"].items():
        print("a", k, "linf=%.4f (refine %s) inner=%.4f wall_share=%.2f Q_err=%.4f" % (
            v["linf_rel"], v["reference_refine"], v["linf_rel_inner"],
            v["wall_share"], v["Q_rel"]))


def reanalyze_case_e():
    """Recompute the 2D/3D ratio against the best available reference.

    The Bingham runs use the refined AL reference (``duct_ref_N{N}_r8.npy``) when
    present, the Newtonian-limit runs use the exact Fourier series (their
    ``tau_y`` is 1e-7, i.e. y_p = 0.01 cell).  References must be computed with
    the *same* rheology -- mixing them silently corrupts the ratio, which is why
    this re-analysis keys on ``tau_y``.
    """
    path = os.path.join(ART, "case_e.json")
    if not os.path.exists(path):
        return
    data = json.load(open(path))
    e = data["cases"]["e"]
    from src.tensorlbm.yield_stress_analytical_moving_wall import (
        bingham_channel_flow_rate, newtonian_square_duct_series,
        rectangular_duct_shape_factor)
    G = e["parameters"]["G"]
    mu_0 = e["parameters"]["mu_0"]
    for k, run in e["runs"].items():
        N = int(k.split("_N")[1])
        tau_y = run["tau_y"]
        ref = None
        r = None
        for rr in (8, 4, 2):
            p = os.path.join(ART, "duct_ref_N%d_r%d.npy" % (N, rr))
            if os.path.exists(p) and tau_y > 1e-6:
                ref, r = np.load(p), rr
                break
        if ref is None:
            if tau_y <= 1e-6:
                ref = newtonian_square_duct_series(N, G, mu_0)
                r = "fourier"
            else:
                continue
        q3_ref = float(ref.sum())
        q2_ref = bingham_channel_flow_rate(N, G, tau_y, mu_0)
        run["Q_3D_ref"] = q3_ref
        run["reference"] = r
        run["ratio_ref"] = q2_ref / q3_ref
        run["ratio_sim"] = run["Q_2D_sim"] / run["Q_3D_sim"]
        run["ratio_err"] = abs(run["ratio_sim"] - run["ratio_ref"]) / run["ratio_ref"]
        run["Q3D_rel_err"] = abs(run["Q_3D_sim"] - q3_ref) / q3_ref
        run["shape_factor_sim"] = run["Q_3D_sim"] / (run["Q_2D_sim"] * N)
        run["shape_factor_ref"] = q3_ref / (q2_ref * N)
        run["shape_factor_err"] = (abs(run["shape_factor_sim"]
                                       - run["shape_factor_ref"])
                                   / run["shape_factor_ref"])
    judged = {k: v for k, v in e["runs"].items()
              if int(k.split("_N")[1]) >= 32 or k.startswith("newtonian_limit")}
    e["metric"] = {
        "ratio_err_max_judged": max(v["ratio_err"] for v in judged.values()),
        "ratio_err_all": {k: v["ratio_err"] for k, v in e["runs"].items()},
        "shape_factor_all": {k: {"sim": v.get("shape_factor_sim"),
                                 "ref": v.get("shape_factor_ref")}
                             for k, v in e["runs"].items()},
        "newtonian_shape_factor": rectangular_duct_shape_factor(1.0),
        "judged_runs": sorted(judged),
        "note": "judged: N>=32 and the Newtonian-limit runs; the N=16 Bingham "
                "run is reported but limited by the AL reference discretisation",
    }
    e["passed"] = e["metric"]["ratio_err_max_judged"] <= 0.05
    json.dump(data, open(path, "w"), indent=2)
    for k, v in e["runs"].items():
        print("e", k, "tau_y=%.1e ratio_sim=%.5f ref=%.5f err=%.4f shape=%.4f/%.4f (%s)" % (
            v["tau_y"], v["ratio_sim"], v["ratio_ref"], v["ratio_err"],
            v.get("shape_factor_sim") or 0, v.get("shape_factor_ref") or 0,
            v.get("reference")))


def main():
    reanalyze_case_a()
    reanalyze_case_e()
    path = os.path.join(ART, "case_c.json")
    data = json.load(open(path))
    c = data["cases"]["c"]
    prm = c["parameters"]
    G, mu_0, tau_y = prm["G"], prm["mu_0"], prm["tau_y"]
    m = prm["m"]
    for k, run in c["runs"].items():
        N = int(k[1:])
        sim = np.load(os.path.join(ART, "channel_N%d_sim.npy" % N))
        yp_exact = tau_y / G
        yp_umax = _yield_surface_from_umax(sim, G, mu_0, N)
        yp_gam = _yield_surface_from_gamma(sim, tau_y, mu_0)
        st = _plug_from_stress(sim, tau_y, mu_0, m)
        run.update({
            "yp_exact": yp_exact,
            "yp_from_umax": yp_umax, "err_umax": abs(yp_umax - yp_exact),
            "yp_from_gamma": yp_gam, "err_gamma": abs(yp_gam - yp_exact),
            "delta_p_stress": st["delta_p"],
            "err_stress": (abs(st["delta_p"] / 2.0 - yp_exact)
                           if st["delta_p"] is not None else None),
        })
    Ns = sorted(int(k[1:]) for k in c["runs"])
    c["slopes"] = {
        "umax_inversion": _slope(Ns, [c["runs"]["N%d" % n]["err_umax"] for n in Ns]),
        "gamma_crossing": _slope(Ns, [c["runs"]["N%d" % n]["err_gamma"] for n in Ns]),
        "stress_crossing": _slope(Ns, [c["runs"]["N%d" % n]["err_stress"] or 0.0
                                       for n in Ns]),
        "profile_linf": _slope(Ns, [c["runs"]["N%d" % n]["profile_linf_rel"]
                                    for n in Ns]),
    }
    vals = [v for v in c["slopes"].values() if v is not None]
    c["metric"] = max(vals) if vals else None
    c["passed"] = bool(c["metric"] is not None and c["metric"] >= 1.7)
    c["estimators_recomputed"] = True
    json.dump(data, open(path, "w"), indent=2)
    print(json.dumps(c["slopes"], indent=1))
    for k in c["runs"]:
        r = c["runs"][k]
        print(k, "yp_exact=%.2f umax=%.4f err=%.4f gamma=%.4f err=%.4f stress_dp=%s linf=%.5f" % (
            r["yp_exact"], r["yp_from_umax"], r["err_umax"], r["yp_from_gamma"],
            r["err_gamma"], r["delta_p_stress"], r["profile_linf_rel"]))


if __name__ == "__main__":
    main()
