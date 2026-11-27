#!/usr/bin/env python3
"""XVAL-LBM-ADDEDMASS-KC-RESPONSE-ENVELOPE - confirmatory driver.

Scope
-----
Follow-up to ``XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY`` (m18997) and its
board audit (m19108, verdict L2 "approve with notes").  The audit's open item is
the LBM arm's honest FAIL: ``oscillatory_cylinder_inertia`` reported
``C_I = 1.55 / 1.83`` (+55 / +83 % vs the analytic 1.0) at KC = 0.49 / 0.94, and
the audit could only confirm that the number is *reproducible*, not that it is
*physics*.  This driver asks what that number actually is.

Hypotheses (fixed before the confirmatory run; exploratory probes are listed in
REPORT.md §2 as such):

H1 (spin-up, primary)  The discarded transient of the shipped arm (1.5 periods)
   is far shorter than the slowest viscous mode of the box,
   ``N_spin ~ beta * (L/2*pi/ (D))^2`` periods.  Prediction: the per-period
   phase drift of the first harmonic goes to zero and the harmonic-fit residual
   falls by >10x once ``transient >= 1.5 * N_spin``.
H2 (beta)   With H1 satisfied, C_I approaches the potential reference as beta
   grows; the *quadrature* force C_Q is not required to vanish (viscous stress).
H3 (KC)     With H1 satisfied, C_I is KC-independent to <5 % for KC <= 0.2.
H4 (falsification) If C_I at converged spin-up stays >15 % above the potential
   reference for every beta >= 100 and KC <= 0.2, then the estimator is biased
   and the honest FAIL cannot be repaired by more spin-up - reported as such.

Usage
-----
    python run_xval.py --group spinup          # one group
    python run_xval.py --all                   # everything (~10 min, 4 workers)
    python run_xval.py --list
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# keep the arm threads honest: never let BLAS/OpenMP oversubscribe the node
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("XVAL_THREADS", "4"))

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

CS = 1.0 / math.sqrt(3.0)


# --------------------------------------------------------------------------- #
# arm specification
# --------------------------------------------------------------------------- #
def _spinup(beta: float, box: float) -> float:
    """Predicted number of periods needed before the periodic state is reached.

    The slowest viscous mode of the periodic box has wavenumber 2*pi/L and decays
    as exp(-nu (2*pi/L)^2 t); expressed in forcing periods T this is
    ``N = nu T (L/2*pi)^2``... with nu T = D^2/beta this gives

        N_spin = beta * (L / (2*pi*D))^2 = beta * (box_over_D / (2*pi))^2 .
    """
    import reference as _ref
    return _ref.spinup_periods(beta, box)


def arms() -> list[dict]:
    """Every confirmatory arm, declared with its own spin-up target."""
    out: list[dict] = []

    def add(group, name, **kw):
        beta = kw.get("beta", 50.0)
        box = kw.get("box", 4.0)
        kw.setdefault("D", 12)
        kw.setdefault("kc", 0.2)
        kw.setdefault("u0", 0.02)
        kw.setdefault("measure", 5.0)
        n_spin = _spinup(beta, box)
        kw.setdefault("transient", round(1.5 * n_spin, 1))
        out.append({"group": group, "name": name, "n_spin_pred": n_spin, **kw})

    # G1 spin-up ladder: the same physical point, only the discarded transient differs
    for tr in (1.5, 2.0, 5.0, 10.0, 25.0, 50.0):
        add("spinup", f"spin{tr:04.1f}", transient=tr, beta=25.0, box=4.0)
    # G2 beta ladder at converged spin-up
    for beta in (25.0, 50.0, 100.0, 200.0, 400.0):
        add("beta", f"beta{beta:05.0f}", beta=beta, box=4.0)
    # G3 KC ladder at converged spin-up
    for kc in (0.05, 0.1, 0.2, 0.4, 0.8):
        add("kc", f"kc{kc:04.2f}", kc=kc, beta=50.0, box=4.0)
    # G4 box ladder (confinement) at converged spin-up
    for box in (4.0, 6.0, 8.0):
        add("box", f"box{box:02.0f}", box=box, beta=50.0)
    # G5 resolution at converged spin-up
    for D in (8, 12, 16):
        add("resolution", f"D{D:02d}", D=D, beta=50.0, box=4.0)
    # G6 the shipped arm vs the same physics point with enough spin-up
    add("repro", "shipped_kc05", kc=0.5, beta=50.0, box=4.0, transient=1.5, measure=3.0)
    add("repro", "fixed_kc05", kc=0.5, beta=50.0, box=4.0)
    add("repro", "shipped_kc10", kc=1.0, beta=50.0, box=4.0, transient=1.5, measure=3.0)
    add("repro", "fixed_kc10", kc=1.0, beta=50.0, box=4.0)
    return out


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def run_one(arm: dict) -> dict:
    from tensorlbm.oscillatory_cylinder_inertia import ArmCase, harmonic_decompose, run_arm

    import reference as ref

    case = ArmCase(
        arm["name"], D=arm["D"], kc=arm["kc"], beta=arm["beta"], u0=arm["u0"],
        box=arm["box"], transient=arm["transient"], measure=arm["measure"],
    )
    t0 = time.time()
    result, trace = run_arm(case, threads=int(os.environ.get("XVAL_THREADS", 4)))
    wall = time.time() - t0
    n = result["numerics"]
    T = n["T_steps"]

    # ---- spin-up diagnostics: per-period first harmonic over the measured window
    per_period = []
    n_win = max(1, int(round(case.measure)))
    for j in range(n_win):
        a = n["measure_start_step"] + j * T
        d = harmonic_decompose(trace["t"], trace["F_mem"], trace["omega"], t0=a, t1=a + T, nh=2)
        per_period.append({
            "window": j,
            "t0": a,
            "C_I": d["cos"] / (n["a0"] * result["geometry"]["n_solid"]),
            "C_Q": d["sin"] / (n["a0"] * result["geometry"]["n_solid"]),
            "resid_rel": d["resid_rel"],
            "phase_deg": math.degrees(math.atan2(d["sin"], d["cos"])),
        })
    ph = np.array([p["phase_deg"] for p in per_period], dtype=float)
    ci = np.array([p["C_I"] for p in per_period], dtype=float)
    drift = float(np.polyfit(np.arange(ph.size), ph, 1)[0]) if ph.size > 1 else 0.0

    # ---- references
    R = result["geometry"]["D_area"] / 2.0
    L = float(n["nx"])
    refs = {
        "c_a_unbounded": ref.c_a_unbounded(),
        "c_a_annulus_inscribed": ref.c_a_annulus_closed(R, L / 2.0),
        "c_a_annulus_areaeq": ref.c_a_annulus_closed(R, L / math.sqrt(math.pi)),
        "c_a_same_mask_area": result["reference"]["potential_same_mask_area"],
        "c_a_same_mask_halfway": result["reference"]["potential_same_mask_halfway"],
    }
    kpi = result["kpi"]
    dec = {"E1_mem_area": kpi["E1_mem_area"], "E2_trac_area": kpi["E2_trac_area"],
           "E1_mem_halfway": kpi["E1_mem_halfway"], "E2_trac_halfway": kpi["E2_trac_halfway"]}
    errs = {}
    for tag, v in dec.items():
        errs[tag] = {
            key: 100.0 * (v["C_I"] - val) / val
            for key, val in (("vs_unbounded", refs["c_a_unbounded"]),
                             ("vs_annulus_inscribed", refs["c_a_annulus_inscribed"]),
                             ("vs_annulus_areaeq", refs["c_a_annulus_areaeq"]),
                             ("vs_same_mask", result["reference"]["potential_same_mask_area"]))
        }

    out = {
        "arm": arm,
        "case": result["case"],
        "geometry": result["geometry"],
        "numerics": n,
        "achieved": result["achieved"],
        "kpi": dec,
        "harmonic": {"E1_mem": result["harmonic"]["E1_mem"], "U": result["harmonic"]["U"]},
        "references": refs,
        "errors_pct": errs,
        "per_period": per_period,
        "diagnostics": {
            "phase_drift_deg_per_period": drift,
            "C_I_per_period_spread": float(ci.max() - ci.min()),
            "resid_rel": result["harmonic"]["E1_mem"]["resid_rel"],
            "estimator_spread": result["estimator_spread_pct"],
            "acoustic_number_omega_L_over_cs": n["omega"] * n["nx"] / CS,
            "stokes_layer_over_R": math.sqrt(n["nu"] * T / math.pi) / R,
            "n_steps": n["n_steps"],
            "wall_s_rerun": round(wall, 2),
            "n_spin_pred": arm["n_spin_pred"],
            "transient_over_predicted": arm["transient"] / max(arm["n_spin_pred"], 1e-9),
        },
        "status": "ok",
    }
    dwell = max(1, len(trace["t"]) // 400)
    out["trace"] = {
        "t": trace["t"][::dwell],
        "F_mem": trace["F_mem"][::dwell],
        "U": trace["U"][::dwell],
        "omega": trace["omega"],
        "a0": trace["a0"],
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "runs"))
    args = ap.parse_args(argv)

    todo = arms()
    if args.list or not (args.all or args.group or args.name):
        for a in todo:
            print(f"{a['group']:>11s} {a['name']:>14s} D={a['D']:>2d} box={a['box']:g} "
                  f"kc={a['kc']:g} beta={a['beta']:g} u0={a['u0']:g} "
                  f"transient={a['transient']:g} (N_spin_pred={a['n_spin_pred']:.1f})")
        return 0
    if args.group:
        todo = [a for a in todo if a["group"] == args.group]
    if args.name:
        todo = [a for a in todo if a["name"] == args.name]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    # references-only artifacts (analytic + repo potential study) live in refs.json
    refs_path = HERE / "refs.json"
    if not refs_path.exists():
        write_reference_study(refs_path)

    t0 = time.time()
    done = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(run_one, todo):
                done.append(res)
                (out_root / f"{res['arm']['name']}.json").write_text(json.dumps(res, indent=1))
                print(f"[ok] {res['arm']['name']:>14s} C_I(E1 area)={res['kpi']['E1_mem_area']['C_I']:+.4f} "
                      f"C_Q={res['kpi']['E1_mem_area']['C_Q']:+.4f} "
                      f"drift={res['diagnostics']['phase_drift_deg_per_period']:+.3f} deg/per "
                      f"resid={res['diagnostics']['resid_rel']:.3f} "
                      f"wall={res['diagnostics']['wall_s_rerun']:.1f}s", flush=True)
    else:
        for a in todo:
            res = run_one(a)
            done.append(res)
            (out_root / f"{res['arm']['name']}.json").write_text(json.dumps(res, indent=1))
            print(f"[ok] {res['arm']['name']}", flush=True)

    print(f"[done] {len(done)} arms in {time.time()-t0:.1f}s")
    return 0


def write_reference_study(path: Path) -> None:
    """Reference-side numbers: analytic anchors + refinement of the repo potential."""
    import reference as ref

    study = {"analytic": ref.reference_table(), "annulus_checks": [], "repo_potential": []}
    for (R, ro) in ((6.0, 24.0), (8.0, 64.0), (8.0, 96.0)):
        row = {"R": R, "R_out": ro, "c_a_closed": ref.c_a_annulus_closed(R, ro)}
        for n in (2000, 8000, 32000):
            row[f"c_a_numeric_n{n}"] = ref.c_a_annulus_numeric(R, ro, n=n)["c_a_numeric"]
        study["annulus_checks"].append(row)
    # the repo same-mask periodic solve, refined at fixed L/R = 12 and 4
    for box in (4.0, 6.0, 12.0):
        for R in (8.0, 12.0, 16.0):
            nx = int(round(box * 2 * R))
            if nx > 96:
                continue
            try:
                rp = ref.repo_potential(nx, nx, R)
            except Exception as exc:  # pragma: no cover - recorded, not hidden
                study["repo_potential"].append({"box": box, "R": R, "status": repr(exc)})
                continue
            study["repo_potential"].append({
                "box": box, "R": R, "nx": nx, "status": "ok",
                "c_a_raw": rp["c_a"],
                "c_a_area_normalised": rp["c_a"] * math.pi * R**2 / rp["body_area"],
                "body_area": rp["body_area"],
                "c_a_annulus_inscribed": ref.c_a_annulus_closed(R, nx / 2.0),
                "c_a_annulus_areaeq": ref.c_a_annulus_closed(R, nx / math.sqrt(math.pi)),
            })
    path.write_text(json.dumps(study, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())