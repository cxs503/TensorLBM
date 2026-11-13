#!/usr/bin/env python3
"""XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — harness (round 2).

Anchor
------
Two *independent* implementations of one static contact-angle wall condition
have to reproduce the prescribed Young angle on the same benchmark and agree
with each other.  The open-source origin of the anchor is the dual
contact-angle machinery of ``lssfau/walberla`` (``oss_solutions`` row 281): a
free-surface side that builds a *geometric* ghost from cached sin/cos and the
wall normal, and a phase-field side that derives a wall condition for the order
parameter from the interface thickness and the angle — two code paths, one
physics.

Arms (see :mod:`tensorlbm.wetting.two_impl_contact_angle`)
----------------------------------------------------------
``dirichlet``        wall value ``psi_w`` from ``cos(theta) = (3p-p^3)/2``.
``neumann_flux``     wall flux ``kappa d_n psi = (3/4) sigma cos(theta)(1-psi^2)``.
``geometric_ghost``  geometric ghost ``psi_0 + cot(theta)|d_t psi|`` (no wall
                     energy at all).  ``neumann_flux`` and ``geometric_ghost``
                     are the predeclared *independent pair*.

Usage
-----
    python3 run_xval.py sweep          # run the predeclared matrix (process pool)
    python3 run_xval.py summarize      # summary.json + gates
    python3 run_xval.py one --arm neumann_flux --theta 90 --grid baseline

Every number in the report is rebuilt from ``runs/<tag>/result.json``; nothing
is copied by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

PREDECLARED = json.loads((HERE / "predeclared.json").read_text())
RUNS = HERE / "runs"
GRIDS = {
    "baseline": PREDECLARED["matrix"]["baseline_grid"],
    "refined": PREDECLARED["matrix"]["refined_grid"],
    "fine_interface": PREDECLARED["matrix"]["fine_interface_grid"],
}


def tag_of(arm: str, theta_deg: float, grid: str, xi_override: float | None = None) -> str:
    xi = GRIDS[grid]["xi"] if xi_override is None else xi_override
    return f"{arm}__th{theta_deg:05.1f}__{grid}__xi{xi:g}"


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def run_one(args) -> dict:
    arm, theta_deg, grid, xi_override = args
    import torch  # imported inside the worker: one thread per process

    torch.set_num_threads(1)
    from tensorlbm.wetting.two_impl_contact_angle import TwoImplConfig, run_arm

    g = dict(GRIDS[grid])
    xi = g.pop("xi")
    if xi_override is not None:
        xi = float(xi_override)
    radius = g.pop("radius")
    cfg = TwoImplConfig(interface_width=xi, **g)
    theta = math.radians(theta_deg)
    t0 = time.time()
    out = run_arm(
        arm,
        theta,
        cfg=cfg,
        radius=radius,
        max_iter=PREDECLARED["numerics"]["max_iter"],
        omega=PREDECLARED["numerics"]["omega"],
        tol=PREDECLARED["numerics"]["tol"],
    )
    psi = out.pop("psi")
    out.pop("cfg", None)
    th = PREDECLARED["predeclared_thresholds"]

    apex_rel = out["apex_height"] / cfg.interface_width
    rms_rel = out["circle_rms"] / cfg.interface_width
    finite = all(math.isfinite(out[k]) for k in ("theta_cap_deg", "theta_circle_deg"))
    measurable = bool(
        finite
        and apex_rel >= th["measurable_apex_over_xi"]
        and rms_rel <= th["measurable_circle_rms_over_xi"]
    )
    err_cap = (
        abs(out["theta_cap_deg"] - theta_deg) / theta_deg * 100.0 if math.isfinite(out["theta_cap_deg"]) else None
    )
    err_circ = (
        abs(out["theta_circle_deg"] - theta_deg) / theta_deg * 100.0
        if math.isfinite(out["theta_circle_deg"])
        else None
    )
    passed = bool(
        measurable
        and err_cap is not None and err_circ is not None
        and err_cap <= th["accept_pct"] and err_circ <= th["accept_pct"]
    )

    payload = {
        "tag": tag_of(arm, theta_deg, grid, xi_override),
        "anchor": PREDECLARED["anchor"],
        "arm": arm,
        "grid": grid,
        "geometry": {"ny": g["ny"], "nx": g["nx"], "radius": radius, "surface_tension": g["surface_tension"]},
        "interface_width": xi,
        "parameters": {
            "a": cfg.a,
            "kappa": cfg.kappa,
            "sigma": cfg.surface_tension,
            "xi": xi,
            "xi_over_radius": xi / radius,
            "omega": PREDECLARED["numerics"]["omega"],
            "dyadic": {"Laplacian": 0, "viscosity": 0, "Reynolds": 0, "capillary": 0},
        },
        "theta_prescribed_deg": theta_deg,
        "theta_ref_deg": theta_deg,  # analytic Young law: theta_ref == theta of the wall energy
        "theta_cap_deg": out["theta_cap_deg"],
        "theta_circle_deg": out["theta_circle_deg"],
        "err_cap_pct": err_cap,
        "err_circle_pct": err_circ,
        "estimator_gap_deg": out["estimator_gap_deg"],
        "estimators_agree": bool(
            math.isfinite(out["estimator_gap_deg"]) and out["estimator_gap_deg"] <= th["estimator_agreement_deg"]
        ),
        "measurable": measurable,
        "apex_over_xi": apex_rel,
        "circle_rms_over_xi": rms_rel,
        "apex_height": out["apex_height"],
        "R_curv": out["R_curv"],
        "circle_rms": out["circle_rms"],
        "gas_supersaturation": out["gas_supersaturation"],
        "wall_angle_readback_deg": out["wall_angle_readback_deg"],
        "wall_value_mean": out["wall_value_mean"],
        "iterations": out["iterations"],
        "converged": out["converged"],
        "drift": out["drift"],
        "mass_error": out["mass_error"],
        "psi_min": out["psi_min"],
        "psi_max": out["psi_max"],
        "pass": passed,
        "wall_seconds": time.time() - t0,
    }
    _write(payload)
    if theta_deg in (70.0, 110.0) and grid in ("baseline", "refined"):
        np.savez(RUNS / payload["tag"] / "psi.npz", psi=psi.detach().cpu().numpy())
    return payload


def _write(payload: dict) -> None:
    d = RUNS / payload["tag"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# sweep / summary
# ---------------------------------------------------------------------------
def sweep(workers: int = 8) -> None:
    jobs = []
    angles = PREDECLARED["matrix"]["angles_deg"]
    for arm in PREDECLARED["arms"]:
        for th in angles:
            jobs.append((arm, th, "baseline", None))
    for arm in PREDECLARED["arms"]:
        for th in PREDECLARED["matrix"]["angles_refined_deg"]:
            jobs.append((arm, th, "refined", None))
    # interface-width sensitivity at fixed drop radius (baseline grid geometry)
    for arm in PREDECLARED["independent_pair"]:
        for xi in PREDECLARED["matrix"]["xi_sensitivity"]:
            for th in (70.0, 110.0):
                jobs.append((arm, th, "baseline", xi))
    # shallow-cap window opened by refining the interface at fixed theta/R
    for arm in PREDECLARED["independent_pair"]:
        jobs.append((arm, 45.0, "fine_interface", None))

    print(f"{len(jobs)} runs on {workers} workers", flush=True)
    with Pool(workers) as pool:
        for res in pool.imap_unordered(_report, jobs):
            print(res, flush=True)


def _report(args) -> str:
    p = run_one(args)
    return (
        f"{p['tag']:52s} cap={p['theta_cap_deg']:7.2f} circ={p['theta_circle_deg']:7.2f} "
        f"gap={p['estimator_gap_deg']:5.2f} meas={int(p['measurable'])} pass={int(p['pass'])} "
        f"gas={p['gas_supersaturation']:+.3f} it={p['iterations']:6d} {p['wall_seconds']:.0f}s"
    )


def summarize() -> dict:
    th = PREDECLARED["predeclared_thresholds"]
    rows = []
    for f in sorted(RUNS.glob("*/result.json")):
        rows.append(json.loads(f.read_text()))
    if not rows:
        raise SystemExit("no runs")

    measurable = [r for r in rows if r["measurable"] and r["grid"] in ("baseline", "refined")]
    pair = PREDECLARED["independent_pair"]
    arms = PREDECLARED["arms"]

    def cell(arm, grid, theta):
        for r in rows:
            if r["arm"] == arm and r["grid"] == grid and r["theta_prescribed_deg"] == theta:
                return r
        return None

    cross = []
    for grid in ("baseline", "refined"):
        for theta in PREDECLARED["matrix"]["angles_deg"]:
            a, b = cell(pair[0], grid, theta), cell(pair[1], grid, theta)
            if a is None or b is None or not (a["measurable"] and b["measurable"]):
                continue
            d = abs(a["theta_cap_deg"] - b["theta_cap_deg"]) / theta * 100.0
            cross.append(
                {
                    "grid": grid,
                    "theta_prescribed_deg": theta,
                    "theta_neumann_flux_deg": a["theta_cap_deg"],
                    "theta_geometric_ghost_deg": b["theta_cap_deg"],
                    "diff_pct": d,
                    "pass": bool(d <= th["cross_arm_pct"]),
                }
            )

    per_arm = {}
    for arm in arms:
        sub = [r for r in measurable if r["arm"] == arm]
        errs = [max(r["err_cap_pct"], r["err_circle_pct"]) for r in sub if r["err_cap_pct"] is not None]
        per_arm[arm] = {
            "n_measurable": len(sub),
            "n_pass": sum(1 for r in sub if r["pass"]),
            "median_err_pct": float(np.median(errs)) if errs else None,
            "max_err_pct": float(max(errs)) if errs else None,
            "n_not_measurable": sum(1 for r in rows if r["arm"] == arm and not r["measurable"]),
        }

    anchor_pass = bool(
        all(per_arm[a]["n_measurable"] > 0 and per_arm[a]["n_pass"] == per_arm[a]["n_measurable"] for a in pair)
        and cross
        and all(c["pass"] for c in cross)
    )
    summary = {
        "anchor": PREDECLARED["anchor"],
        "n_rows": len(rows),
        "thresholds": th,
        "per_arm": per_arm,
        "cross_arm": cross,
        "anchor_pass": anchor_pass,
        "verdict": "PASS" if anchor_pass else "FAIL",
        "rows": [
            {
                k: r[k]
                for k in (
                    "tag", "arm", "grid", "interface_width", "theta_prescribed_deg",
                    "theta_cap_deg", "theta_circle_deg", "err_cap_pct", "err_circle_pct",
                    "estimator_gap_deg", "estimators_agree", "measurable", "pass",
                    "gas_supersaturation", "iterations", "converged", "drift",
                    "mass_error", "wall_angle_readback_deg", "apex_over_xi",
                    "circle_rms_over_xi", "wall_seconds",
                )
            }
            for r in sorted(rows, key=lambda r: (r["arm"], r["grid"], r["interface_width"], r["theta_prescribed_deg"]))
        ],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "summarize", "one"])
    ap.add_argument("--arm", default="neumann_flux")
    ap.add_argument("--theta", type=float, default=90.0)
    ap.add_argument("--grid", default="baseline", choices=sorted(GRIDS))
    ap.add_argument("--xi", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "sweep":
        sweep(args.workers)
    elif args.cmd == "summarize":
        s = summarize()
        print(json.dumps({k: s[k] for k in ("verdict", "per_arm", "cross_arm")}, indent=2))
    else:
        print(_report((args.arm, args.theta, args.grid, args.xi)))


if __name__ == "__main__":
    main()