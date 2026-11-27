"""Round-4 XVAL driver: **transient / impulsive-start** verification of the
periodic-x + constant-dp/dx channel (``joconnor22/LIFE`` tag
``single-phase-periodic-x-plus-constant-dpdx-driving``).

Run (cluster):  python benchmarks/cross_validation/periodic_channel_dpdx/run_xval_r4.py --workers 8
Quick (laptop): python benchmarks/cross_validation/periodic_channel_dpdx/run_xval_r4.py --quick
Writes:         benchmarks/cross_validation/periodic_channel_dpdx/out/xval_life_periodic_dpdx_r4.json

Why a fourth round
------------------
Rounds 1-3 verified the **steady** state: both ported arms reach the exact
plane-Poiseuille parabola to <1 %, second order in the wall distance ``H``.
A steady benchmark cannot see a time error in the driving: a constant
``O(\\Delta t)`` offset in the reported velocity, or a lag in the force
injection, leaves the steady field unchanged.  This round therefore compares
the **whole start-up history** with the exact Fourier-series solution of the
impulsively started channel (:func:`tensorlbm.periodic_channel_dpdx.
poiseuille_transient`).

Three scans
-----------
``transient_line``  ``ny in {17, 33, 65}`` x both arms, sampled at the *same
                    nondimensional diffusive time* ``tau = nu t / H^2 in
                    {0.02, 0.05, 0.1, 0.2, 0.4, 0.8}``.  Two error norms are
                    stored: relative to the steady solution's L2 norm (the
                    meaningful one -- the instantaneous norm goes to zero at
                    ``t -> 0`` and would make any early-time error look 100 %)
                    and relative to the instantaneous norm (reported for
                    completeness, expected to be large at small ``tau``).
``resolution_order`` the same ``tau = 0.1`` rows across ``H`` give the observed
                    order of the transient error, through the resolution-
                    explicit :func:`observed_order` introduced in round 3.
``early_time``      the first six steps of the run, where the only analytic
                    content is ``u = g t + const``: measures the *constant*
                    velocity offset that each force convention reports, and
                    checks it against the closed form ``(1 - omega)/2 * g``
                    for the OSS ``guo_half_frozen`` arm and zero for the
                    repo's ``split_guo`` arm.

All numbers land in the JSON artifact; the report and the platform payload are
generated from it (``make_r4_report.py``), never hand-typed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.periodic_channel_dpdx import (  # noqa: E402
    FORCE_GUO_HALF_FROZEN,
    FORCE_SPLIT_GUO,
    WALL_HALFWAY_BB,
    WALL_REGULARISED,
    DpdxChannel,
    observed_order,
    poiseuille_analytic,
    poiseuille_transient,
)

TAU = 0.5375
NU = (TAU - 0.5) / 3.0
U_TARGET = 0.01
GATE = 0.15
TARGET_3PCT = 0.03

ARMS = {
    "PORT-REG": {"wall": WALL_REGULARISED, "scheme": FORCE_GUO_HALF_FROZEN},
    "PORT-BB": {"wall": WALL_HALFWAY_BB, "scheme": FORCE_SPLIT_GUO},
}
TAUS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def h_for(ny: int, wall: str) -> float:
    return float(ny - 1) if wall == WALL_REGULARISED else float(ny - 2)


def g_for(ny: int, wall: str, u_target: float = U_TARGET) -> float:
    return 8.0 * NU * u_target / h_for(ny, wall) ** 2


# ---------------------------------------------------------------------------
# one transient run
# ---------------------------------------------------------------------------
def run_transient(arm: str, ny: int) -> dict:
    wall = ARMS[arm]["wall"]
    h = h_for(ny, wall)
    g = g_for(ny, wall)
    ch = DpdxChannel(2, ny, TAU, g, wall=wall, force_scheme=ARMS[arm]["scheme"])
    u_ss = poiseuille_analytic(ny, NU, g, wall=wall)[1]
    norm_ss = float(np.linalg.norm(u_ss))
    m0 = ch.mass()
    rows = []
    for tau_nd in TAUS:
        nsteps = int(round(tau_nd * h * h / NU))
        while ch.steps < nsteps:
            ch.step()
        _, ua, _, _, modes = poiseuille_transient(ny, NU, g, float(ch.steps), wall=wall)
        y = ch.y_centres()
        u = ch.profile()
        y_c = u.size // 2
        rows.append(
            {
                "tau_nd": tau_nd,
                "y": [float(v) for v in y],
                "u_profile": [float(v) for v in u],
                "u_analytic": [float(v) for v in ua],
                "steps": ch.steps,
                "rel_l2_over_steady": float(np.linalg.norm(u - ua) / norm_ss),
                "rel_l2_over_instant": float(np.linalg.norm(u - ua) / np.linalg.norm(ua)),
                "u_centre": float(u[y_c]),
                "u_centre_analytic": float(ua[y_c]),
                "uc_ratio": float(u.max() / u_ss.max()),
                "uc_ratio_analytic": float(ua.max() / u_ss.max()),
                "mass_drift": float((ch.mass() - m0) / m0),
                "modes": modes,
            }
        )
    return {"arm": arm, "ny": ny, "h": h, "g": g, "rows": rows}


def run_early(arm: str, tau_relax: float, nsteps: int = 6) -> dict:
    """First ``nsteps`` steps: reported ``u/g - t`` must be a constant offset."""
    wall = ARMS[arm]["wall"]
    ny = 17
    h = h_for(ny, wall)
    nu = (tau_relax - 0.5) / 3.0
    g = 8.0 * nu * U_TARGET / h**2
    ch = DpdxChannel(2, ny, tau_relax, g, wall=wall, force_scheme=ARMS[arm]["scheme"])
    hist = []
    for _ in range(nsteps):
        ch.step()
        u = ch.profile()
        hist.append(float(u[u.size // 2] / g - ch.steps))
    omega = 1.0 / tau_relax
    predicted = 0.0 if arm == "PORT-BB" else 0.5 * (1.0 - omega)
    return {
        "arm": arm,
        "tau": tau_relax,
        "omega": omega,
        "offset_measured": hist[-1],
        "offset_span": hist,
        "offset_predicted": predicted,
        "offset_error": hist[-1] - predicted,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--taus", default="", help="comma list of nondimensional times, e.g. 0.02,0.05,0.1,0.2"
    )
    args = ap.parse_args()

    global TAUS
    if args.taus:
        TAUS = tuple(float(v) for v in args.taus.split(","))
    nys = (17, 33) if args.quick else (17, 33, 65)
    t0 = time.time()
    jobs = [(arm, ny) for arm in ARMS for ny in nys]
    results = []
    if args.quick or args.workers <= 1:
        for arm, ny in jobs:
            results.append(run_transient(arm, ny))
            print("done", arm, ny, flush=True)
    else:
        from concurrent.futures import as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_transient, arm, ny): (arm, ny) for arm, ny in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())
                print("done", futs[fut], flush=True)
    early = [
        run_early(arm, tau_r)
        for arm in ARMS
        for tau_r in ((TAU, 0.6) if args.quick else (TAU, 0.6, 0.8))
    ]

    # -- resolution order of the transient error (fixed nondimensional time) --
    order_rows = []
    for arm in ARMS:
        by_ny = {r["ny"]: r for r in results if r["arm"] == arm}
        for k in range(len(nys) - 1):
            coarse, fine = by_ny[nys[k]], by_ny[nys[k + 1]]
            rc = next(r for r in coarse["rows"] if r["tau_nd"] == 0.1)
            rf = next(r for r in fine["rows"] if r["tau_nd"] == 0.1)
            order_rows.append(
                {
                    "arm": arm,
                    "ny_pair": [nys[k], nys[k + 1]],
                    "h_pair": [coarse["h"], fine["h"]],
                    "p_rel_l2": observed_order(
                        rc["rel_l2_over_steady"], rf["rel_l2_over_steady"], coarse["h"], fine["h"]
                    ),
                    "p_uc": observed_order(
                        abs(rc["uc_ratio"] - rc["uc_ratio_analytic"]),
                        abs(rf["uc_ratio"] - rf["uc_ratio_analytic"]),
                        coarse["h"],
                        fine["h"],
                    ),
                }
            )

    # centreline-velocity growth error vs the analytic value (u_c/u_ss)
    for r in results:
        for row in r["rows"]:
            row["uc_ratio_err"] = row["uc_ratio"] - row["uc_ratio_analytic"]

    flat = [
        row | {"arm": r["arm"], "ny": r["ny"], "h": r["h"]} for r in results for row in r["rows"]
    ]
    worst = max(flat, key=lambda x: x["rel_l2_over_steady"])
    worst_uc = max(flat, key=lambda x: abs(x["uc_ratio_err"]))
    mass_worst = max(abs(x["mass_drift"]) for x in flat)

    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"

    rep = {
        "meta": {
            "round": 4,
            "anchor": "XVAL-LBM-SINGLE-PHASE-PERIODIC-X-PLUS-CONSTANT-DP",
            "oss_reference": "joconnor22/LIFE single-phase-periodic-x-plus-constant-dpdx-driving",
            "generated_unix": time.time(),
            "wall_seconds": time.time() - t0,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "git_commit": commit,
            "workers": args.workers,
            "quick": args.quick,
        },
        "lattice_units": {
            "tau": TAU,
            "nu": NU,
            "u_max_target": U_TARGET,
            "gate": GATE,
            "target_3pct": TARGET_3PCT,
            "nondim_times": list(TAUS),
        },
        "arms": {k: {"wall": v["wall"], "force_scheme": v["scheme"]} for k, v in ARMS.items()},
        "transient_line": results,
        "transient_rows": flat,
        "resolution_order": order_rows,
        "early_time": early,
        "verdict": {
            "worst_rel_l2_over_steady": worst["rel_l2_over_steady"],
            "worst_row": {k: worst[k] for k in ("arm", "ny", "tau_nd")},
            "gate_15pct_pass": bool(worst["rel_l2_over_steady"] < GATE),
            "gate_3pct_pass": bool(worst["rel_l2_over_steady"] < TARGET_3PCT),
            "worst_uc_ratio_abs_error": abs(worst_uc["uc_ratio_err"]),
            "worst_uc_row": {k: worst_uc[k] for k in ("arm", "ny", "tau_nd")},
            "worst_mass_drift": mass_worst,
            "min_order_p_rel_l2": min(o["p_rel_l2"] for o in order_rows),
            "early_offset_max_abs_error": max(abs(e["offset_error"]) for e in early),
        },
    }
    out = pathlib.Path(__file__).resolve().parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("xval_life_periodic_dpdx_r4" + ("_quick" if args.quick else "") + ".json")
    path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["verdict"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
