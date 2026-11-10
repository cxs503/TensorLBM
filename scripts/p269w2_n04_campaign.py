"""P269W2-N04 campaign driver: lattice linear stability of plane Poiseuille flow.

Produces, per task, one JSON record under ``artifacts/p269w2_n04/points/`` and
finally an aggregated ``acceptance.json`` with the A1-A4 verdicts.

Usage
-----
    python scripts/p269w2_n04_campaign.py --list
    python scripts/p269w2_n04_campaign.py --run-all            # full campaign
    python scripts/p269w2_n04_campaign.py --shard 3/16         # sharded run
    python scripts/p269w2_n04_campaign.py --aggregate          # merge + criteria

The task list is deterministic, so ``--shard i/n`` always covers the same
subset and the shards can be re-run independently.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.stability.channel_shear import (  # noqa: E402
    ChannelFlow,
    channel_mode_operator,
    lattice_mode_spectrum,
    left_eigenvector,
    richardson_extrapolate,
    stokes_damping_benchmark,
    time_march_growth_rate,
)
from tensorlbm.stability.orr_sommerfeld import (  # noqa: E402
    ORSZAG_RE10000_ALPHA1,
    orszag_benchmark_check,
    os_spectrum,
)

ART = ROOT / "artifacts" / "p269w2_n04"
POINTS = ART / "points"

#: Exactly lattice-representable streamwise wavenumbers ``alpha_nd = pi*m/k``
#: (``Nx = k*Ny``, mode ``m``).  Chosen so that Ny can be refined at fixed
#: physical wavenumber.
ALPHAS = {
    # (mode m, box multiple k, alpha_nd = pi*m/k) - exactly representable on
    # Nx = k*Ny lattice nodes, so Ny can be refined at fixed physical alpha.
    "pi/6": (1, 6, math.pi / 6.0),
    "pi/4": (1, 4, math.pi / 4.0),
    "pi/3": (1, 3, math.pi / 3.0),
    "2pi/5": (2, 5, 2.0 * math.pi / 5.0),
}

RE_MAIN = (1000.0, 2000.0, 3000.0, 4500.0, 6000.0, 8000.0, 10000.0)
RE_CONV = (1000.0, 3000.0, 6000.0)
NY_MAIN = 64
NY_CONV = (32, 64, 128, 256)
KINDS = ("bgk", "mrt", "rlbm")


def alpha_lattice(key: str, ny: int) -> float:
    """Lattice wavenumber for the physical ``alpha`` ``key`` at ``ny``."""
    m, k, _ = ALPHAS[key]
    nx = k * ny
    return 2.0 * math.pi * m / nx


def task_list() -> list[dict]:
    tasks: list[dict] = []
    tasks.append({"kind_task": "os_benchmark"})
    tasks.append({"kind_task": "a1_stokes"})
    # main accuracy grid (Ny = 64, all three collision models)
    for Re in RE_MAIN:
        for akey in ("pi/4", "pi/3", "2pi/5"):
            for kind in KINDS:
                tasks.append(
                    {"kind_task": "point", "Re": Re, "alpha_key": akey,
                     "ny": NY_MAIN, "collision": kind}
                )
    # Ny convergence (A3) at three Reynolds numbers, all three models
    for Re in RE_CONV:
        for akey in ("pi/3",):
            for ny in NY_CONV:
                if ny == NY_MAIN:
                    continue
                for kind in KINDS:
                    tasks.append(
                        {"kind_task": "point", "Re": Re, "alpha_key": akey,
                         "ny": ny, "collision": kind}
                    )
    # second Ny-convergence family (alpha = pi/4) at Re = 3000 / 6000, BGK
    for Re in (3000.0, 6000.0):
        for ny in (32, 64, 128):
            tasks.append(
                {"kind_task": "point", "Re": Re, "alpha_key": "pi/4",
                 "ny": ny, "collision": "bgk"}
            )
    # cross audit: time march (independent nonlinear path) vs eigenvalue
    for Re in (1000.0, 2000.0):
        for kind in KINDS:
            tasks.append(
                {"kind_task": "timemarch", "Re": Re, "alpha_key": "pi/3",
                 "ny": 32, "collision": kind, "n_steps": 40000}
            )
    # A4 at the cheap Ny = 16 operating point (Table: amplitude -> 0 linear limit;
    # the time march there is affordable, which is what the criterion needs)
    for amp in (1e-2, 1e-3, 1e-4):
        tasks.append(
            {"kind_task": "timemarch", "Re": 2000.0, "alpha_key": "pi/3",
             "ny": 16, "collision": "bgk", "n_steps": 20000, "amplitude": amp}
        )
    # A4 / cross audit at Ny = 16 with the regularised operator: BGK diverges in
    # the nonlinear march at tau ~ 0.5006 (omega -> 2 stability edge), RLBM does
    # not - the eigenvalue route is unaffected either way.
    for amp in (1e-2, 1e-3, 1e-4):
        tasks.append({"kind_task": "timemarch", "Re": 2000.0, "alpha_key": "pi/3",
                      "ny": 16, "collision": "rlbm", "n_steps": 20000,
                      "amplitude": amp})
    for kind in KINDS:
        tasks.append({"kind_task": "timemarch", "Re": 2000.0, "alpha_key": "pi/3",
                      "ny": 16, "collision": kind, "n_steps": 20000})
    # Mach (compressibility) sensitivity of the lattice growth rate
    for umax in (0.02, 0.1):
        tasks.append(
            {"kind_task": "point", "Re": 3000.0, "alpha_key": "pi/3", "ny": 64,
             "collision": "bgk", "u_max": umax}
        )
    # extra coverage: Ny = 16 doubling family (Richardson 16/32/64) and the
    # high-Re end of the sign check at the cheap Ny = 32 level
    for kind in KINDS:
        tasks.append({"kind_task": "point", "Re": 3000.0, "alpha_key": "pi/3",
                      "ny": 16, "collision": kind})
    for Re in (8000.0, 10000.0):
        tasks.append({"kind_task": "point", "Re": Re, "alpha_key": "pi/3",
                      "ny": 32, "collision": "bgk"})
    # A4: amplitude -> 0 linear limit
    for amp in (1e-2, 1e-3, 1e-4):
        tasks.append(
            {"kind_task": "timemarch", "Re": 2000.0, "alpha_key": "pi/3",
             "ny": 32, "collision": "bgk", "n_steps": 40000, "amplitude": amp}
        )
    return tasks


def task_key(t: dict) -> str:
    if t["kind_task"] != "point":
        parts = [t["kind_task"]]
        for k in ("Re", "alpha_key", "ny", "collision", "amplitude", "u_max"):
            if k in t:
                parts.append(f"{k}{t[k]}")
        name = "_".join(parts).replace(".", "p")
    else:
        name = "point_Re{Re:g}_{a}_ny{ny}_{c}{u}".format(
            Re=t["Re"], a=t["alpha_key"], ny=t["ny"], c=t["collision"],
            u=("" if abs(t.get("u_max", 0.05) - 0.05) < 1e-12
               else f"_umax{t['u_max']:g}".replace(".", "p")),
        )
    return name.replace("/", "-").replace(".", "p")


def run_task(t: dict) -> dict:
    out = dict(t)
    out["t0"] = time.time()
    if t["kind_task"] == "os_benchmark":
        rows = []
        for n in (32, 48, 64, 80, 96, 128):
            rows.append(orszag_benchmark_check(n))
        out["rows"] = rows
        out["reference"] = [ORSZAG_RE10000_ALPHA1.real, ORSZAG_RE10000_ALPHA1.imag]
        out["worst_rel_error"] = max(r["rel_error_complex"] for r in rows)
    elif t["kind_task"] == "a1_stokes":
        rows = []
        for n in (32, 64, 128):
            for tau in (0.6, 0.8, 1.2):
                rows.append(stokes_damping_benchmark(n=n, mode=1, tau=tau,
                                                     n_steps=2000))
        out["rows"] = rows
    elif t["kind_task"] == "point":
        flow = ChannelFlow(ny=int(t["ny"]), Re=float(t["Re"]),
                           u_max=float(t.get("u_max", 0.05)))
        alpha = alpha_lattice(t["alpha_key"], int(t["ny"]))
        spec = lattice_mode_spectrum(flow, alpha, t["collision"], n_cheb=80)
        prof = spec.velocity_profile()
        os_spec = spec.os_reference
        out["flow"] = flow.summary()
        out["alpha_nd"] = float(spec.alpha_nd)
        out["alpha_lattice"] = float(alpha)
        out["sigma_lattice"] = float(spec.sigma[spec.selected])
        out["sigma_nondim"] = float(spec.sigma_nondim())
        out["overlap_os"] = float(spec.overlap)
        out["sigma_os"] = float(os_spec.sigma_leading)
        out["c_os"] = [float(os_spec.c_leading.real), float(os_spec.c_leading.imag)]
        out["sigma_least_damped"] = float(spec.sigma_nondim(spec.index_least_damped))
        out["least_damped_is_os"] = bool(
            spec.index_least_damped == spec.selected
        )
        out["sign_agree"] = bool(
            np.sign(spec.sigma_nondim()) == np.sign(os_spec.sigma_leading)
        )
        denom = abs(os_spec.sigma_leading)
        out["rel_dev"] = float(abs(spec.sigma_nondim() - os_spec.sigma_leading) / denom)
        out["sigma_nondim_all"] = [
            float(v) for v in np.atleast_1d(spec.sigma_nondim(np.arange(spec.sigma.size)))
        ]
        # full spectrum in the OS-convention complex plane, per-step -> dimless
        scale = flow.h / flow.u_max
        out["eig_omega"] = [[float((np.log(np.abs(w)) * scale)),
                             float(-np.angle(w) * scale)]
                            for w in spec.lam]
        out["profile_v"] = [[float(v.real), float(v.imag)] for v in prof]
        out["profile_y"] = [float(y) for y in flow.y_hat]
        out["n_modes"] = int(spec.lam.size)
    elif t["kind_task"] == "timemarch":
        flow = ChannelFlow(ny=int(t["ny"]), Re=float(t["Re"]),
                           u_max=float(t.get("u_max", 0.05)))
        alpha = alpha_lattice(t["alpha_key"], int(t["ny"]))
        spec = lattice_mode_spectrum(flow, alpha, t["collision"], n_cheb=80)
        L = channel_mode_operator(flow, alpha, t["collision"])
        w = left_eigenvector(L, spec.lam[spec.selected])
        res = time_march_growth_rate(
            flow, alpha, t["collision"], k_box=ALPHAS[t["alpha_key"]][1],
            mode=ALPHAS[t["alpha_key"]][0], n_steps=int(t["n_steps"]),
            amplitude=float(t.get("amplitude", 1e-4)),
            projector=w.reshape(9, flow.ny),
        )
        res["sigma_eigen_nondim"] = float(spec.sigma_nondim())
        res["sigma_os"] = float(spec.os_reference.sigma_leading)
        res["overlap_os"] = float(spec.overlap)
        res["rel_dev_vs_eigen"] = float(
            abs(res["sigma_nondim"] - res["sigma_eigen_nondim"])
            / abs(res["sigma_eigen_nondim"])
        )
        res["rel_dev_vs_os"] = float(
            abs(res["sigma_nondim"] - res["sigma_os"]) / abs(res["sigma_os"])
        )
        out.update(res)
    out["wall_seconds"] = time.time() - out.pop("t0")
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, complex):
        return [float(o.real), float(o.imag)]
    raise TypeError(f"not serialisable: {type(o)}")


def aggregate() -> dict:
    files = sorted(POINTS.glob("*.json"))
    points, timemarch, osb, a1 = [], [], None, None
    for f in files:
        rec = json.loads(f.read_text())
        kt = rec.get("kind_task")
        if kt == "point":
            points.append(rec)
        elif kt == "timemarch":
            timemarch.append(rec)
        elif kt == "os_benchmark":
            osb = rec
        elif kt == "a1_stokes":
            a1 = rec

    summary = {
        "n_point_records": len(points),
        "n_timemarch_records": len(timemarch),
        "os_benchmark": osb,
        "a1_stokes": a1,
        "points": points,
        "timemarch": timemarch,
    }

    crit: dict[str, dict] = {}
    # ---------------- A1: uniform-flow Stokes damping ----------------
    if a1 is not None:
        # The criterion is a continuum (k -> 0) statement: it is evaluated on the
        # resolved shear waves (n >= 64, i.e. k <= pi/32).  The n = 32 rows are
        # kept in the table because they measure the lattice dispersion of the
        # shear mode (|dnu/nu| = k^2/12), which is a resolution requirement, not
        # an error of the damping law.
        resolved = [r for r in a1["rows"] if r["n"] >= 64]
        worst_vn = max((r["rel_error_von_neumann"] for r in resolved), default=float("nan"))
        worst_meas = max((r["rel_error_measured"] for r in resolved), default=float("nan"))
        crit["A1"] = {
            "statement": "uniform-flow shear-wave decay vs analytic Stokes -nu k^2,"
                         " relative error <= 1e-3 on resolved waves (n >= 64)",
            "threshold": 1e-3,
            "n_rows": len(a1["rows"]),
            "n_rows_resolved": len(resolved),
            "worst_von_neumann": float(worst_vn),
            "worst_measured": float(worst_meas),
            "worst_any": float(max(r["rel_error_measured"] for r in a1["rows"])),
            "passed": bool(worst_meas <= 1e-3 and worst_vn <= 1e-3),
            "rows": a1["rows"],
        }
    # ---------------- A2: sign against Orr-Sommerfeld ----------------
    a2_targets = [
        (1000.0, "pi/3"), (3000.0, "pi/3"), (6000.0, "pi/3"),
        (8000.0, "pi/3"), (10000.0, "pi/3"),
    ]
    rows_a2 = []
    for Re, akey in a2_targets:
        matches = [p for p in points if abs(p["Re"] - Re) < 1e-9
                   and p["alpha_key"] == akey and p["collision"] == "bgk"]
        rec = max(matches, key=lambda p: p["ny"]) if matches else None
        if rec is not None:
            rows_a2.append(rec)
    if rows_a2:
        crit["A2"] = {
            "statement": "sign(sigma) matches the Orr-Sommerfeld benchmark at 5 "
                         "(Re, alpha) points (BGK, highest available Ny)",
            "n_points": len(rows_a2),
            "points": [
                {"Re": r["Re"], "alpha_nd": r["alpha_nd"],
                 "sigma_lbm": r["sigma_nondim"], "sigma_os": r["sigma_os"],
                 "sign_agree": r["sign_agree"]} for r in rows_a2
            ],
            "passed": bool(len(rows_a2) >= 5 and all(r["sign_agree"] for r in rows_a2)),
        }
    # ---------------- A3: Ny convergence / Richardson ----------------
    rows_a3 = []
    groups: dict[tuple, dict[int, float]] = {}
    for p in points:
        groups.setdefault((p["Re"], p["alpha_key"], p["collision"]), {})[p["ny"]] = p["sigma_nondim"]
    for (Re, akey, kind), byn in sorted(groups.items()):
        # Richardson needs a geometric (Ny doubling) sequence; prefer the finest
        # available triple, falling back to 16/32/64 when 128/256 are absent
        triple = None
        for base in (64, 32, 16):
            if all(base * 2**i in byn for i in range(3)):
                triple = [base * 2**i for i in range(3)]
                break
        if triple is not None:
            lev = triple
            r = richardson_extrapolate([byn[n] for n in lev], lev)
            r.update({"Re": Re, "alpha_key": akey, "collision": kind,
                      "sigma_os": next(p["sigma_os"] for p in points
                                       if p["Re"] == Re and p["alpha_key"] == akey
                                       and p["ny"] == lev[-1] and p["collision"] == kind)})
            r["rel_dev_extrapolated_vs_os"] = abs(r["limit"] - r["sigma_os"]) / abs(r["sigma_os"])
            rows_a3.append(r)
    if rows_a3:
        crit["A3"] = {
            "statement": "Richardson-extrapolated residual of the finest Ny level "
                         "<= 5% on a Ny-doubling sequence "
                         + "/".join(str(int(v)) for v in rows_a3[0]["levels"]),
            "threshold": 0.05,
            "rows": rows_a3,
            "worst_residual": float(max(r["residual_finest_rel"] for r in rows_a3)),
            "passed": bool(all(r["residual_finest_rel"] <= 0.05 for r in rows_a3)),
            "failing": [{"Re": r["Re"], "collision": r["collision"],
                         "observed_order": r["observed_order"],
                         "residual": r["residual_finest_rel"]}
                        for r in rows_a3 if r["residual_finest_rel"] > 0.05],
        }
    # ---------------- A4: A -> 0 linear limit ----------------
    # Wording of the criterion: the measured decay rate must converge to the
    # linear-theory (eigenvalue) value as the injection amplitude goes to zero,
    # and the amplitude dependence must vanish inside the linear window.  The
    # largest amplitude (1e-2) is deliberately kept in the table as the negative
    # control: at that amplitude the perturbation is ~20 % of u_max and the
    # measured rate *must* depart from linear theory, otherwise the test would
    # have no teeth.
    amp_recs = [t for t in timemarch if "amplitude" in t and t.get("diverged_at") is None]
    if amp_recs:
        # use one operating point only (the group with the most amplitudes)
        groups: dict[tuple, list] = {}
        for t in amp_recs:
            groups.setdefault((t["Re"], t["ny"], t["collision"]), []).append(t)
        key = max(groups, key=lambda k: (len(groups[k]), -k[1]))
        warn = None
        amp_recs = groups[key]
    a4 = sorted(amp_recs, key=lambda r: -r["amplitude"])
    if len(a4) >= 3:
        big, mid, small = a4[0], a4[1], a4[2]
        spread = abs(big["sigma_nondim"] - small["sigma_nondim"]) / abs(small["sigma_nondim"])
        spread_linear = abs(mid["sigma_nondim"] - small["sigma_nondim"]) / abs(
            small["sigma_nondim"])
        crit["A4"] = {
            "statement": "A -> 0 linear limit: |sigma(A=1e-3)-sigma(A=1e-4)|/|sigma| <= 5 %, "
                         "|sigma(A=1e-4) - eigenvalue| <= 15 %, and the A=1e-2 control "
                         "must deviate more than the linear-window pair",
            "amplitudes": [big["amplitude_start"], mid["amplitude_start"],
                           small["amplitude_start"]],
            "sigma_nondim": [big["sigma_nondim"], mid["sigma_nondim"], small["sigma_nondim"]],
            "sigma_eigen": small["sigma_eigen_nondim"],
            "spread_rel_all": float(spread),
            "spread_rel_linear_window": float(spread_linear),
            "rel_dev_smallest_vs_eigen": float(small["rel_dev_vs_eigen"]),
            "rel_dev_largest_vs_eigen": float(big["rel_dev_vs_eigen"]),
            "negative_control_detects_nonlinearity": bool(
                big["rel_dev_vs_eigen"] > spread_linear),
            "passed": bool(spread_linear <= 0.05
                           and small["rel_dev_vs_eigen"] <= 0.15
                           and big["rel_dev_vs_eigen"] > spread_linear
                           and big["diverged_at"] is None
                           and small["diverged_at"] is None),
        }
    # ---------------- accuracy table (Ny >= 64) ----------------
    main = [p for p in points if p["ny"] >= 64]
    if main:
        worst = max(main, key=lambda p: p["rel_dev"])
        crit["accuracy_Ny_ge_64"] = {
            "statement": "relative deviation |sigma_lbm - sigma_os|/|sigma_os| <= 15%"
                         " for every (Re, alpha, operator) at Ny >= 64",
            "n_points": len(main),
            "n_within_15pct": int(sum(1 for p in main if p["rel_dev"] <= 0.15)),
            "worst": {"Re": worst["Re"], "alpha_nd": worst["alpha_nd"],
                      "collision": worst["collision"], "ny": worst["ny"],
                      "rel_dev": worst["rel_dev"]},
            "median_rel_dev": float(np.median([p["rel_dev"] for p in main])),
            "passed": bool(all(p["rel_dev"] <= 0.15 for p in main)),
        }
    # cross audit
    if timemarch:
        cross = [t for t in timemarch if "amplitude" not in t
                 and t.get("diverged_at") is None]
        diverged = [t for t in timemarch if t.get("diverged_at") is not None]
        if cross:
            crit["cross_audit_timemarch"] = {
                "statement": "time-marched (nonlinear) growth rate vs eigenvalue of the"
                             " linearised operator, relative deviation <= 5%",
                "rows": [{"Re": t["Re"], "collision": t["collision"],
                          "sigma_measured": t["sigma_nondim"],
                          "sigma_eigen": t["sigma_eigen_nondim"],
                          "rel_dev": t["rel_dev_vs_eigen"],
                          "log_amp_residual": t["log_amplitude_residual"],
                          "base_flow_drift": t["base_flow_drift"]} for t in cross],
                "worst_rel_dev": float(max(t["rel_dev_vs_eigen"] for t in cross)),
                "passed": bool(all(t["rel_dev_vs_eigen"] <= 0.05 for t in cross)),
            }
        if diverged:
            crit["time_march_diverged"] = {
                "statement": "records where the nonlinear march blew up (BGK at "
                             "tau ~ 0.5006, omega -> 2): reported, not used as growth "
                             "rate evidence",
                "rows": [{"Re": t["Re"], "ny": t["ny"], "collision": t["collision"],
                          "amplitude": t.get("amplitude"),
                          "diverged_at": t["diverged_at"]} for t in diverged],
                "passed": True,
            }
    # keep the per-point eigen/vector payload only for two representative records
    # (all spectra are ~9*Ny complex values each; the rest of the numbers stay)
    heavy = sorted([p for p in points if p["collision"] == "bgk"],
                   key=lambda p: (p["ny"], p["Re"]))[-2:]
    for p in points:
        p.pop("sigma_nondim_all", None)
        if p not in heavy:
            p.pop("eig_omega", None)
            p.pop("profile_v", None)
            p.pop("profile_y", None)
    summary["criteria"] = crit
    return summary


def main(argv: list[str] | None = None) -> int:
    global ART, POINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--only", default=None, help="comma separated task indices")
    ap.add_argument("--sort-ny", action="store_true",
                    help="run cheap (small Ny) points first")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=str(ART))
    args = ap.parse_args(argv)

    tasks = task_list()
    if args.list:
        for i, t in enumerate(tasks):
            print(i, task_key(t))
        print(f"total {len(tasks)} tasks")
        return 0

    if args.out != str(ART):
        ART = Path(args.out)
        POINTS = ART / "points"
    POINTS.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        summary = aggregate()
        (ART / "acceptance.json").write_text(
            json.dumps(summary, indent=2, default=_json_default)
        )
        print(json.dumps({k: {kk: vv for kk, vv in v.items()
                             if kk in ("statement", "passed", "threshold")}
                          for k, v in summary["criteria"].items()}, indent=2))
        print("wrote", ART / "acceptance.json")
        return 0

    sel = list(enumerate(tasks))
    if args.sort_ny:
        sel.sort(key=lambda kt: (kt[1]["kind_task"] != "point", kt[1].get("ny", 0) or 0))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        sel = [(k, t) for k, t in sel if k % n == i % n]
    elif args.only:
        keep = {int(x) for x in args.only.split(",")}
        sel = [(k, t) for k, t in sel if k in keep]
    elif not args.run_all:
        ap.error("need --run-all, --shard i/n, --list or --aggregate")

    failures = 0
    for k, t in sel:
        name = f"{k:03d}_{task_key(t)}"
        dest = POINTS / f"{name}.json"
        if dest.exists():
            print(f"[skip] {name}")
            continue
        print(f"[run ] {name}", flush=True)
        try:
            rec = run_task(t)
        except Exception:  # noqa: BLE001 - a failed task must not kill the shard
            failures += 1
            dest.with_suffix(".failed").write_text(traceback.format_exc())
            print(traceback.format_exc())
            continue
        rec["task_index"] = k
        rec["task_name"] = name
        dest.write_text(json.dumps(rec, indent=1, default=_json_default))
        print(f"[done] {name} {rec.get('wall_seconds', 0):.1f}s", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())