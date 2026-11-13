#!/usr/bin/env python
"""XVAL-LBM-BITFLAG-NEIGHBOUR-MASK benchmark runner.

Subcommands (each prints one JSON object on stdout and appends it to
``runs/xval_bitflag_neighbour_mask/<case>.jsonl``):

``unit``        operator equivalence (mask vs OSS-style scan), mask statistics,
                scalar-kernel work accounting + timings
``poiseuille``  inlet-driven plane channel vs the analytic parabola (both modes)
``couette``     moving-lid Couette channel vs the analytic linear profile
``sweep``       inlet-flux linearity (three inlet amplitudes, one mode)
``cylinder``    Schäfer-Turek 2D-1 in-channel cylinder: Cd convergence history
``aggregate``   build ``result.json`` (error table + pass/fail), figures, report

Every physical comparison is computed here, never typed in: the reference drag is
imported from :mod:`tensorlbm.cylinder_st2d1` (``REFERENCE['cd']``), which points
at the in-repo verified anchor ``benchmarks/verified/cylinder/re20_st/result.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.bitflag_d2q9 import (  # noqa: E402
    BitflagD2Q9,
    make_channel,
    make_cylinder_channel,
)
from tensorlbm.bitflag_neighbour_mask import (  # noqa: E402
    StreamStats,
    mask_stats,
    open_cells,
    neighbour_mask,
    oss_border_guard_wraps,
    stream_mask_scalar,
    stream_mask_vector,
    stream_scan_scalar,
    stream_scan_vector,
)

OUT_DIR = ROOT / "runs" / "xval_bitflag_neighbour_mask"


def _emit(case: str, payload: dict) -> dict:
    payload = dict(payload)
    payload.setdefault("case", case)
    payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    payload.setdefault("host", platform.node())
    payload.setdefault("python", platform.python_version())
    payload.setdefault("numpy", np.__version__)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{case}.jsonl").open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return payload


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _poiseuille_inlet(ny: int, u_max: float) -> np.ndarray:
    """Parabolic x-velocity with peak *u_max* and zero at y = 1/2, ny - 1/2."""
    prof = np.zeros(ny)
    h = ny - 1.0
    y = np.arange(1, ny - 1) - 0.5
    prof[1:-1] = u_max * 4.0 * y * (h - y) / h**2
    return prof


def _profile_error(got: np.ndarray, ref: np.ndarray) -> dict:
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    scale = np.abs(ref).max()
    return {
        "max_abs": float(np.abs(got - ref).max()),
        "max_rel_pct": float(100.0 * np.abs(got - ref).max() / scale) if scale else float("inf"),
        "l2_rel_pct": float(100.0 * np.sqrt(np.mean((got - ref) ** 2)) / scale) if scale else float("inf"),
        "ref_scale": float(scale),
    }


def _channel_case(nx: int, ny: int, u_max: float, tau: float, mode: str, outlet: str):
    ct = make_channel(nx, ny, inlet=True, outlet=True)
    return BitflagD2Q9(
        ct,
        tau=tau,
        mode=mode,
        inlet_profile=_poiseuille_inlet(ny, u_max),
        outlet_mode=outlet,
    )


# ---------------------------------------------------------------------------
# case: unit / operator equivalence
# ---------------------------------------------------------------------------
def case_unit(args) -> dict:
    rng = np.random.default_rng(20261113)
    result: dict = {"checks": {}, "geometry": {}, "work": {}}

    # 1. random geometry + random populations: mask kernel == scan kernel,
    #    both scalar and vectorised, over a channel-with-cylinder geometry.
    nx, ny = 40, 24
    ct = make_cylinder_channel(nx=nx, ny=ny, centre=(12.5, 12.5), radius=4.0)
    f_post = rng.random((9, nx, ny))
    mask = neighbour_mask(ct)
    vec_mask = stream_mask_vector(f_post, ct, mask)
    vec_scan = stream_scan_vector(f_post, ct)
    scal_mask = stream_mask_scalar(f_post, ct, mask)
    scal_scan = stream_scan_scalar(f_post, ct)
    op = open_cells(ct)[None, :, :]
    def _cmp(a, b):
        """Compare on open cells only: closed cells are frozen garbage by design."""
        return bool(np.array_equal(np.where(op, a, 0.0), np.where(op, b, 0.0)))
    result["checks"]["vector_mask_eq_vector_scan"] = _cmp(vec_mask, vec_scan)
    result["checks"]["scalar_mask_eq_scalar_scan"] = _cmp(scal_mask, scal_scan)
    result["checks"]["vector_mask_eq_scalar_mask"] = _cmp(vec_mask, scal_mask)
    result["max_abs_diff_open_cells"] = float(np.abs(np.where(op, vec_mask, 0.0) - np.where(op, vec_scan, 0.0)).max())

    # 2. moving-wall term: mask and scan must agree with a nonzero wall velocity
    uw = np.full((nx, ny), 0.02)
    uw[:, : ny // 2] = 0.0
    wall_dot = np.stack([np.array([0, 0, 0, -1, -1, -1, 1, 1, 1])[i] * uw for i in range(9)])
    mv_mask = stream_mask_vector(f_post, ct, mask, wall_velocity=wall_dot)
    mv_scan = stream_scan_vector(f_post, ct, wall_velocity=wall_dot)
    result["checks"]["moving_wall_mask_eq_scan"] = _cmp(mv_mask, mv_scan)

    # 3. mask statistics on the geometry families used by the benchmark
    for name, ct_case in {
        "channel_80x22": make_channel(80, 22),
        "channel_80x22_io": make_channel(80, 22, inlet=True, outlet=True),
        "cylinder_D12": make_cylinder_channel(
            nx=22 * 12 + 1, ny=int(4.1 * 12) + 1, centre=(2 * 12 + 0.5, 2 * 12 + 0.5), radius=6.0
        ),
        "cylinder_D16": make_cylinder_channel(
            nx=22 * 16 + 1, ny=int(4.1 * 16) + 1, centre=(2 * 16 + 0.5, 2 * 16 + 0.5), radius=8.0
        ),
        "cylinder_D20": make_cylinder_channel(
            nx=22 * 20 + 1, ny=int(4.1 * 20) + 1, centre=(2 * 20 + 0.5, 2 * 20 + 0.5), radius=10.0
        ),
    }.items():
        result["geometry"][name] = mask_stats(ct_case).as_dict()

    # 4. work accounting: neighbour decisions per streaming sweep
    st_scan = StreamStats()
    st_mask = StreamStats()
    for _ in range(3):
        stream_scan_scalar(f_post, ct, stats=st_scan)
        stream_mask_scalar(f_post, ct, mask, stats=st_mask)
    s_scan, s_mask = st_scan.as_dict(), st_mask.as_dict()
    result["work"]["scalar_scan"] = {k: v // 3 for k, v in s_scan.items()}
    result["work"]["scalar_mask"] = {k: v // 3 for k, v in s_mask.items()}
    result["work"]["decision_ratio_scan_over_mask"] = (
        float(s_scan["decision_count"] / s_mask["decision_count"]) if s_mask["decision_count"] else None
    )
    # 5. wall-clock of the two scalar sweeps on a fixed workload (Python-level,
    #    interpreter-dominated -> reported as a ratio, not an absolute speed-up)
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        stream_scan_scalar(f_post, ct)
    t_scan = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        stream_mask_scalar(f_post, ct, mask)
    t_mask = (time.perf_counter() - t0) / reps
    result["work"]["scalar_sweep_seconds_scan"] = t_scan
    result["work"]["scalar_sweep_seconds_mask"] = t_mask
    result["work"]["scalar_speedup"] = t_scan / t_mask if t_mask else None
    t0 = time.perf_counter()
    for _ in range(reps):
        stream_scan_vector(f_post, ct)
    t_vscan = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        stream_mask_vector(f_post, ct, mask)
    t_vmask = (time.perf_counter() - t0) / reps
    result["work"]["vector_sweep_seconds_scan"] = t_vscan
    result["work"]["vector_sweep_seconds_mask"] = t_vmask
    result["work"]["vector_speedup"] = t_vscan / t_vmask if t_vmask else None

    # 6. reference-implementation caveat: the OSS border guard
    #    (lbm/streamcollide.py:76-78) does not catch negative indices
    result["checks"]["oss_guard_skips_upper"] = bool(oss_border_guard_wraps(41, 5, (40, 24)))
    result["checks"]["oss_guard_wraps_lower"] = not bool(oss_border_guard_wraps(-1, 5, (40, 24)))
    return _emit("unit", result)


# ---------------------------------------------------------------------------
# case: plane Poiseuille
# ---------------------------------------------------------------------------
def case_poiseuille(args) -> dict:
    nx, ny = args.nx, args.ny
    tau = args.tau
    u_max = args.u_max
    nu = (1.0 / tau - 0.5) / 3.0
    prof = _poiseuille_inlet(ny, u_max)
    out: dict = {"params": {"nx": nx, "ny": ny, "tau": tau, "nu": nu, "u_max": u_max,
                            "re_h": u_max * (ny - 1) / nu, "steps": args.steps}}
    hist = []
    solvers = {}
    for mode in ("mask", "scan"):
        s = BitflagD2Q9(
            make_channel(nx, ny, inlet=True, outlet=True),
            tau=tau,
            mode=mode,
            inlet_profile=prof,
            outlet_mode=args.outlet,
        )
        solvers[mode] = s
        t0 = time.perf_counter()
        for _ in range(args.steps):
            s.step()
        elapsed = time.perf_counter() - t0
        rho, ux, uy = s.macros()
        mid = ux[nx // 2, 1:-1]
        err = _profile_error(mid, prof[1:-1])
        flux_in = float(ux[1, 1:-1].sum())
        flux_out = float(ux[nx - 2, 1:-1].sum())
        hist.append(
            {
                "mode": mode,
                "max_rel_pct": err["max_rel_pct"],
                "l2_rel_pct": err["l2_rel_pct"],
                "u_max_got": float(mid.max()),
                "u_max_ana": float(prof[1:-1].max()),
                "flux_in": flux_in,
                "flux_out": flux_out,
                "flux_drift_pct": 100.0 * (flux_out - flux_in) / flux_in,
                "rho_min": float(rho[1:-1, 1:-1].min()),
                "rho_max": float(rho[1:-1, 1:-1].max()),
                "uy_max": float(np.abs(uy[1:-1, 1:-1]).max()),
                "seconds": elapsed,
            }
        )
    fm, fs = solvers["mask"].f, solvers["scan"].f
    out["modes"] = hist
    out["mask_vs_scan_max_abs_diff"] = float(np.abs(fm - fs).max())
    out["mask_vs_scan_bit_identical"] = bool(np.array_equal(fm, fs))
    np.save(OUT_DIR / f"poiseuille_profile_{args.tag}.npy", solvers["mask"].macros()[1][nx // 2, :])
    np.save(OUT_DIR / f"poiseuille_analytic_{args.tag}.npy", prof)
    return _emit(f"poiseuille_{args.tag}", out)


# ---------------------------------------------------------------------------
# case: Couette
# ---------------------------------------------------------------------------
def case_couette(args) -> dict:
    nx, ny = args.nx, args.ny
    u_wall = args.u_wall
    h = ny - 1.0
    y = np.arange(ny) - 0.5
    prof = np.zeros(ny)
    prof[1:-1] = u_wall * (np.arange(1, ny - 1) - 0.5) / h
    uw = np.zeros((nx, ny))
    uw[:, ny - 1] = u_wall
    hist = []
    for mode in ("mask", "scan"):
        s = BitflagD2Q9(
            make_channel(nx, ny, inlet=True, outlet=True),
            tau=args.tau,
            mode=mode,
            inlet_profile=prof,
            u_wall_x=uw,
            outlet_mode="zou_he",
        )
        for _ in range(args.steps):
            s.step()
        rho, ux, uy = s.macros()
        mid = ux[:, nx // 2]
        err = _profile_error(mid[1 : ny - 1], prof[1:-1])
        hist.append({"mode": mode, "max_rel_pct": err["max_rel_pct"], "l2_rel_pct": err["l2_rel_pct"],
                     "u_max_got": float(mid[1 : ny - 1].max()), "u_max_ana": float(prof.max())})
    out = {"params": {"nx": nx, "ny": ny, "tau": args.tau, "u_wall": u_wall, "steps": args.steps},
           "modes": hist}
    np.save(OUT_DIR / "couette_profile.npy", s.macros()[1][:, nx // 2])
    np.save(OUT_DIR / "couette_analytic.npy", prof)
    return _emit("couette", out)


# ---------------------------------------------------------------------------
# case: inlet-flux linearity
# ---------------------------------------------------------------------------
def case_sweep(args) -> dict:
    rows = []
    for u_max in args.u_max_list:
        s = _channel_case(args.nx, args.ny, u_max, args.tau, "mask", "zero_gradient_all")
        for _ in range(args.steps):
            s.step()
        rho, ux, uy = s.macros()
        mid = ux[args.nx // 2, 1:-1]
        prof = _poiseuille_inlet(args.ny, u_max)
        err = _profile_error(mid, prof[1:-1])
        rows.append({"u_max_in": u_max, "u_max_got": float(mid.max()),
                     "ratio": float(mid.max() / u_max), "max_rel_pct": err["max_rel_pct"]})
    return _emit("sweep", {"params": {"nx": args.nx, "ny": args.ny, "tau": args.tau,
                                      "steps": args.steps}, "rows": rows})


# ---------------------------------------------------------------------------
# case: Schäfer-Turek 2D-1 cylinder
# ---------------------------------------------------------------------------
def case_cylinder(args) -> dict:
    from tensorlbm.cylinder_st2d1 import GEOMETRY, REFERENCE

    D = args.D
    nx = 22 * D + 1
    ny = int(round(4.1 * D)) + 1
    rho_scale = 1.0
    u_mean = args.u_mean  # lattice mean inlet velocity
    u_max = 1.5 * u_mean
    nu = u_mean * D / GEOMETRY["re"]  # Re = u_mean * D / nu  (D in cells)
    tau = 0.5 + 3.0 * nu
    h = ny - 1.0
    prof = np.zeros(ny)
    prof[1:-1] = u_max * 4.0 * ((np.arange(1, ny - 1) - 0.5) * (h - (np.arange(1, ny - 1) - 0.5))) / h**2
    centre = (2 * D + 0.5, 2 * D + 0.5)
    radius = 0.5 * D
    ct = make_cylinder_channel(nx=nx, ny=ny, centre=centre, radius=radius)
    s = BitflagD2Q9(ct, tau=tau, mode=args.mode, inlet_profile=prof, outlet_mode=args.outlet)
    norm = 0.5 * rho_scale * u_mean**2 * D

    history = []
    window = max(200, args.steps // 20)
    cd_now = None
    converged = False
    t0 = time.perf_counter()
    for k in range(1, args.steps + 1):
        s.step()
        if k % args.sample_every == 0 or k == args.steps:
            fx, fy = s.cylinder_force(centre, radius)
            cd_now = abs(fx) / norm
            rho, ux, uy = s.macros()
            history.append({"step": k, "cd": cd_now, "cl": fy / norm,
                            "u_max": float(np.abs(ux[2:-2, 1:-1]).max())})
            if len(history) >= 3 and k > 2 * args.sample_every:
                tail = [h["cd"] for h in history[-(window // args.sample_every + 1):]]
                if len(tail) >= 2:
                    spread = (max(tail) - min(tail)) / (np.mean(tail) if np.mean(tail) else 1.0)
                    if spread < 1e-3 and k >= args.min_steps:
                        converged = True
                        break
    rho, ux, uy = s.macros()
    cd_samples = [h["cd"] for h in history[-max(3, len(history) // 5):]]
    cd = float(np.mean(cd_samples))
    cd_ref = float(REFERENCE["cd"])
    payload = {
        "params": {
            "D_cells": D, "nx": nx, "ny": ny, "u_mean": u_mean, "u_max": u_max,
            "nu": nu, "tau": tau, "re": GEOMETRY["re"], "blockage": D / h,
            "mode": args.mode, "outlet": args.outlet, "max_steps": args.steps,
            "steps_run": history[-1]["step"] if history else 0, "sample_every": args.sample_every,
        },
        "cd": cd, "cd_ref": cd_ref,
        "cd_err_pct": 100.0 * (cd - cd_ref) / cd_ref,
        "cl": float(np.mean([h["cl"] for h in history[-max(3, len(history) // 5):]])),
        "converged": converged,
        "history": history,
        "seconds": time.perf_counter() - t0,
        "mask": mask_stats(ct).as_dict(),
        "stream_stats_last": s.last_stream_stats.as_dict() if s.last_stream_stats else None,
        "reference_source": REFERENCE["source"],
    }
    np.save(OUT_DIR / f"cylinder_D{D}_{args.mode}_ux.npy", s.macros()[1])
    np.save(OUT_DIR / f"cylinder_D{D}_{args.mode}_mask.npy", s.mask)
    return _emit(f"cylinder_D{D}_{args.mode}_{args.outlet}", payload)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "operator_bit_identical": True,   # mask == scan, exact
    "poiseuille_max_rel_pct": 3.0,
    "couette_max_rel_pct": 3.0,
    "cylinder_cd_err_pct": 15.0,      # owner accuracy standard
    "flux_drift_pct": 0.5,
}


def case_aggregate(args) -> dict:  # pragma: no cover - reporting helper
    recs: dict[str, list[dict]] = {}
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        key = path.stem
        recs[key] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    unit = recs["unit"][-1]
    pois = recs["poiseuille"][-1]
    cou = recs.get("couette", [None])[-1]
    cyl = sorted(recs["cylinder"], key=lambda r: (r["params"]["D_cells"], r["params"]["outlet"]))
    sweep = recs.get("sweep", [None])[-1]

    table = []

    def row(name, ref_name, got, ref, unit_, criterion, passed, note=""):
        table.append({"quantity": name, "reference": ref_name, "got": got, "ref": ref,
                      "error_pct": 100.0 * (got - ref) / ref if ref else None,
                      "criterion": criterion, "pass": bool(passed), "note": note})

    for m in pois["modes"]:
        row(f"Poiseuille u_max [{m['mode']}]", "analytic parabola (wall at y=1/2)",
            m["u_max_got"], m["u_max_ana"], "-", f"|err| < {THRESHOLDS['poiseuille_max_rel_pct']}%",
            m["max_rel_pct"] < THRESHOLDS["poiseuille_max_rel_pct"],
            f"max profile error {m['max_rel_pct']:.3f}%, flux drift {m['flux_drift_pct']:.3f}%")
    if cou:
        for m in cou["modes"]:
            row(f"Couette u_max [{m['mode']}]", "analytic linear profile", m["u_max_got"],
                m["u_max_ana"], "-", f"|err| < {THRESHOLDS['couette_max_rel_pct']}%",
                m["max_rel_pct"] < THRESHOLDS["couette_max_rel_pct"],
                f"max profile error {m['max_rel_pct']:.3f}%")
    for r in cyl:
        row(f"Cd cylinder D={r['params']['D_cells']} [{r['params']['mode']}, {r['params']['outlet']}]",
            "Schäfer-Turek 2D-1 (REFERENCE['cd'])", r["cd"], r["cd_ref"], "-",
            f"|err| < {THRESHOLDS['cylinder_cd_err_pct']}%",
            abs(r["cd_err_pct"]) < THRESHOLDS["cylinder_cd_err_pct"],
            f"err {r['cd_err_pct']:+.2f}%, converged={r['converged']}")

    result = {
        "case": "aggregate",
        "thresholds": THRESHOLDS,
        "operator_equivalence": {
            "bit_identical": unit["checks"]["vector_mask_eq_vector_scan"],
            "scalar_bit_identical": unit["checks"]["scalar_mask_eq_scalar_scan"],
            "moving_wall_bit_identical": unit["checks"]["moving_wall_mask_eq_scan"],
            "max_abs_diff": unit["max_abs_diff"],
        },
        "mask_vs_scan_poiseuille_bit_identical": pois["mask_vs_scan_bit_identical"],
        "mask_vs_scan_poiseuille_max_abs_diff": pois["mask_vs_scan_max_abs_diff"],
        "work": unit["work"],
        "geometry": unit["geometry"],
        "table": table,
        "cylinder_runs": [
            {k: r[k] for k in ("case", "params", "cd", "cd_ref", "cd_err_pct", "converged", "history")}
            for r in cyl
        ],
        "poiseuille": pois,
        "couette": cou,
        "sweep": sweep,
        "all_pass": all(t["pass"] for t in table) and unit["checks"]["vector_mask_eq_vector_scan"],
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: result[k] for k in ("all_pass", "operator_equivalence", "work")}, indent=2))
    print(json.dumps(result["table"], indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("unit").set_defaults(func=case_unit)

    p = sub.add_parser("poiseuille")
    p.add_argument("--nx", type=int, default=80)
    p.add_argument("--ny", type=int, default=22)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--u-max", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--outlet", default="zou_he",
                   choices=["outflow", "zero_gradient_all", "zero_gradient", "zou_he"])
    p.add_argument("--tag", default="base")
    p.set_defaults(func=case_poiseuille)

    c = sub.add_parser("couette")
    c.add_argument("--nx", type=int, default=40)
    c.add_argument("--ny", type=int, default=22)
    c.add_argument("--tau", type=float, default=0.8)
    c.add_argument("--u-wall", type=float, default=0.02)
    c.add_argument("--steps", type=int, default=20000)
    c.set_defaults(func=case_couette)

    sw = sub.add_parser("sweep")
    sw.add_argument("--nx", type=int, default=80)
    sw.add_argument("--ny", type=int, default=22)
    sw.add_argument("--tau", type=float, default=0.8)
    sw.add_argument("--steps", type=int, default=20000)
    sw.add_argument("--u-max-list", type=float, nargs="+", default=[0.01, 0.02, 0.04])
    sw.set_defaults(func=case_sweep)

    cy = sub.add_parser("cylinder")
    cy.add_argument("--D", type=int, default=16)
    cy.add_argument("--u-mean", type=float, default=0.05)
    cy.add_argument("--steps", type=int, default=120000)
    cy.add_argument("--min-steps", type=int, default=20000)
    cy.add_argument("--sample-every", type=int, default=200)
    cy.add_argument("--mode", default="mask", choices=["mask", "scan"])
    cy.add_argument("--outlet", default="zou_he",
                    choices=["outflow", "zero_gradient_all", "zero_gradient", "zou_he"])
    cy.set_defaults(func=case_cylinder)

    sub.add_parser("aggregate").set_defaults(func=case_aggregate)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())