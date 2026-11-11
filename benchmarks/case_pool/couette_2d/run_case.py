#!/usr/bin/env python3
"""CASE-COUETTE-2D — case-pool driver for plane Couette flow (TensorLBM).

Case pool record ``COUETTE-2D`` (repo TensorLBM, tol <2%):

    name      库埃特剪切流 (plane Couette flow)
    ref       线性解析解  ->  u(y) = U0 (y - y_b) / H,  tau_w = mu U0 / H

This driver does **not** re-implement a solver.  It reuses two capabilities that
already live in the repository and compares both against the closed-form
reference, so the case is a *reuse* test of existing code, not a new
registration:

Route A (primary, D2Q9 moving-wall bounce-back)
    ``benchmarks/verified/couette_2d/run.py::run_case`` — the repository's
    validated 2D Couette benchmark (library primitives ``d2q9`` + ``solver``,
    pre-streaming half-way bounce-back with moving-wall momentum injection).

Route B (independent force route, D3Q19 interpolated bounce-back)
    ``tensorlbm.curved_boundary_calibration.run_plane_couette`` — quasi-2D
    plane Couette whose wall shear is measured twice, by link-wise
    momentum exchange and by control-surface stress integration.

Reference numbers are **generated from the closed form** (never transcribed):
for a given (U0, H_eff, nu) the target profile, wall shear, and momentum
integral are evaluated from the analytic expression at the same lattice
coordinates the solver reports.  See ``README.md`` for the predeclared
thresholds and the decision rule.

Usage
-----
    PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/couette_2d/run_case.py \
        --out-dir benchmarks/case_pool/couette_2d/artifacts

    # fast subset used by tests/test_case_pool_couette_2d.py
    ... run_case.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

# ---------------------------------------------------------------------------
# PREDECLARED decision rule (fixed before any run; not tuned after inspection)
# ---------------------------------------------------------------------------

CASE_CODE = "COUETTE-2D"
TOL_PCT = 2.0  # case-pool tolerance policy: <2 %

# Each entry: quantity -> tolerance in percent relative error (None = informational)
THRESHOLDS: dict[str, float | None] = {
    # Route A — primary field/derived quantities
    "profile_l2_rel_pct": TOL_PCT,          # ||u_num - u_ref||_2 / ||u_ref||_2
    "profile_max_rel_pct": TOL_PCT,         # max pointwise, u_ref > 0.2 U0
    "wall_shear_rel_pct": TOL_PCT,          # tau_num vs mu U0 / H
    "top_row_u_rel_pct": TOL_PCT,           # u at last fluid row vs analytic
    "momentum_integral_rel_pct": TOL_PCT,   # int u dy vs U0 H_eff / 2
    "mass_drift_pct": 1e-3,                 # |dm|/m0 of total populations
    # Route B — independent force route
    "force_me_rel_pct": TOL_PCT,            # momentum-exchange wall force
    "force_cv_rel_pct": TOL_PCT,            # control-surface stress force
    "force_two_route_rel_pct": TOL_PCT,     # ME vs CV consistency
    "force_bottom_rel_pct": TOL_PCT,        # stationary-wall force (=-top)
    "profile_max_rel_pct_routeB": TOL_PCT,  # Route B profile max pointwise error
}

#: Two-route (A vs B) shear consistency tolerance.
CROSS_ROUTE_TOL_PCT = TOL_PCT

#: Convergence: L2 error is accepted as "not deteriorating" if the observed
#: order stays non-negative within this slack (errors are at round-off level,
#: so a plateau - not a slope - is the expected signature).
ORDER_SLACK = 0.25


# ---------------------------------------------------------------------------
# analytic reference (closed form, evaluated programmatically)
# ---------------------------------------------------------------------------

REF_SOURCE = "线性解析解 (case pool ref_source)"
REF_FORMULA = (
    "u(y) = U0 (y - y_b) / H_eff ; tau_w = mu U0 / H_eff, mu = rho nu ; "
    "int_0^H u dy = U0 H_eff / 2 ; rho = const"
)
REF_CITATION_NOTE = (
    "闭式解由 N-S 方程在 Couette 边界条件下直接求得，目标数值由本文件的 "
    "analytic_* 函数按闭式求值（零手抄，无表格转录）。教材/手册级页码引用在"
    "离线环境不可核验（本机无外网），故不著录页码，以免伪造出处；"
    "证据面=self-contained 闭式 + 两条独立数值路径。"
)


def analytic_profile(U0: float, H_eff: float, y: np.ndarray) -> np.ndarray:
    return U0 * np.asarray(y, dtype=np.float64) / H_eff


def analytic_wall_shear(nu: float, U0: float, H_eff: float) -> float:
    return nu * U0 / H_eff


def analytic_momentum_integral(U0: float, H_eff: float) -> float:
    return 0.5 * U0 * H_eff


# ---------------------------------------------------------------------------
# route A: reuse benchmarks/verified/couette_2d/run.py
# ---------------------------------------------------------------------------


def _load_verified_runner():
    path = REPO_ROOT / "benchmarks" / "verified" / "couette_2d" / "run.py"
    spec = importlib.util.spec_from_file_location("couette_2d_verified_run", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_route_a(
    H: int,
    tau: float,
    U0: float,
    collision: str,
    out_dir: Path,
    min_steps: int,
    max_steps: int,
    compile_mode: str | None = "eager",
    force: bool = False,
) -> dict:
    """One Route-A Couette run + ref/got comparison against the closed form."""
    tag = f"H{H}_tau{tau:g}_U{U0:g}_{collision}"
    out_json = out_dir / f"routeA_{tag}.json"
    # Deterministic reuse: an existing per-case artifact is reloaded instead of
    # re-simulated (raw measurements only; every comparison below is recomputed).
    if out_json.exists() and not force:
        raw = json.loads(out_json.read_text())
        reused = True
    else:
        mod = _load_verified_runner()
        raw = mod.run_case(
            H, tau, U0, collision, min_steps, max_steps, str(out_json), 0, compile_mode
        )
        reused = False

    nu = (tau - 0.5) / 3.0
    H_eff = float(H)
    u_num = np.asarray(raw["u_profile"], dtype=np.float64)
    # same lattice coordinates the solver reports: fluid rows sit at y = 0.5 .. H-0.5
    y_phys = np.arange(1, H + 1, dtype=np.float64) - 0.5
    u_ref = analytic_profile(U0, H_eff, y_phys)

    # derived diagnostics recomputed here against the closed form (zero hand-copy)
    # Discrete momentum per unit width: the two wall nodes carry the exact
    # no-slip / moving-wall velocities (u(y=0)=0, u(y=H)=U0), the H fluid rows
    # sit at y = 0.5 .. H-0.5.  Trapezoid over the closed interval [0, H].
    mom_ref = analytic_momentum_integral(U0, H_eff)
    u_nodes = np.concatenate(([0.0], u_num, [U0]))
    y_nodes = np.concatenate(([0.0], y_phys, [float(H)]))
    mom_got = float(np.trapezoid(u_nodes, y_nodes))
    mom_rel = abs(mom_got - mom_ref) / mom_ref * 100.0

    rows = {
        "profile_l2_rel_pct": float(raw["l2_rel_err"]) * 100.0,
        "profile_max_rel_pct": float(raw["max_rel_err_masked_pct"]),
        "wall_shear_rel_pct": abs(float(raw["tau_err_pct"])),
        "top_row_u_rel_pct": abs(float(raw["u_top_row_err_pct"])),
        "momentum_integral_rel_pct": mom_rel,
        "mass_drift_pct": abs(float(raw["mass_drift_pct"])),
    }

    ref_vs_got = {
        "u_top_row": {"ref": float(u_ref[-1]), "got": float(u_num[-1])},
        "wall_shear": {
            "ref": analytic_wall_shear(nu, U0, H_eff),
            "got": float(raw["tau_num"]),
        },
        "momentum_integral": {"ref": mom_ref, "got": mom_got},
        "profile_l2_norm": {
            "ref": float(np.linalg.norm(u_ref)),
            "got": float(np.linalg.norm(u_num)),
        },
    }

    return {
        "route": "A_d2q9_moving_wall_bounce_back",
        "impl": "benchmarks/verified/couette_2d/run.py::run_case",
        "tag": tag,
        "H": H,
        "tau": tau,
        "nu": nu,
        "U0": U0,
        "Re": U0 * H_eff / nu,
        "Ma": U0 / math.sqrt(1.0 / 3.0),
        "collision": collision,
        "n_steps": int(raw["n_steps"]),
        "steady": bool(raw["steady"]),
        "reused_cached_artifact": reused,
        "elapsed_s": raw["elapsed_s"],
        "profile": [
            {"y": float(y), "u_num": float(a), "u_ref": float(b)}
            for y, a, b in zip(y_phys, u_num, u_ref)
        ],
        "ref_vs_got": ref_vs_got,
        "errors_pct": rows,
        "thresholds_pct": {k: THRESHOLDS[k] for k in rows},
        "pass": {k: (v <= THRESHOLDS[k]) for k, v in rows.items() if THRESHOLDS.get(k) is not None},
        "artifact": str(out_json.relative_to(REPO_ROOT)) if out_json.is_relative_to(REPO_ROOT) else str(out_json),
    }


# ---------------------------------------------------------------------------
# route B: reuse tensorlbm.curved_boundary_calibration.run_plane_couette
# ---------------------------------------------------------------------------


def run_route_b(
    ny: int,
    wall_speed: float,
    tau: float,
    fmt: str,
    out_dir: Path,
    steps: int,
) -> dict:
    from tensorlbm.curved_boundary_calibration import run_plane_couette

    tag = f"ny{ny}_tau{tau:g}_U{wall_speed:g}_{fmt}"
    res = run_plane_couette(
        wall_speed=wall_speed,
        tau=tau,
        format=fmt,
        ny=ny,
        nx=2,
        nz=2,
        steps=steps,
        collision="regularized",
    )
    out_json = out_dir / f"routeB_{tag}.json"
    out_json.write_text(json.dumps(res, indent=2))

    got = {
        "force_me_rel_pct": float(res["force_rel_err"]) * 100.0,
        "force_bottom_rel_pct": float(res["force_bottom_rel_err"]) * 100.0,
        "force_cv_rel_pct": float(res["cv_rel_err"]) * 100.0,
        "force_two_route_rel_pct": float(res["me_vs_cv_rel_diff"]) * 100.0,
        "profile_max_rel_pct": float(res["profile_max_rel_err"]) * 100.0,
    }
    measured = {k: got[k] for k in got if k != "profile_max_rel_pct"}
    got["profile_max_rel_pct_routeB"] = got.pop("profile_max_rel_pct")
    measured["profile_max_rel_pct_routeB"] = got["profile_max_rel_pct_routeB"]
    return {
        "route": "B_d3q19_interpolated_bounce_back_force_routes",
        "impl": "tensorlbm.curved_boundary_calibration.run_plane_couette",
        "tag": tag,
        "format": fmt,
        "ny": ny,
        "height": res["height"],
        "tau": res["tau"],
        "nu": res["nu"],
        "wall_speed": res["wall_speed"],
        "steps": res["steps"],
        "collision": res["collision"],
        "ref_vs_got": {
            "force_top_x": {"ref": res["force_analytic"], "got": res["force_top_x"]},
            "force_bottom_x": {"ref": -res["force_analytic"], "got": res["force_bottom_x"]},
            "force_cv_x": {"ref": res["force_analytic"], "got": res["cv_flux_midplane"]},
        },
        "errors_pct": got,
        "thresholds_pct": {k: THRESHOLDS[k] for k in measured},
        "pass": {k: (v <= THRESHOLDS[k]) for k, v in measured.items()},
        "artifact": str(out_json.relative_to(REPO_ROOT)) if out_json.is_relative_to(REPO_ROOT) else str(out_json),
    }


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------


def environment() -> dict:
    import torch

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - git absent
        commit = "unknown"
    return {
        "repo": "TensorLBM",
        "git_commit": commit,
        "branch": os.environ.get("CASE_POOL_BRANCH", ""),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "arch": platform.machine(),
        "numpy": np.__version__,
        "compile_mode": "eager",
        "seed": 0,
        "threads": torch.get_num_threads(),
    }


def _fmt_row(name: str, d: dict) -> str:
    tolp = THRESHOLDS.get(name)
    ok = d["pass"].get(name)
    mark = "✓" if ok else "✗"
    tol = f"{tolp:g}%" if tolp is not None else "-"
    return f"| {name} | {tol} | {d['errors_pct'][name]:.3e}% | {mark} |"


def main() -> int:
    ap = argparse.ArgumentParser(description="CASE-COUETTE-2D case-pool driver")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--quick", action="store_true", help="fast subset (tests)")
    ap.add_argument("--preset", choices=["full", "pool"], default="full",
                    help="pool = reference grid points H=16/32 + all sensitivity axes at H=16")
    ap.add_argument("--force", action="store_true", help="re-simulate even if the artifact exists")
    ap.add_argument("--min-steps", type=int, default=10000)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--route-b-steps", type=int, default=8000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.quick:
        grid = [{"H": 16, "tau": 0.8, "U0": 0.05, "collision": "bgk"}]
        b_cases = [{"ny": 18, "wall_speed": 0.05, "tau": 0.8, "fmt": "halfway"}]
    elif args.preset == "pool":
        # two grid points (H = 16, 32) + all sensitivity axes kept on the cheap
        # H = 16 grid (the sensitivity question is scale-independent: it is about
        # Re and Ma, not about resolution)
        grid = [
            {"H": 16, "tau": 0.8, "U0": 0.05, "collision": "bgk"},
            {"H": 32, "tau": 0.8, "U0": 0.05, "collision": "bgk"},
            {"H": 16, "tau": 0.6, "U0": 0.05, "collision": "bgk", "axis": "tau"},
            {"H": 16, "tau": 1.4, "U0": 0.05, "collision": "bgk", "axis": "tau"},
            {"H": 16, "tau": 0.8, "U0": 0.01, "collision": "bgk", "axis": "U0"},
            {"H": 16, "tau": 0.8, "U0": 0.10, "collision": "bgk", "axis": "U0"},
            {"H": 16, "tau": 0.8, "U0": 0.05, "collision": "mrt", "axis": "collision"},
        ]
        b_cases = [
            {"ny": 18, "wall_speed": 0.05, "tau": 0.8, "fmt": "halfway"},
            {"ny": 34, "wall_speed": 0.05, "tau": 0.8, "fmt": "bouzidi_quadratic"},
        ]
    else:
        # grid-convergence column (fixed tau, U0)
        grid = [
            {"H": 16, "tau": 0.8, "U0": 0.05, "collision": "bgk"},
            {"H": 32, "tau": 0.8, "U0": 0.05, "collision": "bgk"},
            {"H": 64, "tau": 0.8, "U0": 0.05, "collision": "bgk"},
        ]
        # viscosity (tau -> Re) sensitivity at fixed H
        grid += [
            {"H": 32, "tau": 0.6, "U0": 0.05, "collision": "bgk", "axis": "tau"},
            {"H": 32, "tau": 1.4, "U0": 0.05, "collision": "bgk", "axis": "tau"},
        ]
        # compressibility (U0 -> Ma) sensitivity at fixed H, tau
        grid += [
            {"H": 32, "tau": 0.8, "U0": 0.01, "collision": "bgk", "axis": "U0"},
            {"H": 32, "tau": 0.8, "U0": 0.10, "collision": "bgk", "axis": "U0"},
        ]
        # collision-model cross-check at the reference grid point
        grid += [{"H": 32, "tau": 0.8, "U0": 0.05, "collision": "mrt", "axis": "collision"}]
        b_cases = [
            {"ny": 18, "wall_speed": 0.05, "tau": 0.65, "fmt": "halfway"},
            {"ny": 34, "wall_speed": 0.05, "tau": 0.8, "fmt": "bouzidi_quadratic"},
        ]

    route_a = []
    for case in grid:
        r = run_route_a(
            H=case["H"],
            tau=case["tau"],
            U0=case["U0"],
            collision=case["collision"],
            out_dir=out_dir,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            compile_mode="eager",
            force=args.force,
        )
        r["axis"] = case.get("axis", "grid")
        route_a.append(r)
        print(
            f"[A] {r['tag']:28s} l2={r['errors_pct']['profile_l2_rel_pct']:.3e}% "
            f"max={r['errors_pct']['profile_max_rel_pct']:.3e}% "
            f"tau={r['errors_pct']['wall_shear_rel_pct']:.3e}% "
            f"steps={r['n_steps']} t={r['elapsed_s']}s pass={all(r['pass'].values())}",
            flush=True,
        )

    route_b = []
    for case in b_cases:
        r = run_route_b(
            ny=case["ny"],
            wall_speed=case["wall_speed"],
            tau=case["tau"],
            fmt=case["fmt"],
            out_dir=out_dir,
            steps=args.route_b_steps,
        )
        route_b.append(r)
        print(
            f"[B] {r['tag']:28s} ME={r['errors_pct']['force_me_rel_pct']:.3e}% "
            f"CV={r['errors_pct']['force_cv_rel_pct']:.3e}% "
            f"2route={r['errors_pct']['force_two_route_rel_pct']:.3e}% "
            f"pass={all(r['pass'].values())}",
            flush=True,
        )

    # ---- convergence order from the L2 column (H = 16, 32, 64) ----
    grid_pts = sorted(
        [r for r in route_a if r["axis"] == "grid" and r["collision"] == "bgk" and r["tau"] == 0.8 and r["U0"] == 0.05],
        key=lambda r: r["H"],
    )
    convergence = []
    for a, b in zip(grid_pts, grid_pts[1:]):
        e1 = a["errors_pct"]["profile_l2_rel_pct"]
        e2 = b["errors_pct"]["profile_l2_rel_pct"]
        order = math.log(e1 / e2) / math.log(b["H"] / a["H"]) if e2 > 0 else float("inf")
        convergence.append(
            {"H_from": a["H"], "H_to": b["H"], "l2_pct_from": e1, "l2_pct_to": e2, "observed_order": order}
        )

    # ---- two-route shear consensus (A@H=32 vs B) ----
    a_ref = next(
        (r for r in route_a if r["H"] == 32 and r["tau"] == 0.8 and r["U0"] == 0.05 and r["collision"] == "bgk"),
        None,
    )
    cross = []
    if a_ref is not None:
        for b in route_b:
            tau_a = a_ref["ref_vs_got"]["wall_shear"]["got"]
            # run_plane_couette uses nx=nz=2 -> wall area = 4; tau = |F| / area
            tau_b = abs(b["ref_vs_got"]["force_top_x"]["got"]) / 4.0
            # Routes run at DIFFERENT gap heights (H=32 vs ny-2), so the physical
            # shears are not comparable one-to-one.  The comparison is made on the
            # dimensionless wall-shear coefficient C = tau_w H / (nu U0), which is
            # 1 for the analytic Couette solution at every height.
            c_a = tau_a * a_ref["H"] / (a_ref["nu"] * a_ref["U0"])
            c_b = tau_b * b["height"] / (b["nu"] * b["wall_speed"])
            rel = abs(c_a - c_b) / max(abs(c_a), 1e-30) * 100.0
            rel_raw = abs(tau_a - tau_b) / max(abs(tau_a), 1e-30) * 100.0
            cross.append(
                {
                    "route_a": a_ref["tag"],
                    "route_b": b["tag"],
                    "tau_route_a": tau_a,
                    "tau_route_b": tau_b,
                    "shear_coeff_route_a": c_a,
                    "shear_coeff_route_b": c_b,
                    "rel_diff_pct": rel,
                    "rel_diff_raw_dimfull_pct_v1": rel_raw,
                    "tol_pct": CROSS_ROUTE_TOL_PCT,
                    "pass": rel <= CROSS_ROUTE_TOL_PCT,
                }
            )

    sensitivity = []
    for axis in ("tau", "U0", "collision"):
        pts = [r for r in route_a if r["axis"] == axis]
        sensitivity.append(
            {
                "axis": axis,
                "points": [
                    {
                        "tag": p["tag"],
                        "tau": p["tau"],
                        "Re": p["Re"],
                        "Ma": p["Ma"],
                        "collision": p["collision"],
                        "profile_l2_rel_pct": p["errors_pct"]["profile_l2_rel_pct"],
                        "profile_max_rel_pct": p["errors_pct"]["profile_max_rel_pct"],
                        "wall_shear_rel_pct": p["errors_pct"]["wall_shear_rel_pct"],
                        "mass_drift_pct": p["errors_pct"]["mass_drift_pct"],
                        "pass": all(p["pass"].values()),
                    }
                    for p in pts
                ],
            }
        )

    failing = []
    for r in route_a + route_b:
        failing += [f"{r['route']}:{r['tag']}:{k}" for k, ok in r["pass"].items() if not ok]
    failing += [f"two_route:{c['route_b']}" for c in cross if not c["pass"]]
    for c in convergence:
        if c["observed_order"] < -ORDER_SLACK:
            failing.append(f"convergence:{c['H_from']}->{c['H_to']}")

    # split the predeclared criteria into the two families they belong to:
    # (a) accuracy vs the closed form (<2 % pool tolerance), (b) self-imposed
    # guards/diagnostics (mass-drift band, convergence-order sign).
    ACC_KEYS = (
        "profile_l2_rel_pct", "profile_max_rel_pct", "wall_shear_rel_pct",
        "top_row_u_rel_pct", "momentum_integral_rel_pct", "force_me_rel_pct",
        "force_cv_rel_pct", "force_two_route_rel_pct", "force_bottom_rel_pct",
        "profile_max_rel_pct_routeB",
    )
    tol_failing = [
        f for f in failing
        if any(f.endswith(k) for k in ACC_KEYS) or f.startswith("two_route:")
    ]

    summary = {
        "case_code": CASE_CODE,
        "case": "steady plane Couette flow between parallel plates (moving top wall, x-periodic channel)",
        "repo": "TensorLBM",
        "ref": {
            "source": REF_SOURCE,
            "formula": REF_FORMULA,
            "citation_note": REF_CITATION_NOTE,
            "generator": "analytic_profile / analytic_wall_shear / analytic_momentum_integral (closed form, evaluated in code)",
        },
        "tol_policy_pct": TOL_PCT,
        "predeclared_thresholds_pct": THRESHOLDS,
        "cross_route_tol_pct": CROSS_ROUTE_TOL_PCT,
        "route_a": route_a,
        "route_b": route_b,
        "convergence": convergence,
        "two_route_consensus": cross,
        "sensitivity": sensitivity,
        "verdict": {
            "pass": not failing,
            "tolerance_failing": tol_failing,
            "pool_tolerance_pass": not tol_failing,
            "guard_or_diagnostic_failing": [f for f in failing if f not in tol_failing],
            "failing": failing,
            "decision_rule": "<2% on every listed quantity at the reference grid point (H=32, tau=0.8, U0=0.05); "
            "non-deteriorating L2 with resolution; two independent routes must agree <2%",
        },
        "environment": environment(),
        "wall_clock_s": round(time.time() - t_start, 1),
        "repro_cmd": "PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/couette_2d/run_case.py "
        "--preset pool --out-dir benchmarks/case_pool/couette_2d/artifacts",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== ref vs got (reference grid point H=32, tau=0.8, U0=0.05) ===")
    if a_ref is not None:
        print("| quantity | tol | rel err | pass |")
        print("|---|---|---|---|")
        for k in THRESHOLDS:
            if k in a_ref["errors_pct"]:
                print(_fmt_row(k, a_ref))
    print("\n=== route B (independent force route) ===")
    for b in route_b:
        print(f"-- {b['tag']}")
        for k in b["errors_pct"]:
            if k in b["thresholds_pct"]:
                print(_fmt_row(k, b))
    print("\nconvergence:", json.dumps(convergence))
    print("two-route:", json.dumps(cross))
    print("VERDICT pass =", summary["verdict"]["pass"], "failing =", failing)
    print("summary ->", out_dir / "summary.json", f"({summary['wall_clock_s']}s)")
    return 0 if summary["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())