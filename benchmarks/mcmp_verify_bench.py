#!/usr/bin/env python3
"""FEAT-LBM-VERIFY-BENCH runner: MCMP calibration + falsification baseline.

Produces, from a single command, the three deliverables of the mission:

1. a Young-Laplace calibration over >= 3 surface-tension levels and >= 3 radii
   per level, with the acceptance metric of
   :func:`tensorlbm.mcmp_verification.young_laplace_calibration`;
2. a spurious-current baseline table over a (grid x coupling) matrix, plus the
   interface thickness measured on the same runs;
3. the contact-angle / interface-thickness calibration on analytic fields with
   known answers (no solver in the loop, so it is a *measurement* calibration).

All numbers are written to ``artifacts/mcmp_verify_bench/results.json`` and the
markdown tables in the report are rendered **from that json** - nothing in the
report is typed by hand.  Fields of the "hero" run are stored as ``.npz`` so the
figures are generated from the same data the tables come from.

Usage::

    python benchmarks/mcmp_verify_bench.py --out artifacts/mcmp_verify_bench \
        --jobs 8 [--quick]

``--quick`` shrinks the protocol (fewer steps, smaller grids) for a smoke run;
the acceptance numbers in the report must come from a full run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tensorlbm.mcmp_verification as mv  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol (pre-registered: fixed before the numbers were produced)
# ---------------------------------------------------------------------------

#: Coupling levels.  The valid window at rho_total ~ 1, tau = 1 was mapped by
#: the guardrails themselves: G_12 = -2.2 dissolves the droplet, -2.3 inverts
#: it (the interior expels component 1), -3.5 and below diverge within 250
#: steps.  The three levels below are inside the window.
G_LEVELS = [-2.5, -2.75, -3.0]

#: Droplet radii of the Laplace series (all <= 0.3 n for the n used).
RADII = [14.0, 19.0, 24.0]

#: (grid, coupling) matrix of the spurious-current table, fixed physical radius.
SPURIOUS_GRIDS = [48, 64, 96]
SPURIOUS_G = [-2.5, -2.75, -3.0]
SPURIOUS_RADIUS = 10.0

#: Analytic contact-angle calibration: true angles and the cap geometry.
CONTACT_ANGLES = [30.0, 45.0, 60.0, 90.0, 120.0, 135.0]
CONTACT_CAP = {"n": 96, "radius": 25.0, "wall_rows": 6, "width": 1.5}

#: Analytic interface-thickness calibration: tanh profiles with known w.
THICKNESS_W = [1.0, 1.5, 2.0, 3.0]

FULL = {"n_laplace": 96, "steps_laplace": 12000, "steps_coex": 12000, "steps_spurious": 12000}
QUICK = {"n_laplace": 64, "steps_laplace": 4000, "steps_coex": 4000, "steps_spurious": 4000}


# ---------------------------------------------------------------------------
# Jobs (module level so they pickle for multiprocessing)
# ---------------------------------------------------------------------------


def _quiet_threads() -> None:
    torch.set_num_threads(1)


def job_coexistence(args: tuple) -> dict:
    G, n, steps = args
    _quiet_threads()
    out = mv.estimate_coexistence(G_12=G, n=n, geometry="planar", steps=steps)
    out["flat_interface_dp"] = out.get("flat_interface_dp")
    return out


def job_laplace(args: tuple) -> dict:
    G, n, radius, steps, coex = args
    _quiet_threads()
    run = mv.simulate_static_droplet(
        G_12=G,
        n=n,
        radius_init=radius,
        rho1_in=coex["rho1_liq"],
        rho2_in=coex["rho2_liq"],
        rho1_out=coex["rho1_gas"],
        rho2_out=coex["rho2_gas"],
        steps=steps,
    )
    case = mv.measure_laplace(run)
    case["history"] = run["history"]
    case["coexistence"] = {k: v for k, v in coex.items()}
    return case


def job_spurious(args: tuple) -> dict:
    n, G, radius, steps, hero = args
    _quiet_threads()
    run = mv.simulate_static_droplet(G_12=G, n=n, radius_init=radius, steps=steps)
    case = mv.measure_laplace(run)
    rho_bar = float((run["rho1"] + run["rho2"]).mean())
    mu = rho_bar * (run["tau1"] - 0.5) / 3.0
    sigma = case["sigma"]
    thickness = mv.interface_thickness(run["rho1"])
    row = {
        "n": n,
        "G_12": G,
        "radius_init": radius,
        "steps_run": case["steps_run"],
        "converged": case["converged"],
        "u_max": case["u_max"],
        "u_max_over_cs": case["u_max"] / math.sqrt(mv.CS2),
        "sigma": sigma,
        "ca_spurious": None if not sigma else mu * case["u_max"] / sigma,
        "mu_lattice": mu,
        "dp": case["dp"],
        "radius": case["radius"],
        "density_ratio": case["density_ratio"],
        "w_tanh": thickness["w_tanh"],
        "width_10_90": thickness["width_10_90"],
        "thickness_r2": thickness["r2"],
        "stencil": mv.FORCE_STENCIL,
        "guardrails": case["guardrails"] + thickness["guardrails"],
    }
    if hero:
        row["_fields"] = {
            "rho1": run["rho1"].numpy(),
            "rho2": run["rho2"].numpy(),
            "ux": run["ux"].numpy(),
            "uy": run["uy"].numpy(),
        }
    return row


def job_contact(args: tuple) -> dict:
    theta, cap = args
    _quiet_threads()
    rho, solid = mv.synthetic_cap_field(
        n=cap["n"],
        radius=cap["radius"],
        theta_deg=theta,
        wall_rows=cap["wall_rows"],
        width=cap["width"],
    )
    return {
        "theta_true": theta,
        "height": -cap["radius"] * math.cos(math.radians(theta)),
        "geometric": mv.geometric_contact_angle(rho, solid, order=2, first_layer=2),
        "circle_fit": mv.circle_fit_contact_angle(rho, solid),
        "cap_geometry": mv.cap_geometry_contact_angle(rho, solid),
    }


def job_thickness(args: tuple) -> dict:
    (w,) = args
    _quiet_threads()
    n = 128
    centre = (n - 1) / 2.0
    yy, xx = mv._grid(n, torch.device("cpu"))
    r = torch.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    rho = 0.14 + 0.78 * 0.5 * (1.0 - torch.tanh((r - 20.0) / w))
    out = mv.interface_thickness(rho)
    out["w_true"] = w
    out["width_10_90_true"] = mv.interface_width_10_90(w)
    out.pop("profile_r", None)
    out.pop("profile_rho", None)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_all(jobs, fn, workers: int):
    if workers <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(processes=workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def build(out_dir: Path, workers: int, quick: bool) -> dict:
    proto = QUICK if quick else FULL
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_lap = proto["n_laplace"]

    # 1. coexistence (one flat slab per coupling level) ---------------------
    log(f"coexistence: {len(G_LEVELS)} planar slab runs at n={n_lap}")
    coex_rows = _run_all(
        [(G, n_lap, proto["steps_coex"]) for G in G_LEVELS], job_coexistence, workers
    )
    coex = {row["G_12"]: row for row in coex_rows}
    for row in coex_rows:
        log(
            f"  G={row['G_12']}: rho1_liq={row['rho1_liq']:.4f} rho1_gas={row['rho1_gas']:.4f} "
            f"flat_dp={row['flat_interface_dp']:.2e} umax={row['u_max']:.4f}"
        )

    # 2. Young-Laplace series ----------------------------------------------
    log(f"laplace: {len(G_LEVELS)} levels x {len(RADII)} radii at n={n_lap}")
    lap_rows = _run_all(
        [(G, n_lap, R, proto["steps_laplace"], coex[G]) for G in G_LEVELS for R in RADII],
        job_laplace,
        workers,
    )
    levels = []
    for G in G_LEVELS:
        cases = [c for c in lap_rows if c["G_12"] == G]
        levels.append(mv._fit_laplace_level(G, cases, 0.05))
    laplace = {
        "protocol": {
            "n": n_lap,
            "steps": proto["steps_laplace"],
            "coexist_steps": proto["steps_coex"],
            "radii": RADII,
            "G_values": G_LEVELS,
            "threshold": 0.05,
            "coexistence_geometry": "planar",
            "metric": "one-parameter Young-Laplace Delta p = 2 sigma / R, sigma "
            "fitted through the origin; per-case relative error",
        },
        "levels": levels,
        "cases": lap_rows,
        "coexistence": coex_rows,
        "pass": all(level["pass"] for level in levels),
    }
    for level in levels:
        log(
            f"  G={level['G_12']}: sigma_origin={level['sigma_origin']:.5f} "
            f"max_rel_err={level['max_rel_err']:.3%} pass={level['pass']} "
            f"guards={[g['code'] for g in level['guardrails']]}"
        )

    # 3. spurious currents --------------------------------------------------
    sp_jobs = []
    hero_cfg = (96, -3.0)
    for G in SPURIOUS_G:
        for n in SPURIOUS_GRIDS:
            hero = (n, G) == hero_cfg and not quick
            sp_jobs.append((n, G, SPURIOUS_RADIUS, proto["steps_spurious"], hero))
    log(f"spurious currents: {len(sp_jobs)} runs")
    sp_rows = _run_all(sp_jobs, job_spurious, workers)
    hero_fields = None
    for row in sp_rows:
        if "_fields" in row:
            hero_fields = row.pop("_fields")
            row["hero"] = True
        else:
            row["hero"] = False
        log(
            f"  n={row['n']} G={row['G_12']}: |u|max={row['u_max']:.4e} "
            f"Ca_sp={row['ca_spurious'] if row['ca_spurious'] is None else round(row['ca_spurious'], 4)} "
            f"w={row['w_tanh'] if row['w_tanh'] is None else round(row['w_tanh'], 2)} "
            f"guards={[g['code'] for g in row['guardrails']]}"
        )
    spurious = {
        "stencil": mv.FORCE_STENCIL,
        "radius": SPURIOUS_RADIUS,
        "grids": SPURIOUS_GRIDS,
        "G_values": SPURIOUS_G,
        "rows": sp_rows,
    }

    # 4. analytic calibration of the fast checks ----------------------------
    log("contact angle: analytic caps")
    contact = _run_all([(th, CONTACT_CAP) for th in CONTACT_ANGLES], job_contact, workers)
    log("interface thickness: analytic tanh profiles")
    thickness = _run_all([(w,) for w in THICKNESS_W], job_thickness, workers)

    # 5. guardrail map (falsification: which parameters are NOT usable) ------
    if quick:
        probe = []
    else:
        log("guardrail map: out-of-window couplings")
        probe_G = [-2.2, -2.3, -3.5]
        probe_runs = _run_all([(64, G, 14.0, 6000, False) for G in probe_G], job_spurious, workers)
        probe = [
            {
                "G_12": row["G_12"],
                "u_max": row["u_max"],
                "radius": row["radius"],
                "dp": row["dp"],
                "converged": row["converged"],
                "guardrails": row["guardrails"],
            }
            for row in probe_runs
        ]
        for row in probe:
            log(
                f"  G={row['G_12']}: radius={row['radius']} guards="
                f"{[g['code'] for g in row['guardrails']]}"
            )

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {"quick": quick, **proto},
        "library": {"version": _version(), "torch": torch.__version__},
        "laplace": laplace,
        "spurious": spurious,
        "contact_angle": contact,
        "interface_thickness": thickness,
        "guardrail_probe": probe,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if hero_fields is not None:
        import numpy as np

        np.savez_compressed(
            out_dir / "hero_fields.npz",
            **{k: v for k, v in hero_fields.items()},
            G_12=hero_cfg[1],
            n=hero_cfg[0],
        )
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=_json_safe))
    log(f"wrote {out_dir / 'results.json'}")
    return results


def _json_safe(obj):  # numpy scalars / tuples
    try:
        return obj.item()
    except AttributeError:
        return list(obj) if isinstance(obj, tuple) else str(obj)


def _version() -> str:
    try:
        from tensorlbm import __version__ as v

        return str(v)
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "mcmp_verify_bench"))
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 4)))
    ap.add_argument("--quick", action="store_true", help="smoke protocol, not for acceptance")
    ap.add_argument("--report", action="store_true", help="also render docs/MCMP_VERIFY_BENCH.md")
    args = ap.parse_args()
    out = Path(args.out)
    build(out, args.jobs, args.quick)
    if args.report:
        from tensorlbm.mcmp_verify_report import render_report

        render_report(out, REPO_ROOT / "docs" / "MCMP_VERIFY_BENCH.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
