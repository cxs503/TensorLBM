#!/usr/bin/env python3
"""Incremental runner for 3-D MCMP spurious-current scaling.

Runs cases one at a time, appending results to a JSON file after each case.
This allows recovery if the process is interrupted.

Usage::

    python benchmarks/mcmp3d_spurious_scaling/run_incremental.py \
        --out artifacts/mcmp3d_spurious_scaling --threads 2

"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Protocol parameters (frozen, same as run.py)
GRIDS = [32, 48, 64]
G_LEVELS = [-4.4, -4.7, -5.0, -5.3, -5.6]
FORCINGS = ["kupershtokh", "velocity_shift"]
FORM = "difference"
G_CONTROL_ZERO = -4.0
G_CONTROL_WB = -5.0
TAU = 1.0
STEPS_FULL = 6000
CHECK_EVERY = 200
STEADY_TOL = 1e-3
STEADY_WINDOW = 3
WIDTH_INIT = 2.0


def radius_of(n: int) -> float:
    return n / 8.0


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:
        return ""


def coexistence_table(g_levels: list[float]) -> dict:
    from tensorlbm.pseudopotential_edm3d import maxwell_coexistence
    table = {}
    for G in sorted(set(g_levels)):
        coex = maxwell_coexistence(G)
        table[str(G)] = coex
        print(f"[coex] G_eff={G:+.2f}  rho_l={coex['rho_l']:.5f} "
              f"rho_v={coex['rho_v']:.6f}  drho={coex['rho_l'] - coex['rho_v']:.5f} "
              f"ratio={coex['density_ratio']:.3f}  converged={coex['converged']}",
              flush=True)
    return table


def run_case(case: dict, threads: int = 2) -> dict:
    import torch
    from tensorlbm.pseudopotential_edm3d import static_droplet

    torch.set_num_threads(threads)
    t0 = time.time()
    out = static_droplet(
        R=radius_of(case["grid"]),
        n=case["grid"],
        G_eff=case["G"],
        tau=case["tau"],
        stencil="d3q19",
        form=case["form"],
        forcing=case["forcing"],
        rho_l=case["rho_l"],
        rho_v=case["rho_v"],
        width=WIDTH_INIT,
        steps=case["steps"],
        check_every=case["check_every"],
        steady_tol=case["steady_tol"],
        steady_window=case["steady_window"],
        thread_count=threads,
    )
    tail = out["history"][-case.get("steady_window", STEADY_WINDOW):]
    rec = {
        "id": case["id"],
        "kind": case["kind"],
        "grid": case["grid"],
        "G_eff": case["G"],
        "forcing": case["forcing"],
        "form": case["form"],
        "tau": case["tau"],
        "R": radius_of(case["grid"]),
        "W": WIDTH_INIT,
        "rho_l_maxwell": case["rho_l"],
        "rho_v_maxwell": case["rho_v"],
        "drho_maxwell": case["rho_l"] - case["rho_v"],
        "ratio_maxwell": case["rho_l"] / case["rho_v"] if case["rho_v"] > 0 else float("inf"),
        "u_max": out["u_max"],
        "u_max_iface": out["u_max_iface"],
        "u_max_core": out["u_max_core"],
        "dp": out["dp"],
        "sigma": out["sigma"],
        "R_eq": out["R_eq"],
        "mass_drift_rel": out["mass_drift_rel"],
        "rho_min_tail": float(min(h["rho_min"] for h in tail)) if tail else float("nan"),
        "rho_max_tail": float(max(h["rho_max"] for h in tail)) if tail else float("nan"),
        "drho_measured": (
            float(max(h["rho_max"] for h in tail) - min(h["rho_min"] for h in tail))
            if tail else float("nan")
        ),
        "u_max_spread_rel": (
            (max(h["u_max"] for h in tail) - min(h["u_max"] for h in tail))
            / abs(sum(h["u_max"] for h in tail) / len(tail))
            if tail and any(h["u_max"] for h in tail) else float("nan")
        ),
        "steps": out["steps"],
        "converged": bool(out["converged"]),
        "finite": bool(out["finite"]),
        "wall_s": time.time() - t0,
        "history": out["history"],
    }
    return rec


def build_cases(grids, g_levels, forcings, controls: bool = True) -> list[dict]:
    cases = []
    for n in grids:
        for G in g_levels:
            for f in forcings:
                cases.append({"id": f"main_n{n}_G{G}_{f}", "kind": "main",
                              "grid": n, "G": G, "forcing": f, "form": FORM})
    if not controls:
        return cases
    for n in grids:
        cases.append({"id": f"expl_psi2_n{n}_G{G_CONTROL_WB}", "kind": "exploratory_psi2",
                      "grid": n, "G": G_CONTROL_WB, "forcing": "kupershtokh", "form": "psi2"})
        cases.append({"id": f"ctrlZ_n{n}_G{G_CONTROL_ZERO}", "kind": "control_zero",
                      "grid": n, "G": G_CONTROL_ZERO, "forcing": "kupershtokh", "form": FORM})
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/mcmp3d_spurious_scaling")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=STEPS_FULL)
    ap.add_argument("--check-every", type=int, default=CHECK_EVERY)
    ap.add_argument("--steady-tol", type=float, default=STEADY_TOL)
    ap.add_argument("--resume", action="store_true", help="Resume from existing results.json")
    args = ap.parse_args(argv)

    import torch
    from tensorlbm import __version__ as tlbm_version

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    results_path = outdir / "results.json"

    # Coexistence table
    all_g = list(G_LEVELS) + [G_CONTROL_ZERO]
    coex = coexistence_table(all_g)

    # Build cases
    cases = build_cases(GRIDS, G_LEVELS, FORCINGS, controls=True)
    for c in cases:
        pair = coex.get(str(c["G"]))
        c["rho_l"] = float(pair["rho_l"]) if pair else float("nan")
        c["rho_v"] = float(pair["rho_v"]) if pair else float("nan")
        c["tau"] = TAU
        c["steps"] = args.steps
        c["check_every"] = args.check_every
        c["steady_tol"] = args.steady_tol
        c["steady_window"] = STEADY_WINDOW

    # Load existing results if resuming
    done_ids = set()
    existing_cases = []
    if args.resume and results_path.exists():
        existing = json.loads(results_path.read_text())
        existing_cases = existing.get("cases", [])
        done_ids = {c["id"] for c in existing_cases if "error" not in c}
        print(f"[resume] Found {len(done_ids)} completed cases", flush=True)

    # Filter out done cases
    pending = [c for c in cases if c["id"] not in done_ids]
    print(f"[bench] {len(pending)} pending cases out of {len(cases)} total, threads={args.threads}", flush=True)

    t0_total = time.time()
    all_cases = list(existing_cases)

    for i, case in enumerate(pending):
        case_t0 = time.time()
        print(f"[{i+1}/{len(pending)}] Running {case['id']}...", flush=True)
        try:
            rec = run_case(case, threads=args.threads)
            all_cases.append(rec)
            print(f"  -> u_max={rec['u_max']:.6e} steps={rec['steps']} "
                  f"converged={rec['converged']} ({rec['wall_s']:.1f}s)", flush=True)
        except Exception as exc:
            rec = {"id": case["id"], "kind": case["kind"], "grid": case["grid"],
                   "G_eff": case["G"], "forcing": case["forcing"], "form": case["form"],
                   "error": f"{type(exc).__name__}: {exc}", "u_max": float("nan")}
            all_cases.append(rec)
            print(f"  -> ERROR: {exc}", flush=True)

        # Save after each case
        payload = {
            "meta": {
                "gap": "GAP-PP-3-3DMCMP-SPURIOUS-SCALING",
                "protocol": "benchmarks/mcmp3d_spurious_scaling/README.md",
                "grids": GRIDS,
                "g_levels": G_LEVELS,
                "forcings": FORCINGS,
                "form": FORM,
                "tau": TAU,
                "steps_cap": args.steps,
                "check_every": args.check_every,
                "steady_tol": args.steady_tol,
                "steady_window": STEADY_WINDOW,
                "radius_rule": "R = n/8",
                "width_rule": f"W_init = {WIDTH_INIT} (constant, lattice units)",
                "quick": False,
                "engine": "tensorlbm.pseudopotential_edm3d.static_droplet (D3Q19, float64)",
                "wall_s": time.time() - t0_total,
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "tensorlbm": tlbm_version,
                "git_commit": _git(["rev-parse", "HEAD"]),
                "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "command": " ".join([sys.executable, "benchmarks/mcmp3d_spurious_scaling/run_incremental.py"] + (argv or [])),
            },
            "coexistence": coex,
            "cases": all_cases,
        }
        results_path.write_text(json.dumps(payload, indent=2))
        print(f"  -> Saved results.json ({len(all_cases)} cases total)", flush=True)

    print(f"\n[bench] All {len(all_cases)} cases completed in {time.time()-t0_total:.0f}s", flush=True)
    print(f"[bench] Results saved to {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
