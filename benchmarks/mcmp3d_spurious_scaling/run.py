#!/usr/bin/env python3
"""3-D MCMP spurious-current scaling bench (gap ``GAP-PP-3-3DMCMP-SPURIOUS-SCALING``).

Measures ``u_max`` of a static droplet for a matrix of

    (grid n)  x  (coupling G_eff -> density difference Δρ)  x  (forcing: EDM | classic)

plus the two falsification controls described in ``README.md`` (W: well-balanced
``psi2`` force; Z: no-interface critical coupling), and writes every number to a
single JSON artifact.  Nothing downstream (tables, figures, verdict) is typed by
hand: :mod:`fit_report` renders it all from that JSON.

The protocol (grids, couplings, steady-state criterion) is pre-registered in
``README.md`` and is *not* tuned after looking at the results.

Usage::

    python benchmarks/mcmp3d_spurious_scaling/run.py \
        --out artifacts/mcmp3d_spurious_scaling --jobs 32 [--quick]

``--quick`` shrinks everything to a smoke test (one grid, one coupling, 600
steps); the acceptance numbers must come from the full protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Pre-registered protocol (README.md; frozen before production numbers)
# ---------------------------------------------------------------------------

GRIDS = [32, 48, 64]                     # 3 resolutions
G_LEVELS = [-4.4, -4.7, -5.0, -5.3, -5.6]  # 5 Maxwell pairs
FORCINGS = ["kupershtokh", "velocity_shift"]  # EDM vs classic
FORM = "difference"                      # spatial force form (see README)
G_CONTROL_ZERO = -4.0                    # no-interface control (Δρ ~ 2e-3)
G_CONTROL_WB = -5.0                      # well-balanced control coupling
TAU = 1.0

STEPS_FULL = 6000
CHECK_EVERY = 200
STEADY_TOL = 1e-3
STEADY_WINDOW = 3

QUICK = dict(grids=[24], g_levels=[-5.0], forcings=["kupershtokh"],
             steps=600, check_every=100, controls=False)


def radius_of(n: int) -> float:
    """Self-similar droplet radius ``R = n/8`` (box ratio 0.125)."""
    return n / 8.0


#: Initial tanh width of the droplet profile, in lattice units, at every grid.
#: The *equilibrium* interface thickness is set by the model (it is ~constant in
#: lattice units for fixed G_eff), so ``W_init`` only seeds the transient; the
#: grid ladder refines the box/radius while the interface stays equally
#: resolved.  This is the standard spurious-current resolution ladder.
WIDTH_INIT = 2.0


def width_of(n: int) -> float:  # noqa: ARG001 - kept for a uniform call signature
    """Initial interface width (lattice units), constant over the grid ladder."""
    return WIDTH_INIT


# ---------------------------------------------------------------------------
# Single case
# ---------------------------------------------------------------------------


def run_case(case: dict) -> dict:
    """March one static droplet to steady state and return its measured record."""
    import torch

    from tensorlbm.pseudopotential_edm3d import static_droplet

    torch.set_num_threads(int(case.get("threads", 4)))
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
        width=width_of(case["grid"]),
        steps=case["steps"],
        check_every=case["check_every"],
        steady_tol=case["steady_tol"],
        steady_window=case["steady_window"],
        thread_count=int(case.get("threads", 4)),
    )
    tail = out["history"][-out.get("steady_window", STEADY_WINDOW):]
    rec = {
        "id": case["id"],
        "kind": case["kind"],
        "grid": case["grid"],
        "G_eff": case["G"],
        "forcing": case["forcing"],
        "form": case["form"],
        "tau": case["tau"],
        "R": radius_of(case["grid"]),
        "W": width_of(case["grid"]),
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
        "laplace_residual": None,  # dp = 2 sigma / R_eq is exact by construction
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


def _worker(payload: tuple[dict, dict]) -> dict:
    case, common = payload
    case = {**case, **common}
    try:
        return run_case(case)
    except Exception as exc:  # noqa: BLE001 - a failed case must not kill the campaign
        return {"id": case["id"], "kind": case["kind"], "grid": case["grid"],
                "G_eff": case["G"], "forcing": case["forcing"], "form": case["form"],
                "error": f"{type(exc).__name__}: {exc}", "u_max": float("nan")}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_cases(grids, g_levels, forcings, controls: bool = True) -> list[dict]:
    cases: list[dict] = []
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


def coexistence_table(g_levels) -> dict:
    from tensorlbm.pseudopotential_edm3d import maxwell_coexistence

    table = {}
    for G in sorted(set(list(g_levels) + [G_CONTROL_ZERO])):
        t0 = time.time()
        coex = maxwell_coexistence(G)
        table[str(G)] = {**coex, "wall_s": time.time() - t0}
        print(
            f"[coex] G_eff={G:+.2f}  rho_l={coex['rho_l']:.5f} "
            f"rho_v={coex['rho_v']:.6f}  drho={coex['rho_l'] - coex['rho_v']:.5f} "
            f"ratio={coex['density_ratio']:.3f}  converged={coex['converged']}",
            flush=True,
        )
    return table


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/mcmp3d_spurious_scaling")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS_FULL)
    ap.add_argument("--check-every", type=int, default=CHECK_EVERY)
    ap.add_argument("--steady-tol", type=float, default=STEADY_TOL)
    # Debug-only overrides (never used for the acceptance numbers).
    ap.add_argument("--grids", type=int, nargs="+", default=None)
    ap.add_argument("--g-levels", type=float, nargs="+", default=None)
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args(argv)

    import torch

    from tensorlbm import __version__ as tlbm_version

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        grids, g_levels, forcings = QUICK["grids"], QUICK["g_levels"], QUICK["forcings"]
        steps, check_every, controls = QUICK["steps"], QUICK["check_every"], QUICK["controls"]
    else:
        grids = list(args.grids) if args.grids else GRIDS
        g_levels = list(args.g_levels) if args.g_levels else G_LEVELS
        forcings = FORCINGS
        steps, check_every = args.steps, args.check_every
        controls = not args.no_controls

    coex = coexistence_table(g_levels)
    ctrl_coex = coexistence_table([G_CONTROL_ZERO]) if controls else {}

    known = {**coex, **ctrl_coex}
    cases = build_cases(grids, g_levels, forcings, controls=controls)
    for c in cases:
        pair = known.get(str(c["G"]))
        c["rho_l"] = float(pair["rho_l"]) if pair else float("nan")
        c["rho_v"] = float(pair["rho_v"]) if pair else float("nan")
        c["tau"] = TAU
        c["steps"] = steps
        c["check_every"] = check_every
        c["steady_tol"] = args.steady_tol
        c["steady_window"] = STEADY_WINDOW

    common = {"threads": args.threads}
    payloads = [(c, common) for c in cases]

    print(f"[bench] {len(cases)} cases, jobs={args.jobs}, threads={args.threads}, "
          f"steps<={steps}", flush=True)
    t0 = time.time()
    results = []
    if args.jobs <= 1:
        for p in payloads:
            r = _worker(p)
            print(f"[case] {r['id']}: u_max={r.get('u_max')} steps={r.get('steps')} "
                  f"converged={r.get('converged')} ({r.get('wall_s', 0):.1f}s)", flush=True)
            results.append(r)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.jobs) as pool:
            for r in pool.imap_unordered(_worker, payloads):
                print(f"[case] {r['id']}: u_max={r.get('u_max')} steps={r.get('steps')} "
                      f"converged={r.get('converged')} ({r.get('wall_s', 0):.1f}s)", flush=True)
                results.append(r)
    results.sort(key=lambda r: r["id"])

    payload = {
        "meta": {
            "gap": "GAP-PP-3-3DMCMP-SPURIOUS-SCALING",
            "protocol": "benchmarks/mcmp3d_spurious_scaling/README.md",
            "grids": grids,
            "g_levels": g_levels,
            "forcings": forcings,
            "form": FORM,
            "tau": TAU,
            "steps_cap": steps,
            "check_every": check_every,
            "steady_tol": args.steady_tol,
            "steady_window": STEADY_WINDOW,
            "radius_rule": "R = n/8",
            "width_rule": f"W_init = {WIDTH_INIT} (constant, lattice units)",
            "quick": bool(args.quick),
            "engine": "tensorlbm.pseudopotential_edm3d.static_droplet (D3Q19, float64)",
            "wall_s": time.time() - t0,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "tensorlbm": tlbm_version,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "command": " ".join([sys.executable, *sys.argv]),
            "notes": "u_max = mean of the last 3 steady checks; deterministic (no RNG)",
        },
        "coexistence": known,
        "cases": results,
    }
    path = outdir / "results.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=False))
    print(f"[bench] wrote {path} ({time.time() - t0:.1f}s wall)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())