#!/usr/bin/env python3
"""Stability map of the D3Q19 wall-modelled channel (criterion a resolved tier).

Criterion (a) asks for the velocity profile in ``y+ = 30...300``.  That band only
lies inside the log layer when ``Re_tau >= ~1000``; the profile above the first
off-wall node is *resolved* by the LES, so a high-``Re_tau`` run is the only way
to measure the criterion the way a WMLES user would.  This script maps where the
solver actually converges, so the acceptance record can say *why* a tier is
missing instead of asserting that it is.

Each case is one ``(Re_tau, Cs)`` pair run in its own process (the runner is
stateful and heavy), and the per-case JSON carries the whole time history so the
first non-finite step is visible.  ``--summarise`` merges the cases.

Usage
-----
    python benchmarks/wall_model_les/stability_map.py --re-tau 1000 --cs 0.05 --steps 2000
    python benchmarks/wall_model_les/stability_map.py --summarise
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.channel import (  # noqa: E402
    ChannelWMLESConfig,
    run_channel_wmles,
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_case(re_tau: float, cs: float, *, steps: int, out_dir: Path, nx: int, nz: int, ny: int) -> dict:
    config = ChannelWMLESConfig(
        nx=nx,
        nz=nz,
        ny=ny,
        re_tau=float(re_tau),
        smagorinsky_cs=float(cs),
        steps=int(steps),
        averaging_start=int(0.6 * steps),
        log_interval=max(steps // 20, 1),
        wall_model="equilibrium",
        wall_coupling="assimilate",
    )
    started = time.time()
    error = None
    stats = None
    try:
        stats = run_channel_wmles(config)
    except Exception as exc:  # noqa: BLE001 - a crash is a result here
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    history = (stats or {}).get("history") or []
    first_nonfinite = None
    for row in history:
        values = [v for k, v in row.items() if k != "step"]
        if any(isinstance(v, float) and not math.isfinite(v) for v in values):
            first_nonfinite = row.get("step")
            break
    converged = bool(
        stats
        and _finite(stats.get("u_tau_model_mean"))
        and _finite(stats.get("u_tau_model_over_balance"))
        and first_nonfinite is None
    )
    record = {
        "re_tau_target": float(re_tau),
        "smagorinsky_cs": float(cs),
        "steps": int(steps),
        "grid": {"nx": nx, "nz": nz, "ny": ny, "delta_y_plus": config.delta_y_plus,
                 "y_plus_exchange": config.y_of_node(config.j_exchange) * config.u_tau / config.nu},
        "elapsed_s": round(elapsed, 1),
        "error": error,
        "first_nonfinite_step": first_nonfinite,
        "converged": converged,
        "u_tau_model_over_balance": (stats or {}).get("u_tau_model_over_balance"),
        "re_tau_measured_last": (
            history[-1].get("Re_tau") if history and isinstance(history[-1], dict) else None
        ),
        "n_history": len(history),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"re{int(re_tau)}_cs{cs:g}"
    (out_dir / f"stability_{tag}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def summarise(out_dir: Path) -> dict:
    cases = []
    for path in sorted(out_dir.glob("stability_re*.json")):
        cases.append(json.loads(path.read_text()))
    cases.sort(key=lambda c: (c["re_tau_target"], c["smagorinsky_cs"]))
    limit = None
    for case in cases:
        if not case["converged"]:
            limit = case["re_tau_target"]
            break
    record = {
        "purpose": (
            "map the Re_tau range in which the D3Q19 wall-modelled channel of this "
            "repository produces a finite, statistically usable profile; the "
            "y+ = 30...300 criterion band needs Re_tau >= ~1000"
        ),
        "grid_rule": "nx=nz=32, ny chosen so that dy+ ~ 40...60 (WMLES resolution)",
        "cases": cases,
        "highest_converged_re_tau": max(
            [c["re_tau_target"] for c in cases if c["converged"]], default=None
        ),
        "lowest_diverged_re_tau": limit,
        "band_reachable": bool(
            max([c["re_tau_target"] for c in cases if c["converged"]], default=0.0) >= 1000.0
        ),
        "criterion_band_note": (
            "y+ = 300 is 0.3h+ at Re_tau = 1000 and 0.15h+ at Re_tau = 2000; below "
            "Re_tau ~ 1000 the top of the criterion band sits in the wake"
        ),
    }
    (out_dir / "stability_map.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--re-tau", type=float)
    parser.add_argument("--cs", type=float)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--ny", type=int, default=33)
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--out", default="artifacts/wall_model_les")
    args = parser.parse_args()
    out_dir = Path(args.out)
    if args.summarise:
        print(json.dumps(summarise(out_dir), indent=2, default=str))
        return
    if args.re_tau is None or args.cs is None:
        parser.error("--re-tau and --cs are required unless --summarise is given")
    record = run_case(
        args.re_tau,
        args.cs,
        steps=args.steps,
        out_dir=out_dir,
        nx=args.nx,
        nz=args.nz,
        ny=args.ny,
    )
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()