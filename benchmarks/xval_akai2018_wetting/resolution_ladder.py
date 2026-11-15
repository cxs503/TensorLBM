#!/usr/bin/env python3
"""XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING, round R5: resolution ladder.

Round R4 left the theta = 30 deg arm at +32 % and explained it by the *cap
geometry being thinner than the interface* -- proposing, but never testing, the
criterion ``R (1 - cos theta) / xi >= 2.5``.  Its two cluster jobs (cj3864 with
R = 20 / 128x96 / 40000 steps, cj3865 with R = 40 / 224x160 / 30000 steps) were
never scheduled, so the criterion stayed a hypothesis (open item in
``docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING-R4-OPP-FIX.md`` section 6).

The R5 analytic study (``estimator_analytic_study.py``) removed the estimator
half of the explanation: on *exact* diffused caps the circle-fit estimator reads
the prescribed angle to within 1.7 deg down to ``h/xi = 0.68``, i.e. at the
R = 14 / theta = 30 deg geometry (``h/xi = 0.94``) the estimator is not the
problem.  What is left is the physics: does the settled shape become a 30 deg
cap when the cap grows, or does it stay at the ~39 deg that the estimator
reports?

This driver answers that by holding ``xi``, ``A``, ``beta``, ``tau`` fixed and
varying only ``R`` (14 -> 20 -> 40) and ``theta`` (30/60/90), each arm run to
``dphi`` convergence.  It reuses the production arm driver
(``run.py:run_arm``) unchanged -- no second solver, no second post-processor.

The grid is grown with the cap (``nx ~ 2R sin theta + 24``, ``ny ~ R(1-cos
theta) + 24``) so the vapour margin is constant and the wall confinement does
not change with R, and the step budget grows as ``R^2`` (viscous time at
``nu = (tau - 1/2)/3 = 1/6``), with an early exit at ``dphi < 1e-9``.

Usage
-----
    python resolution_ladder.py [--shard I --nshards N] [--out DIR] [--steps-cap N]
    python resolution_ladder.py --merge [--out DIR]      # combine shard JSONs

Every case writes ``arm_R{R}_t{T}.json`` plus the settled field
``_last_phi_R{R}_t{T}.npy`` into ``--out``; ``--merge`` writes ``ladder.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run as akai  # noqa: E402  (production arm driver, reused verbatim)

# radius x contact angle; xi = 2 throughout so h/xi spans 0.94 -> 2.68
CASES: list[tuple[float, float]] = [
    (14.0, 30.0),
    (14.0, 60.0),
    (14.0, 90.0),
    (20.0, 30.0),
    (20.0, 60.0),
    (20.0, 90.0),
    (40.0, 30.0),
    (40.0, 60.0),
    (40.0, 90.0),
]


def arm_name(radius: float, theta: float) -> str:
    return f"R{radius:.0f}_t{theta:.0f}"


def build_spec(radius: float, theta: float, steps_cap: int) -> akai.ArmSpec:
    th = math.radians(theta)
    height = radius * (1.0 - math.cos(th))
    width = 2.0 * radius * math.sin(th)
    nx = int(max(96, math.ceil(width + 24.0)))
    ny = int(max(48, math.ceil(height + 24.0)))
    # viscous time ~ R^2 / nu with nu = (tau - 1/2)/3 = 1/6
    steps = int(min(steps_cap, max(6000, 6.0 * radius * radius / (1.0 / 6.0))))
    return akai.ArmSpec(
        name=arm_name(radius, theta),
        theta_target=theta,
        radius=radius,
        nx=nx,
        ny=ny,
        xi=2.0,
        steps=steps,
    )


def merge(out: Path) -> int:
    cases = []
    for p in sorted(out.glob("arm_*.json")):
        cases.append(json.loads(p.read_text()))
    cases.sort(key=lambda c: (c["spec"]["radius"], c["theta_target"]))
    rows = []
    for c in cases:
        R = c["spec"]["radius"]
        th = c["theta_target"]
        h = R * (1.0 - math.cos(math.radians(th)))
        rows.append(
            {
                "arm": c["arm"],
                "radius": R,
                "theta_target": th,
                "xi": c["spec"]["xi"],
                "h_over_xi": h / c["spec"]["xi"],
                "nx": c["spec"]["nx"],
                "ny": c["spec"]["ny"],
                "steps_run": c["steps_run"],
                "theta_circle": c["theta_circle"],
                "err_pct_circle": c["err_pct_circle"],
                "theta_cap": c.get("theta_cap"),
                "theta_tail_std": c["theta_tail_std"],
                "converged_dphi": c["converged_dphi"],
                "u_max_fluid": c["u_max_fluid"],
                "mass_drift_r": c["mass_drift_r"],
                "interface_width_10_90": c["interface_width_10_90"],
                "pass_15pct": c["pass_15pct"],
                "phi_field_sha256": c["phi_field_sha256"],
                "wall_time_s": c["wall_time_s"],
            }
        )
    # per-angle trend: does the error shrink with R?
    trend = {}
    for th in sorted({r["theta_target"] for r in rows}):
        sub = [r for r in rows if r["theta_target"] == th]
        trend[f"{th:.0f}"] = {
            "h_over_xi": [r["h_over_xi"] for r in sub],
            "err_pct": [r["err_pct_circle"] for r in sub],
            "pass15": [r["pass_15pct"] for r in sub],
        }
    (out / "ladder.json").write_text(json.dumps({"cases": rows, "trend": trend}, indent=1))
    print(f"{'arm':8s} {'h/xi':>6s} {'steps':>7s} {'theta':>8s} {'err%':>8s}  pass")
    for r in rows:
        print(
            f"{r['arm']:8s} {r['h_over_xi']:6.2f} {r['steps_run']:7d} "
            f"{r['theta_circle']:8.2f} {r['err_pct_circle']:+8.2f}  {r['pass_15pct']}"
        )
    print(f"wrote {out / 'ladder.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[1] / "artifacts" / "m22257_akai_r5" / "ladder"))
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--steps-cap", type=int, default=60000)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(out)

    mine = [c for i, c in enumerate(CASES) if i % args.nshards == (args.shard - 1) % args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} case(s) {mine}", flush=True)
    # run_arm writes the settled field to module-level HERE; redirect it to --out
    akai.HERE = out
    for radius, theta in mine:
        spec = build_spec(radius, theta, args.steps_cap)
        target = out / f"arm_{spec.name}.json"
        if target.is_file():
            print(f"  [skip] {spec.name} already done", flush=True)
            continue
        print(
            f"== {spec.name}: R={radius} theta={theta} grid={spec.nx}x{spec.ny} "
            f"steps={spec.steps} h/xi={radius * (1 - math.cos(math.radians(theta))) / spec.xi:.2f}",
            flush=True,
        )
        res = akai.run_arm(spec, sample_every=1000)
        target.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())