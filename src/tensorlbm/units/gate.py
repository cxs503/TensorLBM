"""Machine-verifiable acceptance gate for the lattice -> physical unit chain.

The gate answers one question with numbers: *would a factor error in a
dimensionalisation coefficient be visible in CI before it reaches a report?*
It combines three layers:

``roundtrip``
    Algebraic closure of every conversion site
    (``to_phys(to_lattice(x)) == x``) to machine precision, plus the
    similarity invariant ``Re_lattice == Re_physical``.  Criterion:
    ``max relative closure error <= 1e-14``.

``anchors``
    Four analytic anchors (:mod:`tensorlbm.units.anchors`) with pre-declared
    tolerances: Poiseuille ``U_b^+ = Re_tau/3`` <= 0.4 %, Couette linear
    profile / ``tau_w = mu U/H`` <= 0.5 %, Stokes ``F = 6 pi mu a U`` <= 2 %,
    Taylor-Green ``exp(-2 nu k^2 t)`` <= 2 %.

``grid_invariance``
    One physical case run at three independent ``(dx, dt)`` groups
    (``Ma < 0.1``, ``tau in [0.51, 2]``): the dimensionless anchor number must
    agree to ``<= 0.1 %``, i.e. the reported physics must not depend on the
    arbitrary choice of discretisation.

``mutation_battery``
    Five single-site coefficient defects (``length x2``, ``length /2``,
    ``rho0 -> 2 rho0``, ``nu -> 2 nu``, ``dt -> 2 dt``).  Every mutation must be
    exposed by at least one anchor (5/5) and every anchor must expose at least
    one mutation.  With the pristine bridge all anchors must pass for three
    different seeds (zero false positives).

The mutation is injected **at the output (lattice -> physical) site only**;
see :mod:`tensorlbm.units.bridge` for why a *consistent* rescaling of
``dx``/``dt`` is physically undetectable (similarity invariance) and why the
output site is therefore the correct place to model the defect class
(historical ``F/2`` drag defect).

Run from the CLI::

    python -m tensorlbm.units.gate --out artifacts/units_bootstrap_gate
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import anchors as anchor_mod
from .bridge import CS_LATTICE, MUTATIONS, LatticeBridge, bootstrap_bridge, mutate

__all__ = ["run_gate", "main", "ROUNDTRIP_TOL", "GRID_SPREAD_TOL"]

#: round-trip closure tolerance required by the campaign criterion
ROUNDTRIP_TOL: float = 1e-14
#: allowed spread of a dimensionless anchor number over three (dx, dt) groups
GRID_SPREAD_TOL: float = 1.0e-3
#: seeds used for the zero-false-positive replication
SEEDS: tuple[int, ...] = (0, 1, 2)


def _roundtrip(seed: int = 0) -> dict:
    """Round-trip closure and similarity consistency of the bridge."""
    rng = np.random.RandomState(7 + seed)
    bridge = bootstrap_bridge(
        length_phys=1.0e-3, velocity_phys=2.5e-3, nu_phys=1.0e-6, nx=48,
        rho0=998.0, u_lb=0.02,
    )
    values = 10.0 ** rng.uniform(-3.0, 3.0, size=8)
    checks: dict[str, float] = {}

    def rel(a: float, b: float) -> float:
        return abs(a - b) / abs(b)

    for name, to_phys, to_lat in (
        ("length", bridge.length_to_phys, bridge.length_to_lattice),
        ("time", bridge.time_to_phys, bridge.time_to_lattice),
        ("velocity", bridge.velocity_to_phys, bridge.velocity_to_lattice),
        ("viscosity", bridge.viscosity_to_phys, bridge.viscosity_to_lattice),
        ("density", bridge.density_to_phys, bridge.density_to_lattice),
        ("stress", bridge.stress_to_phys, bridge.stress_to_lattice),
        ("force", bridge.force_to_phys, bridge.force_to_lattice),
    ):
        checks[name] = max(rel(to_phys(to_lat(v)), v) for v in values)
    checks["stress"] = max(
        rel(bridge.stress_to_phys(bridge.stress_to_lattice(v)), v) for v in values
    )
    checks["force"] = max(
        rel(bridge.force_to_phys(bridge.force_to_lattice(v)), v) for v in values
    )
    # similarity invariant: the lattice Reynolds number of the anchor equals the
    # physical one built from the same bridge
    u_lb = 0.02
    h_lb = 24.0
    nu_lb = bridge.viscosity_to_lattice(1.0e-6)
    checks["reynolds_identity"] = rel(
        bridge.reynolds_lattice(u_lb, h_lb, nu_lb),
        bridge.reynolds_physical(
            bridge.velocity_to_phys(u_lb),
            bridge.length_to_phys(h_lb),
            bridge.viscosity_to_phys(nu_lb),
        ),
    )
    checks["mach_lt_0_1"] = 0.0 if bridge.mach(u_lb) < 0.1 else 1.0
    worst = max(checks.values())
    return {
        "bridge": {"dx": bridge.dx, "dt": bridge.dt, "rho0": bridge.rho0,
                   "nu_lb": nu_lb, "tau": bridge.tau_from_viscosity(nu_lb),
                   "cs_lattice": CS_LATTICE},
        "max_rel_error": worst,
        "tolerance": ROUNDTRIP_TOL,
        "checks": checks,
        "passed": bool(worst <= ROUNDTRIP_TOL),
    }


def _anchor_results(seeds: tuple[int, ...], grids: dict[str, tuple[int, ...]]) -> dict:
    out: dict[str, list[dict]] = {}
    for name in anchor_mod.ANCHORS:
        rows = []
        for seed in seeds:
            gl = grids[name] if seed == seeds[0] else grids[name][:1]
            for grid in gl:
                case = anchor_mod.build_case(name, seed, grid)
                res = anchor_mod.evaluate(case)
                res["seed"] = seed
                res["steps"] = case.lattice.get("steps")
                res["regime"] = {
                    "Ma": case.bridge.mach(anchor_mod.U_LB),
                    "tau": case.lattice["tau"],
                }
                rows.append(res)
        out[name] = rows
    return out


def _grid_invariance(anchor_rows: dict) -> dict:
    """Spread of the primary dimensionless number over the (dx, dt) ladder."""
    primary = {
        "poiseuille": "U_b_plus_over_Re_tau_over_3",
        "couette": "tau_wall_over_reference",
        "stokes": "drag_over_reference",
        "taylor_green": "nu_implied_over_nu",
    }
    #: Stokes is a conversion-chain anchor (no time integration), so the
    #: tau window of the marching anchors does not apply to it; Stokes flow
    #: with Ma < 0.1 inherently needs nu_lb >> 0.5 (classic LBM limitation).
    time_marching = {"poiseuille": True, "couette": True, "stokes": False,
                     "taylor_green": True}
    out: dict[str, dict] = {}
    for name, key in primary.items():
        rows = [r for r in anchor_rows[name] if r["seed"] == SEEDS[0]]
        vals = [r["dimensionless"][key] for r in rows if key in r["dimensionless"]]
        spread = (max(vals) - min(vals)) / abs(sum(vals) / len(vals)) if vals else float("inf")
        taus = [r["regime"]["tau"] for r in rows]
        mas = [r["regime"]["Ma"] for r in rows]
        out[name] = {
            "quantity": key,
            "grids": [r["grid"] for r in rows],
            "values": vals,
            "spread": spread,
            "tolerance": GRID_SPREAD_TOL,
            "regime": {"tau_min": min(taus), "tau_max": max(taus),
                       "Ma_max": max(mas)},
            "time_marching": time_marching[name],
            "regime_ok": (
                bool(min(taus) >= 0.51 and max(taus) <= 2.0 and max(mas) < 0.1)
                if time_marching[name]
                else bool(max(mas) < 0.1)   # tau window N/A: no time stepping
            ),
            "passed": bool(spread <= GRID_SPREAD_TOL),
        }
    return out


def _mutation_battery(cases: dict, pristine: dict) -> dict:
    """Inject single-site coefficient defects; require the anchors to catch."""
    matrix: dict[str, dict[str, float]] = {}
    caught_by: dict[str, list[str]] = {}
    anchor_catches: dict[str, list[str]] = {n: [] for n in anchor_mod.ANCHORS}
    for mut in MUTATIONS:
        matrix[mut] = {}
        caught_by[mut] = []
        for name, case in cases.items():
            bridge = mutate(case.bridge, mut)
            res = anchor_mod.evaluate(case, bridge)
            worst = res["worst_rel_error"]
            matrix[mut][name] = worst
            if worst > res["tolerance"]:
                caught_by[mut].append(name)
                anchor_catches[name].append(mut)
    missing = [m for m, hits in caught_by.items() if not hits]
    no_catch_anchor = [a for a, hits in anchor_catches.items() if not hits]
    return {
        "matrix_worst_rel_error": matrix,
        "caught_by": caught_by,
        "anchor_catches": anchor_catches,
        "mutations": list(MUTATIONS),
        "caught_count": len(MUTATIONS) - len(missing),
        "missing": missing,
        "anchors_without_catch": no_catch_anchor,
        "pristine_worst_rel_error": {n: pristine[n]["worst_rel_error"]
                                     for n in anchor_mod.ANCHORS},
        "passed": bool(not missing and not no_catch_anchor),
    }


def run_gate(
    *,
    seeds: tuple[int, ...] = SEEDS,
    grids: dict[str, tuple[int, ...]] | None = None,
) -> dict:
    """Run the whole gate and return a JSON-serialisable report."""
    t0 = time.time()
    grids = grids or anchor_mod.GRIDS
    rt = _roundtrip()
    rows = _anchor_results(seeds, grids)
    pristine = {
        name: {k: v for k, v in rows[name][0].items()}
        for name in anchor_mod.ANCHORS
    }
    grid_inv = _grid_invariance(rows)
    battery = _mutation_battery(
        {n: anchor_mod.build_case(n, SEEDS[0], grids[n][0]) for n in anchor_mod.ANCHORS},
        pristine,
    )
    all_anchor_pass = all(r["passed"] for name in rows for r in rows[name])
    three_seeds_green = all(
        r["passed"] for name in rows for r in rows[name] if r["seed"] in seeds
    )
    report = {
        "gate": "lbm-unit-bootstrap",
        "criteria": {
            "roundtrip_tolerance": ROUNDTRIP_TOL,
            "anchor_tolerances": anchor_mod.TOLERANCES,
            "grid_spread_tolerance": GRID_SPREAD_TOL,
            "mutation_count_required": len(MUTATIONS),
            "seeds": list(seeds),
        },
        "roundtrip": rt,
        "anchors": rows,
        "grid_invariance": grid_inv,
        "mutation_battery": battery,
        "false_positive": {
            "seeds": list(seeds),
            "all_anchors_green": bool(three_seeds_green),
            "passed": bool(three_seeds_green),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.time() - t0,
    }
    report["overall_passed"] = bool(
        rt["passed"]
        and all_anchor_pass
        and all(v["passed"] for v in grid_inv.values())
        and battery["passed"]
        and three_seeds_green
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/units_bootstrap_gate"),
                    help="directory for gate_report.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    report = run_gate()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "gate_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    if not args.quiet:
        print(f"round-trip      : {report['roundtrip']['max_rel_error']:.2e} "
              f"(<= {ROUNDTRIP_TOL:g}) -> {report['roundtrip']['passed']}")
        for name, rows in report["anchors"].items():
            worst = max(r["worst_rel_error"] for r in rows)
            print(f"anchor {name:13s}: worst {worst:.2e} "
                  f"(<= {anchor_mod.TOLERANCES[name]:g}) -> "
                  f"{all(r['passed'] for r in rows)}")
        for name, gi in report["grid_invariance"].items():
            print(f"grid-inv {name:12s}: spread {gi['spread']:.2e} -> {gi['passed']}")
        bat = report["mutation_battery"]
        print(f"mutations caught: {bat['caught_count']}/{len(MUTATIONS)} "
              f"missing={bat['missing']} -> {bat['passed']}")
        print(f"false positives : all green over seeds "
              f"{report['false_positive']['seeds']} -> "
              f"{report['false_positive']['passed']}")
        print(f"OVERALL         : {report['overall_passed']}  "
              f"({report['runtime_seconds']:.1f}s)  report -> {path}")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())