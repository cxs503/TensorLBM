#!/usr/bin/env python3
"""2-D dam-break verification face: Ritter analytic anchor, mass audit, order.

Two routes are measured against the *same* metrics and gates:

``swe``
    The 2-D shallow-water finite-volume solver
    (:mod:`tensorlbm.shallow_water2d`): planar dam break, MUSCL-Hancock +
    hydrostatic reconstruction, exactly mass conservative.  This is the route
    whose model assumptions match Ritter's (hydrostatic, inviscid), so it is the
    analytic-anchor route.

``fslbm``
    The production Körner free-surface LBM (:mod:`tensorlbm.free_surface_lbm`,
    D3Q19) in a thin free-slip slab, i.e. the lattice-Boltzmann free-surface
    (Navier-Stokes) route requested by the node.  It solves the full
    incompressible NS equations, so the shallow-water anchor is only an
    approximation for it -- the measured deviation is reported, not tuned.

Usage::

    PYTHONPATH=src python benchmarks/dam_break/2d/run.py --out result.json
    PYTHONPATH=src python benchmarks/dam_break/2d/run.py --fast   # smoke
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # benchmarks/dam_break
sys.path.insert(0, str(_HERE.parents[2] / "src"))  # src

import common  # noqa: E402

CHECKPOINTS = (0.4, 0.6, 0.8)
"""``t* = t sqrt(g/h0)`` checkpoints (``c0 t* = 0.4..0.8 h0 < a``: anchor valid)."""

T_STAR_END = 0.8

REYNOLDS_TARGET = 150.0
"""Fixed dam-break Reynolds number ``Re = c0 h0 / nu`` of the FS-LBM study.

Fixing ``tau`` instead would let ``Re`` grow with the grid (``Re ~ h0``) and mix
discretisation error with a viscosity change, so the slab study is run at
constant ``Re = 150`` (``tau = 0.526 / 0.538 / 0.551`` on ``h0 = 16 / 24 / 32``).

``Re`` is the binding constraint of this route: ``Re = 300`` (``tau >= 0.51``)
was the first attempt and is already unstable in the three-dimensional box
(``TopologyTransactionError``), and ``Re = 150`` is two orders of magnitude
below the inviscid shallow-water limit that the Ritter anchor assumes -- one of
the documented reasons this route cannot certify the anchor
(``docs/verification/DAM_BREAK_LBM.md`` §5).
"""
"""End time in ``t*``.  ``c0 t = 0.8 h0 < a = h0`` keeps the analytic anchor valid."""


# ---------------------------------------------------------------------------
# Route A: 2-D shallow water (analytic-anchor route)
# ---------------------------------------------------------------------------
def run_swe(nx_list: list[int], t_star_end: float = T_STAR_END) -> dict:
    from tensorlbm.shallow_water2d import (
        dam_break_initial_state,
        depth_profile_x,
        mass_drift,
        simulate,
    )

    h0, a, g = 1.0, 1.0, 1.0
    domain_x = 4.0
    runs: list[dict] = []
    profiles: dict[str, np.ndarray] = {}
    x_by_grid: dict[int, np.ndarray] = {}
    for nx in nx_list:
        dx = domain_x / nx
        state = dam_break_initial_state(8, nx, depth=h0, column_width=a, dx=dx, dy=0.2, g=g)
        start = time.time()
        state = simulate(state, t_end=t_star_end, cfl=0.4, order=2)
        elapsed = time.time() - start
        x = (np.arange(nx) + 0.5) * dx
        h = depth_profile_x(state)
        profiles[str(nx)] = h
        x_by_grid[nx] = x
        metrics = common.profile_metrics(x, h, t_star_end, h0, g, a)
        front = common.front_metrics(x, h, t_star_end, h0, g, a)
        runs.append(
            {
                "grid": {"nx": nx, "ny": 8, "dx": dx},
                "n_steps": len(state.history["t"]) - 1,
                "order_scheme": 2,
                "cfl": 0.4,
                "runtime_s": round(elapsed, 2),
                "profile": metrics,
                "front": front,
                "mass_drift": common.mass_drift_series(
                    [
                        (m - state.history["mass"][0]) / state.history["mass"][0]
                        for m in state.history["mass"]
                    ]
                ),
                "solver_mass_drift": mass_drift(state),
            }
        )
    errors = [run["profile"]["rel_l1"] for run in runs]
    ratios = [nx_list[i] / nx_list[i + 1] for i in range(len(nx_list) - 1)]
    convergence = common.convergence_order(errors, [1.0 / r for r in ratios])
    convergence["spacings"] = [run["grid"]["dx"] for run in runs]
    convergence["grids"] = [run["grid"]["nx"] for run in runs]
    return {
        "route": "swe2d",
        "solver": "tensorlbm.shallow_water2d (MUSCL-Hancock + hydrostatic reconstruction)",
        "runs": runs,
        "convergence": convergence,
        "_profiles": profiles,
        "_x_by_grid": x_by_grid,
    }


# ---------------------------------------------------------------------------
# Route B: Körner free-surface LBM in a free-slip slab (planar realisation)
# ---------------------------------------------------------------------------
def run_fslbm(h0_list: list[int], t_star_end: float = T_STAR_END) -> dict:
    from fslbm_driver import FSConfig
    from fslbm_driver import run as run_fs

    runs: list[dict] = []
    profiles: dict[str, np.ndarray] = {}
    x_by_grid: dict[int, np.ndarray] = {}
    for h0 in h0_list:
        config = FSConfig(
            h0=h0,
            span_cells=4,
            span_free_slip=True,
            domain_x=4.0,
            domain_y=2.0,
            reynolds_target=REYNOLDS_TARGET,  # constant-Re refinement study
        )
        checkpoint_times = tuple(sorted(set(CHECKPOINTS) | {t_star_end}))
        result = run_fs(config, checkpoints=checkpoint_times)
        t_scale = h0 / config.c0  # lattice steps per unit t*
        x = np.asarray(result["x"])
        key = f"{t_star_end:.3f}"
        depth = np.asarray(result["profiles"][key]["depth_x"])
        profiles[str(h0)] = depth
        x_by_grid[h0] = x
        t_lattice = t_star_end * t_scale
        metrics = common.profile_metrics(x, depth, t_lattice, h0, config.g, h0)
        front = common.front_metrics(x, depth, t_lattice, h0, config.g, h0)
        depth_xz = np.asarray(result["snapshots"].get("depth_xz", np.zeros((1, x.size))))
        span_spread = (
            float(
                np.max(np.abs(depth_xz - depth_xz.mean(axis=0))) / max(float(depth_xz.max()), 1e-30)
            )
            if depth_xz.shape[0] > 1
            else 0.0
        )
        checkpoints = {
            t_star: common.profile_metrics(
                x,
                np.asarray(result["profiles"][f"{t_star:.3f}"]["depth_x"]),
                t_star * t_scale,
                h0,
                config.g,
                h0,
            )
            for t_star in checkpoint_times
            if f"{t_star:.3f}" in result["profiles"]
        }
        runs.append(
            {
                "grid": result["grid"],
                "h0": h0,
                "n_steps": result["n_steps"],
                "tau": config.tau_effective,
                "nu": config.nu,
                "reynolds": config.reynolds,
                "c0": config.c0,
                "g": config.g,
                "t_star_end": result["t_star_end"],
                "profile": metrics,
                "profile_checkpoints": checkpoints,
                "front": front,
                "mass_drift": {
                    "max_abs": result["mass_drift_max_abs"],
                    "n_samples": len(result["mass_drift_series"]),
                },
                "inventory_drift_max_abs": result["inventory_drift_max_abs"],
                "spanwise_spread": span_spread,
                "nan_step": result["nan_step"],
            }
        )
    errors = [run["profile"]["rel_l1"] for run in runs]
    spacings = [1.0 / run["h0"] for run in runs]
    convergence = common.convergence_order(
        errors, [spacings[i] / spacings[i + 1] for i in range(len(spacings) - 1)]
    )
    convergence["spacings"] = spacings
    convergence["grids"] = [run["h0"] for run in runs]
    return {
        "route": "fslbm_planar_slab",
        "solver": "tensorlbm.free_surface_lbm (Körner D3Q19, free-slip slab)",
        "runs": runs,
        "convergence": convergence,
        "_profiles": profiles,
        "_x_by_grid": x_by_grid,
    }


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def evaluate(route: dict, *, finest_key: str) -> dict:
    """Evaluate the four predeclared gates on the finest grid of a route."""
    runs = route["runs"]
    finest = runs[-1]
    grid = finest.get("h0", finest["grid"].get("nx"))
    return {
        "grid_label": grid,
        "P1_profile": {
            "value": finest["profile"]["rel_l1"],
            "limit": common.PROFILE_REL_ERROR_MAX,
            "pass": bool(finest["profile"]["rel_l1"] <= common.PROFILE_REL_ERROR_MAX),
            "metric": "relative L1 depth-profile deviation from Ritter on the wet region",
        },
        "P2_front": {
            "value": finest["front"]["rel_error"],
            "limit": common.FRONT_REL_ERROR_MAX,
            "pass": bool(finest["front"]["rel_error"] <= common.FRONT_REL_ERROR_MAX),
            "metric": "front position error relative to the Ritter front advance",
        },
        "P3_mass": {
            "value": finest["mass_drift"]["max_abs"],
            "limit": common.MASS_DRIFT_MAX,
            "pass": bool(finest["mass_drift"]["max_abs"] <= common.MASS_DRIFT_MAX),
            "metric": "max relative mass drift over the whole run",
        },
        "P4_order": {
            "value": route["convergence"]["order_mean"],
            "limit": common.CONVERGENCE_ORDER_MIN,
            "pass": bool(
                route["convergence"]["order_mean"] is not None
                and not math.isnan(route["convergence"]["order_mean"])
                and route["convergence"]["order_mean"] >= common.CONVERGENCE_ORDER_MIN
            ),
            "metric": "observed order p on the P1 metric over three grids",
        },
        "all_pass": bool(
            finest["profile"]["rel_l1"] <= common.PROFILE_REL_ERROR_MAX
            and finest["front"]["rel_error"] <= common.FRONT_REL_ERROR_MAX
            and finest["mass_drift"]["max_abs"] <= common.MASS_DRIFT_MAX
            and route["convergence"]["order_mean"] >= common.CONVERGENCE_ORDER_MIN
        ),
        "finest": finest_key,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_HERE / "result.json"))
    parser.add_argument("--fig-dir", default=str(_HERE / "figures"))
    parser.add_argument("--fast", action="store_true", help="tiny grids for a smoke run")
    parser.add_argument("--routes", default="swe,fslbm")
    args = parser.parse_args(argv)

    if args.fast:
        swe_grids, fslbm_grids = [100, 200], [8, 12]
    else:
        swe_grids, fslbm_grids = [200, 400, 800], [16, 24, 32]

    payload: dict = {
        "benchmark": "dam_break_2d",
        "title": "2-D dam break: Ritter analytic anchor, mass audit, grid order",
        "nondimensionalisation": (
            "lattice units; h0 = water depth, a = h0 (square column), c0 = sqrt(g h0) = 0.08 "
            "for the LBM route (g = c0^2/h0), t* = t sqrt(g/h0)"
        ),
        "checkpoints_t_star": list(CHECKPOINTS),
        "t_star_end": T_STAR_END,
        "gates": common.gates(),
        "routes": {},
        "verdict": {},
    }

    if "swe" in args.routes:
        payload["routes"]["swe2d"] = run_swe(swe_grids)
        payload["verdict"]["swe2d"] = evaluate(
            payload["routes"]["swe2d"], finest_key=str(swe_grids[-1])
        )
    if "fslbm" in args.routes:
        payload["routes"]["fslbm_planar_slab"] = run_fslbm(fslbm_grids)
        payload["verdict"]["fslbm_planar_slab"] = evaluate(
            payload["routes"]["fslbm_planar_slab"], finest_key=str(fslbm_grids[-1])
        )

    # --- figures + numeric payload separation ---------------------------------
    fig_dir = Path(args.fig_dir)
    figures: list[str] = []
    swe = payload["routes"].get("swe2d")
    if swe is not None:
        finest_nx = swe_grids[-1]
        x = swe["_x_by_grid"][finest_nx]
        t_lattice = T_STAR_END
        analytic = common.ritter_depth(x, t_lattice, 1.0, 1.0, 1.0)
        profiles = {
            f"SWE nx={nx}": (swe["_x_by_grid"][nx], swe["_profiles"][str(nx)]) for nx in swe_grids
        }
        figures.append(
            str(
                common.plot_profiles(
                    fig_dir / "2d_profiles_swe.png",
                    x,
                    analytic,
                    profiles,
                    title=f"2-D dam break (SWE), t*={T_STAR_END}: depth profile vs Ritter",
                    xlabel="x / h0",
                    ylabel="h / h0",
                )
            )
        )
        spacings = swe["convergence"]["spacings"]
        figures.append(
            str(
                common.plot_convergence(
                    fig_dir / "2d_convergence_swe.png",
                    {"SWE (MUSCL-Hancock)": (spacings, swe["convergence"]["errors"])},
                    title="2-D dam break: grid convergence of the Ritter profile error",
                )
            )
        )
    fslbm = payload["routes"].get("fslbm_planar_slab")
    if fslbm is not None:
        finest_h0 = fslbm_grids[-1]
        x = fslbm["_x_by_grid"][finest_h0]
        config_g = (0.08**2) / finest_h0
        analytic = common.ritter_depth(
            x, T_STAR_END * finest_h0 / 0.08, finest_h0, config_g, finest_h0
        )
        profiles = {
            f"FS-LBM h0={h0}": (fslbm["_x_by_grid"][h0], fslbm["_profiles"][str(h0)])
            for h0 in fslbm_grids
        }
        figures.append(
            str(
                common.plot_profiles(
                    fig_dir / "2d_profiles_fslbm.png",
                    x,
                    analytic,
                    profiles,
                    title=f"2-D dam break (FS-LBM slab), t*={T_STAR_END}: depth vs Ritter",
                )
            )
        )
        figures.append(
            str(
                common.plot_convergence(
                    fig_dir / "2d_convergence_fslbm.png",
                    {
                        "FS-LBM (Körner, free-slip slab)": (
                            fslbm["convergence"]["spacings"],
                            fslbm["convergence"]["errors"],
                        )
                    },
                    title="2-D dam break (FS-LBM): grid convergence of the profile error",
                )
            )
        )

    for route in list(payload["routes"].values()):
        route.pop("_profiles", None)
        route.pop("_x_by_grid", None)
    payload["figures"] = figures

    out = common.write_json(args.out, payload)
    print(f"wrote {out}")
    for name, verdict in payload["verdict"].items():
        print(f"  {name}: all_pass={verdict['all_pass']}")
        for key in ("P1_profile", "P2_front", "P3_mass", "P4_order"):
            item = verdict[key]
            print(
                f"    {key:<11} value={item['value']!r} limit={item['limit']!r} pass={item['pass']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
