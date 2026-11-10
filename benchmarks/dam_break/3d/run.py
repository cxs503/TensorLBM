#!/usr/bin/env python3
"""3-D dam-break verification face: square column collapse, mass audit, order.

Geometry: a square column ``a x a x h0`` (``a = h0``) of still water collapsing
under gravity in a ``4 h0 (streamwise) x 2 h0 (vertical) x h0 (span)`` box with
no-slip walls, integrated with the production Körner free-surface LBM
(:mod:`tensorlbm.free_surface_lbm`, D3Q19) -- the repository's free-surface
model.

Reported quantities (identical metric code as the 2-D face, see
``benchmarks/dam_break/common.py``):

* the span-averaged streamwise depth profile against the Ritter (1892) profile
  (a **reference**, not a certified anchor: the mid-plane of a collapsing
  three-dimensional column is not the planar shallow-water solution);
* the front position;
* the mass drift (tracked Körner mass ledger) and the liquid inventory drift;
* the observed grid-convergence order of the profile metric over three grids;
* the water-surface map ``h_s(x, z)`` at the checkpoints (3-D evolution).

Usage::

    PYTHONPATH=src python benchmarks/dam_break/3d/run.py --out result.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # benchmarks/dam_break
sys.path.insert(0, str(_HERE.parents[2] / "src"))  # src

import common  # noqa: E402

CHECKPOINTS = (0.4, 0.8)
REYNOLDS_TARGET: float | None = None
"""Reynolds target of the 3-D study; ``None`` = fixed ``tau = 0.52``.

The delivered 3-D artifact is a *fixed-tau* study (``Re = 144 / 192 / 240`` on
``h0 = 12 / 16 / 20``).  A constant-Re study of the three-dimensional box is not
deliverable with the BGK free-surface model: at ``Re = 150`` the finest grid
needs ``tau = 0.56`` and the run aborts with
``TopologyTransactionError: topology candidate fill is outside [-0.1, 10]``.
The consequence (mixing discretisation with viscosity, hence no formal
convergence rate) is recorded in ``docs/verification/DAM_BREAK_LBM.md`` §5.
"""
T_STAR_END = 0.8
C0 = 0.08


def run_case(h0: int) -> dict:
    """One 3-D free-surface dam-break run plus all metrics."""
    from fslbm_driver import FSConfig
    from fslbm_driver import run as run_fs

    config = FSConfig(
        h0=h0,
        span_cells=h0,
        span_free_slip=False,
        span_ratio=1.0,
        domain_x=4.0,
        domain_y=2.0,
        reynolds_target=REYNOLDS_TARGET,
    )
    result = run_fs(config, checkpoints=CHECKPOINTS)
    x = np.asarray(result["x"])
    t_scale = h0 / config.c0
    key = f"{T_STAR_END:.3f}"
    depth = np.asarray(result["profiles"][key]["depth_x"])
    t_lattice = T_STAR_END * t_scale
    case = {
        "grid": result["grid"],
        "h0": h0,
        "n_steps": result["n_steps"],
        "tau": config.tau_effective,
        "nu": config.nu,
        "reynolds": config.reynolds,
        "c0": config.c0,
        "g": config.g,
        "profile": common.profile_metrics(x, depth, t_lattice, h0, config.g, h0),
        "profile_checkpoints": {
            cp: common.profile_metrics(
                x,
                np.asarray(result["profiles"][f"{cp:.3f}"]["depth_x"]),
                cp * t_scale,
                h0,
                config.g,
                h0,
            )
            for cp in CHECKPOINTS
            if f"{cp:.3f}" in result["profiles"]
        },
        "front": common.front_metrics(x, depth, t_lattice, h0, config.g, h0),
        "mass_drift": {
            "max_abs": result["mass_drift_max_abs"],
            "final": result["mass_drift_series"][-1] if result["mass_drift_series"] else 0.0,
            "n_samples": len(result["mass_drift_series"]),
        },
        "inventory_drift_max_abs": result["inventory_drift_max_abs"],
        "nan_step": result["nan_step"],
        "_depth": depth,
        "_x": x,
        "_surface": result["profiles"][key].get("depth_xz", result["snapshots"].get("depth_xz")),
        "_surface_early": result["profiles"].get(f"{CHECKPOINTS[0]:.3f}", {}).get("depth_xz"),
    }
    return case


def plot_surface(path: Path, surface: list, t_star: float) -> Path:
    """Water-column height map ``h_s(x, z)`` at one time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(surface, dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=140)
    im = ax.imshow(
        z, origin="lower", aspect="auto", extent=[0, z.shape[1], 0, z.shape[0]], cmap="viridis"
    )
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("z (span) [cells]")
    ax.set_title(f"3-D dam break: water-column height h_s(x,z), t*={t_star:g}")
    fig.colorbar(im, ax=ax, label="h_s [cells]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(_HERE / "result.json"))
    parser.add_argument("--fig-dir", default=str(_HERE / "figures"))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--grids", default=None, help="comma separated h0 list")
    args = parser.parse_args(argv)

    grids = (
        [int(v) for v in args.grids.split(",")]
        if args.grids
        else ([8, 10] if args.fast else [12, 16, 20])
    )

    cases = []
    for h0 in grids:
        case = run_case(h0)
        cases.append(case)
        print(
            f"  h0={h0:3d} grid={case['grid']} steps={case['n_steps']} "
            f"P1={100 * case['profile']['rel_l1']:.2f}% "
            f"front_err={100 * case['front']['rel_error']:.2f}% "
            f"mass_drift={100 * case['mass_drift']['max_abs']:.3f}% "
            f"inv_drift={100 * case['inventory_drift_max_abs']:.3f}%",
            flush=True,
        )

    payload: dict = {
        "benchmark": "dam_break_3d",
        "title": (
            "3-D square-column dam break (Körner free-surface LBM): mass audit, "
            "Ritter reference profile, grid order"
        ),
        "geometry": "column a x a x h0 (a = h0) in a 4h0 x 2h0 x h0 box, no-slip walls",
        "nondimensionalisation": (
            "lattice units; c0 = sqrt(g h0) = 0.08, t* = t sqrt(g/h0); "
            "the Ritter profile is a reference, not a certified anchor"
        ),
        "checkpoints_t_star": list(CHECKPOINTS),
        "t_star_end": T_STAR_END,
        "gates": common.gates(),
        "runs": [{k: v for k, v in case.items() if not k.startswith("_")} for case in cases],
    }

    errors = [case["profile"]["rel_l1"] for case in cases]
    spacings = [1.0 / case["h0"] for case in cases]
    convergence = common.convergence_order(
        errors, [spacings[i] / spacings[i + 1] for i in range(len(spacings) - 1)]
    )
    convergence["spacings"] = spacings
    convergence["grids"] = [case["h0"] for case in cases]
    payload["convergence"] = convergence

    finest = cases[-1]
    verdict = {
        "grid_label": finest["h0"],
        "P1_profile": {
            "value": finest["profile"]["rel_l1"],
            "limit": common.PROFILE_REL_ERROR_MAX,
            "pass": bool(finest["profile"]["rel_l1"] <= common.PROFILE_REL_ERROR_MAX),
            "metric": "relative L1 depth-profile deviation from Ritter (reference, not anchor)",
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
            "value": convergence["order_mean"],
            "limit": common.CONVERGENCE_ORDER_MIN,
            "pass": bool(
                convergence["order_mean"] is not None
                and not math.isnan(convergence["order_mean"])
                and convergence["order_mean"] >= common.CONVERGENCE_ORDER_MIN
            ),
            "metric": "observed order p on the P1 metric over three grids",
        },
    }
    verdict["all_pass"] = all(
        verdict[key]["pass"] for key in ("P1_profile", "P2_front", "P3_mass", "P4_order")
    )
    payload["verdict"] = verdict

    # --- figures -------------------------------------------------------------
    fig_dir = Path(args.fig_dir)
    h0_finest = finest["h0"]
    x = finest["_x"]
    g_finest = C0**2 / h0_finest
    analytic = common.ritter_depth(x, T_STAR_END * h0_finest / C0, h0_finest, g_finest, h0_finest)
    figures: list[str] = [
        str(
            common.plot_profiles(
                fig_dir / "3d_profiles.png",
                x,
                analytic,
                {f"FS-LBM 3D h0={h0_finest} (span-averaged)": (x, finest["_depth"])},
                title=f"3-D dam break h0={h0_finest}: span-averaged depth vs Ritter reference",
            )
        )
    ]
    if finest["_surface"] is not None:
        figures.append(
            str(plot_surface(fig_dir / "3d_surface.png", finest["_surface"], T_STAR_END))
        )
    if finest["_surface_early"] is not None:
        figures.append(
            str(
                plot_surface(
                    fig_dir / "3d_surface_early.png", finest["_surface_early"], CHECKPOINTS[0]
                )
            )
        )
    figures.append(
        str(
            common.plot_convergence(
                fig_dir / "3d_convergence.png",
                {"FS-LBM 3D (Körner)": (spacings, errors)},
                title="3-D dam break: grid convergence of the Ritter-reference profile error",
            )
        )
    )
    payload["figures"] = figures

    out = common.write_json(args.out, payload)
    print(f"wrote {out}")
    for key in ("P1_profile", "P2_front", "P3_mass", "P4_order"):
        item = verdict[key]
        print(f"    {key:<11} value={item['value']!r} limit={item['limit']!r} pass={item['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
