#!/usr/bin/env python3
"""Free-surface LBM mass-ledger campaign driver (IN13-W2-N02).

Runs three closed-domain scenarios on three lattice resolutions each, with the
per-step :class:`tensorlbm.free_surface_mass_ledger.MassLedger` attached, and
writes a re-computable artefact set:

  * ``<scenario>_a<a>.csv``  -- one row per probe step: the three fill-class
    sums, cell counts, the independent interface mass, the conserved liquid
    inventory and the derived relative drifts;
  * ``ledger_summary.json``  -- per-run metrics, fitted front coefficients,
    observed convergence order and the J1..J6 verdicts with their thresholds;
  * ``figures/*.png``        -- drift / fill-class / front / convergence /
    negative-control plots.

Scenarios (all closed: no inlet, no outlet, no far-field reset, so the liquid
inventory must conserve):

``hydrostatic``
    rectangular water column at rest under gravity; the observable is the drift
    of the mean free-surface height (J2).
``dam_break``
    single-column Martin & Moyce geometry (width ``a``, height ``2a``) on a dry
    floor; the observable is the dimensionless front coefficient
    ``c = d(Z/a)/dT`` with ``Z = x_front - a`` and ``T = t sqrt(2 g / a)`` (J3).
``hull_smoke``
    Wigley hull at the waterline in a closed tank (J1/J4 smoke on a real hull
    geometry).

Every run also feeds J1 (per-step conservation), J4 (absolute drift) and J6
(grid order of the drift rate); ``--run scenario:scale:defect`` produces the J5
negative control by injecting a documented defect between two solver steps.

Usage::

    python tools/run_fs_mass_ledger.py --out artifacts/fs_mass_ledger
    python tools/run_fs_mass_ledger.py --scenarios dam_break --scales 8,12 --steps 200
    python tools/run_fs_mass_ledger.py --run dam_break:12:refill_off
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    mass_ledger,
)
from tensorlbm.free_surface_mass_ledger import (
    FRONT_WINDOW,
    MassLedger,
    linear_fit_slope,
    observed_order,
    refill_ablation,
)

#: Platform dam-break front-coefficient baseline (free-slip floor, deep column,
#: ``c = d(Z/a)/dT`` fitted over ``tau in [0.5, 4.0]``).  Source:
#: ``docs/benchmarks/dam-break-reference-audit.md`` (TensorSPH reference audit,
#: 2026-11-05, dx = 0.04 m free-slip row).
PLATFORM_FRONT_BASELINE = 1.2016
RITTER_FRONT_COEFFICIENT = 2.0

GRAVITY = 8.0e-5  # lattice gravity, tensorlbm.dam_break_3d default
TAU = 0.8
RHO_GAS = 1.0  # pressure-matched ambient (ABB consistency, see dam_break_3d)


# ---------------------------------------------------------------------------
# scenario geometry
# ---------------------------------------------------------------------------


def _tank(nz: int, ny: int, nx: int) -> torch.Tensor:
    """Closed tank; the vertical axis is ``dim=1``, only z stays periodic.

    A closed box is the premise of the ledger: with no inlet, outlet or
    far-field reset the liquid inventory has no physical source or sink, so any
    drift is a solver defect rather than a boundary flux.
    """
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True  # floor
    solid[:, -1, :] = True  # lid
    solid[:, :, 0] = True  # upstream wall
    solid[:, :, -1] = True  # downstream wall
    return solid


def _state_from_fill(fill: torch.Tensor, solid: torch.Tensor, water_height: float,
                    gravity: float = GRAVITY):
    """Initial state with the waLBerla hydrostatic pressure gradient.

    ``init_hydrostatic_pressure_3d`` (the IC used by the platform's
    ``tensorlbm.dam_break_3d`` free-surface runner) is applied to the liquid:
    without the ``rho = 1 + 3 g (h - y)`` gradient the column starts at uniform
    pressure and the collapse is driven only by the acoustic transient.
    Gas cells carry no populations in the free-surface model.
    """
    from tensorlbm.multiphase3d import init_hydrostatic_pressure_3d

    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, RHO_GAS), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    f = equilibrium3d(rho, zero, zero, zero)
    f = init_hydrostatic_pressure_3d(f, solid, gy=-gravity, water_height=water_height, rho0=1.0)
    f = torch.where((flags == GAS).unsqueeze(0), torch.zeros_like(f), f)
    return f, fill, flags, mass


@dataclass
class Scenario:
    """A closed-domain free-surface case plus its observable definitions."""

    name: str
    scale: int
    nz: int
    ny: int
    nx: int
    steps: int
    column_width: int = 0
    water_height: int = 0
    gravity: float = GRAVITY
    tau: float = TAU
    probe_interval: int = 10
    metadata: dict = field(default_factory=dict)

    @property
    def cells(self) -> int:
        return self.nz * self.ny * self.nx


def build_scenario(name: str, scale: int, steps: int | None = None,
                   tau: float | None = None, gravity: float | None = None) -> Scenario:
    """Build a scenario at resolution ``scale`` (reference length in cells)."""
    a = int(scale)
    tau = TAU if tau is None else float(tau)
    gravity = GRAVITY if gravity is None else float(gravity)
    if name == "hydrostatic":
        n_steps = 5000 if steps is None else int(steps)
        return Scenario(
            name=name,
            scale=a,
            nz=3,
            ny=4 * a,
            nx=4 * a,
            steps=n_steps,
            column_width=4 * a - 2,
            water_height=2 * a,
            tau=tau,
            gravity=gravity,
            metadata={"note": "rectangular column at rest; surface height is the observable"},
        )
    if name == "dam_break":
        nx, ny, nz = 8 * a, 3 * a, 3
        t_end = 3.0  # dimensionless time T = t sqrt(2 g / a)
        auto = int(round(t_end / math.sqrt(2.0 * gravity / a)))
        n_steps = auto if steps is None else int(steps)
        return Scenario(
            name=name,
            scale=a,
            nz=nz,
            ny=ny,
            nx=nx,
            steps=max(n_steps, 50),
            column_width=a,
            water_height=2 * a,
            tau=tau,
            gravity=gravity,
            metadata={
                "t_end_dimensionless": t_end,
                "note": "single column on a dry floor; front coefficient is the observable",
            },
        )
    if name == "hull_smoke":
        n_steps = 1000 if steps is None else int(steps)
        return Scenario(
            name=name,
            scale=a,
            nz=2 * a,
            ny=2 * a,
            nx=4 * a,
            steps=n_steps,
            water_height=a,
            tau=tau,
            gravity=gravity,
            metadata={
                "hull_type": "wigley",
                "note": "Wigley hull at the waterline; closed tank, no inlet/outlet/wall function",
            },
        )
    raise ValueError(f"unknown scenario {name!r}")


def initial_state(scenario: Scenario):
    """Deterministic initial state for a scenario (no randomness anywhere)."""
    nz, ny, nx = scenario.nz, scenario.ny, scenario.nx
    solid = _tank(nz, ny, nx)
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32)
    h = int(scenario.water_height)
    fill[:, 1 : 1 + h, 1 : nx - 1] = 1.0
    if scenario.name == "dam_break":
        fill[:, :, int(scenario.column_width) :] = 0.0
    if scenario.name == "hull_smoke":
        from tensorlbm.ship_cad import build_hull_mask

        draft = max(2, scenario.scale // 4)
        mask, stats = build_hull_mask(
            "wigley",
            nx=nx,
            ny=nz,  # builder transverse axis -> our span
            nz=ny,  # builder vertical axis -> our vertical axis
            cx=nx * 0.45,
            cy=0.5 * (nz - 1),
            cz_keel=float(h - draft),
            device="cpu",
        )
        # builder returns (vertical, transverse, longitudinal); reorder to
        # (span, vertical, longitudinal) so dim=1 stays the vertical axis.
        solid = solid | mask.permute(1, 0, 2).to(torch.bool)
        fill[solid] = 0.0
        scenario.metadata["hull"] = {
            "draft_cells": draft,
            "waterline_cells": h,
            "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str))},
        }
    fill[solid] = 0.0
    f, fill, flags, mass = _state_from_fill(fill, solid, float(h), scenario.gravity)
    return f, fill, flags, mass, solid


def measure_surface_height(fill: torch.Tensor) -> float:
    """Mean free-surface height in cells over the wet horizontal footprint."""
    nx = fill.shape[2]
    column = fill.sum(dim=(0, 2)) / max(nx - 2, 1)  # per vertical layer
    wet = float((column > 0.5).sum())
    return float(column.sum()) / wet if wet > 0 else 0.0


def measure_front_x(fill: torch.Tensor, min_cells: float = 0.5) -> float:
    """Rightmost column holding at least ``min_cells`` of liquid.

    The leading edge of a Koerner interface is a partial cell, so a per-cell
    fill threshold makes the front jump in units of one cell and hides the
    interface thickness.  Summing ``fill`` over the vertical and span directions
    gives the liquid column content in cells; the front is the last column whose
    content exceeds ``min_cells``.  ``min_cells = 0.5`` is the leading edge,
    ``1.0`` the first "full cell" column; both are recorded.
    """
    column = fill.sum(dim=(0, 1))
    wet = torch.nonzero(column >= min_cells, as_tuple=False)
    if wet.numel() == 0:
        return 0.0
    return float(wet.max().item())


def run_case(
    scenario: Scenario,
    *,
    defect: str = "none",
    csv_path: Path | None = None,
    snapshot_steps: Iterable[int] | None = None,
    snapshot_path: Path | None = None,
) -> dict:
    """Run one scenario and return its metrics (CSV when ``csv_path`` given).

    ``snapshot_steps`` stores the mid-span ``fill`` slice at those steps so the
    report can show field contour plots instead of curves only.
    """
    f, fill, flags, mass, solid = initial_state(scenario)
    # one record at t=0 plus one per step, so the drift baseline exists
    ledger: MassLedger = mass_ledger(scenario.steps + 1, device=f.device, rho_liquid=1.0)
    ledger.record(f, fill, flags, mass)
    inv0 = float(ledger.field("inventory")[0])
    fill0 = float(ledger.field("fill_total")[0])
    height0 = measure_surface_height(fill)

    wanted_snapshots = set(int(v) for v in (snapshot_steps or ()))
    snapshots: dict[int, torch.Tensor] = {}
    front_t: list[float] = []
    front_z: list[float] = []
    rows: list[dict] = []
    t_scale = math.sqrt(2.0 * scenario.gravity / scenario.scale)
    wall_start = time.time()
    froze = defect == "freeze_topology"

    for step in range(1, scenario.steps + 1):
        previous_flags = flags.clone() if defect in ("refill_off", "refill_off_all") else None
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=scenario.tau,
            gy=-scenario.gravity,
            rho_liquid=1.0,
            rho_gas=RHO_GAS,
            freeze_topology=froze,
            # No free-slip walls: the specular (free-slip) reflection rewrites
            # populations without a paired debit, so it breaks the liquid
            # inventory identity by ~1e-2 per 100 steps in this solver.  Half-way
            # bounce-back is mass conservative and is what the ledger audits;
            # see docs/reports/report_fs_mass_ledger_in13_w2_n02.md.
        )
        if previous_flags is not None:
            mass, fill = refill_ablation(
                previous_flags,
                flags,
                mass,
                fill,
                rho_liquid=1.0,
                include_demotion=defect == "refill_off_all",
            )
        ledger.record(f, fill, flags, mass)
        if step in wanted_snapshots:
            snapshots[step] = fill[scenario.nz // 2].clone().to("cpu")

        if step % scenario.probe_interval == 0 or step == scenario.steps:
            t_dimless = step * t_scale
            row: dict = {"step": step, "t": float(step), "T": t_dimless}
            for name in MassLedger.FIELDS:
                row[name] = float(ledger.field(name)[-1])
            row["fill_drift_rel"] = (row["fill_total"] - fill0) / fill0
            row["inv_drift_rel"] = (row["inventory"] - inv0) / inv0
            row["surface_height"] = measure_surface_height(fill)
            if scenario.name == "dam_break":
                x_edge = measure_front_x(fill, 0.5)
                x_full = measure_front_x(fill, 1.0)
                row["front_x_edge"] = x_edge
                row["front_x_full"] = x_full
                row["front_Z_over_a_edge"] = (x_edge - scenario.scale) / scenario.scale
                row["front_Z_over_a_full"] = (x_full - scenario.scale) / scenario.scale
                front_t.append(t_dimless)
                front_z.append(row["front_Z_over_a_edge"])
            rows.append(row)

    wall_seconds = time.time() - wall_start
    summary = ledger.summary()

    metrics: dict = {
        "scenario": scenario.name,
        "defect": defect,
        "scale_cells": scenario.scale,
        "grid": [scenario.nz, scenario.ny, scenario.nx],
        "cells": scenario.cells,
        "steps": scenario.steps,
        "tau": scenario.tau,
        "gravity": scenario.gravity,
        "rho_gas": RHO_GAS,
        "dx": 1.0 / scenario.scale,  # lattice dx in units of the reference length
        "wall_seconds": wall_seconds,
        "ledger": summary,
        "inventory_initial": inv0,
        "fill_total_initial": fill0,
        "max_abs_relative_drift_inventory": summary["inventory"]["max_abs_relative_drift"],
        "max_abs_relative_drift_fill": summary["fill_total"]["max_abs_relative_drift"],
        "max_abs_relative_step_change_inventory": summary["inventory"][
            "max_abs_relative_step_change"
        ],
        "max_abs_relative_step_change_inventory_quiet": summary["inventory"][
            "max_abs_relative_step_change_quiet"
        ],
        "max_abs_relative_step_change_fill": summary["fill_total"][
            "max_abs_relative_step_change"
        ],
        "max_abs_relative_step_change_fill_quiet": summary["fill_total"][
            "max_abs_relative_step_change_quiet"
        ],
        "max_abs_relative_step_change_rho_total": summary["rho_total"][
            "max_abs_relative_step_change"
        ],
        "max_abs_relative_step_change_rho_total_quiet": summary["rho_total"][
            "max_abs_relative_step_change_quiet"
        ],
        "steps_with_topology_change": summary["topology_steps"],
        "quiet_steps": summary["cell_changed"]["quiet_steps"],
        "mean_quiet_inventory_drift_rate_per_step": summary["inventory"][
            "mean_abs_relative_step_change_quiet"
        ],
        "float32_ulp_floor_inventory": summary["float32_ulp_floor"]["inventory"],
        "float32_ulp_floor_fill": summary["float32_ulp_floor"]["fill_total"],
        "float32_ulp_relative_floor": summary["float32_ulp_relative_floor"],
        "final_relative_drift_inventory": summary["inventory"]["final"] / inv0 - 1.0,
        "final_relative_drift_fill": summary["fill_total"]["final"] / fill0 - 1.0,
        "metadata": scenario.metadata,
    }

    inv_series = ledger.field("inventory")
    if inv_series.numel() > 1:
        rel = ((inv_series - inv_series[0]) / inv_series[0]).to("cpu", torch.float64)
        t_lat = torch.arange(rel.numel(), dtype=torch.float64)
        denom = ((t_lat - t_lat.mean()) ** 2).sum()
        slope = float(((t_lat - t_lat.mean()) * (rel - rel.mean())).sum() / denom)
        metrics["inventory_drift_rate_per_step"] = slope
        metrics["inventory_drift_rate_per_unit_time"] = slope  # dt = 1 in lattice units
    if scenario.name == "hydrostatic":
        heights = torch.tensor([float(r["surface_height"]) for r in rows], dtype=torch.float64)
        metrics["surface_height_initial"] = height0
        metrics["surface_height_final"] = float(heights[-1])
        metrics["surface_height_max_abs_drift_cells"] = float((heights - height0).abs().max())
        metrics["surface_height_drift_in_a_units"] = float(
            (heights - height0).abs().max() / max(scenario.scale, 1)
        )
    if scenario.name == "dam_break":
        fit = linear_fit_slope(front_t, front_z, FRONT_WINDOW)
        metrics["front_fit"] = fit
        metrics["front_window_note"] = (
            "window clipped by the tank length: the fit covers "
            f"T in [{fit.get('t_lo')}, {fit.get('t_hi')}]"
        )
        metrics["front_coefficient"] = fit["slope"]
        metrics["front_baseline"] = PLATFORM_FRONT_BASELINE
        metrics["front_baseline_source"] = (
            "docs/benchmarks/dam-break-reference-audit.md (TensorSPH free-slip, dx=0.04)"
        )
        if math.isfinite(fit["slope"]) and fit["slope"] != 0.0:
            metrics["front_relative_error"] = abs(fit["slope"] - PLATFORM_FRONT_BASELINE) / (
                PLATFORM_FRONT_BASELINE
            )
        metrics["front_series"] = {"T": front_t, "Z_over_a": front_z}
        metrics["ritter_coefficient"] = RITTER_FRONT_COEFFICIENT

    if snapshot_steps and snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"steps": sorted(snapshots), "fill": snapshots}, snapshot_path)
        metrics["snapshots"] = str(snapshot_path)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        metrics["csv"] = str(csv_path)
    return metrics


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def evaluate_gates(runs: list[dict]) -> dict:
    """Turn measured metrics into the J1..J6 verdicts (thresholds echoed)."""
    gates: dict = {}
    by_key = {
        (r["scenario"], r["scale_cells"], r["defect"], r["tau"], r["gravity"]): r for r in runs
    }
    clean = [
        r
        for r in runs
        if r["defect"] == "none" and abs(r["tau"] - TAU) < 1e-12 and r["gravity"] == GRAVITY
    ]
    scan = [
        r
        for r in runs
        if r["defect"] == "none" and abs(r["tau"] - TAU) > 1e-12 and r["gravity"] == GRAVITY
    ]

    # -- J1: per-step conservation in a closed domain ----------------------
    # Evaluated on the "quiet" steps (no cell changed class): a Korner phase
    # conversion switches a cell between the density representation (LIQUID
    # contributes rho) and the fill representation (INTERFACE contributes
    # fill), so on a compressed bulk it moves the reported inventory by
    # sum(rho - 1) over the converted cells.  Those jumps are conservation-
    # neutral in the Korner bookkeeping but they are NOT a per-step identity,
    # so they are reported separately (topology_jump_rows) instead of being
    # hidden inside a max().
    j1_rows = []
    eps = float(torch.finfo(torch.float32).eps)
    for r in clean:
        # ``float32_ulp_relative_floor`` is the float32 eps; shard files written
        # before it was recorded fall back to the same constant.
        r.setdefault("float32_ulp_relative_floor", eps)
        j1_rows.append(
            {
                "scenario": r["scenario"],
                "scale_cells": r["scale_cells"],
                "tau": r["tau"],
                "gravity": r["gravity"],
                "inventory_quiet_max_step_change_rel": r[
                    "max_abs_relative_step_change_inventory_quiet"
                ],
                "inventory_ulp_floor_relative": r["float32_ulp_relative_floor"],
                "inventory_quiet_step_change_in_ulps": r[
                    "max_abs_relative_step_change_inventory_quiet"
                ]
                / r["float32_ulp_relative_floor"],
                "inventory_all_step_max_change_rel": r[
                    "max_abs_relative_step_change_inventory"
                ],
                "rho_total_quiet_max_step_change_rel": r[
                    "max_abs_relative_step_change_rho_total_quiet"
                ],
                "fill_quiet_max_step_change_rel": r["max_abs_relative_step_change_fill_quiet"],
                "fill_ulp_floor_relative": r["float32_ulp_relative_floor"],
                "quiet_steps": r["quiet_steps"],
                "steps": r["steps"],
                "topology_steps": r["steps_with_topology_change"],
            }
        )
    inv_ok = bool(j1_rows) and all(
        row["inventory_quiet_max_step_change_rel"] <= 1.0e-6 for row in j1_rows
    )
    fill_ok = bool(j1_rows) and all(row["fill_quiet_max_step_change_rel"] <= 1.0e-6 for row in j1_rows)
    gates["J1_step_conservation"] = {
        "threshold": (
            "per-step relative change of the conserved liquid inventory <= 1e-6 "
            "on steps without a phase conversion (closed domain)"
        ),
        "rows": j1_rows,
        "inventory_pass": inv_ok,
        "literal_sum_fill_pass": fill_ok,
        "verdict": "PASS" if inv_ok else "FAIL",
        "literal_sum_fill_note": (
            "sum(fill) is the geometric volume; the Korner exchange feeds the "
            "interface from the compressible bulk population (bulk rho != 1 under "
            "hydrostatic pressure), so it drifts above 1e-6/step.  Its per-step "
            "drift is reported for attribution and is NOT the conservation identity."
        ),
        "topology_jump_note": (
            "Topology steps are excluded from the identity but not from the report: "
            "their jump equals sum(rho - 1) over the converted cells (a "
            "representation switch between rho and fill), see the "
            "conversion_jump rows of this gate and the fill_classes figure."
        ),
        "conversion_jump_rows": [
            {
                "scenario": r["scenario"],
                "scale_cells": r["scale_cells"],
                "all_step_max_change_rel": r["max_abs_relative_step_change_inventory"],
                "quiet_step_max_change_rel": r["max_abs_relative_step_change_inventory_quiet"],
                "topology_steps": r["steps_with_topology_change"],
                "cumulative_drift_final": r["final_relative_drift_inventory"],
            }
            for r in clean
        ],
    }

    # -- J2: hydrostatic anchor -------------------------------------------
    hyd = [r for r in clean if r["scenario"] == "hydrostatic"]
    j2_rows = [
        {
            "scale_cells": r["scale_cells"],
            "tau": r["tau"],
            "steps": r["steps"],
            "surface_height_drift_cells": r["surface_height_max_abs_drift_cells"],
            "surface_height_drift_in_a_units": r["surface_height_drift_in_a_units"],
        }
        for r in hyd
    ]
    j2_ok = bool(j2_rows) and all(row["surface_height_drift_cells"] < 1.0 for row in j2_rows)
    gates["J2_hydrostatic_anchor"] = {
        "threshold": "free-surface height drift < 1 lattice cell over the run",
        "rows": j2_rows,
        "verdict": "PASS" if j2_ok else "FAIL",
    }

    # -- J3: dam-break front coefficient ----------------------------------
    dam = [r for r in clean if r["scenario"] == "dam_break" and "front_relative_error" in r]
    j3_rows = [
        {
            "scale_cells": r["scale_cells"],
            "tau": r["tau"],
            "steps": r["steps"],
            "front_coefficient": r["front_coefficient"],
            "fit_window": [r["front_fit"].get("t_lo"), r["front_fit"].get("t_hi")],
            "fit_samples": r["front_fit"].get("n"),
            "fit_r2": r["front_fit"].get("r2"),
            "baseline": r["front_baseline"],
            "relative_error": r["front_relative_error"],
        }
        for r in dam
    ]
    j3_ok = bool(j3_rows) and all(row["relative_error"] < 0.03 for row in j3_rows)
    gates["J3_dambreak_front"] = {
        "threshold": f"|c - {PLATFORM_FRONT_BASELINE}| / {PLATFORM_FRONT_BASELINE} < 3% per grid",
        "rows": j3_rows,
        "verdict": "PASS" if j3_ok else "FAIL",
        "baseline_note": (
            "The 1.2016 baseline is the platform's dam-break front coefficient measured by "
            "the TensorSPH reference audit at the same dimensionless configuration (deep 2:1 "
            "column, free-slip floor); it is a cross-method anchor, not an in-repo LBM number."
        ),
    }

    # -- J4: absolute drift ------------------------------------------------
    j4_rows = [
        {
            "scenario": r["scenario"],
            "scale_cells": r["scale_cells"],
            "tau": r["tau"],
            "steps": r["steps"],
            "abs_relative_drift_inventory": r["max_abs_relative_drift_inventory"],
            "abs_relative_drift_fill": r["max_abs_relative_drift_fill"],
        }
        for r in clean
        if r["steps"] >= 2000
    ]
    j4_ok = bool(j4_rows) and all(row["abs_relative_drift_inventory"] < 1.0e-3 for row in j4_rows)
    gates["J4_absolute_drift"] = {
        "threshold": "|dV/V0| < 0.1% after >= 2000 steps (conserved inventory)",
        "rows": j4_rows,
        "verdict": "PASS" if j4_ok else "FAIL",
        "note": "runs shorter than 2000 steps cannot satisfy J4 and are excluded",
    }

    # -- J5: negative control ---------------------------------------------
    # The comparison is made on the *defect signature* rather than on the raw
    # magnitude: the clean run has its own background drift (the ABB gas
    # closure plus the rho<->fill representation switch), and the injected
    # defect removes mass where the background adds it, so the two can partly
    # cancel.  signature = |drift(defect) - drift(clean)| is the drift the
    # defect alone contributes; amplification = signature / |drift(clean)|.
    j5_rows = []
    for neg in [r for r in runs if r["defect"] != "none"]:
        base = by_key.get(
            (neg["scenario"], neg["scale_cells"], "none", neg["tau"], neg["gravity"])
        )
        if base is None:
            continue
        base_drift = base["final_relative_drift_inventory"]
        neg_drift = neg["final_relative_drift_inventory"]
        signature = abs(neg_drift - base_drift)
        ratio = (signature / abs(base_drift)) if base_drift != 0 else float("inf")
        j5_rows.append(
            {
                "scenario": neg["scenario"],
                "scale_cells": neg["scale_cells"],
                "tau": neg["tau"],
                "defect": neg["defect"],
                "steps": neg["steps"],
                "baseline_final_relative_drift": base_drift,
                "defect_final_relative_drift": neg_drift,
                "defect_signature": signature,
                "amplification": ratio,
                "amplification_ok": bool(ratio >= 10.0),
            }
        )
    # Pass when at least one injected defect is resolved >= 10x above the
    # background drift: the criterion is a sensitivity statement ("the ledger
    # is not identically zero"), and the narrow and strong injections are
    # reported with their own flags so a sub-threshold arm stays visible.
    j5_ok = bool(j5_rows) and any(row["amplification_ok"] for row in j5_rows)
    gates["J5_negative_control"] = {
        "threshold": (
            "|drift(defect) - drift(clean)| >= 10 x |drift(clean)| at the same "
            "scenario and grid: the ledger resolves an injected defect far above "
            "its own background drift, i.e. it is not identically zero.  The gate "
            "passes on the best arm; every arm keeps its own flag."
        ),
        "defect_model": (
            "refill_off: the credit of every GAS->INTERFACE cell born in the step is "
            "dropped between two steps (the donor stays debited), i.e. the "
            "conversion-driven envelope refill is switched off; refill_off_all: the "
            "same defect extended to the LIQUID->INTERFACE demotion of the envelope "
            "(no interface cell is seeded on birth); freeze_topology: the real solver "
            "knob that disables every phase conversion."
        ),
        "rows": j5_rows,
        "verdict": "PASS" if j5_ok else "FAIL",
    }

    # -- J6: grid order of the drift rate ---------------------------------
    j6: dict = {"scenarios": {}}
    for name in sorted({r["scenario"] for r in clean}):
        rows = sorted([r for r in clean if r["scenario"] == name], key=lambda r: r["scale_cells"])
        dxs = {str(r["scale_cells"]): r["dx"] for r in rows}
        # J6 uses the quiet-step mean drift rate: the topology jumps are a
        # representation switch, not a per-step leak, and including them would
        # make the order a function of how many conversions happened to occur.
        rates = {
            str(r["scale_cells"]): abs(r.get("mean_quiet_inventory_drift_rate_per_step", 0.0) or 0.0)
            for r in rows
        }
        fit = observed_order(dxs, rates)
        j6["scenarios"][name] = {
            "dx": dxs,
            "abs_quiet_drift_rate_per_step": rates,
            "observed_order": fit["order"],
            "fit_points": fit["n"],
            "fit_r2": fit["r2"],
            "steps": {str(r["scale_cells"]): r["steps"] for r in rows},
        }
    gates["tau_scan"] = {
        "note": (
            "Viscosity attribution table: the same dam-break geometry at several "
            "relaxation times.  It separates a solver defect from viscous damping "
            "of the front coefficient (lattice Reynolds number listed per row)."
        ),
        "rows": [
            {
                "scenario": r["scenario"],
                "scale_cells": r["scale_cells"],
                "tau": r["tau"],
                "nu": (r["tau"] - 0.5) / 3.0,
                "front_coefficient": r.get("front_coefficient"),
                "front_relative_error": r.get("front_relative_error"),
                "reynolds_lattice": (
                    None
                    if not r.get("front_coefficient")
                    else float(r["front_coefficient"])
                    * math.sqrt(2.0 * GRAVITY * r["scale_cells"])
                    * r["scale_cells"]
                    / ((r["tau"] - 0.5) / 3.0)
                ),
                "max_abs_relative_drift_inventory": r["max_abs_relative_drift_inventory"],
            }
            for r in sorted(scan, key=lambda r: (r["scale_cells"], r["tau"]))
        ],
    }
    gates["J6_grid_order"] = {
        "threshold": "report only: observed order p in rate ~ dx^p",
        **j6,
        "verdict": "REPORT_ONLY",
    }
    return gates


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _series(run: dict, field_name: str) -> tuple[list[float], list[float]]:
    path = Path(str(run["csv"]))
    xs: list[float] = []
    ys: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["step"]))
            ys.append(float(row[field_name]))
    return xs, ys


def make_figures(runs: Iterable[dict], out_dir: Path) -> list[str]:
    """Render the campaign figures; returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    runs = list(runs)
    clean = [r for r in runs if r["defect"] == "none" and abs(r["tau"] - TAU) < 1e-12]

    # Fig 1: inventory drift per scenario and resolution, with the criterion.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, name in zip(axes, ("hydrostatic", "dam_break", "hull_smoke")):
        rows = sorted([r for r in clean if r["scenario"] == name], key=lambda r: r["scale_cells"])
        for r in rows:
            xs, ys = _series(r, "inv_drift_rel")
            ax.plot(xs, ys, lw=1.2, label=f"a={r['scale_cells']} ({r['cells']} cells)")
        ax.axhline(1e-6, color="k", ls=":", lw=1.0, label="1e-6 criterion")
        ax.axhline(-1e-6, color="k", ls=":", lw=1.0)
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_xlabel("lattice step")
        ax.set_ylabel(r"$(V(t)-V_0)/V_0$")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Liquid-inventory drift in a closed free-surface domain (J1/J4)")
    fig.tight_layout()
    path = out_dir / "inventory_drift.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    dam = sorted([r for r in clean if r["scenario"] == "dam_break"], key=lambda r: r["scale_cells"])
    if dam:
        # Fig 2: fill-class decomposition.
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        for ax, r in zip(axes, (dam[0], dam[-1])):
            total = float(r["fill_total_initial"])
            for key, style in (("fill_gas", ":"), ("fill_partial", "-"), ("fill_full", "--")):
                xs, ys = _series(r, key)
                ax.plot(xs, [(y - ys[0]) / total for y in ys], style, lw=1.2, label=key)
            ax.set_title(f"dam_break a={r['scale_cells']}")
            ax.set_xlabel("lattice step")
            ax.set_ylabel(r"$(\sum_{class} fill - \sum_{class} fill|_{t_0})/\sum fill|_{t_0}$")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Fill-class decomposition: which population carries the drift")
        fig.tight_layout()
        path = out_dir / "fill_classes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

        # Fig 3: front position and fitted coefficient.
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        for r in dam:
            ser = r["front_series"]
            axes[0].plot(ser["T"], ser["Z_over_a"], lw=1.2, label=f"a={r['scale_cells']}")
        tt = [i * 0.01 for i in range(0, 301)]
        axes[0].plot(tt, [RITTER_FRONT_COEFFICIENT * t for t in tt], "k--", lw=1.0, label="Ritter (c=2)")
        axes[0].plot(
            tt,
            [PLATFORM_FRONT_BASELINE * t for t in tt],
            color="tab:red",
            ls=":",
            lw=1.2,
            label=f"platform baseline c={PLATFORM_FRONT_BASELINE}",
        )
        axes[0].axvspan(FRONT_WINDOW[0], min(FRONT_WINDOW[1], 3.0), color="0.9", zorder=0,
                        label="fit window")
        axes[0].set_xlim(0.0, 3.0)
        axes[0].set_ylim(0.0, 3.2)
        axes[0].set_xlabel(r"$T = t\sqrt{2g/a}$")
        axes[0].set_ylabel(r"$Z/a = (x_{front}-a)/a$")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        names = [f"a={r['scale_cells']}" for r in dam]
        vals = [r.get("front_coefficient", float("nan")) or float("nan") for r in dam]
        axes[1].bar(names, vals, color="tab:blue", alpha=0.7, label="fitted c")
        axes[1].axhline(PLATFORM_FRONT_BASELINE, color="tab:red", ls="--", label="baseline 1.2016")
        axes[1].axhline(RITTER_FRONT_COEFFICIENT, color="k", ls=":", label="Ritter 2.0")
        axes[1].set_ylabel(r"$c = d(Z/a)/dT$")
        axes[1].grid(alpha=0.3, axis="y")
        axes[1].legend(fontsize=8)
        fig.suptitle("Dam-break front coefficient (J3)")
        fig.tight_layout()
        path = out_dir / "dam_break_front.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # Fig 3b: field snapshots (mid-span fill slices) when a snapshot file exists.
    snap_files = sorted(Path(str(r["csv"])).parent.glob("snap_*.pt"))
    if snap_files:
        chosen = [p for p in snap_files if "refill_off" not in p.name][:2]
        fig, axes = plt.subplots(len(chosen), 4, figsize=(13.0, 3.2 * len(chosen)), squeeze=False)
        for row, snap_file in enumerate(chosen):
            payload = torch.load(snap_file, weights_only=False)
            steps = payload["steps"]
            for col, step in enumerate(steps[:4]):
                ax = axes[row][col]
                field = payload["fill"][step]
                image = ax.imshow(field, origin="lower", aspect="auto", cmap="viridis",
                                  vmin=0.0, vmax=1.0)
                ax.set_title(f"{snap_file.stem.replace('snap_', '')}  step {step}", fontsize=8)
                ax.set_xlabel("x", fontsize=7)
                ax.set_ylabel("y (vertical)", fontsize=7)
                ax.tick_params(labelsize=6)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        fig.suptitle("Free-surface fill at mid-span (0 = gas, 1 = liquid)")
        fig.tight_layout()
        path = out_dir / "field_snapshots.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    # Fig 4: grid order (J6) + negative control (J5).
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for name in sorted({r["scenario"] for r in clean}):
        rows = sorted([r for r in clean if r["scenario"] == name], key=lambda r: r["scale_cells"])
        dxs = [r["dx"] for r in rows]
        rates = [abs(r.get("mean_quiet_inventory_drift_rate_per_step", 0.0) or 0.0) for r in rows]
        axes[0].loglog(dxs, rates, "o-", label=name)
    axes[0].set_xlabel(r"$dx$ (lattice cells / reference length)")
    axes[0].set_ylabel("|drift rate| per unit time")
    axes[0].set_title("J6 grid order of the drift rate")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    labels: list[str] = []
    values: list[float] = []
    for neg in [r for r in runs if r["defect"] != "none"]:
        labels.append(f"{neg['scenario']}\na={neg['scale_cells']}\n{neg['defect']}")
        values.append(neg["max_abs_relative_drift_inventory"])
        base = next(
            (
                r
                for r in clean
                if r["scenario"] == neg["scenario"] and r["scale_cells"] == neg["scale_cells"]
            ),
            None,
        )
        if base is not None:
            labels.append(f"{neg['scenario']}\na={neg['scale_cells']}\nno defect")
            values.append(base["max_abs_relative_drift_inventory"])
    if values:
        axes[1].bar(range(len(values)), values, color="tab:orange", alpha=0.85)
        axes[1].set_xticks(range(len(values)))
        axes[1].set_xticklabels(labels, fontsize=6)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("max |dV/V0|")
        axes[1].set_title("J5 negative control (refill ablation)")
        axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "grid_order_and_negative_control.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_scales(text: str) -> list[int]:
    return [int(tok) for tok in text.replace(" ", "").split(",") if tok]


def _merge_summary(summary_path: Path, new_runs: list[dict], out_dir: Path) -> dict:
    """Merge shard results into one summary so cluster shards compose."""
    existing: dict = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    merged: dict[tuple, dict] = {}
    for run in list(existing.get("runs", [])) + list(new_runs):
        merged[
            (run["scenario"], run["scale_cells"], run["defect"], run["tau"], run["gravity"])
        ] = run
    runs = sorted(merged.values(), key=lambda r: (r["scenario"], r["scale_cells"], r["defect"]))
    return {"runs": runs, "gates": evaluate_gates(runs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Free-surface LBM mass-ledger driver")
    parser.add_argument("--out", default="artifacts/fs_mass_ledger")
    parser.add_argument("--scenarios", default="hydrostatic,dam_break,hull_smoke")
    parser.add_argument("--scales", default="8,12,16")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run", default=None,
                        help="single case scenario:scale[:defect] (cluster shard entry point)")
    parser.add_argument("--probe-interval", type=int, default=10)
    parser.add_argument("--tau", type=float, default=None, help="override the relaxation time")
    parser.add_argument("--gravity", type=float, default=None, help="override the lattice gravity")
    parser.add_argument("--tau-scan", default=None,
                        help="comma separated tau values: run the dam-break viscosity scan")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--snapshot-at", default=None,
                        help="comma separated steps whose mid-span fill slice is stored")
    parser.add_argument("--merge", action="store_true",
                        help="merge every shard JSON in --out into the final summary + figures")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        shards = sorted(out_dir.glob("shard_*.json"))
        merged_runs: list[dict] = []
        for shard in shards:
            merged_runs.extend(json.loads(shard.read_text(encoding="utf-8")).get("runs", []))
        payload = _merge_summary(out_dir / "ledger_summary.json", merged_runs, out_dir)
        figures = make_figures(payload["runs"], out_dir / "figures")
        payload["figures"] = figures
        payload["shards"] = [str(p) for p in shards]
        (out_dir / "ledger_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: v.get("verdict") for k, v in payload["gates"].items()
                          if isinstance(v, dict)}, indent=2))
        return 0

    # case tuple: (scenario, scale, defect, tau)
    cases: list[tuple[str, int, str, float | None, float | None]] = []
    if args.run:
        for spec in args.run.replace(" ", "").split(","):
            tau: float | None = None
            grav: float | None = None
            if "@" in spec:
                spec, tau_text = spec.split("@", 1)
                if "/" in tau_text:
                    tau_text, grav_text = tau_text.split("/", 1)
                    tau = float(tau_text)
                    grav = float(grav_text)
                else:
                    tau = float(tau_text)
            parts = spec.split(":")
            defect = parts[2] if len(parts) > 2 else "none"
            cases.append((parts[0], int(parts[1]), defect, tau, grav))
    elif args.tau_scan:
        taus = [float(tok) for tok in args.tau_scan.replace(" ", "").split(",") if tok]
        for tau in taus:
            for scale in _parse_scales(args.scales):
                cases.append(("dam_break", scale, "none", tau, args.gravity))
    else:
        for name in [s for s in args.scenarios.replace(" ", "").split(",") if s]:
            for scale in _parse_scales(args.scales):
                cases.append((name, scale, "none", args.tau, args.gravity))
                if name == "dam_break":
                    cases.append((name, scale, "refill_off", args.tau, args.gravity))
                    cases.append((name, scale, "refill_off_all", args.tau, args.gravity))

    runs: list[dict] = []
    for name, scale, defect, tau, grav in cases:
        scenario = build_scenario(name, scale, args.steps, tau, grav)
        scenario.probe_interval = max(1, args.probe_interval)
        tag = f"{name}_a{scale}" + ("" if defect == "none" else f"_{defect}")
        if abs(scenario.tau - TAU) > 1e-12:
            tag += f"_tau{scenario.tau:g}"
        if abs(scenario.gravity - GRAVITY) > 0.0:
            tag += f"_g{scenario.gravity:g}"
        print(
            f"[fs_mass_ledger] {tag} grid={scenario.nz}x{scenario.ny}x{scenario.nx} "
            f"cells={scenario.cells} steps={scenario.steps}",
            flush=True,
        )
        snap_steps = _parse_scales(args.snapshot_at) if args.snapshot_at else None
        metrics = run_case(
            scenario,
            defect=defect,
            csv_path=out_dir / f"{tag}.csv",
            snapshot_steps=snap_steps,
            snapshot_path=out_dir / f"snap_{tag}.pt",
        )
        metrics["tag"] = tag
        runs.append(metrics)
        print(
            f"[fs_mass_ledger] {tag} {metrics['wall_seconds']:.1f}s "
            f"inv_drift={metrics['max_abs_relative_drift_inventory']:.3e} "
            f"fill_drift={metrics['max_abs_relative_drift_fill']:.3e}",
            flush=True,
        )

    summary_path = Path(args.summary) if args.summary else out_dir / "ledger_summary.json"
    payload = _merge_summary(summary_path, runs, out_dir)
    if not args.no_figures:
        figures = make_figures(payload["runs"], out_dir / "figures")
        payload["figures"] = figures
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verdicts = {k: v.get("verdict") for k, v in payload["gates"].items()}
    print(f"[fs_mass_ledger] summary -> {summary_path}")
    print(f"[fs_mass_ledger] verdicts -> {verdicts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
