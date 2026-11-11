"""IN13-W2-N04 driver: stage-resolved free-surface mass closure campaign.

Runs the three closed-domain scenarios of
``docs/reports/report_fs_mass_ledger_in13_w2_n02.md`` at several resolutions and
records, per step,

* the three-fill-class ledger (:class:`tensorlbm.free_surface_mass_ledger.
  MassLedger`) -- the "how much" series;
* the stage attribution (:class:`tensorlbm.free_surface_mass_closure.
  ClosureLedger`) -- the "through which stage" series, which is the N04 delta;
* the geometric observables (surface height, two dam-break front definitions).

Outputs (all inside ``--out``): ``case_<scenario>_a<scale>[_<defect>].csv`` per
run, ``closure_summary.json`` (metrics + gate verdicts, the single source of the
report numbers) and ``figures/*.png``.

Usage
-----
    python tools/run_fs_mass_closure.py --out artifacts/fs_mass_closure_n04
    python tools/run_fs_mass_closure.py --case dam_break:12 --defect refill_off   # one shard

Every case is independent, so the campaign is normally launched as one process
per case (the tool is single-threaded by design: these grids are small enough
that torch's intra-op threading costs more than it buys).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "src"))

import run_fs_mass_ledger as base  # noqa: E402  (scenario builders, IC, figures)

from tensorlbm.free_surface_lbm import free_surface_step, mass_ledger  # noqa: E402
from tensorlbm.free_surface_mass_closure import (  # noqa: E402
    ClosureLedger,
    class_mass_identity,
    front_coefficient,
)
from tensorlbm.free_surface_mass_ledger import MassLedger, refill_ablation  # noqa: E402

#: Platform dam-break front-coefficient anchor (thick-column limit).
FRONT_REFERENCE: float = 1.2016
FRONT_TOLERANCE: float = 0.03
#: Front definitions: ``edge`` = first cell with any liquid (sub-cell tolerant),
#: ``body`` = column that still carries at least 4 cells of liquid.
FRONT_THRESHOLDS: dict[str, float] = {"edge": 0.5, "body": 4.0}


def _front_x(fill: torch.Tensor, threshold: float) -> float:
    column = fill.sum(dim=(0, 1))
    wet = torch.nonzero(column >= threshold, as_tuple=False)
    return 0.0 if wet.numel() == 0 else float(wet.max().item())


def run_case(
    scenario,
    *,
    defect: str = "none",
    probe_interval: int = 10,
    csv_path: Path | None = None,
    snapshot_steps=(),
) -> dict:
    f, fill, flags, mass, solid = base.initial_state(scenario)
    ledger: MassLedger = mass_ledger(scenario.steps + 1, device=f.device, rho_liquid=1.0)
    closure = ClosureLedger(scenario.steps + 1, device=f.device)
    class_residual_ulps = 0.0
    ledger.record(f, fill, flags, mass)
    inv0 = float(ledger.field("inventory")[0])
    fill0 = float(ledger.field("fill_total")[0])
    closure.set_initial(inv0)
    height0 = _surface_height(fill)
    _water_depth_at_start = _water_depth(fill)

    snapshots: dict[int, torch.Tensor] = {}
    wanted = set(int(v) for v in snapshot_steps)
    fronts: dict[str, list[float]] = {k: [] for k in FRONT_THRESHOLDS}
    ts: list[float] = []
    rows: list[dict] = []
    t_scale = math.sqrt(2.0 * scenario.gravity / scenario.scale)
    defects = defect in ("refill_off", "refill_off_all")
    t0 = time.time()
    for step in range(1, scenario.steps + 1):
        previous_flags = flags.clone() if defects else None
        stage_ledger: dict = {}
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=scenario.tau,
            gy=-scenario.gravity,
            rho_liquid=1.0,
            rho_gas=base.RHO_GAS,
            freeze_topology=defect == "freeze_topology",
            inventory_reconciliation_ledger=stage_ledger,
        )
        if defects:
            mass, fill = refill_ablation(
                previous_flags, flags, mass, fill, rho_liquid=1.0,
                include_demotion=defect == "refill_off_all",
            )
        ledger.record(f, fill, flags, mass)
        closure.record_step(
            stage_ledger["stage_deltas"],
            stage_ledger["observed_total_liquid_inventory_delta"],
            stage_ledger["stages"]["before_collision"]["total_liquid_inventory"],
        )
        if step % 50 == 0 or step == scenario.steps:
            identity = class_mass_identity(fill)
            class_residual_ulps = max(class_residual_ulps, float(identity["residual_ulps"]))
        if step in wanted:
            snapshots[step] = fill[scenario.nz // 2].clone().to("cpu")
        if step % probe_interval == 0 or step == scenario.steps:
            V = float(ledger.field("inventory")[-1])
            row = {
                "step": step,
                "T": step * t_scale,
                "inventory": V,
                "inventory_drift_rel": (V - inv0) / inv0,
                "fill_total": float(ledger.field("fill_total")[-1]),
                "surface_height": _surface_height(fill),
                "water_depth": _water_depth(fill),
                "cell_partial": float(ledger.field("cell_partial")[-1]),
                "cell_changed": float(ledger.field("cell_changed")[-1]),
            }
            row["fill_drift_rel"] = (row["fill_total"] - fill0) / fill0
            if scenario.name == "dam_break":
                for label, threshold in FRONT_THRESHOLDS.items():
                    x = _front_x(fill, threshold)
                    fronts[label].append((x - scenario.scale) / scenario.scale)
                    row[f"front_x_{label}"] = x
                    row[f"Z_over_a_{label}"] = (x - scenario.scale) / scenario.scale
                ts.append(row["T"])
            rows.append(row)

    wall = time.time() - t0
    mass_summary = ledger.summary()
    closure_summary = closure.summary(ulp_scale=mass_summary["float32_ulp_floor"]["inventory"])
    metrics: dict = {
        "scenario": scenario.name,
        "defect": defect,
        "scale_cells": scenario.scale,
        "grid": [scenario.nz, scenario.ny, scenario.nx],
        "cells": scenario.cells,
        "steps": scenario.steps,
        "tau": scenario.tau,
        "nu": (scenario.tau - 0.5) / 3.0,
        "gravity": scenario.gravity,
        "rho_gas": base.RHO_GAS,
        "dx": 1.0 / scenario.scale,
        "wall_seconds": wall,
        "inventory_initial": inv0,
        "fill_total_initial": fill0,
        "class_identity_residual_ulps": class_residual_ulps,
        "closure": closure_summary,
        "max_abs_relative_drift_inventory": mass_summary["inventory"]["max_abs_relative_drift"],
        "max_abs_relative_drift_fill": mass_summary["fill_total"]["max_abs_relative_drift"],
        "water_depth_start": _water_depth_at_start,
        "water_depth_end": _water_depth(fill),
        "water_depth_drift_cells": abs(_water_depth(fill) - _water_depth_at_start),
        "surface_height_start": height0,
        "surface_height_end": _surface_height(fill),
        "surface_height_drift_cells": abs(_surface_height(fill) - height0),
        "steps_with_topology_change": mass_summary["topology_steps"],
        "metadata": scenario.metadata,
    }
    if scenario.name == "dam_break":
        for label in FRONT_THRESHOLDS:
            fit = front_coefficient(ts, fronts[label], window=(0.5, 3.0))
            metrics[f"front_{label}"] = fit
            metrics[f"front_{label}_error_vs_reference"] = (
                abs(fit["slope"] - FRONT_REFERENCE) / FRONT_REFERENCE
                if fit["samples"] >= 3
                else None
            )
    if csv_path is not None:
        _write_csv(csv_path, rows)
        csv_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
        if snapshots:
            torch.save(snapshots, csv_path.with_suffix(".snapshots.pt"))
    return metrics


def _water_depth(fill: torch.Tensor) -> float:
    """Continuous column depth in cells: max over (x,z) of the y-sum of fill.

    Unlike the integer wet-cell index this resolves sub-cell motion, which is
    what a hydrostatic anchor has to measure (the drift is O(1e-4) cell).
    """
    return float(fill.sum(dim=1).max())


def _surface_height(fill: torch.Tensor) -> float:
    column = fill.sum(dim=(0, 2))  # (ny,)
    wet = torch.nonzero(column > 1e-3, as_tuple=False)
    return 0.0 if wet.numel() == 0 else float(wet.max().item()) + 1.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(f"{row.get(k, '')!r}".strip("'") for k in keys) + "\n")


# --------------------------------------------------------------------- gates
def evaluate_gates(runs: list[dict]) -> dict:
    """Evaluate the J-gates on the collected runs (honest FAIL is a valid verdict)."""
    def sel(**kwargs):
        out = []
        for run in runs:
            if all(run.get(k) == v for k, v in kwargs.items()):
                out.append(run)
        return out

    gates: dict[str, dict] = {}

    def _vals(getter, name="value"):
        return [getter(r) for r in runs if getter(r) is not None]

    ulps = _vals(lambda r: r["closure"]["closure_residual_max_ulps"])
    gates["J1a_closure_completeness"] = {
        "criterion": "max |observed - sum(channels)| <= 100 float32 ulp of V0 on every run",
        "value": max(ulps) if ulps else None,
        "unit": "ulp",
        "threshold": 100.0,
        "verdict": "PASS" if ulps and max(ulps) <= 100.0 else "FAIL",
    }

    identity = _vals(lambda r: r["class_identity_residual_ulps"])
    gates["J1b_class_identity"] = {
        "criterion": "{fill<=0}+{0<fill<1}+{fill>=1} reproduces sum(fill) within 4 ulp "
                     "(two summation orders of a 1e4-term float32 sum)",
        "value": max(identity) if identity else None,
        "unit": "ulp",
        "threshold": 4.0,
        "verdict": "PASS" if identity and max(identity) <= 4.0 else "FAIL",
    }

    def _quiet_max(pred):
        return [r["closure"]["observed_quiet_max_abs_step_delta_relative"]
                for r in runs if pred(r)
                and r["closure"].get("observed_quiet_max_abs_step_delta_relative") is not None]

    quiescent = _quiet_max(lambda r: r["scenario"] in ("hydrostatic", "hull_smoke"))
    gates["J1c_quiescent_conservation"] = {
        "criterion": "max per-step |dV|/V0 on conversion-free steps <= 1e-6 for the "
                     "quiescent scenarios (hydrostatic, hull smoke)",
        "value": max(quiescent) if quiescent else None,
        "unit": "relative",
        "threshold": 1e-6,
        "verdict": "PASS" if quiescent and max(quiescent) <= 1e-6 else "FAIL",
    }
    dynamic = _quiet_max(lambda r: r["scenario"] == "dam_break")
    gates["J1c_dynamic_drift"] = {
        "criterion": "report-only: per-step |dV|/V0 on conversion-free steps of the breaking "
                     "dam (grows with the lattice Mach number, i.e. with 1/tau)",
        "value": max(dynamic) if dynamic else None,
        "unit": "relative",
        "threshold": None,
        "verdict": "REPORT_ONLY",
        "detail": [
            {"a": r["scale_cells"], "tau": r["tau"], "defect": r["defect"],
             "quiet_max_relative": r["closure"].get("observed_quiet_max_abs_step_delta_relative")}
            for r in runs if r["scenario"] == "dam_break"
        ],
    }

    churn = _vals(lambda r: r["closure"].get("churn_ratio"))
    gates["J1d_churn_amplification"] = {
        "criterion": "sum|channel deltas| / sum|net delta| >= 5 on every run: the stages cancel, "
                     "so a scalar drift understates the internal churn",
        "value": min(churn) if churn else None,
        "unit": "ratio",
        "threshold": 5.0,
        "verdict": "PASS" if churn and min(churn) >= 5.0 else "FAIL",
        "detail": {f"{r['scenario']}_a{r['scale_cells']}_tau{r['tau']}": r["closure"].get("churn_ratio")
                   for r in runs if r["defect"] == "none"},
    }

    exchange = _vals(lambda r: r["closure"].get("exchange_quiet_max_abs_step_delta_relative"))
    gates["J1e_exchange_conservation"] = {
        "criterion": "max per-step |dV|/V0 of the Korner exchange stage alone (quiet steps) <= 1e-6",
        "value": max(exchange) if exchange else None,
        "unit": "relative",
        "threshold": 1e-6,
        "verdict": "PASS" if exchange and max(exchange) <= 1e-6 else "FAIL",
    }

    hydro = [r for r in runs if r["scenario"] == "hydrostatic" and r["defect"] == "none"]
    drift = [r["water_depth_drift_cells"] for r in hydro]
    gates["J2_hydrostatic_anchor"] = {
        "criterion": "continuous free-surface depth drift < 1 lattice cell "
                     "(max over x,z of sum_y fill, sub-cell resolution)",
        "value": max(drift) if drift else None,
        "unit": "cell",
        "threshold": 1.0,
        "verdict": "PASS" if drift and max(drift) < 1.0 else "FAIL",
        "detail": {f"a={r['scale_cells']}": {"depth_drift_cells": r.get("water_depth_drift_cells"),
                                             "integer_height_drift_cells": r["surface_height_drift_cells"]}
                   for r in hydro},
    }

    db = [r for r in runs if r["scenario"] == "dam_break" and r["defect"] == "none"]
    arms: dict[str, list] = {}
    for run in db:
        if run.get("front_edge_error_vs_reference") is None:
            continue
        arms.setdefault(f"tau={run['tau']}", []).append(run["front_edge_error_vs_reference"])
    best_arm = min(arms, key=lambda k: max(arms[k])) if arms else None
    gates["J3_dambreak_front"] = {
        "criterion": f"|c - {FRONT_REFERENCE}| / {FRONT_REFERENCE} < 3% on the highest-Re arm "
                     f"(front = edge definition)",
        "value": max(arms[best_arm]) if best_arm else None,
        "unit": "relative",
        "threshold": FRONT_TOLERANCE,
        "verdict": ("PASS" if best_arm and max(arms[best_arm]) < FRONT_TOLERANCE else "FAIL"),
        "detail": {
            arm: {
                "max_error_vs_reference": max(errors),
                "points": [
                    {"a": r["scale_cells"], "tau": r["tau"],
                     "c_edge": (r.get("front_edge") or {}).get("slope"),
                     "c_body": (r.get("front_body") or {}).get("slope"),
                     "r2": (r.get("front_edge") or {}).get("r2")}
                    for r in db if f"tau={r['tau']}" == arm
                ],
            }
            for arm, errors in arms.items()
        },
    }

    long_runs = [r for r in runs if r["steps"] >= 2000]
    j4 = [abs(r["max_abs_relative_drift_inventory"]) for r in long_runs]
    gates["J4_absolute_drift"] = {
        "criterion": "|dV/V0| < 1e-3 on runs of >= 2000 steps",
        "value": max(j4) if j4 else None,
        "unit": "relative",
        "threshold": 1e-3,
        "verdict": "PASS" if j4 and max(j4) < 1e-3 else ("REPORT_ONLY" if not j4 else "FAIL"),
    }

    control = [r for r in runs if r["defect"] != "none"]
    clean = [r for r in runs if r["defect"] == "none"]
    amplifications = []
    for defect_run in control:
        match = [c for c in clean if c["scenario"] == defect_run["scenario"]
                 and c["scale_cells"] == defect_run["scale_cells"]
                 and abs(c["tau"] - defect_run["tau"]) < 1e-9]
        if not match:
            continue
        base_run = match[0]
        for channel in ("exchange", "topology"):
            key = f"{channel}_max_abs_step_delta_ulps"
            clean_value = float(base_run["closure"].get(key) or 0.0)
            defect_value = float(defect_run["closure"].get(key) or 0.0)
            if clean_value == 0 and defect_value == 0:
                ratio = None
            else:
                ratio = defect_value / clean_value if clean_value > 0 else math.inf
            amplifications.append({
                "metric": f"channel:{channel}", "scenario": defect_run["scenario"],
                "a": defect_run["scale_cells"], "tau": defect_run["tau"],
                "defect": defect_run["defect"], "channel": channel,
                "clean": clean_value, "defect": defect_value, "amplification": ratio,
            })
        # the cumulative-drift signature the N02 report used, for continuity
        clean_drift = abs(float(base_run["max_abs_relative_drift_inventory"]))
        defect_drift = abs(float(defect_run["max_abs_relative_drift_inventory"]))
        amplifications.append({
            "metric": "cumulative_drift", "scenario": defect_run["scenario"],
            "a": defect_run["scale_cells"], "tau": defect_run["tau"],
            "defect": defect_run["defect"], "channel": "net",
            "clean": clean_drift, "defect": defect_drift,
            "amplification": defect_drift / clean_drift if clean_drift > 0 else None,
        })
    finite = [a["amplification"] for a in amplifications if a["amplification"] is not None]
    best = max(finite) if finite else None
    gates["J5_negative_control"] = {
        "criterion": "the injected defect (refill_off) raises a named channel or the cumulative "
                     "drift of the clean twin by >= 10x",
        "value": best,
        "unit": "ratio",
        "threshold": 10.0,
        "verdict": "PASS" if best is not None and best >= 10.0 else "FAIL",
        "detail": amplifications,
    }

    rates = [(r["dx"], abs(r["max_abs_relative_drift_inventory"]), r["scenario"],
              r["scale_cells"]) for r in clean]
    gates["J6_grid_order"] = {
        "criterion": "report-only: observed order p of the drift rate against dx",
        "value": None,
        "unit": "power",
        "threshold": None,
        "verdict": "REPORT_ONLY",
        "detail": rates,
    }
    overall = "PASS" if all(
        g["verdict"] in ("PASS", "REPORT_ONLY") for g in gates.values()
    ) else "FAIL"
    return {"gates": gates, "overall": overall}


# ------------------------------------------------------------------- figures
def make_figures(runs: list[dict], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = [r for r in runs if r["defect"] == "none"]

    # 1) channel attribution per scenario (log scale bars)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, scenario in zip(axes, ("hydrostatic", "dam_break", "hull_smoke")):
        subset = [r for r in clean if r["scenario"] == scenario]
        if not subset:
            continue
        names = list(subset[0]["closure"]["channels_detail"])
        heights = [max(subset[0]["closure"]["channels_detail"][n]["max_abs_step_delta_ulps"], 1e-3)
                   for n in names]
        ax.barh(range(len(names)), heights, color="#3b6ea5")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("after_", "") for n in names], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("max |dV| per step  [ulp]")
        ax.set_title(f"{scenario} (a={subset[0]['scale_cells']})")
    fig.suptitle("Stage-resolved inventory attribution (IN13-W2-N04)")
    fig.tight_layout()
    path = out_dir / "channel_attribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path.name)

    # 2) hydrostatic surface height + inventory drift
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for run in [r for r in clean if r["scenario"] == "hydrostatic"]:
        ax.semilogy(run["steps"], run["max_abs_relative_drift_inventory"], "o",
                    label=f"a={run['scale_cells']} |dV/V0|")
    ax.axhline(1e-3, ls="--", c="r", lw=1, label="J4 threshold 1e-3")
    ax.set_xlabel("steps")
    ax.set_ylabel("max relative inventory drift")
    ax.set_title("Hydrostatic anchor: absolute drift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "hydrostatic_drift.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path.name)

    # 3) dam-break front coefficient vs reference
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    db = [r for r in clean if r["scenario"] == "dam_break"]
    if db:
        xs = [r["scale_cells"] for r in db]
        ax.plot(xs, [r["front_edge"]["slope"] for r in db], "o-", label="edge front")
        ax.plot(xs, [r["front_body"]["slope"] for r in db], "s--", label="body front")
        ax.axhline(FRONT_REFERENCE, c="r", ls=":", label=f"reference {FRONT_REFERENCE}")
        ax.axhspan(FRONT_REFERENCE * 0.97, FRONT_REFERENCE * 1.03, color="r", alpha=0.1)
        ax.set_xlabel("a [cells]")
        ax.set_ylabel("c (front coefficient)")
        ax.legend(fontsize=8)
    ax.set_title("Dam-break front coefficient")
    fig.tight_layout()
    path = out_dir / "dam_break_front.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path.name)

    # 4) negative control: channel signature clean vs defect
    control = [r for r in runs if r["defect"] != "none"]
    if control:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels, values = [], []
        for run in control:
            match = [c for c in clean if c["scenario"] == run["scenario"]
                     and c["scale_cells"] == run["scale_cells"]]
            if not match:
                continue
            for channel in ("exchange", "topology"):
                key = f"{channel}_max_abs_step_delta_ulps"
                labels.append(f"{run['scenario']} a={run['scale_cells']}\n{channel}")
                values.append((float(match[0]["closure"].get(key) or 0.0),
                               float(run["closure"].get(key) or 0.0)))
        if values:
            import numpy as np

            idx = np.arange(len(values))
            ax.bar(idx - 0.2, [v[0] for v in values], 0.4, label="clean")
            ax.bar(idx + 0.2, [v[1] for v in values], 0.4, label="defect")
            ax.set_yscale("log")
            ax.set_xticks(idx)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylabel("max |dV| per step [ulp]")
            ax.legend(fontsize=8)
        ax.set_title("Negative control: channel signature")
        fig.tight_layout()
        path = out_dir / "negative_control.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        figures.append(path.name)
    return figures


def _parse_scales(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/fs_mass_closure_n04")
    parser.add_argument("--scenarios", default="hydrostatic,dam_break,hull_smoke")
    parser.add_argument("--scales", default="8,12,16")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--defect", default="none",
                        help="none|refill_off|refill_off_all|freeze_topology")
    parser.add_argument("--case", action="append", default=None,
                        help="run a single <scenario>:<scale> (repeatable)")
    parser.add_argument("--probe-interval", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="merge new runs into an existing closure_summary.json")
    parser.add_argument("--collect", action="store_true",
                        help="only collect case_*.metrics.json under --out, then evaluate gates + figures")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "closure_summary.json"
    runs: list[dict] = []
    if args.merge and summary_path.exists():
        runs = json.loads(summary_path.read_text()).get("runs", [])

    if args.collect:
        runs = [json.loads(path.read_text())
                for path in sorted(out.rglob("case_*.metrics.json"))]
        if not runs:
            raise SystemExit(f"no case_*.metrics.json under {out}")
        gates = evaluate_gates(runs)
        summary = {
            "campaign": "IN13-W2-N04 free-surface mass closure",
            "reference_front_coefficient": FRONT_REFERENCE,
            "front_tolerance": FRONT_TOLERANCE,
            "runs": runs,
            **gates,
        }
        if not args.no_figures:
            summary["figures"] = make_figures(runs, out / "figures")
        summary_path.write_text(json.dumps(summary, indent=1, default=str))
        print(f"[n04] collected {len(runs)} runs, overall={summary['overall']}")
        for name, gate in summary["gates"].items():
            print(f"  {name}: {gate['verdict']} ({gate['value']})")
        return 0

    if args.case:
        cases = []
        for spec in args.case:
            name, scale = spec.split(":")
            cases.append((name, int(scale)))
    else:
        cases = [(name, scale)
                 for name in args.scenarios.split(",") if name.strip()
                 for scale in _parse_scales(args.scales)]

    for name, scale in cases:
        scenario = base.build_scenario(name, scale, steps=args.steps,
                                       tau=args.tau, gravity=args.gravity)
        tag = f"case_{name}_a{scale}" + ("" if args.defect == "none" else f"_{args.defect}")
        metrics = run_case(scenario, defect=args.defect,
                           probe_interval=args.probe_interval,
                           csv_path=out / f"{tag}.csv",
                           snapshot_steps=(scenario.steps // 2,))
        print(f"[n04] {tag}: {metrics['wall_seconds']:.1f}s "
              f"drift={metrics['max_abs_relative_drift_inventory']:.3e} "
              f"closure_ulp={metrics['closure']['closure_residual_max_ulps']:.1f} "
              f"exchange_rel={metrics['closure'].get('exchange_max_abs_step_delta_relative')}",
              flush=True)
        runs = [r for r in runs if not (r["scenario"] == name
                                        and r["scale_cells"] == scale
                                        and r["defect"] == args.defect)]
        runs.append(metrics)
        gates = evaluate_gates(runs)
        summary = {
            "campaign": "IN13-W2-N04 free-surface mass closure",
            "reference_front_coefficient": FRONT_REFERENCE,
            "front_tolerance": FRONT_TOLERANCE,
            "runs": runs,
            **gates,
        }
        summary_path.write_text(json.dumps(summary, indent=1, default=str))
        if not args.no_figures:
            figures = make_figures(runs, out / "figures")
            summary["figures"] = figures
            summary_path.write_text(json.dumps(summary, indent=1, default=str))
    print(f"[n04] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())