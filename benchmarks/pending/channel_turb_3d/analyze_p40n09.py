#!/usr/bin/env python3
"""P40-N09 analysis: decompose the channel u+ error into statistics and bias.

Reads the campaign arms written by :mod:`run_p40n09` and produces

* ``p40n09_numbers.json`` -- every number quoted by the report (zero hand copy),
* ``gates_p40n09.json``     -- the six machine-checkable gates, fail-closed,
* ``figures_p40n09/*.png``  -- the report figures.

The gates (declared in ``p40n09_prereg.json`` before the runs):

======================================  =========================================
``G_A_steady_state_half_window``        two adjacent half windows of the u+
                                        profile agree within 1 u+ unit
``G_B_statistical_ci_3pct``             the 95 % batch-means CI half-width of u+
                                        is <= 3 % (per point, comparison window)
``G_C_domain_length_3pct``              Lx = 2 pi delta vs 4 pi delta on the same
                                        grid agree within 3 % (relative L2)
``G_D_attribution_verdict``             binary verdict: the part of the error
                                        outside U_SN is either below the 3 %
                                        scale (statistics dominated) or not
                                        (model / domain dominated)
``G_E_negative_control_sgs``            the C_s = 0 / C_s = 0.1 u+ profiles differ
                                        by more than the pair's combined CI
``G_F_budget_extrapolation``            err(T_obs) curve measured + a machine-time
                                        forecast for the MKM box
======================================  =========================================

Fail-closed rules: an arm that is missing -> ``withheld``; averaging below three
eddy turnovers -> ``WITHHELD`` (the CI is not estimable); a gate whose evidence
file is absent is never reported as passed.

Units: ``u+ = u / u_tau`` with ``u_tau = sqrt(a h / 2)`` (the exact momentum
balance of the forced channel, independent of the profile), ``y+ = (y - y_w)
u_tau / nu`` with ``y_w`` measured on the laminar twin at the *same* viscosity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from tensorlbm.validation.channel_wall_units import fold_channel_profile
from tensorlbm.validation.statistical_error import (
    batch_means_summary,
    budget_extrapolation,
    convergence_eligibility,
    decompose_error,
    domain_length_check,
    half_window_max_diff,
    metric_from_batches,
    negative_control_check,
    relative_l2_error,
)

REF_CSV = Path(__file__).resolve().parent / "dns_ref" / "mkm180.csv"
# P20-N01 laminar calibration of the production wall format (cumulant / natural
# KBC agree to 0.002 cell): wall stress anchored estimate at tau = 0.50155.
P20N01_WALL_ORIGIN = 0.6437
COMPARE_WINDOW = (10.0, 170.0)  # y+ window of the MKM comparison (buffer+log+outer)
MKMB_BOX_CELLS = 400 * 64 * 132  # MKM Re_tau=180 box quoted by P20-N01
MKM_STEPS = 120_000


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 1], data[:, 2]


def load_arm(run_dir: Path, case: str) -> dict | None:
    result = run_dir / f"{case}_result.json"
    meta = run_dir / f"{case}_meta.json"
    if not result.exists():
        return None
    payload = {
        "case": case,
        "result": json.loads(result.read_text()),
        "meta": json.loads(meta.read_text()) if meta.exists() else {},
        "result_path": str(result),
        "meta_path": str(meta) if meta.exists() else None,
    }
    return payload


def wall_units_for_arm(arm: dict, y_w: float) -> dict:
    """Fold the mean and every batch onto the lower half channel in wall units."""
    result = arm["result"]
    config = result["configuration"]
    stats = result["statistics"]
    # u_tau from the exact momentum balance tau_w = a h / 2 of the forced channel
    derived = result["derived"]
    u_tau = math.sqrt(
        float(derived["body_force_acceleration"]) * float(derived["height"]) / 2.0
    )
    nu = float(derived["nu"])
    y, u, fold_info = fold_channel_profile(stats["mean_velocity_profile"], y_w=y_w)
    y_plus = y * u_tau / nu
    u_plus = u / u_tau

    batches = stats.get("profile_batch_moment_means") or []
    counts = stats.get("profile_batch_counts") or []
    batch_u_plus = []
    for index, moment_mean in enumerate(batches):
        if counts and counts[index] == 0:
            continue
        _, u_b, _ = fold_channel_profile(moment_mean[0], y_w=y_w)
        batch_u_plus.append((u_b / u_tau).tolist())
    return {
        "u_tau": u_tau,
        "nu": nu,
        "re_tau": float(config["re_tau"]),
        "y_w": y_w,
        "y": y.tolist(),
        "y_plus": y_plus.tolist(),
        "u_plus": u_plus.tolist(),
        "u_plus_batches": batch_u_plus,
        "batch_counts": counts,
        "fold_info": fold_info,
        "averaging_eddy_turnovers": float(
            arm["meta"].get("plan", {}).get(
                "averaging_eddy_turnovers",
                stats.get("statistics_eddy_turnovers", 0.0) / 2.0,
            )
        ),
        "t_int_ledger": arm["meta"].get("t_int_ledger"),
        "cell_updates_per_s": arm["meta"].get("cell_updates_per_s"),
        "nx": arm["meta"].get("plan", {}).get("nx"),
        "ny": arm["meta"].get("plan", {}).get("ny"),
        "nz": arm["meta"].get("plan", {}).get("nz"),
        "delta": arm["meta"].get("plan", {}).get("delta"),
        "dy_plus": arm["meta"].get("plan", {}).get("dy_plus"),
        "box_over_delta": arm["meta"].get("plan", {}).get("streamwise_over_delta"),
        "elapsed_s": arm["meta"].get("elapsed_s"),
        "cells": arm["meta"].get("plan", {}).get("cells"),
        "eddy_turnover_steps": arm["meta"].get("plan", {}).get("eddy_turnover_steps"),
        "stationarity_half_window_drift_fraction": stats.get(
            "stationarity_half_window_drift_fraction"
        ),
        "crossflow_rms_over_u_tau": stats.get("recent_crossflow_rms_over_u_tau"),
    }


def _gate(name: str, status: str, evidence: list, problems: list | None = None) -> dict:
    return {
        "gate": name,
        "status": status,  # "pass" | "fail" | "withheld"
        "passed": status == "pass",
        "evidence": evidence,
        "problems": problems or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs_p40n09"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures_p40n09"))
    parser.add_argument("--wall-origin", type=float, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    ref_y_plus, ref_u_plus = load_reference(REF_CSV)

    # ---- wall origin: prefer the laminar twin measured at this viscosity
    wall_origin_source = "p20n01_laminar_table"
    y_w = P20N01_WALL_ORIGIN
    lam = load_arm(args.run_dir, "LAM_yw")
    if args.wall_origin is not None:
        y_w = float(args.wall_origin)
        wall_origin_source = "cli_override"
    elif lam is not None and lam["meta"].get("laminar_wall_origin"):
        y_w = float(lam["meta"]["laminar_wall_origin"]["wall_position_low"])
        wall_origin_source = "laminar_twin_this_campaign"

    arms = {}
    for case in ("D4_cs0", "D2_cs0", "SGS_cs0", "SGS_cs01"):
        arm = load_arm(args.run_dir, case)
        if arm is not None:
            arms[case] = wall_units_for_arm(arm, y_w)

    numbers: dict = {
        "node": "P40-N09",
        "wall_origin": {"value": y_w, "source": wall_origin_source},
        "reference": str(REF_CSV),
        "compare_window_y_plus": list(COMPARE_WINDOW),
        "arms": {},
    }
    gates: list[dict] = []

    def summarise(case: str, arm: dict) -> dict:
        batches = np.asarray(arm["u_plus_batches"], dtype=float)
        y_plus = np.asarray(arm["y_plus"], dtype=float)
        u_plus = np.asarray(arm["u_plus"], dtype=float)
        re_tau_arm = float(arm.get("re_tau") or 180.0)
        if re_tau_arm < 179.0:
            # off-design arms (the SGS negative control) live at a lower Re_tau,
            # so the MKM comparison window does not apply; use the whole profile.
            window = (y_plus >= 0.5) & (y_plus <= 0.95 * re_tau_arm)
        else:
            window = (y_plus >= COMPARE_WINDOW[0]) & (y_plus <= COMPARE_WINDOW[1])
        entry = {
            "y_plus": arm["y_plus"],
            "u_plus": arm["u_plus"],
            "n_batches": int(batches.shape[0]) if batches.size else 0,
            "averaging_eddy_turnovers": arm["averaging_eddy_turnovers"],
            "t_int_ledger": arm["t_int_ledger"],
            "stationarity_half_window_drift_fraction": arm[
                "stationarity_half_window_drift_fraction"
            ],
            "crossflow_rms_over_u_tau": arm["crossflow_rms_over_u_tau"],
            "cell_updates_per_s": arm["cell_updates_per_s"],
            "cells": arm["cells"],
            "nx": arm.get("nx"),
            "ny": arm.get("ny"),
            "nz": arm.get("nz"),
            "delta": arm.get("delta"),
            "dy_plus": arm.get("dy_plus"),
            "box_over_delta": arm.get("box_over_delta"),
            "u_tau": arm.get("u_tau"),
            "nu": arm.get("nu"),
            "elapsed_s": arm["elapsed_s"],
            "eddy_turnover_steps": arm["eddy_turnover_steps"],
            "window_mask": window.tolist(),
            "re_tau": arm.get("re_tau"),
        }
        mkm_comparable = abs(float(arm.get("re_tau") or 0.0) - 180.0) < 1.0e-9
        if batches.size:
            ci = batch_means_summary(batches)
            rel = np.asarray(ci["ci_half_relative"], dtype=float)
            finite = np.isfinite(rel) & window
            entry["batch_ci"] = {
                "n_batches": ci["n_batches"],
                "dof": ci["dof"],
                "t_critical": ci["t_critical"],
                "max_ci_half": ci["max_ci_half"],
                "max_ci_half_relative_window": float(np.max(rel[finite])) if finite.any() else None,
                "mean_ci_half_relative_window": float(np.mean(rel[finite])) if finite.any() else None,
                "ci_half": ci["ci_half"],
                "ci_half_relative": ci["ci_half_relative"],
            }
            gate = half_window_max_diff(batches)
            entry["half_window"] = {
                "max_diff_u_plus": gate["half_window_max_diff"],
                "rms_diff_u_plus": gate["half_window_rms_diff"],
                "max_consecutive_batch_diff_u_plus": gate["max_consecutive_batch_diff"],
                "split_index": gate["split_index"],
            }
            ref_on_grid = np.interp(y_plus, ref_y_plus, ref_u_plus)
            metric = metric_from_batches(batches, ref_on_grid)
            if mkm_comparable:
                entry["mkm"] = {
                    "relative_l2": relative_l2_error(u_plus, ref_on_grid),
                    "error_mean": metric["error_mean"],
                    "error_ci_half": metric["error_ci_half"],
                    "error_ci_low": metric["error_ci_low"],
                    "error_ci_high": metric["error_ci_high"],
                    "u_sn": metric["u_sn"],
                    "per_batch_relative_l2": metric["per_batch"],
                    "max_pointwise_ci_half_u_plus": metric["pointwise_max_ci_half"],
                    "residual_u_plus": (u_plus - ref_on_grid).tolist(),
                    "pointwise_ci_half_u_plus": metric["pointwise_ci_half"],
                    "window_relative_l2": relative_l2_error(
                        u_plus[window], ref_on_grid[window]
                    )
                    if window.any()
                    else None,
                }
        return entry

    for case, arm in arms.items():
        numbers["arms"][case] = summarise(case, arm)

    # ---------------------------------------------------------------- gate A/B
    for case in ("D4_cs0", "D2_cs0"):
        entry = numbers["arms"].get(case)
        if entry is None or "half_window" not in entry:
            gates.append(
                _gate(
                    f"G_A_steady_state_half_window[{case}]",
                    "withheld",
                    [{"v": None, "f": str(args.run_dir / f'{case}_result.json'), "k": "missing arm"}],
                    ["arm absent or carries no batch record"],
                )
            )
            gates.append(
                _gate(f"G_B_statistical_ci_3pct[{case}]", "withheld", [], ["arm absent"])
            )
            continue
        eligibility = convergence_eligibility(entry["averaging_eddy_turnovers"])
        diff = entry["half_window"]["max_diff_u_plus"]
        status = "pass" if diff <= 1.0 else "fail"
        if not eligibility["eligible"]:
            status = "withheld"
        gates.append(
            _gate(
                f"G_A_steady_state_half_window[{case}]",
                status,
                [
                    {
                        "v": diff,
                        "f": f"{case}:half_window.max_diff_u_plus",
                        "k": "u+ units",
                        "ok": diff <= 1.0,
                    },
                    {
                        "v": entry["averaging_eddy_turnovers"],
                        "f": f"{case}:averaging_eddy_turnovers",
                        "ok": eligibility["eligible"],
                    },
                ],
                [] if status == "pass" else [eligibility["reason"]],
            )
        )
        rel = entry.get("batch_ci", {}).get("max_ci_half_relative_window")
        if rel is None:
            gates.append(
                _gate(f"G_B_statistical_ci_3pct[{case}]", "withheld", [], ["no CI"])
            )
        else:
            status = "pass" if rel <= 0.03 else "fail"
            if not eligibility["eligible"]:
                status = "withheld"
            gates.append(
                _gate(
                    f"G_B_statistical_ci_3pct[{case}]",
                    status,
                    [
                        {
                            "v": rel,
                            "f": f"{case}:batch_ci.max_ci_half_relative_window",
                            "ok": rel <= 0.03,
                        },
                        {
                            "v": entry["batch_ci"]["max_ci_half"],
                            "f": f"{case}:batch_ci.max_ci_half",
                            "k": "u+ units",
                            "ok": None,
                        },
                        {
                            "v": entry["averaging_eddy_turnovers"],
                            "f": f"{case}:averaging_eddy_turnovers",
                            "ok": eligibility["eligible"],
                        },
                    ],
                    [] if status == "pass" else [f"CI half-width {rel:.4%} vs 3 % gate"],
                )
            )

    # ---------------------------------------------------------------- gate C/D
    domain_result = None
    d4 = numbers["arms"].get("D4_cs0")
    d2 = numbers["arms"].get("D2_cs0")
    if d4 and d2 and "batch_ci" in d4 and "batch_ci" in d2:
        domain_result = domain_length_check(
            d2["u_plus"],
            d4["u_plus"],
            short_ci_half=d2["batch_ci"]["ci_half"],
            long_ci_half=d4["batch_ci"]["ci_half"],
            tolerance_relative=0.03,
        )
        numbers["domain_length"] = domain_result
        status = "pass" if domain_result["within_tolerance"] else "fail"
        if not (convergence_eligibility(d4["averaging_eddy_turnovers"])["eligible"]):
            status = "withheld"
        gates.append(
            _gate(
                "G_C_domain_length_3pct",
                status,
                [
                    {
                        "v": domain_result["relative_l2_difference"],
                        "f": "domain_length.relative_l2_difference",
                        "ok": domain_result["within_tolerance"],
                    },
                    {
                        "v": domain_result["max_abs_difference"],
                        "f": "domain_length.max_abs_difference",
                        "k": "u+ units",
                        "ok": None,
                    },
                    {
                        "v": domain_result["resolved"],
                        "f": "domain_length.resolved",
                        "k": "difference exceeds the combined CI",
                        "ok": None,
                    },
                ],
                [] if status == "pass" else ["box effect exceeds the 3 % gate"],
            )
        )
    else:
        gates.append(
            _gate("G_C_domain_length_3pct", "withheld", [], ["one of the domain arms is absent"])
        )

    attribution = None
    if d4 and "mkm" in d4:
        metric_like = {
            "error_mean": d4["mkm"]["error_mean"],
            "error_ci_low": d4["mkm"]["error_ci_low"],
            "error_ci_high": d4["mkm"]["error_ci_high"],
            "u_sn": d4["mkm"]["u_sn"],
        }
        domain_shift = None
        if domain_result is not None and domain_result["resolved"]:
            domain_shift = domain_result["relative_l2_difference"]
        ref_on_grid = np.interp(
            np.asarray(d4["y_plus"]), ref_y_plus, ref_u_plus
        )
        attribution = decompose_error(
            np.asarray(d4["u_plus"]),
            ref_on_grid,
            metric_like,
            tolerance_relative=0.03,
            domain_shift_relative=domain_shift,
        )
        numbers["attribution"] = attribution
        # Fail-closed: an attribution computed on a window that has not reached
        # the steady state gate is not admitted -- the batch scatter would mix a
        # transient into what is supposed to be a statistical error.
        steady_gate = None
        for gate in gates:
            if gate["gate"].startswith("G_A_steady_state_half_window[D4_cs0]"):
                steady_gate = gate
        attribution_status = "pass"
        attribution_problems: list[str] = []
        if steady_gate is None or steady_gate["status"] != "pass":
            attribution_status = "withheld"
            attribution_problems.append(
                "D4_cs0 failed (or was withheld by) the steady-state gate; "
                "the statistical part cannot be separated from a residual transient"
            )
        gates.append(
            _gate(
                "G_D_attribution_verdict",
                attribution_status,  # the gate asks for a *binary* verdict, not for a small error
                [
                    {
                        "v": attribution["observed_relative_l2"],
                        "f": "attribution.observed_relative_l2",
                        "ok": None,
                    },
                    {"v": attribution["u_sn"], "f": "attribution.u_sn", "ok": None},
                    {
                        "v": attribution["residual_beyond_ci"],
                        "f": "attribution.residual_beyond_ci",
                        "ok": None,
                    },
                    {
                        "v": attribution["verdict"],
                        "f": "attribution.verdict",
                        "ok": attribution["verdict"]
                        in {"statistics_dominated", "model_or_domain_dominated"},
                    },
                ],
                attribution_problems,
            )
        )
    else:
        gates.append(
            _gate("G_D_attribution_verdict", "withheld", [], ["no reference arm with CI"])
        )

    # ---------------------------------------------------------------- gate E
    sgs0 = numbers["arms"].get("SGS_cs0")
    sgs1 = numbers["arms"].get("SGS_cs01")
    if sgs0 and sgs1 and "batch_ci" in sgs0 and "batch_ci" in sgs1:
        neg = negative_control_check(
            sgs0["u_plus"],
            sgs1["u_plus"],
            control_ci_half=sgs0["batch_ci"]["ci_half"],
            variant_ci_half=sgs1["batch_ci"]["ci_half"],
        )
        neg["control_case"] = "SGS_cs0"
        neg["variant_case"] = "SGS_cs01"
        numbers["negative_control"] = neg
        gates.append(
            _gate(
                "G_E_negative_control_sgs",
                "pass" if neg["separated"] else "fail",
                [
                    {
                        "v": neg["max_abs_difference"],
                        "f": "negative_control.max_abs_difference",
                        "k": "u+ units",
                        "ok": neg["separated"],
                    },
                    {
                        "v": neg.get("max_combined_ci_half"),
                        "f": "negative_control.max_combined_ci_half",
                        "k": "u+ units",
                        "ok": None,
                    },
                    {
                        "v": neg.get("separation_in_ci_units"),
                        "f": "negative_control.separation_in_ci_units",
                        "ok": neg["separated"],
                    },
                ],
                [] if neg["separated"] else ["C_s pair not separated beyond its CI"],
            )
        )
    else:
        gates.append(
            _gate(
                "G_E_negative_control_sgs",
                "withheld",
                [],
                ["no stable cumulant arm pair: the SGS separation could not be run"],
            )
        )

    # ---------------------------------------------------------------- gate F
    budget = None
    if d4 and "mkm" in d4 and d4.get("batch_ci"):
        batches = np.asarray(arms["D4_cs0"]["u_plus_batches"], dtype=float)
        y_plus = np.asarray(d4["y_plus"], dtype=float)
        ref_on_grid = np.interp(y_plus, ref_y_plus, ref_u_plus)
        total = batches.shape[0]
        turnover = d4["averaging_eddy_turnovers"] / total
        curve = []
        ks = sorted({max(4, int(total * frac)) for frac in (0.125, 0.25, 0.5, 0.75, 1.0)})
        for k in ks:
            partial = batches[:k].mean(axis=0)
            curve.append(
                {
                    "batches": int(k),
                    "t_obs_turnovers": k * turnover,
                    "relative_error": relative_l2_error(partial, ref_on_grid),
                }
            )
        numbers["error_vs_t_obs"] = curve
        budget = budget_extrapolation(
            [point["t_obs_turnovers"] for point in curve],
            [point["relative_error"] for point in curve],
            target_relative=0.03,
            throughput_cell_updates_per_s=d4.get("cell_updates_per_s"),
            reference_box_cells=MKMB_BOX_CELLS,
            reference_box_steps=MKM_STEPS,
            own_grid_cells=d4.get("cells"),
            eddy_turnover_steps=d4.get("eddy_turnover_steps"),
        )
        numbers["budget"] = budget
        complete = (
            len(curve) >= 2
            and "reference_box" in budget
            and (
                budget.get("predicted_turnovers_to_target") is not None
                or budget.get("extrapolation_refused_reason") is not None
            )
        )
        gates.append(
            _gate(
                "G_F_budget_extrapolation",
                "pass" if complete else "fail",
                [
                    {
                        "v": [point["relative_error"] for point in curve],
                        "f": "error_vs_t_obs[].relative_error",
                        "ok": None,
                    },
                    {
                        "v": budget["fitted_exponent"],
                        "f": "budget.fitted_exponent",
                        "ok": None,
                    },
                    {
                        "v": budget.get("predicted_turnovers_to_target"),
                        "f": "budget.predicted_turnovers_to_target",
                        "ok": None,
                    },
                    {
                        "v": budget.get("extrapolation_refused_reason"),
                        "f": "budget.extrapolation_refused_reason",
                        "ok": None,
                    },
                    {
                        "v": budget.get("reference_box", {}).get("predicted_hours"),
                        "f": "budget.reference_box.predicted_hours",
                        "ok": None,
                    },
                ],
                [] if len(curve) >= 2 else ["not enough windows to fit a trend"],
            )
        )
    else:
        gates.append(
            _gate("G_F_budget_extrapolation", "withheld", [], ["no reference arm"])
        )

    numbers["gates"] = gates
    numbers["verdict"] = (
        "pass"
        if all(gate["status"] == "pass" for gate in gates)
        else ("withheld" if any(gate["status"] == "withheld" for gate in gates) else "fail")
    )
    gates_path = args.out_dir / "gates_p40n09.json"
    gates_path.write_text(
        json.dumps(
            {"node": "P40-N09", "verdict": numbers["verdict"], "gates": gates},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    numbers_path = args.out_dir / "p40n09_numbers.json"
    numbers_path.write_text(
        json.dumps(numbers, indent=2, allow_nan=False, default=float) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------- figures
    try:
        make_figures(numbers, ref_y_plus, ref_u_plus, args.fig_dir)
    except Exception as exc:  # figures must never block the numbers
        print(f"[figures] failed: {type(exc).__name__}: {exc}", flush=True)

    # ---------------------------------------------------------------- console
    print(json.dumps({"verdict": numbers["verdict"]}, indent=2))
    for gate in gates:
        print(f"{gate['status']:>9}  {gate['gate']}")
        for item in gate["evidence"]:
            print(f"            {item['f']} = {item['v']}")
    if attribution:
        print(
            "attribution:",
            json.dumps(
                {
                    key: attribution[key]
                    for key in (
                        "observed_relative_l2",
                        "u_sn",
                        "residual_beyond_ci",
                        "residual_in_u_sn_units",
                        "domain_shift_relative",
                        "unexplained_model_relative",
                        "verdict",
                    )
                },
                indent=2,
            ),
        )
    return 0


def make_figures(numbers: dict, ref_y_plus, ref_u_plus, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = numbers["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=140)
    mask_ref = (ref_y_plus >= 1.0) & (ref_y_plus <= 180.0)
    for axis, logscale in zip(axes, (False, True)):
        axis.plot(
            ref_y_plus[mask_ref],
            ref_u_plus[mask_ref],
            "k-",
            lw=1.6,
            label="MKM 1999 (Re_tau=180)",
        )
        for case, style in (("D4_cs0", "C0-o"), ("D2_cs0", "C3-s"), ("SGS_cs0", "C2-^"), ("SGS_cs01", "C4-v")):
            entry = arms.get(case)
            if not entry:
                continue
            yp = np.asarray(entry["y_plus"])
            up = np.asarray(entry["u_plus"])
            label = case
            if "batch_ci" in entry:
                ci = np.asarray(entry["batch_ci"]["ci_half"])
                axis.errorbar(
                    yp,
                    up,
                    yerr=ci,
                    fmt=style,
                    ms=4,
                    lw=1.2,
                    capsize=2,
                    label=f"{label} (95 % CI)",
                )
            else:
                axis.plot(yp, up, style, ms=4, lw=1.2, label=label)
        axis.set_xlabel("y+")
        axis.set_ylabel("u+")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        if logscale:
            axis.set_xscale("log")
            axis.set_xlim(1.0, 200.0)
            axis.set_title("log-linear (u+ vs ln y+)")
        else:
            axis.set_title("linear")
    fig.suptitle("P40-N09 channel u+ profiles with batch-means 95 % CI")
    fig.tight_layout()
    fig.savefig(fig_dir / "u_plus_ci.png")
    plt.close(fig)

    if "error_vs_t_obs" in numbers:
        curve = numbers["error_vs_t_obs"]
        budget = numbers.get("budget", {})
        fig, axis = plt.subplots(figsize=(6.2, 4.4), dpi=140)
        t = np.asarray([point["t_obs_turnovers"] for point in curve])
        e = np.asarray([point["relative_error"] for point in curve])
        axis.loglog(t, e, "o-", label="measured err(T_obs)")
        if budget:
            tt = np.linspace(t.min(), max(t.max(), budget.get("predicted_turnovers_to_target") or t.max()) * 1.2, 100)
            fit = budget["fitted_coefficient"] * tt ** (-budget["fitted_exponent"])
            axis.loglog(tt, fit, "k--", label=f"fit err = {budget['fitted_coefficient']:.3f} T^-{budget['fitted_exponent']:.2f}")
            if budget.get("predicted_turnovers_to_target"):
                axis.axvline(
                    budget["predicted_turnovers_to_target"],
                    color="C3",
                    ls=":",
                    label="target 3 % crossover",
                )
        axis.axhline(0.03, color="C7", ls="-.", lw=1.0, label="3 % gate")
        axis.set_xlabel("T_obs [delta/u_tau]")
        axis.set_ylabel("relative L2 error vs MKM")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)
        axis.set_title("P40-N09 error vs averaging window")
        fig.tight_layout()
        fig.savefig(fig_dir / "error_vs_t_obs.png")
        plt.close(fig)

    if "domain_length" in numbers or "negative_control" in numbers:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=140)
        d4 = arms.get("D4_cs0")
        d2 = arms.get("D2_cs0")
        if d4 and d2:
            yp = np.asarray(d4["y_plus"])
            axes[0].plot(yp, np.asarray(d2["u_plus"]) - np.asarray(d4["u_plus"]), "C3-s", label="2 pi d - 4 pi d")
            if "batch_ci" in d4 and "batch_ci" in d2:
                comb = np.sqrt(
                    np.asarray(d4["batch_ci"]["ci_half"]) ** 2
                    + np.asarray(d2["batch_ci"]["ci_half"]) ** 2
                )
                axes[0].plot(yp, comb, "k:", label="combined 95 % CI")
                axes[0].plot(yp, -comb, "k:")
            axes[0].axhline(0.0, color="k", lw=0.8)
            axes[0].set_xscale("log")
            axes[0].set_title("domain-length contrast")
            axes[0].legend(fontsize=8)
        s0 = arms.get("SGS_cs0")
        s1 = arms.get("SGS_cs01")
        if s0 and s1:
            yp = np.asarray(s0["y_plus"])
            axes[1].plot(yp, np.asarray(s1["u_plus"]) - np.asarray(s0["u_plus"]), "C4-v", label="C_s=0.1 - C_s=0")
            comb = np.sqrt(
                np.asarray(s0["batch_ci"]["ci_half"]) ** 2
                + np.asarray(s1["batch_ci"]["ci_half"]) ** 2
            )
            axes[1].plot(yp, comb, "k:", label="combined 95 % CI")
            axes[1].axhline(0.0, color="k", lw=0.8)
            axes[1].set_xscale("log")
            axes[1].set_title("SGS negative control")
            axes[1].legend(fontsize=8)
        for axis in axes:
            axis.set_xlabel("y+")
            axis.set_ylabel("delta u+")
            axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "contrasts.png")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())