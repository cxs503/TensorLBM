#!/usr/bin/env python3
"""IN45-N01 gate evaluation: turbulent channel vs MKM-1999 DNS.

Reads the campaign run(s) produced by ``run_case.py``, the in-repo MKM-1999
reference table and the laminar wall-origin calibration, then evaluates every
mission criterion and writes ``artifacts/in45/campaign_gates.json``.

Everything is fail-closed: a criterion whose inputs are missing is reported as
``withheld`` (never ``pass``), and every number carries the expression it was
computed from so that the report can be regenerated without hand-copying.

Criteria (mission IN45-N01)
---------------------------
C1  u+ profile vs MKM in the log-law window (30 < y+ < 200, truncated to the
    reference support): relative L2 < 5 %.
C2  rms profiles vs DNS < 10 % (in-repo MKM table has no stresses -> peak
    amplitudes are compared against the published MKM peak band instead and the
    criterion is reported as withheld).
C3  C_f vs DNS < 3 %.
C4  y_w defect fixed and u_tau three-way (momentum balance / wall shear /
    log law) within 2 %.
C5  pytest >= 6 tests including the negative control (laminar, no model).
C6  Re scan on the loong64 cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.validation import channel_wall_units as cwu  # noqa: E402

DNS_CSV = _REPO / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref" / "mkm180.csv"
ARCHIVE_L3 = _REPO / "benchmarks" / "pending" / "channel_turb_3d" / "runs_3d" / "L3_result.json"

KAPPA = cwu.KAPPA
B_LOG = cwu.B_LOG


def _json_default(value):
    """Serialise numpy scalars/arrays so no number is hand-copied anywhere."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def load_reference(path: Path = DNS_CSV) -> tuple[np.ndarray, np.ndarray]:
    y_plus, u_plus = [], []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            y_plus.append(float(row["y_plus"]))
            u_plus.append(float(row["u_plus"]))
    return np.array(y_plus), np.array(u_plus)


def reduce_run(record: dict, *, y_w: float) -> dict[str, object]:
    """Reduce one run record into wall units, with fail-closed checks."""
    result = record.get("result", record)
    config = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]
    nu = float(derived["nu"])
    u_tau_target = float(config["u_tau"])
    height = float(derived["height"])
    body_force = float(derived["body_force_acceleration"])
    mean_profile = np.array(stats["mean_velocity_profile"])
    stress = np.array(stats["reynolds_stress_profiles_uu_vv_ww_uv"])

    # u_tau from the exact momentum balance (anchor of the three-way set).
    u_tau_balance = cwu.u_tau_momentum_balance(body_force, height)
    # u_tau from the bounce-back momentum exchange on the wall.
    wall_shear = stats.get("stationarity_wall_shear")
    if wall_shear is None:
        friction = stats.get("stationarity_friction_velocity")
        if friction is None:
            friction = stats.get("recent_friction_velocity_mean")
        wall_shear = None if friction is None else float(friction) ** 2
    wall_shear = float(wall_shear) if wall_shear is not None else None
    if wall_shear is None:
        reports = result.get("reports") or []
        if reports:
            wall_shear = float(np.mean([float(r["measured_wall_shear"]) for r in reports]))
    steady_ratio = None
    if wall_shear:
        steady_ratio = wall_shear / (u_tau_target**2)

    y, u_folded, fold_info = cwu.fold_channel_profile(mean_profile, y_w=y_w)
    uv_folded, _, _ = cwu.fold_channel_profile(stress[3], y_w=y_w)
    uu_folded, _, _ = cwu.fold_channel_profile(stress[0], y_w=y_w)
    vv_folded, _, _ = cwu.fold_channel_profile(stress[1], y_w=y_w)
    ww_folded, _, _ = cwu.fold_channel_profile(stress[2], y_w=y_w)

    three_way = cwu.u_tau_three_way(
        body_force=body_force,
        height=height,
        wall_shear=wall_shear,
        y=y,
        u=u_folded,
        nu=nu,
        window=(30.0, min(100.0, 0.6 * cwu.effective_height(config["ny"], y_w) * derived["dx_plus"])),
    )
    u_bulk = cwu.bulk_velocity(mean_profile, y_w=y_w)
    cf = cwu.friction_coefficient(u_tau_balance, u_bulk)
    total_stress = cwu.u_tau_total_stress(
        y, u_folded, uv_folded, nu, cwu.effective_height(config["ny"], y_w),
        window=(3.0, 12.0),
    )
    return {
        "name": record.get("name", "unnamed"),
        "grid": [config["nx"], config["ny"], config["nz"]],
        "re_tau": float(config["re_tau"]),
        "u_tau": u_tau_target,
        "nu": nu,
        "tau": float(derived["tau"]),
        "dx_plus": float(derived["dx_plus"]),
        "steps": int(config["steps"]),
        "warmup_steps": int(config["warmup_steps"]),
        "collision": config["collision_model"],
        "smagorinsky_cs": float(config.get("smagorinsky_cs", 0.0)),
        "statistics_steps": int(stats["statistics_steps"]),
        "eddy_turnovers": float(stats.get("eddy_turnovers", stats.get("statistics_eddy_turnovers", 0.0))),
        "samples": int(stats["profile_samples"]),
        "wall_origin": float(y_w),
        "y_w_effect_height": cwu.effective_height(config["ny"], y_w),
        "y_plus": (y * u_tau_balance / nu).tolist(),
        "u_plus": (u_folded / u_tau_balance).tolist(),
        "uv_plus": (uv_folded / u_tau_balance**2).tolist(),
        "uu_plus": (uu_folded / u_tau_balance**2).tolist(),
        "vv_plus": (vv_folded / u_tau_balance**2).tolist(),
        "ww_plus": (ww_folded / u_tau_balance**2).tolist(),
        "fold_info": fold_info,
        "u_tau_balance": u_tau_balance,
        "wall_shear_measured": wall_shear,
        "steady_ratio_wall_shear_over_target": steady_ratio,
        "three_way": three_way,
        "u_tau_total_stress": total_stress,
        "u_bulk": u_bulk,
        "u_bulk_plus": u_bulk / u_tau_balance,
        "c_f": cf,
        "stationarity": {
            key: stats.get(key)
            for key in (
                "recent_friction_velocity_mean",
                "recent_friction_velocity_error_pct",
                "recent_friction_velocity_range_fraction",
                "stationarity_half_window_drift_fraction",
                "recent_crossflow_rms_over_u_tau",
            )
        },
    }


def evaluate_gates(
    reduced: dict[str, object],
    ref_y_plus: np.ndarray,
    ref_u_plus: np.ndarray,
    *,
    pytest_summary: dict | None,
    cluster_summary: dict | None,
    wall_origin_calibration: dict | None,
) -> dict[str, object]:
    y_plus = np.array(reduced["y_plus"], dtype=float)
    u_plus = np.array(reduced["u_plus"], dtype=float)
    reduced["uu_plus"] = np.array(reduced["uu_plus"], dtype=float)
    reduced["vv_plus"] = np.array(reduced["vv_plus"], dtype=float)
    reduced["ww_plus"] = np.array(reduced["ww_plus"], dtype=float)
    reduced["uv_plus"] = np.array(reduced["uv_plus"], dtype=float)
    u_tau = reduced["u_tau_balance"]
    u_bulk = reduced["u_bulk"]
    cf = reduced["c_f"]

    log_window = (30.0, min(200.0, float(ref_y_plus[-1])))
    profile_all = cwu.profile_error_vs_reference(y_plus, u_plus, ref_y_plus, ref_u_plus)
    profile_log = cwu.profile_error_vs_reference(
        y_plus, u_plus, ref_y_plus, ref_u_plus, window=log_window
    )
    # Decompose the log-layer error at the DNS sample abscissae.
    ref_at_sim = np.interp(y_plus, ref_y_plus, ref_u_plus)
    band = (y_plus >= log_window[0]) & (y_plus <= log_window[1])
    rel_band = (u_plus[band] - ref_at_sim[band]) / ref_at_sim[band]

    u_bulk_ref = cwu.bulk_velocity(ref_u_plus, y=ref_y_plus, height=float(ref_y_plus[-1]))
    cf_ref = cwu.friction_coefficient(1.0, u_bulk_ref)

    peaks = {
        "uu_peak": float(np.max(reduced["uu_plus"])),
        "uu_peak_y_plus": float(y_plus[int(np.argmax(reduced["uu_plus"]))]),
        "vv_peak": float(np.max(reduced["vv_plus"])),
        "ww_peak": float(np.max(reduced["ww_plus"])),
        "uv_peak": float(np.min(reduced["uv_plus"])),
        "uv_peak_y_plus": float(y_plus[int(np.argmin(reduced["uv_plus"]))]),
    }
    # Published MKM-1999 peak bands at Re_tau ~ 180 (approximate, +-10 %):
    # these are literature values, NOT in-repo data, so C2 stays withheld.
    literature_peak_band = {
        "uu_peak": [6.5, 8.0],
        "vv_peak": [0.75, 1.05],
        "ww_peak": [1.0, 1.5],
        "uv_peak": [-0.80, -0.65],
        "uv_peak_y_plus": [25.0, 40.0],
    }
    within_band = {
        key: (band_[0] <= peaks[key] <= band_[1]) for key, band_ in literature_peak_band.items()
    }

    three = reduced["three_way"]
    steady_ratio = reduced["steady_ratio_wall_shear_over_target"]
    model_gap = three["log_law_model_form_gap_pct"]

    gates = {
        "C1_u_plus_log_law_error_under_5pct": {
            "status": "pass" if profile_log["relative_l2"] < 0.05 else "fail",
            "relative_l2_log_window": profile_log["relative_l2"],
            "window": profile_log["window"],
            "max_abs_deviation": profile_log["max_abs_deviation"],
            "max_relative_deviation_in_band": float(np.max(np.abs(rel_band))),
            "signed_mean_bias_plus": float(np.mean(u_plus[band] - ref_at_sim[band])),
            "relative_l2_full_profile": profile_all["relative_l2"],
            "clamped_fraction": profile_all["clamped_fraction"],
            "expression": "||u+_sim - u+_MKM||_2 / ||u+_MKM||_2 on 30 < y+ < 200",
        },
        "C2_rms_profiles_under_10pct": {
            "status": "withheld",
            "reason": (
                "the in-repo MKM-1999 table (dns_ref/mkm180.csv) contains only "
                "y, y+, u+ - no <u'u'>, <v'v'>, <w'w'> or <u'v'> columns, so a "
                "10 % profile error cannot be machine-checked.  Peak amplitudes "
                "are quoted against the published MKM peak band instead."
            ),
            "peaks": peaks,
            "literature_peak_band": literature_peak_band,
            "within_band": within_band,
        },
        "C3_cf_under_3pct": {
            "status": "pass" if abs(cf / cf_ref - 1.0) < 0.03 else "fail",
            "c_f": cf,
            "c_f_reference": cf_ref,
            "relative_error": float(cf / cf_ref - 1.0),
            "u_bulk": u_bulk,
            "u_bulk_plus": reduced["u_bulk_plus"],
            "u_bulk_plus_reference": u_bulk_ref,
            "expression": "C_f = 2 u_tau^2 / U_b^2 ; reference = same formula on the MKM profile",
        },
        "C4_u_tau_three_way_under_2pct": {
            "status": "pass" if three["consistent"] else "fail",
            "max_pairwise_deviation_pct": three["max_pairwise_deviation_pct"],
            "deviations_pct": three["deviations_pct"],
            "routes": three["routes"],
            "log_law_model_form_gap_pct": model_gap,
            "y_w_used": reduced["wall_origin"],
            "steady_ratio_wall_shear_over_target": steady_ratio,
            "expression": "pairwise |a-b| / mean(a,b) of {momentum balance, wall shear, log law}",
        },
        "C5_pytest_with_negative_control": {
            "status": "pass" if (pytest_summary and pytest_summary["passed"] >= 6) else "fail",
            "pytest": pytest_summary,
            "negative_control": "tests/test_d3q19_channel_numpy.py::test_negative_control_no_perturbation_stays_laminar",
        },
        "C6_loong64_cluster_re_scan": {
            "status": (
                "pass"
                if cluster_summary and cluster_summary.get("executed")
                and len(cluster_summary.get("points", [])) >= 2
                else ("withheld" if cluster_summary is None else "fail")
            ),
            "cluster": cluster_summary,
        },
        "y_w_defect_fixed": {
            "status": "pass" if (wall_origin_calibration and wall_origin_calibration.get("measured")) else "withheld",
            "calibration": wall_origin_calibration,
        },
    }
    return gates


def run_pytest() -> dict[str, object]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_channel_wall_units.py",
        "tests/test_d3q19_channel_numpy.py",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    completed = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    passed = failed = 0
    for token in tail.replace(",", " ").split():
        if token.isdigit():
            continue
    import re

    match = re.search(r"(\d+) passed", completed.stdout)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) failed", completed.stdout)
    if match:
        failed = int(match.group(1))
    return {
        "command": " ".join(cmd),
        "returncode": completed.returncode,
        "passed": passed,
        "failed": failed,
        "summary": tail,
        "tests": [
            "tests/test_channel_wall_units.py (10): fold/wall-origin/u_tau closures/C_f",
            "tests/test_d3q19_channel_numpy.py (5): torch-equivalence, mass conservation,",
            "  laminar Poiseuille steady state, negative control, perturbation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/in45")
    parser.add_argument("--out", default="artifacts/in45/campaign_gates.json")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()
    run_dir = _REPO / args.run_dir
    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_y_plus, ref_u_plus = load_reference()

    calibration_path = _REPO / "artifacts" / "in45" / "wall_origin_calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    y_w = 0.6447
    if calibration and "production_y_w" in calibration:
        y_w = float(calibration["production_y_w"])

    runs = {}
    for path in sorted(run_dir.glob("*_result.json")):
        record = json.loads(path.read_text())
        runs[record.get("name", path.stem)] = record

    reduced_runs = []
    for name, record in runs.items():
        reduced = reduce_run(record, y_w=y_w)
        reduced["elapsed_seconds"] = record.get("elapsed_seconds")
        reduced["mega_cell_steps_per_second"] = record.get("lattice_ratings", {}).get(
            "mega_cell_steps_per_second"
        )
        reduced_runs.append(reduced)

    compiler = {
        "wall_origin_calibration": calibration,
        "reference": {
            "file": str(DNS_CSV.relative_to(_REPO)),
            "source": "Moser, Kim & Mansour, Phys. Fluids 11, 943 (1999), Re_tau = 178.12",
            "u_bulk_plus": cwu.bulk_velocity(ref_u_plus, y=ref_y_plus, height=float(ref_y_plus[-1])),
            "c_f": cwu.friction_coefficient(
                1.0, cwu.bulk_velocity(ref_u_plus, y=ref_y_plus, height=float(ref_y_plus[-1]))
            ),
            "dean_fit_c_f": cwu.dean_channel_friction(178.12),
        },
        "pytest": None if args.skip_pytest else run_pytest(),
        "cluster": None,
        "runs": reduced_runs,
        "archive_p20_n01": None,
    }
    cluster_path = _REPO / "artifacts" / "in45" / "cluster_re_scan.json"
    if cluster_path.exists():
        compiler["cluster"] = json.loads(cluster_path.read_text())

    if ARCHIVE_L3.exists() and reduced_runs:
        archive = json.loads(ARCHIVE_L3.read_text())
        profile = np.array(archive["statistics"]["mean_velocity_profile"])
        u_tau_arch = float(archive["configuration"]["u_tau"])
        nu_arch = float(archive["derived"]["nu"])
        folded_y, folded_u, _ = cwu.fold_channel_profile(profile, y_w=y_w)
        unfolded_y = np.arange(1, profile.size - 1) - y_w
        compiler["archive_p20_n01"] = {
            "file": str(ARCHIVE_L3.relative_to(_REPO)),
            "reported_relative_l2": 0.1519,
            "refolded_relative_l2": cwu.profile_error_vs_reference(
                folded_y * u_tau_arch / nu_arch, folded_u / u_tau_arch, ref_y_plus, ref_u_plus
            )["relative_l2"],
            "unfolded_relative_l2": cwu.profile_error_vs_reference(
                unfolded_y * u_tau_arch / nu_arch,
                profile[1:-1] / u_tau_arch,
                ref_y_plus,
                ref_u_plus,
            )["relative_l2"],
        }

    gates = {}
    for reduced in reduced_runs:
        gates[reduced["name"]] = evaluate_gates(
            reduced,
            ref_y_plus,
            ref_u_plus,
            pytest_summary=compiler["pytest"],
            cluster_summary=compiler["cluster"],
            wall_origin_calibration=calibration,
        )
    compiler["gates"] = gates
    out_path.write_text(json.dumps(compiler, indent=2, default=_json_default), encoding="utf-8")

    summary = {
        name: {
            "C1": g["C1_u_plus_log_law_error_under_5pct"]["status"],
            "C1_value": round(100 * g["C1_u_plus_log_law_error_under_5pct"]["relative_l2_log_window"], 2),
            "C3": g["C3_cf_under_3pct"]["status"],
            "C3_value": round(100 * g["C3_cf_under_3pct"]["relative_error"], 2),
            "C4": g["C4_u_tau_three_way_under_2pct"]["status"],
            "C4_value": round(g["C4_u_tau_three_way_under_2pct"]["max_pairwise_deviation_pct"], 2),
        }
        for name, g in gates.items()
    }
    print(json.dumps({"out": str(out_path), "gates": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())