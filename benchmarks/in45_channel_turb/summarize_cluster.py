#!/usr/bin/env python3
"""Summarise the loong64 cluster Re-scan (C6) into a machine-checked record.

Reads the run records fetched from the cluster (``runs/in45/<name>_result.json``
plus the shard logs) and writes ``artifacts/in45/cluster_re_scan.json`` with

* the executed points (Re_tau, grid, throughput, statistics),
* the wall-unit reduction of each point (three-way u_tau, log-law closure, C_f
  against the Dean-type channel fit - no DNS table exists at Re_tau = 395 in
  this repository, so nothing is compared against a fabricated reference),
* the throughput comparison against the x86 control-plane measurement, which is
  what makes "run it on the cluster" a documented decision rather than a habit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from tensorlbm.validation import channel_wall_units as cwu  # noqa: E402

X86_REFERENCE_MCELL = 0.80  # measured on the control plane (torch cumulant)


def reduce_point(record: dict, y_w: float) -> dict[str, object]:
    result = record.get("result", record)
    config = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]
    nu = float(derived["nu"])
    height = float(derived["height"])
    body_force = float(derived["body_force_acceleration"])
    profile = np.array(stats["mean_velocity_profile"])
    stress = np.array(stats["reynolds_stress_profiles_uu_vv_ww_uv"])
    y, u, _ = cwu.fold_channel_profile(profile, y_w=y_w)
    u_tau_balance = cwu.u_tau_momentum_balance(body_force, height)
    friction = stats.get("stationarity_friction_velocity")
    wall_shear = None if friction is None else float(friction) ** 2
    window = (30.0, min(100.0, 0.6 * derived["dx_plus"] * (0.5 * height)))
    three = cwu.u_tau_three_way(
        body_force=body_force,
        height=height,
        wall_shear=wall_shear,
        y=y,
        u=u,
        nu=nu,
        window=window,
    )
    u_bulk = cwu.bulk_velocity(profile, y_w=y_w)
    return {
        "re_tau_lattice": float(config["re_tau"]),
        "grid": [config["nx"], config["ny"], config["nz"]],
        "u_tau": float(config["u_tau"]),
        "nu": nu,
        "tau": float(derived["tau"]),
        "dx_plus": float(derived["dx_plus"]),
        "steps": int(config["steps"]),
        "statistics_eddy_turnovers": float(
            stats.get("statistics_eddy_turnovers", stats.get("eddy_turnovers", 0.0))
        ),
        "wall_shear_ratio": None if wall_shear is None else wall_shear / float(config["u_tau"]) ** 2,
        "u_bulk_plus": u_bulk / u_tau_balance,
        "c_f": cwu.friction_coefficient(u_tau_balance, u_bulk),
        "c_f_dean_fit": cwu.dean_channel_friction(float(config["re_tau"])),
        "u_tau_routes": three["routes"],
        "max_pairwise_deviation_pct": three["max_pairwise_deviation_pct"],
        "log_law_model_form_gap_pct": three["log_law_model_form_gap_pct"],
        "u_plus_profile": (u / u_tau_balance).tolist(),
        "y_plus_profile": (y * u_tau_balance / nu).tolist(),
        "uv_plus_peak": float(
            np.min(
                cwu.fold_channel_profile(stress[3], y_w=y_w)[1] / u_tau_balance**2
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/in45")
    parser.add_argument("--out", default="artifacts/in45/cluster_re_scan.json")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--nodes", default="n315,n330")
    parser.add_argument("--y-w", type=float, default=None)
    args = parser.parse_args()
    calibration_path = _REPO / "artifacts" / "in45" / "wall_origin_calibration.json"
    y_w = args.y_w
    if y_w is None:
        y_w = 0.6447
        if calibration_path.exists():
            y_w = float(json.loads(calibration_path.read_text())["production_y_w"])
    points = []
    for path in sorted((_REPO / args.run_dir).glob("*_result.json")):
        record = json.loads(path.read_text())
        point = reduce_point(record, y_w)
        point["name"] = record.get("name", path.stem)
        point["throughput_mcell_step_per_s"] = record.get("lattice_ratings", {}).get(
            "mega_cell_steps_per_second"
        )
        point["elapsed_seconds"] = record.get("elapsed_seconds")
        points.append(point)
    throughput = [p["throughput_mcell_step_per_s"] for p in points if p["throughput_mcell_step_per_s"]]
    payload = {
        "schema": "in45-cluster-re-scan-v1",
        "executed": bool(points),
        "job_id": args.job_id,
        "nodes": args.nodes,
        "arch": "loongarch64 (Loongson, 128 cores/node)",
        "y_w_used": y_w,
        "points": points,
        "throughput_mcell_step_per_s": max(throughput) if throughput else None,
        "x86_control_plane_reference_mcell_step_per_s": X86_REFERENCE_MCELL,
        "note": (
            "Re_tau = 395 has no DNS table in this repository, so it is scored "
            "against the Dean-type channel fit for C_f and against its own "
            "three-way u_tau closure - no fabricated reference profile."
        ),
    }
    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "points": [
                    {
                        "name": p["name"],
                        "re_tau": p["re_tau_lattice"],
                        "grid": p["grid"],
                        "throughput": p["throughput_mcell_step_per_s"],
                        "wall_shear_ratio": p["wall_shear_ratio"],
                        "three_way": p["max_pairwise_deviation_pct"],
                        "c_f": p["c_f"],
                        "c_f_dean": p["c_f_dean_fit"],
                    }
                    for p in points
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())