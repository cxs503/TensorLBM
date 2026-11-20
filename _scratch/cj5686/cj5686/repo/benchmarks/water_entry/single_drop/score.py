"""Score only healthy post-contact samples against the locked experimental observations."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def reference(case):
    meta = case["reference_data"]
    path = HERE / meta["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("experimental curve changed; review the protocol")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != meta["markers"]:
        raise ValueError("incomplete reference")
    return tuple(np.array([float(row[k]) for row in rows]) for k in ("time_ms", "depth_over_D"))


def score_run(run, t_ref, h_ref, case):
    rows = run["history"]
    step = run["contact_step"]
    health = case["gates"]
    if not rows or [r["step"] for r in rows] != list(range(len(rows))):
        raise ValueError("expected contiguous, ordered history from initialization")
    valid = []
    for r in rows:
        if any(
            r.get(k) is None or not np.isfinite(r[k])
            for k in ("rho_min", "mach_max", "volume_drift", "time_ms")
        ):
            break
        if (
            r["rho_min"] <= 0
            or not 0 <= r["mach_max"] <= health["maximum_lattice_mach"]
            or not 0 <= r["volume_drift"] <= health["maximum_liquid_volume_relative_drift"]
        ):
            break
        if step is not None and r["step"] >= step:
            if (
                r.get("depth_over_D") is None
                or not np.isfinite(r["depth_over_D"])
                or r["depth_over_D"] < 0
            ):
                break
            valid.append(r)
    points = []
    h_sim = []
    if valid and valid[0]["step"] == step:
        t = np.array([r["time_ms"] - valid[0]["time_ms"] for r in valid])
        if not np.all(np.diff(t) > 0):
            raise ValueError("time must increase strictly")
        h_sim = np.array([r["depth_over_D"] for r in valid])
        for t0, h0 in zip(t_ref, h_ref):
            if t[0] <= t0 <= t[-1]:
                h = float(np.interp(t0, t, h_sim))
                points.append(
                    {
                        "time_ms": float(t0),
                        "experiment": float(h0),
                        "simulation": h,
                        "signed_relative_error": h / h0 - 1,
                        "relative_error": abs(h / h0 - 1),
                    }
                )
    covered = len(points) == len(t_ref)
    peak_error = None
    observed_window_peak = None
    observed_window_peak_time = None
    if covered:
        # Include the simulated peak between experimental frames, over the
        # same observed window; the (interpolated) endpoints are candidates.
        interior = (t > t_ref[0]) & (t < t_ref[-1])
        peak_t = np.r_[t_ref[0], t[interior], t_ref[-1]]
        peak_h = np.r_[points[0]["simulation"], h_sim[interior], points[-1]["simulation"]]
        index = int(np.argmax(peak_h))
        observed_window_peak = float(peak_h[index])
        observed_window_peak_time = float(peak_t[index])
        peak_error = abs(observed_window_peak / max(h_ref) - 1.0)
    u = run.get("impact_speed_m_s")
    speed_error = abs(u / case["velocity_m_s"] - 1) if u is not None and np.isfinite(u) else None
    max_error = max(p["relative_error"] for p in points) if covered else None
    silhouette = run.get("side_view_measurement") or {}
    centre_speed = silhouette.get("fitted_downward_speed_m_s")
    centre_error = (
        abs(centre_speed / case["velocity_m_s"] - 1)
        if centre_speed is not None and np.isfinite(centre_speed)
        else None
    )
    camera_speed = silhouette.get("camera_interval_speed_m_s")
    camera_error = (
        abs(camera_speed / case["velocity_m_s"] - 1)
        if camera_speed is not None and np.isfinite(camera_speed)
        else None
    )
    free_surface = run.get("solver", "").startswith("free_surface_step")
    actual_volume = [
        r["geometric_volume_drift"] for r in rows if r.get("geometric_volume_drift") is not None
    ]
    if not free_surface:
        actual_volume = [r["volume_drift"] for r in rows if r.get("volume_drift") is not None]
    volume_complete = len(actual_volume) == len(rows) and all(
        np.isfinite(v) and v >= 0 for v in actual_volume
    )
    max_volume = max(actual_volume) if volume_complete else None
    mass = [
        r["represented_mass_drift"] for r in rows if r.get("represented_mass_drift") is not None
    ]
    max_mass = max(mass, default=None)
    number_within_tolerance = sum(
        int(p["relative_error"] <= health["maximum_each_point_relative_error"]) for p in points
    )
    numeric_pass = (
        max_volume is not None
        and max_volume <= health["maximum_liquid_volume_relative_drift"]
        and covered
        and not run.get("at_rest_control", False)
        and camera_error is not None
        and camera_error <= health["maximum_impact_speed_relative_error"]
        and max_error <= health["maximum_each_point_relative_error"]
        and peak_error <= health["maximum_peak_depth_relative_error"]
    )
    return {
        "case_id": run.get("case_id", "pilot"),
        "solver": run.get("solver"),
        "span_over_D": run.get("span_over_D"),
        "pool_depth_over_D": run.get("pool_depth_over_D"),
        "lattice_speed": run.get("lattice_speed"),
        "gap_over_D": run.get("gap_over_D"),
        "release_speed_m_s": run.get("release_speed_m_s"),
        "physical_time_step_ms": (run["scales"]["dt_s"] * 1000 if "scales" in run else None),
        "cells_per_D": run.get("cells_per_D"),
        "last_recorded_step": rows[-1]["step"],
        "postcontact_healthy_ms": float(valid[-1]["time_ms"] - valid[0]["time_ms"])
        if valid
        else None,
        "contact_step": step,
        "impact_speed_m_s": u,
        "impact_speed_relative_error": speed_error,
        "side_view_speed_m_s": centre_speed,
        "side_view_speed_relative_error": centre_error,
        "side_view_fit_interval_ms": silhouette.get("recorded_interval_ms"),
        "camera_interval_speed_m_s": camera_speed,
        "camera_speed_relative_error": camera_error,
        "matched_points": len(points),
        "points_within_tolerance": number_within_tolerance,
        "max_geometric_volume_relative_drift": max_volume,
        "max_represented_liquid_mass_relative_drift": max_mass,
        "max_sampled_depth_in_available_window": float(np.max(h_sim)) if len(h_sim) else None,
        "expected_points": len(t_ref),
        "comparisons": points,
        "max_full_curve_relative_error": max_error,
        "peak_relative_error": peak_error,
        "full_window_peak_depth_over_D": observed_window_peak,
        "full_window_peak_time_ms": observed_window_peak_time,
        "numeric_pass": bool(numeric_pass),
        "stop_reason": run["stop_reason"],
    }


def read_run(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def assess(paths, case):
    tref, href = reference(case)
    results = {}
    groups = {}
    for path in paths:
        r = read_run(path)
        key = path.name.split(".")[0]
        if key in results:
            raise ValueError("duplicate run name")
        result = score_run(r, tref, href, case)
        results[key] = result
        if (
            r.get("at_rest_control")
            or r.get("case_id", "pilot") != "santini_case3"
            or r["stop_reason"] == "step_limit"
        ):
            continue
        config = tuple(
            r.get(k)
            for k in (
                "solver",
                "span_over_D",
                "pool_depth_over_D",
                "gap_over_D",
                "lattice_speed",
                "interface_width_cells",
                "hydro_odd_rate",
                "exterior_potential_flow",
                "uniform_drop",
                "stencil_wall",
                "capacity_redistribution",
                "nearest_surface_velocity",
                "simple_mass_exchange",
                "prescribed_release_speed_m_s",
            )
        )
        groups.setdefault(config, []).append((r["cells_per_D"], result))
    converged = []
    for group in groups.values():
        group.sort(key=lambda item: item[0])
        if len(group) < case["gates"]["minimum_grid_count"]:
            continue
        if not all(
            r["max_full_curve_relative_error"] is not None
            and r["max_geometric_volume_relative_drift"] is not None
            and r["max_geometric_volume_relative_drift"]
            <= case["gates"]["maximum_liquid_volume_relative_drift"]
            and r["camera_speed_relative_error"] is not None
            and r["camera_speed_relative_error"]
            <= case["gates"]["maximum_impact_speed_relative_error"]
            for _, r in group
        ):
            continue
        strict = all(
            nf > nc
            and fine["peak_relative_error"] < coarse["peak_relative_error"]
            and fine["max_full_curve_relative_error"] < coarse["max_full_curve_relative_error"]
            for (nc, coarse), (nf, fine) in zip(group, group[1:])
        )
        if strict and group[-1][1]["numeric_pass"]:
            converged.append([n for n, _ in group])
    resolved = bool(
        np.all(
            case["reference_data"]["depth_readout_over_D"] / href
            <= case["gates"]["maximum_each_point_relative_error"]
        )
    )
    length_accuracy = case["case3_measurements"]["length_accuracy_mm"] / (1000 * case["diameter_m"])
    threshold = case["gates"]["maximum_each_point_relative_error"]
    unresolved_times = [
        float(time) for time, depth in zip(tref, href) if length_accuracy > threshold * depth
    ]
    return {
        "passed_3pct_and_converged": bool(converged),
        "status": "eligible_for_review"
        if converged
        and resolved
        and not unresolved_times
        and case["domain_and_material_match_confirmed"]
        else "pending",
        "reference_observed_peak": float(max(href)),
        "time_of_observed_peak_ms": float(tref[np.argmax(href)]),
        "digitization_resolves_3pct_at_every_point": resolved,
        "experimental_length_accuracy_over_D": length_accuracy,
        "times_below_experimental_3pct_resolution_ms": unresolved_times,
        "domain_and_material_match_confirmed": case["domain_and_material_match_confirmed"],
        "converged_grid_pairs": converged,
        "runs": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = assess(args.runs, json.loads((HERE / "protocol.json").read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"])
    raise SystemExit(0 if result["status"] == "eligible_for_review" else 1)
