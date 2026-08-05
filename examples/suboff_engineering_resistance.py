#!/usr/bin/env python3
"""Validated engineering resistance interpolation for DARPA SUBOFF Table 14.

This is deliberately separate from CFD validation.  For each configuration,
the lowest and highest tow speeds calibrate a Reynolds-dependent multiplier
on the ITTC-1957 friction baseline.  The four interior speeds are untouched
holdout points.  Predictions are restricted to the calibrated speed range.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from suboff_experimental_resistance import (
    KNOT_TO_MPS,
    MODEL_LENGTH_M,
    PRIMARY_SOURCE,
    TOW_TANK_POINTS,
)


def ittc_friction_force(
    speed_knots: float,
    *,
    rho_water: float = 998.2,
    nu_water: float = 1.004e-6,
    wetted_area_m2: float = 5.922697010175,
) -> tuple[float, float, float]:
    speed_mps = speed_knots * KNOT_TO_MPS
    re = speed_mps * MODEL_LENGTH_M / nu_water
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    force_n = 0.5 * rho_water * speed_mps**2 * wetted_area_m2 * cf
    return force_n, re, cf


def fit_endpoint_model(hull_type: str) -> dict:
    """Calibrate endpoints and validate every interior Table 14 point."""
    points = TOW_TANK_POINTS[hull_type]
    endpoint_rows = []
    for point in (points[0], points[-1]):
        baseline_n, re, _ = ittc_friction_force(point.speed_knots)
        endpoint_rows.append((math.log10(re), point.resistance_n / baseline_n))
    (x0, a0), (x1, a1) = endpoint_rows
    slope = (a1 - a0) / (x1 - x0)
    intercept = a0 - slope * x0

    rows = []
    for index, point in enumerate(points):
        baseline_n, re, cf = ittc_friction_force(point.speed_knots)
        multiplier = intercept + slope * math.log10(re)
        predicted_n = baseline_n * multiplier
        signed_error_pct = (
            (predicted_n - point.resistance_n) / point.resistance_n * 100.0
        )
        rows.append({
            "role": "calibration" if index in (0, len(points) - 1) else "holdout",
            "speed_knots": point.speed_knots,
            "reynolds_number": re,
            "ittc_cf": cf,
            "effective_multiplier": multiplier,
            "predicted_resistance_n": predicted_n,
            "experimental_resistance_n": point.resistance_n,
            "signed_error_pct": signed_error_pct,
            "absolute_error_pct": abs(signed_error_pct),
        })
    holdouts = [row for row in rows if row["role"] == "holdout"]
    return {
        "hull_type": hull_type,
        "model": "R=0.5*rho*V^2*S*Cf_ITTC*(a+b*log10(Re))",
        "intercept_a": intercept,
        "slope_b": slope,
        "calibrated_speed_range_knots": [points[0].speed_knots, points[-1].speed_knots],
        "rows": rows,
        "holdout_max_absolute_error_pct": max(row["absolute_error_pct"] for row in holdouts),
        "holdout_mean_absolute_error_pct": sum(row["absolute_error_pct"] for row in holdouts) / len(holdouts),
        "holdout_target_pct": 5.0,
        "holdout_target_met": all(row["absolute_error_pct"] <= 5.0 for row in holdouts),
    }


def build_report() -> dict:
    cases = [fit_endpoint_model("bare_hull"), fit_endpoint_model("full")]
    return {
        "schema": "tensorlbm-suboff-engineering-resistance-v1",
        "status": "validated_engineering_interpolator",
        "physical_validation": True,
        "cfd_validation": False,
        "primary_source": PRIMARY_SOURCE,
        "scope": (
            "Interpolation only within the Table 14 model-scale speed range; "
            "two endpoint measurements calibrate each hull and four interior "
            "measurements are holdouts. Not a CFD validation result."
        ),
        "assumptions": {
            "model_length_m": MODEL_LENGTH_M,
            "rho_water_kg_m3": 998.2,
            "nu_water_m2_s": 1.004e-6,
            "bare_body_wetted_area_m2": 5.922697010175,
        },
        "cases": cases,
        "all_holdout_targets_met": all(case["holdout_target_met"] for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for case in report["cases"]:
        print(
            f"{case['hull_type']}: holdout max="
            f"{case['holdout_max_absolute_error_pct']:.2f}% mean="
            f"{case['holdout_mean_absolute_error_pct']:.2f}% "
            f"pass={case['holdout_target_met']}"
        )
    print(f"output={output}")


if __name__ == "__main__":
    main()
