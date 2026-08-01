#!/usr/bin/env python3
"""Audit force observers in a SUBOFF static-AMR checkpoint.

This tool intentionally reports numerical closure separately from agreement
with the towing-tank value.  A force observer can close perfectly while the
flow is still spatially or temporally unconverged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT / "src", ROOT / "examples"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from suboff_experimental_resistance import (  # noqa: E402
    MODEL_LENGTH_M,
    experimental_point,
    force_scale_newton,
)

from tensorlbm.yplus_guide import (  # noqa: E402
    estimate_exchange_yplus,
    plan_exchange_yplus_refinement,
)


def _sample_tensor(state: dict, name: str) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"checkpoint field {name!r} must be an N x 2 tensor")
    return value.detach().to(device="cpu", dtype=torch.float64)


def _group_sample_means(samples: torch.Tensor) -> torch.Tensor:
    """Average repeated fine-substep observations at each coarse-grid step."""
    if samples.numel() == 0:
        return samples.reshape(0, 2)
    grouped: list[tuple[float, float]] = []
    current_step = float(samples[0, 0])
    values: list[float] = []
    for step_tensor, value_tensor in samples:
        step = float(step_tensor)
        value = float(value_tensor)
        if step != current_step:
            grouped.append((current_step, sum(values) / len(values)))
            current_step = step
            values = []
        values.append(value)
    grouped.append((current_step, sum(values) / len(values)))
    return torch.tensor(grouped, dtype=torch.float64)


def _aligned_values(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_by_step = {int(step): float(value) for step, value in left.tolist()}
    right_by_step = {int(step): float(value) for step, value in right.tolist()}
    common = sorted(left_by_step.keys() & right_by_step.keys())
    if not common:
        raise ValueError("force observers have no common sampled steps")
    return (
        torch.tensor([left_by_step[step] for step in common], dtype=torch.float64),
        torch.tensor([right_by_step[step] for step in common], dtype=torch.float64),
    )


def _difference_report(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    *,
    scale: float = 1.0,
) -> dict[str, float | int]:
    if reference.shape != comparison.shape or reference.numel() == 0:
        raise ValueError("force observer vectors must be non-empty and aligned")
    residual = (comparison - reference) * scale
    reference_mean = float(reference.mean()) * scale
    comparison_mean = float(comparison.mean()) * scale
    return {
        "samples": reference.numel(),
        "reference_mean_n": reference_mean,
        "comparison_mean_n": comparison_mean,
        "mean_residual_n": float(residual.mean()),
        "rms_residual_n": float(residual.square().mean().sqrt()),
        "maximum_absolute_residual_n": float(residual.abs().max()),
        "mean_difference_pct": (
            abs(comparison_mean - reference_mean)
            / max(abs(reference_mean), 1.0e-30)
            * 100.0
        ),
    }


def _finite_history_reduction(
    state: dict,
    name: str,
    reduction: str,
) -> float | None:
    value = state.get(name)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    finite = value.detach().cpu().to(torch.float64)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return None
    return float(getattr(finite, reduction)())


def audit_checkpoint(state: dict) -> dict:
    configuration = state.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint has no configuration dictionary")
    required_configuration = (
        "hull_type",
        "speed_knots",
        "hull_length",
        "rho_water",
        "lattice_speed",
        "warmup_steps",
    )
    missing = [name for name in required_configuration if name not in configuration]
    if missing:
        raise ValueError(f"checkpoint configuration is missing {missing}")

    point = experimental_point(
        configuration["hull_type"], configuration["speed_knots"],
    )
    scale = force_scale_newton(
        rho_water=configuration["rho_water"],
        dx_m=MODEL_LENGTH_M / (2.0 * configuration["hull_length"]),
        speed_mps=point.speed_mps,
        lattice_speed=configuration["lattice_speed"],
    )

    force_history = state["force_history"].detach().cpu().to(torch.float64)
    bfl_history = state["bfl_total_history"].detach().cpu().to(torch.float64)
    if force_history.shape != bfl_history.shape or force_history.numel() == 0:
        raise ValueError("checkpoint force histories must be non-empty and aligned")
    history_closure = _difference_report(force_history, bfl_history)
    pressure_history = state["pressure_history"].detach().cpu().to(torch.float64)
    wall_shear_history = state["wall_shear_history"].detach().cpu().to(torch.float64)
    if pressure_history.shape != force_history.shape:
        raise ValueError("checkpoint pressure history is not aligned")
    if wall_shear_history.shape != force_history.shape:
        raise ValueError("checkpoint wall-shear history is not aligned")

    primary = _group_sample_means(_sample_tensor(state, "paired_primary_cv_samples"))
    bfl = _group_sample_means(_sample_tensor(state, "paired_bfl_total_samples"))
    numerical_source = _group_sample_means(
        _sample_tensor(state, "numerical_momentum_source_samples"),
    )
    corrected = _group_sample_means(_sample_tensor(state, "corrected_cv_samples"))
    surface = _group_sample_means(_sample_tensor(state, "surface_total_samples"))

    primary_values, bfl_values = _aligned_values(primary, bfl)
    corrected_values, corrected_bfl_values = _aligned_values(corrected, bfl)
    surface_values, surface_cv_values = _aligned_values(surface, primary)
    source_values, source_cv_values = _aligned_values(numerical_source, primary)

    auxiliary: dict[str, dict[str, float | int]] = {}
    for margin, samples in state.get("auxiliary_cv_samples", {}).items():
        grouped = _group_sample_means(samples.detach().cpu().to(torch.float64))
        primary_for_aux, auxiliary_values = _aligned_values(primary, grouped)
        auxiliary[str(margin)] = _difference_report(
            primary_for_aux, auxiliary_values, scale=scale,
        )

    current_mean = float(force_history.mean())
    wall_exchange_prior = None
    wall_refinement_plan = None
    exchange_distance = configuration.get("stress_exchange_distance")
    physical_viscosity = configuration.get("nu_water")
    if exchange_distance is not None and physical_viscosity is not None:
        physical_reynolds = point.speed_mps * MODEL_LENGTH_M / physical_viscosity
        finest_length_cells = 2.0 * configuration["hull_length"]
        wall_exchange_prior = estimate_exchange_yplus(
            physical_reynolds=physical_reynolds,
            characteristic_length_cells=finest_length_cells,
            exchange_distance_cells=exchange_distance,
        )
        wall_refinement_plan = plan_exchange_yplus_refinement(
            physical_reynolds=physical_reynolds,
            characteristic_length_cells=finest_length_cells,
        )
    return {
        "checkpoint_schema": state.get("schema"),
        "checkpoint_step": int(state["step"]),
        "case": {
            "hull_type": configuration["hull_type"],
            "speed_knots": configuration["speed_knots"],
            "hull_length_lu": configuration["hull_length"],
            "coarse_shape_zyx": configuration.get("coarse_shape_zyx"),
            "collision_model": configuration.get("collision_model"),
            "wall_law": configuration.get("wall_law"),
            "link_force_frame": configuration.get("link_force_frame"),
        },
        "force_scale_n_per_lattice_unit": scale,
        "warmup_steps": int(configuration["warmup_steps"]),
        "post_warmup_history_steps": force_history.numel(),
        "sampled_coarse_steps": primary_values.numel(),
        "force_decomposition": {
            "mean_bfl_link_pressure_n": float(pressure_history.mean()),
            "mean_modeled_wall_shear_n": float(wall_shear_history.mean()),
            "mean_bfl_link_plus_wall_shear_n": float(bfl_history.mean()),
            "mean_cv_minus_modeled_wall_shear_n": (
                float(force_history.mean() - wall_shear_history.mean())
            ),
        },
        "wall_model_applicability": {
            "minimum_observed_y_plus": _finite_history_reduction(
                state, "wall_y_plus_min_history", "min",
            ),
            "mean_observed_y_plus": _finite_history_reduction(
                state, "wall_y_plus_mean_history", "mean",
            ),
            "maximum_observed_y_plus": _finite_history_reduction(
                state, "wall_y_plus_max_history", "max",
            ),
            "maximum_rejected_sample_fraction": _finite_history_reduction(
                state, "wall_rejected_fraction_history", "max",
            ),
            "ittc_exchange_y_plus_prior": wall_exchange_prior,
            "minimum_sample_refinement_plan": wall_refinement_plan,
        },
        "numerical_quality": {
            "maximum_positivity_limited_fraction": state.get(
                "maximum_positivity_limited_fraction",
            ),
            "maximum_reflux_population_residual": state.get(
                "maximum_reflux_population_residual",
            ),
            "maximum_reflux_limited_directions": state.get(
                "maximum_reflux_limited_directions",
            ),
            "maximum_raw_kinetic_mismatch": state.get(
                "maximum_raw_kinetic_mismatch",
            ),
        },
        "observer_closure": {
            "history_cv_vs_bfl": history_closure,
            "sampled_cv_vs_bfl": _difference_report(
                primary_values, bfl_values, scale=scale,
            ),
            "source_corrected_cv_vs_bfl": _difference_report(
                corrected_values, corrected_bfl_values, scale=scale,
            ),
            "surface_vs_cv": _difference_report(
                surface_cv_values, surface_values, scale=scale,
            ),
            "numerical_momentum_source": {
                "samples": source_values.numel(),
                "mean_n": float(source_values.mean()) * scale,
                "fraction_of_cv_mean_pct": (
                    abs(float(source_values.mean()))
                    / max(abs(float(source_cv_values.mean())), 1.0e-30)
                    * 100.0
                ),
            },
            "auxiliary_control_volumes": auxiliary,
        },
        "partial_physical_result": {
            "mean_resistance_n": current_mean,
            "experimental_resistance_n": point.resistance_n,
            "error_pct": (
                abs(current_mean - point.resistance_n)
                / point.resistance_n
                * 100.0
            ),
            "admission_warning": (
                "This is an in-progress checkpoint diagnostic, not a "
                "grid/time-converged CFD result."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    report = audit_checkpoint(state)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
