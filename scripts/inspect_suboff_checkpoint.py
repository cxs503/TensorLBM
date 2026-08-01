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
