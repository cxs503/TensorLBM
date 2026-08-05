#!/usr/bin/env python3
"""Read-only health audit for an active nested SUBOFF checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.population_health import inspect_population_health  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("checkpoint", type=Path)
    result.add_argument("--output", type=Path)
    return result


def inspect_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    schema = state.get("schema")
    if not isinstance(schema, str) or not schema.startswith(
        "tensorlbm-suboff-nested-amr-smoke-checkpoint-v",
    ):
        raise ValueError("not a nested SUBOFF checkpoint")
    populations = state.get("level_populations")
    if not isinstance(populations, list) or len(populations) < 2:
        raise ValueError("checkpoint must contain at least two hierarchy levels")
    health = []
    for level, population in enumerate(populations):
        if not isinstance(population, torch.Tensor):
            raise ValueError(f"level {level} population is not a tensor")
        record = inspect_population_health(population).to_dict()
        record["level"] = level
        record["shape"] = list(population.shape)
        health.append(record)
    step_records = state.get("step_records", [])
    if not isinstance(step_records, list):
        raise ValueError("step_records must be a list")
    last_record = step_records[-1] if step_records else None
    if last_record is not None and not isinstance(last_record, dict):
        raise ValueError("step record must be a mapping")
    return {
        "schema": "tensorlbm-nested-suboff-checkpoint-health-v1",
        "source_schema": schema,
        "source_path": str(path),
        "step": int(state["step"]),
        "configuration": state.get("configuration"),
        "levels": health,
        "all_levels_finite": all(record["finite"] for record in health),
        "step_record_count": len(step_records),
        "last_step_record": last_record,
        "maximum_positivity_limited_fraction": state.get(
            "maximum_limiter_fraction",
        ),
        "maximum_reflux_residual_by_interface": state.get(
            "maximum_reflux_residual",
        ),
        "maximum_reflux_limited_directions_by_interface": state.get(
            "maximum_reflux_limited_directions",
        ),
        "maximum_transfer_limited_fraction_by_interface": state.get(
            "maximum_transfer_limited_fraction",
        ),
        "minimum_transfer_alpha_by_interface": state.get(
            "minimum_transfer_alpha",
        ),
    }


def main() -> None:
    args = parser().parse_args()
    result = inspect_checkpoint(args.checkpoint)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
