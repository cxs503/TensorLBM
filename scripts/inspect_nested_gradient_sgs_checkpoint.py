#!/usr/bin/env python3
"""Report WALE/Vreman SGS viscosity for every nested checkpoint level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorlbm.cumulant import summarize_gradient_sgs_effective_tau_d3q19


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--chunk-cells", type=int, default=262_144)
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    populations = state.get("level_populations")
    health = state.get("health_records")
    configuration = state.get("configuration")
    if not isinstance(populations, list) or not populations:
        raise ValueError("checkpoint has no nested level_populations")
    if not isinstance(health, list) or not health:
        raise ValueError("checkpoint has no health records with collision tau")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint has no configuration")
    tau_by_level = health[-1].get("collision_tau_by_level")
    if not isinstance(tau_by_level, list) or len(tau_by_level) != len(populations):
        raise ValueError("latest health record has no complete collision tau chain")
    collision_model = configuration.get("collision_model")
    if collision_model == "cumulant_wale":
        model = "wale"
        coefficient_name = "wale_cw"
    elif collision_model == "cumulant_vreman":
        model = "vreman"
        coefficient_name = "vreman_cv"
    else:
        raise ValueError("checkpoint collision model must be cumulant_wale or cumulant_vreman")
    coefficient = float(configuration[coefficient_name])
    summaries = [
        summarize_gradient_sgs_effective_tau_d3q19(
            value,
            tau=float(tau),
            model=model,
            coefficient=coefficient,
            chunk_cells=args.chunk_cells,
        )
        for value, tau in zip(populations, tau_by_level, strict=True)
    ]
    result = {
        "schema": "tensorlbm-nested-gradient-sgs-checkpoint-audit-v1",
        "checkpoint": str(checkpoint),
        "step": int(state["step"]),
        "collision_model": collision_model,
        "sgs_model": model,
        "coefficient_name": coefficient_name,
        "coefficient": coefficient,
        "gradient_boundary_treatment": "centered_interior_second_order_one_sided_edges",
        "levels": summaries,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
