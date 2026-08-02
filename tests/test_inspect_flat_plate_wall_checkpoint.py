from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_flat_plate_wall_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_flat_plate_wall_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_flat_plate_checkpoint_audit_is_read_only(tmp_path: Path) -> None:
    shape = (3, 24, 32)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.06)
    zero = torch.zeros(shape)
    populations = equilibrium3d(rho, ux, zero, zero)
    checkpoint = tmp_path / "flat.ckpt"
    torch.save(
        {
            "schema": "tensorlbm-flat-plate-checkpoint-v4",
            "step": 10,
            "configuration": {
                "shape_zyx": list(shape),
                "plate_length": 8,
                "plate_start_fraction": 0.25,
                "reynolds": 1.0e6,
                "lattice_speed": 0.06,
                "wall_law": "musker",
                "stress_exchange_distance": 1.0,
            },
            "populations": populations,
        },
        checkpoint,
    )

    result = MODULE.inspect_checkpoint(checkpoint)

    assert result["source_step"] == 10
    assert result["population_state_advanced"] is False
    assert result["wall_exchange"]["active_nodes"] > 0
    assert result["wall_exchange"]["pressure_gradient_parameter_max"] == 0.0
    gradient = result["wall_exchange"]["pressure_gradient_summary"]
    assert gradient["valid_samples"] == gradient["requested_samples"]
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert torch.equal(restored["populations"], populations)
