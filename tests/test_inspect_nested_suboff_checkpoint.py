from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d

SCRIPT = Path(__file__).parents[1] / "scripts" / "inspect_nested_suboff_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_nested_suboff_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _checkpoint(path: Path) -> None:
    rho = torch.ones((2, 3, 4))
    zero = torch.zeros_like(rho)
    population = equilibrium3d(rho, zero, zero, zero)
    torch.save(
        {
            "schema": "tensorlbm-suboff-nested-amr-smoke-checkpoint-v3",
            "configuration": {"hull_type": "bare_hull"},
            "step": 12,
            "level_populations": [population, population.clone(), population.clone()],
            "step_records": [{"step": 12, "cv_resistance_n": 91.0}],
            "maximum_limiter_fraction": 0.0,
            "maximum_reflux_residual": [1e-8, 2e-8],
            "maximum_reflux_limited_directions": [0, 0],
        },
        path,
    )


def test_nested_checkpoint_health_is_read_only_and_explicit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nested.ckpt"
    _checkpoint(checkpoint)

    result = MODULE.inspect_checkpoint(checkpoint)

    assert result["step"] == 12
    assert result["all_levels_finite"] is True
    assert [record["level"] for record in result["levels"]] == [0, 1, 2]
    assert result["levels"][0]["maximum_speed"] == pytest.approx(0.0, abs=1e-7)
    assert result["last_step_record"]["cv_resistance_n"] == 91.0
    assert result["maximum_transfer_limited_fraction_by_interface"] is None


def test_nested_checkpoint_health_rejects_unrelated_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save({"schema": "other"}, checkpoint)
    with pytest.raises(ValueError, match="not a nested"):
        MODULE.inspect_checkpoint(checkpoint)
