from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "assess_collision_viscosity_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("viscosity_schedule", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_schedule_admits_recovered_viscosities() -> None:
    result = MODULE.assess(
        "natural_kbc", [0.68, 0.8],
        wavelength_cells=24,
        transverse_cells=3,
        amplitude=0.01,
        steps=120,
        fit_start_step=15,
        maximum_relative_error_pct=2.0,
        device="cpu",
        dtype="float64",
    )
    assert result["status"] == "admitted"
    assert result["acceptance"]["configured_reynolds_sequence_admitted"] is True
    assert result["physical_validation"] is False


def test_schedule_rejects_duplicate_or_missing_taus() -> None:
    kwargs = {
        "wavelength_cells": 24,
        "transverse_cells": 3,
        "amplitude": 0.01,
        "steps": 120,
        "fit_start_step": 15,
        "maximum_relative_error_pct": 2.0,
        "device": "cpu",
        "dtype": "float64",
    }
    with pytest.raises(ValueError, match="finite"):
        MODULE.assess("natural_kbc", [], **kwargs)
    with pytest.raises(ValueError, match="unique"):
        MODULE.assess("natural_kbc", [0.8, 0.8], **kwargs)
