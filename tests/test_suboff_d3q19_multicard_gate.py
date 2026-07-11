"""Validity gates for the bounded D3Q19 multi-card SUBOFF diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def module():
    path = Path(__file__).parents[1] / "examples" / "dg_suboff_mrt_d3q19_multicard.py"
    spec = importlib.util.spec_from_file_location("suboff_d3q19_multicard", path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_diagnostic_gate_rejects_underresolved_or_unsettled_runs(module):
    with pytest.raises(ValueError, match="D >="):
        module.validate_suboff_diagnostic(nx=192, ny=72, nz=72, hull_length=96,
                                          n_steps=600, warmup=300, u_in=0.06)
    with pytest.raises(ValueError, match="stern-to-outlet"):
        module.validate_suboff_diagnostic(nx=416, ny=208, nz=208, hull_length=206,
                                          n_steps=1000, warmup=300, u_in=0.06)


def test_diagnostic_gate_accepts_resolved_time_window(module):
    module.validate_suboff_diagnostic(nx=416, ny=208, nz=208, hull_length=206,
                                      n_steps=3600, warmup=2800, u_in=0.06)


def test_multicard_pressure_force_has_body_drag_sign_and_ignores_halos(module):
    solid = torch.tensor([[[False, True, False, False]]])
    pressure = torch.tensor([[[2.0, 0.0, 1.0, 99.0]]])
    interior = torch.tensor([[[True, True, True, False]]])
    assert module.pressure_drag_x_19(pressure, solid, interior).item() == 1.0
