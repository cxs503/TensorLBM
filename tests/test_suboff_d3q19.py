"""Regression tests for D3Q19 SUBOFF drag bookkeeping."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_suboff_module():
    path = Path(__file__).parents[1] / "examples" / "dg_suboff_mrt_d3q19.py"
    spec = importlib.util.spec_from_file_location("suboff_d3q19", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pressure_drag_is_force_on_body_not_force_on_fluid():
    module = _load_suboff_module()
    # A higher pressure on the upstream (+x-neighbour) face must create
    # positive streamwise drag on the body.
    solid = torch.zeros((1, 1, 5), dtype=torch.bool)
    solid[0, 0, 2] = True
    rho = torch.ones_like(solid, dtype=torch.float32)
    rho[0, 0, 1] = 1.3
    velocity = torch.zeros_like(rho)
    f = module.equilibrium3d(rho, velocity, velocity, velocity)
    _, _, pressure_drag = module.wall_function_19(f, solid, nu=0.01)
    assert pressure_drag > 0.0
