"""Lifecycle contracts for the Körner free-surface hull caller."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID
from tensorlbm.hull_free_surface_v2 import HullFreeSurfaceV2Config, run_hull_free_surface_v2


def _config(**overrides):
    kwargs = {
        "nx": 32,
        "ny": 16,
        "nz": 16,
        "n_steps": 2,
        "warmup": 0,
        "output_interval": 1,
        "device": "cpu",
        "use_free_surface": True,
        "use_wall_function": False,
    }
    kwargs.update(overrides)
    return HullFreeSurfaceV2Config(**kwargs)


def test_hull_caller_preserves_returned_mass_and_topology_between_steps(monkeypatch):
    """Mass and flags are caller-owned step state, not reconstructed from fill."""
    import tensorlbm.hull_free_surface_v2 as runner

    real_init_flags = runner.init_flags_from_fill
    init_calls = []
    seen_mass = []
    seen_fill = []
    seen_flags = []
    returned_mass = []
    returned_fill = []
    sentinel_flags = []

    def record_initial_flags(fill, solid):
        init_calls.append((fill.clone(), solid.clone()))
        return real_init_flags(fill, solid)

    def controlled_step(f, fill, flags, solid, *args, **kwargs):
        seen_mass.append(kwargs.get("mass"))
        seen_fill.append(fill.clone())
        seen_flags.append(flags.clone())
        next_mass = kwargs["mass"].clone()
        next_fill = fill.clone()
        next_fill[~solid] = 0.25
        next_flags = torch.full_like(flags, INTERFACE)
        next_flags[solid] = 3
        # Deliberately independent from fill: this must be passed through.
        next_mass[~solid] = 0.875 + 0.125 * len(seen_mass)
        returned_mass.append(next_mass)
        returned_fill.append(next_fill)
        sentinel_flags.append(next_flags)
        return f, next_fill, next_flags, next_mass, torch.tensor(0.0)

    monkeypatch.setattr(runner, "init_flags_from_fill", record_initial_flags)
    monkeypatch.setattr(runner, "free_surface_step", controlled_step)

    result = run_hull_free_surface_v2(_config())

    assert len(init_calls) == 1
    assert len(seen_mass) == 2
    assert seen_mass[0] is not None
    assert seen_mass[1] is returned_mass[0]
    assert torch.equal(seen_fill[1], returned_fill[0])
    assert torch.equal(seen_flags[1], sentinel_flags[0])
    assert result["topology_safety_status"] == "topology safety only"
    assert [entry["step"] for entry in result["topology_safety"]] == [1, 2]


def test_hull_caller_real_two_steps_returns_finite_full18_safe_telemetry():
    """A real small-grid caller run consumes the five-state solver contract."""
    result = run_hull_free_surface_v2(_config())

    telemetry = result["topology_safety"]
    assert len(telemetry) == 2
    assert all(entry["finite"] for entry in telemetry)
    assert all(entry["directLG"] == 0 for entry in telemetry)
    assert result["topology_safety_status"] == "topology safety only"


def test_hull_caller_fails_closed_when_step_returns_direct_liquid_gas_topology(monkeypatch):
    """The caller independently rejects an invalid full-D3Q19 returned topology."""
    import tensorlbm.hull_free_surface_v2 as runner

    def invalid_step(f, fill, flags, solid, *args, **kwargs):
        bad_flags = torch.full_like(flags, GAS)
        bad_flags[solid] = 3
        fluid_cell = (~solid).nonzero(as_tuple=False)[0]
        bad_flags[tuple(fluid_cell.tolist())] = LIQUID
        return f, fill, bad_flags, kwargs["mass"], torch.tensor(0.0)

    monkeypatch.setattr(runner, "free_surface_step", invalid_step)

    with pytest.raises(ValueError, match="direct LIQUID-GAS"):
        run_hull_free_surface_v2(_config(n_steps=1))
