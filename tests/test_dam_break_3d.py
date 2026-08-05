"""W9-04 contract tests for the dam-break 3D free-surface runner."""
from __future__ import annotations

import json

import pytest
import torch

from tensorlbm.dam_break_3d import DamBreak3DConfig, run_dam_break_3d
from tensorlbm.free_surface_lbm import GAS, LIQUID, init_fill_rectangular, init_flags_from_fill


def _fs_config(tmp_path, **overrides):
    kwargs = {
        "nx": 32,
        "ny": 16,
        "nz": 16,
        "dam_width": 8,
        "fill_height": 8,
        "model": "fs",
        "n_steps": 2,
        "output_interval": 1,
        "output_root": tmp_path,
        "run_name": "five_state",
        "overwrite": True,
        "hydrostatic_init": False,
    }
    kwargs.update(overrides)
    return DamBreak3DConfig(**kwargs)


def test_fs_runner_handoffs_f_fill_flags_mass_and_df_for_two_steps(tmp_path):
    """The real FS runner must retain every free_surface_step return state."""
    run_dir = run_dam_break_3d(_fs_config(tmp_path))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    handoff = metadata["fs_handoff"]
    assert [entry["step"] for entry in handoff] == [1, 2]
    for entry in handoff:
        assert entry["f_shape"] == [19, 16, 16, 32]
        assert entry["fill_shape"] == [16, 16, 32]
        assert entry["flags_shape"] == [16, 16, 32]
        assert entry["mass_shape"] == [16, 16, 32]
        assert isinstance(entry["df"], float)
        assert entry["mass_is_independent"] is True


def test_fs_runner_does_not_rebuild_mass_from_fill_between_steps(tmp_path, monkeypatch):
    """A mass/fill mismatch after step 1 must reach step 2 unchanged."""
    import tensorlbm.dam_break_3d as runner

    real_step = runner.free_surface_step
    seen_masses = []
    injected_at = []

    def instrumented_step(f, fill, flags, solid, *args, **kwargs):
        seen_masses.append(kwargs.get("mass"))
        result = real_step(f, fill, flags, solid, *args, **kwargs)
        if len(seen_masses) == 1:
            f_next, fill_next, flags_next, mass_next, df_next = result
            interface = flags_next == 2
            assert bool(interface.any())
            mass_next = mass_next.clone()
            mass_next[interface] += 0.125
            injected_at.append(interface.nonzero(as_tuple=True))
            return f_next, fill_next, flags_next, mass_next, df_next
        return result

    monkeypatch.setattr(runner, "free_surface_step", instrumented_step)
    run_dam_break_3d(_fs_config(tmp_path, run_name="independent_mass"))

    assert seen_masses[0] is not None
    assert seen_masses[1] is not None
    assert injected_at
    assert torch.allclose(seen_masses[1][injected_at[0]], torch.full_like(
        seen_masses[1][injected_at[0]], 0.125
    ), atol=1e-6)


def test_initial_envelope_repair_blocks_direct_liquid_gas_failure():
    """Repair is necessary: removing it recreates direct L/G rejection."""
    fill, solid = init_fill_rectangular(16, 16, 32, 8.0, 8.0, torch.device("cpu"))
    repaired = init_flags_from_fill(fill, solid)
    from tensorlbm.free_surface_lbm import _assert_no_direct_liquid_gas_links

    _assert_no_direct_liquid_gas_links(repaired)

    broken = repaired.clone()
    broken[(broken == 2) & (fill == 0)] = GAS
    with pytest.raises(ValueError, match="direct LIQUID-GAS"):
        _assert_no_direct_liquid_gas_links(broken)


def test_fs_runner_fails_when_envelope_repair_is_mutated_away(tmp_path, monkeypatch):
    """Integration mutation reproduces the solver's direct-L/G guard."""
    import tensorlbm.dam_break_3d as runner

    real_init = runner.init_flags_from_fill

    def without_envelope(fill, solid):
        flags = real_init(fill, solid)
        flags[(flags == 2) & (fill == 0)] = GAS
        return flags

    monkeypatch.setattr(runner, "init_flags_from_fill", without_envelope)
    with pytest.raises(ValueError, match="direct LIQUID-GAS"):
        run_dam_break_3d(_fs_config(tmp_path, run_name="broken_envelope"))
