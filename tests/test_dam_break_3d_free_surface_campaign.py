"""Long free-surface caller campaign ledger regression."""
from __future__ import annotations

import json

import pytest

from tensorlbm.dam_break_3d import DamBreak3DConfig, run_dam_break_3d


def test_free_surface_dam_break_caller_writes_a_101_step_quality_curve(tmp_path) -> None:
    """The real public caller must retain state and publish every step's budget."""
    config = DamBreak3DConfig(
        nx=32, ny=16, nz=16, dam_width=10, fill_height=8,
        model="fs", n_steps=101, output_interval=101,
        gravity=0.0, A=0.0, output_root=tmp_path, run_name="fs_101",
    )
    metadata = json.loads((run_dam_break_3d(config) / "run_metadata.json").read_text())

    curve = metadata["free_surface_quality_curve"]
    assert len(curve) == 101
    assert [record["step"] for record in curve] == list(range(1, 102))
    assert metadata["free_surface_quality_gate"]["passed"] is True
    assert metadata["free_surface_quality_gate"]["topology_changed"] is True

    required = {
        "mass_drift", "unexplained_residual", "paired_residual", "directLG",
        "conversion", "redistribution", "finite", "flags_finite",
    }
    assert all(required <= record.keys() for record in curve)
    assert all(record["directLG"] == 0 and record["finite"] for record in curve)
    assert any(abs(record["conversion"]) > 1.0e-5 for record in curve)
    assert any(abs(record["redistribution"]) > 1.0e-5 for record in curve)


def test_free_surface_caller_fails_closed_when_accounting_tolerance_is_exceeded(tmp_path) -> None:
    config = DamBreak3DConfig(
        nx=32, ny=16, nz=16, dam_width=10, fill_height=8,
        model="fs", n_steps=10, output_root=tmp_path, run_name="fs_fail_closed",
        free_surface_unexplained_tolerance=0.0,
    )
    with pytest.raises(RuntimeError, match="quality gate fail-closed"):
        run_dam_break_3d(config)
