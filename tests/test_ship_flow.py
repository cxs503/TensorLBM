"""End-to-end smoke tests for run_ship_hull_flow.

Config validation and per-function unit tests are in tests/test_marine.py.
This module adds integration-level smoke runs that exercise the full runner
pipeline and verify the output artefacts.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tensorlbm.ship_flow import ShipHullFlowConfig, run_ship_hull_flow

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Smoke run – tiny grid, few steps, default Smagorinsky
# ---------------------------------------------------------------------------

def test_run_ship_hull_flow_smoke(tmp_path: Path) -> None:
    """Full pipeline smoke test: verify run_dir, metadata, and forces CSV."""
    config = ShipHullFlowConfig(
        nx=32, ny=16, nz=16,
        hull_type="wigley",
        u_in=0.05, re=100.0,
        hull_length=12.0, hull_beam=4.0, hull_draft=4.0,
        smagorinsky_cs=0.1,
        n_steps=4, output_interval=4,
        output_root=tmp_path,
        run_name="smoke",
        overwrite=True,
    )
    run_dir = run_ship_hull_flow(config)

    assert run_dir.exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "forces.csv").exists()
    assert (run_dir / "cad_summary.json").exists()
    assert (run_dir / "cad_preview.png").exists()
    assert (run_dir / "postprocess_summary.json").exists()
    assert (run_dir / "wake_profile.csv").exists()

    # Metadata must be valid JSON with expected top-level keys
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "config" in meta
    assert "derived" in meta
    assert "cad" in meta
    assert "diagnostics" in meta
    assert "postprocess" in meta
    assert len(meta["diagnostics"]) >= 1
    assert meta["postprocess"]["acceptance"]["drag_positive"] is True

    # Forces CSV must have the expected header
    lines = (run_dir / "forces.csv").read_text().splitlines()
    assert lines[0] == "step,cd,cs,cl,mx,my,mz"
    assert len(lines) == 2  # header + one data row


def test_run_ship_hull_flow_bgk_fallback(tmp_path: Path) -> None:
    """smagorinsky_cs=0 must fall back to pure BGK without errors."""
    config = ShipHullFlowConfig(
        nx=32, ny=16, nz=16,
        hull_type="series60",
        u_in=0.05, re=100.0,
        hull_length=12.0, hull_beam=4.0, hull_draft=4.0,
        smagorinsky_cs=0.0,
        n_steps=2, output_interval=2,
        output_root=tmp_path,
        run_name="bgk_smoke",
        export_stl=True,
        overwrite=True,
    )
    run_dir = run_ship_hull_flow(config)
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "cad_summary.json").exists()
    assert (run_dir / "hull.stl").exists()
