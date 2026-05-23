"""Smoke tests for the ship hull flow runner."""
from __future__ import annotations

import json

import pytest

from tensorlbm.ship_flow import ShipHullFlowConfig, run_ship_hull_flow


# ---------------------------------------------------------------------------
# ShipHullFlowConfig.validate() – error paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"nx": 8}, "nx ≥ 32"),
        ({"ny": 4}, "nx ≥ 32"),
        ({"nz": 4}, "nx ≥ 32"),
        ({"n_steps": 0}, "n_steps"),
        ({"output_interval": 0}, "output_interval"),
        ({"u_in": -0.05}, "u_in and re"),
        ({"re": 0.0}, "u_in and re"),
        ({"length_lbm": 200}, "length_lbm too large"),
        ({"beam_lbm": 60}, "beam_lbm too large"),
        ({"draft_lbm": 60}, "draft_lbm too large"),
    ],
)
def test_ship_config_validate_raises(overrides: dict, match: str) -> None:
    base = dict(
        nx=60, ny=24, nz=24,
        u_in=0.05, re=200.0,
        length_lbm=20, beam_lbm=4, draft_lbm=4,
        n_steps=5, output_interval=5,
    )
    base.update(overrides)
    cfg = ShipHullFlowConfig(**base)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_ship_config_tau_too_small() -> None:
    cfg = ShipHullFlowConfig(u_in=1e-9, re=1e12, length_lbm=20)
    with pytest.raises(ValueError, match="tau"):
        cfg.validate()


def test_ship_config_resolved_run_name_default() -> None:
    cfg = ShipHullFlowConfig(nx=60, ny=24, nz=24, re=200.0, n_steps=10)
    name = cfg.resolved_run_name()
    assert "wigley" in name
    assert "re200" in name


def test_ship_config_resolved_run_name_custom() -> None:
    cfg = ShipHullFlowConfig(run_name="my_run")
    assert cfg.resolved_run_name() == "my_run"


# ---------------------------------------------------------------------------
# Smoke run – tiny grid, few steps
# ---------------------------------------------------------------------------

def test_run_ship_hull_flow_smoke(tmp_path: pytest.TempPathFactory) -> None:
    config = ShipHullFlowConfig(
        nx=40, ny=20, nz=20,
        u_in=0.05, re=100.0,
        length_lbm=12, beam_lbm=4, draft_lbm=4,
        C_s=0.1,
        n_steps=4, output_interval=4,
        output_root=tmp_path,
        run_name="smoke",
        overwrite=True,
    )
    run_dir = run_ship_hull_flow(config)

    assert run_dir.exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "forces.csv").exists()

    # Metadata must be valid JSON and contain expected keys
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "config" in meta
    assert "derived" in meta
    assert "diagnostics" in meta
    assert len(meta["diagnostics"]) >= 1

    # Forces CSV must have the right header
    lines = (run_dir / "forces.csv").read_text().splitlines()
    assert lines[0] == "step,cd,cl,Mx,My,Mz"
    assert len(lines) == 2  # header + one data row


def test_run_ship_hull_flow_bgk_fallback(tmp_path: pytest.TempPathFactory) -> None:
    """C_s=0 must fall back to pure BGK without errors."""
    config = ShipHullFlowConfig(
        nx=40, ny=20, nz=20,
        u_in=0.05, re=100.0,
        length_lbm=12, beam_lbm=4, draft_lbm=4,
        C_s=0.0,
        n_steps=2, output_interval=2,
        output_root=tmp_path,
        run_name="bgk_smoke",
        overwrite=True,
    )
    run_dir = run_ship_hull_flow(config)
    assert (run_dir / "run_metadata.json").exists()
