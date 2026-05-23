"""Tests for the Backward-Facing Step benchmark.

Covers:
* make_bfs_solid_mask – geometry
* measure_reattachment_length – detection logic
* BackwardFacingStepConfig – validation, properties, round-trip
* run_backward_facing_step – smoke test
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import (
    BackwardFacingStepConfig,
    make_bfs_solid_mask,
    measure_reattachment_length,
    run_backward_facing_step,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# make_bfs_solid_mask
# ---------------------------------------------------------------------------


class TestMakeBfsSolidMask:
    def test_shape(self) -> None:
        ny, nx, sh, xs = 40, 200, 20, 40
        mask = make_bfs_solid_mask(ny, nx, sh, xs, torch.device("cpu"))
        assert mask.shape == (ny, nx)

    def test_top_wall_fully_solid(self) -> None:
        ny, nx, sh, xs = 40, 200, 20, 40
        mask = make_bfs_solid_mask(ny, nx, sh, xs, torch.device("cpu"))
        assert mask[-1, :].all(), "Top wall should be fully solid"

    def test_bottom_wall_after_step(self) -> None:
        ny, nx, sh, xs = 40, 200, 20, 40
        mask = make_bfs_solid_mask(ny, nx, sh, xs, torch.device("cpu"))
        assert mask[0, xs:].all(), "Bottom wall after step should be solid"

    def test_bottom_wall_before_step_not_solid(self) -> None:
        """The bottom row INSIDE the step block is part of the step solid, but
        that is captured by the step_block condition (xx < x_step & yy < step_h).
        Rows inside the step block are solid; rows of the bottom wall before the
        step that are above step_h are NOT solid."""
        ny, nx, sh, xs = 40, 200, 10, 40
        mask = make_bfs_solid_mask(ny, nx, sh, xs, torch.device("cpu"))
        # At x=0, y=ny//2 (above step) should be fluid (not solid)
        assert not mask[ny // 2, 0].item(), "Cell above step at x=0 should be fluid"

    def test_step_block_is_solid(self) -> None:
        ny, nx, sh, xs = 40, 200, 20, 40
        mask = make_bfs_solid_mask(ny, nx, sh, xs, torch.device("cpu"))
        # Check a cell inside the step block
        assert mask[sh // 2, xs // 2].item(), "Cell inside step block should be solid"

    def test_post_step_interior_is_fluid(self) -> None:
        ny, nx, sh, xs = 40, 200, 20, 40
        mask = make_bfs_solid_mask(ny, nx, sh, xs, torch.device("cpu"))
        # Interior cell well inside the post-step channel
        mid_y = (sh + ny - 1) // 2
        mid_x = (xs + nx - 1) // 2
        assert not mask[mid_y, mid_x].item(), "Interior post-step cell should be fluid"


# ---------------------------------------------------------------------------
# measure_reattachment_length
# ---------------------------------------------------------------------------


class TestMeasureReattachmentLength:
    def test_zero_when_no_recirculation(self) -> None:
        """If ux > 0 everywhere, reattachment is immediately at the step."""
        ny, nx = 40, 200
        ux = torch.ones((ny, nx)) * 0.05
        xr = measure_reattachment_length(ux, x_step=40, step_h=20)
        assert xr == 0.0

    def test_detects_reattachment(self) -> None:
        """Construct a synthetic ux field with a clear reattachment at column 100."""
        ny, nx = 40, 200
        x_step = 40
        step_h = 20
        ux = torch.ones((ny, nx)) * 0.05
        # Simulate recirculation zone at row 1, columns x_step to 99
        ux[1, x_step:x_step + 60] = -0.01  # negative (recirculation)
        ux[1, x_step + 60:] = 0.05          # positive (reattached)
        xr = measure_reattachment_length(ux, x_step=x_step, step_h=step_h)
        expected = 60.0 / step_h
        assert abs(xr - expected) < 1.0 / step_h + 0.01

    def test_returns_zero_when_fully_recirculating(self) -> None:
        """If the bottom row never goes positive, return 0."""
        ny, nx = 40, 200
        ux = torch.ones((ny, nx)) * 0.05
        ux[1, :] = -0.01
        xr = measure_reattachment_length(ux, x_step=40, step_h=20)
        assert xr == 0.0


# ---------------------------------------------------------------------------
# BackwardFacingStepConfig
# ---------------------------------------------------------------------------


class TestBackwardFacingStepConfig:
    def test_defaults(self) -> None:
        cfg = BackwardFacingStepConfig()
        assert cfg.nx == 400
        assert cfg.ny == 80
        assert cfg.step_h == 40
        assert cfg.re == 100.0

    def test_nu_property(self) -> None:
        cfg = BackwardFacingStepConfig(u_in=0.05, step_h=40, re=100.0)
        assert abs(cfg.nu - 0.05 * 40 / 100.0) < 1e-10

    def test_tau_property(self) -> None:
        cfg = BackwardFacingStepConfig(u_in=0.05, step_h=40, re=100.0)
        assert abs(cfg.tau - (3.0 * cfg.nu + 0.5)) < 1e-10

    def test_validate_bad_step_h(self) -> None:
        with pytest.raises(ValueError, match="step_h"):
            BackwardFacingStepConfig(ny=20, step_h=18).validate()

    def test_validate_zero_u_in_raises(self) -> None:
        with pytest.raises(ValueError, match="u_in"):
            BackwardFacingStepConfig(u_in=0.0, step_h=40, re=100.0).validate()

    def test_validate_bad_x_step(self) -> None:
        with pytest.raises(ValueError, match="x_step"):
            BackwardFacingStepConfig(nx=20, x_step=0).validate()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        cfg = BackwardFacingStepConfig(nx=200, re=200.0, run_name="rt")
        p = tmp_path / "bfs_cfg.json"
        cfg.save(p)
        cfg2 = BackwardFacingStepConfig.load(p)
        assert cfg2.nx == cfg.nx
        assert cfg2.re == cfg.re
        assert cfg2.run_name == cfg.run_name


# ---------------------------------------------------------------------------
# run_backward_facing_step – smoke test
# ---------------------------------------------------------------------------


def test_backward_facing_step_smoke(tmp_path: Path) -> None:
    """Smoke test: minimal grid, few steps, check required output files exist."""
    config = BackwardFacingStepConfig(
        nx=60,
        ny=20,
        step_h=10,
        x_step=12,
        u_in=0.05,
        re=50.0,
        n_steps=4,
        output_interval=2,
        output_root=tmp_path / "outputs",
        run_name="smoke",
        overwrite=True,
    )
    run_dir = run_backward_facing_step(config)
    assert run_dir.exists()

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["config"]["n_steps"] == 4
    assert meta["diagnostics"]
    assert "final_reattachment_xr_star" in meta
    assert (run_dir / "snapshot_000004.png").exists()
    assert (run_dir / "reattachment.csv").exists()


def test_backward_facing_step_reattachment_key(tmp_path: Path) -> None:
    """Metadata must contain the reattachment length key."""
    config = BackwardFacingStepConfig(
        nx=60,
        ny=20,
        step_h=10,
        x_step=12,
        u_in=0.05,
        re=50.0,
        n_steps=2,
        output_interval=2,
        output_root=tmp_path / "out",
        run_name="r",
        overwrite=True,
    )
    run_dir = run_backward_facing_step(config)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert "final_reattachment_xr_star" in meta
    # The value should be a non-negative float
    assert meta["final_reattachment_xr_star"] >= 0.0


def test_backward_facing_step_overwrite(tmp_path: Path) -> None:
    """Running twice with overwrite=True does not raise."""
    config = BackwardFacingStepConfig(
        nx=60,
        ny=20,
        step_h=10,
        x_step=12,
        u_in=0.05,
        re=50.0,
        n_steps=2,
        output_interval=2,
        output_root=tmp_path / "out",
        run_name="ow",
        overwrite=True,
    )
    run_backward_facing_step(config)
    run_backward_facing_step(config)  # should not raise
