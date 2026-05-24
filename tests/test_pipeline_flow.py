"""Tests for the pipeline-flow benchmark."""
from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import (
    PipelineFlowConfig,
    make_pipeline_wall_mask,
    measure_strouhal,
    run_pipeline_flow,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestMeasureStrouhal:
    def test_short_series_returns_zero(self) -> None:
        assert measure_strouhal([0.0] * 10, 0.05, 6.0) == 0.0

    def test_sinusoid_returns_finite_float(self) -> None:
        series = [math.sin(2.0 * math.pi * 0.1 * idx) for idx in range(64)]
        value = measure_strouhal(series, 0.05, 6.0)
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value > 0.0


class TestMakePipelineWallMask:
    def test_shape_dtype_and_rows(self) -> None:
        obstacle = torch.zeros((10, 16), dtype=torch.bool)
        mask = make_pipeline_wall_mask(10, 16, obstacle, torch.device("cpu"))
        assert mask.shape == (10, 16)
        assert mask.dtype == torch.bool
        assert mask[0, :].all()
        assert mask[-1, :].all()


class TestPipelineFlowConfig:
    def test_valid_config(self) -> None:
        config = PipelineFlowConfig(nx=80, ny=60, diameter=8.0, gap_ratio=0.5)
        config.validate()
        assert config.nu > 0.0
        assert config.tau > 0.5

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"diameter": 1.0}, "diameter"),
            ({"gap_ratio": 3.0}, "gap_ratio"),
            ({"u_in": 0.05, "re": 1e16}, "tau"),
        ],
    )
    def test_validate_raises(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PipelineFlowConfig(**kwargs).validate()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        config = PipelineFlowConfig(nx=72, run_name="pipe")
        path = tmp_path / "pipeline.json"
        config.save(path)
        loaded = PipelineFlowConfig.load(path)
        assert loaded.nx == config.nx
        assert loaded.run_name == config.run_name


class TestRunPipelineFlow:
    def test_smoke(self, tmp_path: Path) -> None:
        config = PipelineFlowConfig(
            nx=60,
            ny=40,
            diameter=6.0,
            gap_ratio=0.5,
            n_steps=5,
            output_interval=5,
            output_root=tmp_path / "outputs",
            run_name="smoke",
            overwrite=True,
        )
        run_dir = run_pipeline_flow(config)
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "forces.csv").exists()
        assert (run_dir / "strouhal.json").exists()
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert metadata["config"]["n_steps"] == 5
