"""Smoke tests for the DG-LBM hybrid SUBOFF case.

Covers:
- ``DGLBMSuboffConfig`` derived properties and validation.
- ``build_dg_hull_band_mask`` geometry correctness.
- ``run_dg_lbm_suboff_flow`` end-to-end smoke run with artefact checks.
- Resume-from-checkpoint behaviour.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm import DGLBMSuboffConfig, build_dg_hull_band_mask, run_dg_lbm_suboff_flow


# ---------------------------------------------------------------------------
# DGLBMSuboffConfig
# ---------------------------------------------------------------------------


class TestDGLBMSuboffConfig:
    def test_post_init_normalises(self, tmp_path: Path) -> None:
        cfg = DGLBMSuboffConfig(output_root=str(tmp_path), device="CPU")
        assert isinstance(cfg.output_root, Path)
        assert cfg.device == "cpu"

    def test_post_init_normalises_checkpoint(self, tmp_path: Path) -> None:
        cfg = DGLBMSuboffConfig(resume_checkpoint=str(tmp_path))
        assert isinstance(cfg.resume_checkpoint, Path)

    def test_derived_properties(self) -> None:
        cfg = DGLBMSuboffConfig(u_in=0.06, hull_length=120.0, re=200.0)
        assert cfg.nu == pytest.approx(0.06 * 120.0 / 200.0)
        assert cfg.tau == pytest.approx(3.0 * cfg.nu + 0.5)

    def test_run_name_default_contains_hull_type(self) -> None:
        cfg = DGLBMSuboffConfig(
            nx=40, ny=20, nz=20,
            re=200.0, u_in=0.06, hull_length=24.0,
            dg_band=4.0, n_steps=10,
        )
        name = cfg.resolved_run_name()
        assert "re200" in name
        assert "bare_hull" in name
        assert "dg4.0" in name

    def test_run_name_custom(self) -> None:
        assert DGLBMSuboffConfig(run_name="myrun").resolved_run_name() == "myrun"

    def test_validate_raises_small_grid(self) -> None:
        with pytest.raises(ValueError, match="nx, ny, nz"):
            DGLBMSuboffConfig(nx=4).validate()

    def test_validate_raises_dg_band_zero(self) -> None:
        with pytest.raises(ValueError, match="dg_band"):
            DGLBMSuboffConfig(dg_band=0.0).validate()

    def test_validate_raises_bad_dg_order(self) -> None:
        with pytest.raises(ValueError, match="dg_order"):
            DGLBMSuboffConfig(dg_order=2).validate()

    def test_validate_passes_defaults(self) -> None:
        DGLBMSuboffConfig().validate()


# ---------------------------------------------------------------------------
# build_dg_hull_band_mask
# ---------------------------------------------------------------------------


class TestBuildDGHullBandMask:
    def _sphere_solid(self, nx: int = 32, ny: int = 32, nz: int = 32) -> torch.Tensor:
        """Small sphere solid mask for geometry tests."""
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        return r2 <= 5.0 ** 2

    def test_band_excludes_solid(self) -> None:
        solid = self._sphere_solid()
        band = build_dg_hull_band_mask(solid, dg_band=4.0)
        assert not (band & solid).any(), "DG band must not overlap solid"

    def test_band_is_nonempty(self) -> None:
        solid = self._sphere_solid()
        band = build_dg_hull_band_mask(solid, dg_band=4.0)
        assert band.any(), "DG band should contain at least one cell"

    def test_band_shape_preserved(self) -> None:
        solid = self._sphere_solid(20, 24, 28)
        band = build_dg_hull_band_mask(solid, dg_band=3.0)
        assert band.shape == solid.shape

    def test_band_dtype_is_bool(self) -> None:
        solid = self._sphere_solid()
        band = build_dg_hull_band_mask(solid, dg_band=4.0)
        assert band.dtype == torch.bool

    def test_thicker_band_covers_more_cells(self) -> None:
        solid = self._sphere_solid()
        band2 = build_dg_hull_band_mask(solid, dg_band=2.0)
        band6 = build_dg_hull_band_mask(solid, dg_band=6.0)
        assert band6.sum() >= band2.sum()

    def test_empty_solid_gives_empty_band(self) -> None:
        solid = torch.zeros(10, 10, 10, dtype=torch.bool)
        band = build_dg_hull_band_mask(solid, dg_band=4.0)
        assert not band.any()


# ---------------------------------------------------------------------------
# run_dg_lbm_suboff_flow – smoke tests
# ---------------------------------------------------------------------------


class TestRunDGLBMSuboffFlow:
    def _smoke_cfg(self, tmp_path: Path, **overrides) -> DGLBMSuboffConfig:
        kwargs: dict = {
            "nx": 48, "ny": 24, "nz": 24,
            "u_in": 0.05, "re": 100.0,
            "hull_length": 28.0, "hull_type": "bare_hull",
            "dg_band": 3.0,
            "n_steps": 4, "output_interval": 2,
            "output_root": tmp_path, "run_name": "smoke", "overwrite": True,
        }
        kwargs.update(overrides)
        return DGLBMSuboffConfig(**kwargs)

    def test_smoke_run_produces_artifacts(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path)
        run_dir = run_dg_lbm_suboff_flow(cfg)

        assert run_dir == tmp_path / "dg_lbm_suboff" / "smoke"
        assert run_dir.is_dir()

        meta_path = run_dir / "run_metadata.json"
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["config"]["n_steps"] == 4
        assert "diagnostics" in metadata
        assert len(metadata["diagnostics"]) == 2  # steps 2 and 4
        assert "hull_stats" in metadata

        for entry in metadata["diagnostics"]:
            for key in ("step", "mass", "mass_drift", "max_speed", "mean_rho"):
                assert key in entry

        assert (run_dir / "flow_step_000002.png").exists()
        assert (run_dir / "flow_step_000004.png").exists()
        assert (run_dir / "checkpoint_f.pt").exists()
        assert (run_dir / "checkpoint_meta.json").exists()

    def test_diagnostics_are_finite(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path, run_name="finite")
        run_dir = run_dg_lbm_suboff_flow(cfg)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        for entry in metadata["diagnostics"]:
            for key in ("mass", "mass_drift", "max_speed", "mean_rho"):
                value = entry[key]
                assert isinstance(value, (int, float))
                assert math.isfinite(value), f"non-finite {key}={value}"

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        cfg1 = self._smoke_cfg(tmp_path, run_name="first")
        run_dir1 = run_dg_lbm_suboff_flow(cfg1)
        assert (run_dir1 / "checkpoint_f.pt").exists()

        cfg2 = DGLBMSuboffConfig(
            nx=48, ny=24, nz=24,
            u_in=0.05, re=100.0,
            hull_length=28.0, hull_type="bare_hull", dg_band=3.0,
            n_steps=6, output_interval=2,
            output_root=tmp_path, run_name="second", overwrite=True,
            resume_checkpoint=run_dir1,
        )
        run_dir2 = run_dg_lbm_suboff_flow(cfg2)
        metadata = json.loads((run_dir2 / "run_metadata.json").read_text(encoding="utf-8"))
        steps = [entry["step"] for entry in metadata["diagnostics"]]
        assert steps == [6]

    def test_smoke_run_with_sail(self, tmp_path: Path) -> None:
        """Check that with_sail hull type also produces valid outputs."""
        cfg = self._smoke_cfg(tmp_path, run_name="with_sail", hull_type="with_sail")
        run_dir = run_dg_lbm_suboff_flow(cfg)
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert meta["config"]["hull_type"] == "with_sail"
        assert len(meta["diagnostics"]) == 2
