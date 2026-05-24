"""Tests for the 3-D D3Q19 sphere-flow runner.

Targets ``tensorlbm.sphere_flow`` which previously had ~45% coverage
because ``run_sphere_flow`` was not exercised by any test.

Covers:
- ``SphereFlowConfig`` derived properties and run-name formatting.
- ``run_sphere_flow`` smoke run with output artifact checks.
- Resume from checkpoint path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm import SphereFlowConfig, run_sphere_flow

# ---------------------------------------------------------------------------
# Config-level coverage
# ---------------------------------------------------------------------------


class TestSphereFlowConfig:
    def test_post_init_normalises_output_root_and_device(self, tmp_path: Path) -> None:
        cfg = SphereFlowConfig(output_root=str(tmp_path), device="CPU")
        assert isinstance(cfg.output_root, Path)
        assert cfg.output_root == tmp_path
        assert cfg.device == "cpu"

    def test_post_init_normalises_resume_checkpoint(self, tmp_path: Path) -> None:
        cfg = SphereFlowConfig(resume_checkpoint=str(tmp_path))
        assert isinstance(cfg.resume_checkpoint, Path)
        assert cfg.resume_checkpoint == tmp_path

    def test_derived_properties(self) -> None:
        cfg = SphereFlowConfig(u_in=0.06, radius=8.0, re=50.0)
        assert cfg.nu == pytest.approx(0.06 * 2 * 8.0 / 50.0)
        assert cfg.tau == pytest.approx(3.0 * cfg.nu + 0.5)

    def test_resolved_run_name_default_integer_re(self) -> None:
        cfg = SphereFlowConfig(nx=40, ny=20, nz=20, re=50.0, u_in=0.06, n_steps=10)
        name = cfg.resolved_run_name()
        assert name == "nx40_ny20_nz20_re50_uin0.060_steps10"

    def test_resolved_run_name_default_non_integer_re(self) -> None:
        cfg = SphereFlowConfig(nx=40, ny=20, nz=20, re=12.5, u_in=0.06, n_steps=10)
        name = cfg.resolved_run_name()
        assert "re12.5" in name

    def test_resolved_run_name_custom(self) -> None:
        cfg = SphereFlowConfig(run_name="custom")
        assert cfg.resolved_run_name() == "custom"


# ---------------------------------------------------------------------------
# Runner smoke tests
# ---------------------------------------------------------------------------


class TestRunSphereFlow:
    def _smoke_cfg(self, tmp_path: Path, **overrides) -> SphereFlowConfig:
        kwargs = {
            "nx": 32,
            "ny": 16,
            "nz": 16,
            "u_in": 0.05,
            "re": 50.0,
            "radius": 3.0,
            "n_steps": 4,
            "output_interval": 2,
            "output_root": tmp_path,
            "run_name": "smoke",
            "overwrite": True,
        }
        kwargs.update(overrides)
        return SphereFlowConfig(**kwargs)

    def test_smoke_run_produces_expected_artifacts(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path)
        run_dir = run_sphere_flow(cfg)

        # Returned path is correctly resolved.
        assert run_dir == tmp_path / "sphere_flow" / "smoke"
        assert run_dir.is_dir()

        # Metadata file written.
        meta_path = run_dir / "run_metadata.json"
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["config"]["n_steps"] == 4
        assert metadata["derived"]["nu"] == pytest.approx(cfg.nu)
        assert metadata["derived"]["tau"] == pytest.approx(cfg.tau)
        assert "diagnostics" in metadata
        # Diagnostics are recorded every output_interval steps (and at the end).
        # With n_steps=4 and output_interval=2 we expect entries at 2 and 4.
        assert len(metadata["diagnostics"]) == 2
        for entry in metadata["diagnostics"]:
            for key in ("step", "mass", "mass_drift", "max_speed", "mean_rho"):
                assert key in entry

        # PNG snapshots written at output steps.
        assert (run_dir / "flow_step_000002.png").exists()
        assert (run_dir / "flow_step_000004.png").exists()

        # Checkpoints saved at every output step.
        assert (run_dir / "checkpoint_f.pt").exists()
        assert (run_dir / "checkpoint_meta.json").exists()

    def test_smoke_run_finite_diagnostics(self, tmp_path: Path) -> None:
        """Mass and velocity diagnostics should be finite, non-NaN values."""
        import math

        cfg = self._smoke_cfg(tmp_path, run_name="finite_check")
        run_dir = run_sphere_flow(cfg)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        for entry in metadata["diagnostics"]:
            for key in ("mass", "mass_drift", "max_speed", "mean_rho"):
                value = entry[key]
                assert isinstance(value, (int, float))
                assert math.isfinite(value), f"non-finite {key}={value}"

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        """Re-running with a resume_checkpoint pointed at a previous run continues from step+1."""
        # First run: produces a checkpoint at step 4.
        cfg1 = self._smoke_cfg(tmp_path, run_name="first")
        run_dir1 = run_sphere_flow(cfg1)
        assert (run_dir1 / "checkpoint_f.pt").exists()

        # Second run: resume from that checkpoint, do 2 more steps.
        cfg2 = SphereFlowConfig(
            nx=32, ny=16, nz=16,
            u_in=0.05, re=50.0, radius=3.0,
            n_steps=6, output_interval=2,
            output_root=tmp_path,
            run_name="second",
            overwrite=True,
            resume_checkpoint=run_dir1,
        )
        run_dir2 = run_sphere_flow(cfg2)

        metadata = json.loads((run_dir2 / "run_metadata.json").read_text(encoding="utf-8"))
        # We resumed at step=4 so only step 6 should be recorded.
        steps = [entry["step"] for entry in metadata["diagnostics"]]
        assert steps == [6]
