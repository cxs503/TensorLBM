"""Tests for tensorlbm.dam_break – DamBreakConfig and run_dam_break."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from tensorlbm import DamBreakConfig, run_dam_break

# ---------------------------------------------------------------------------
# DamBreakConfig validation
# ---------------------------------------------------------------------------


class TestDamBreakConfigValidation:
    def _base(self, **overrides: object) -> DamBreakConfig:
        kwargs: dict = {
            "nx": 40, "ny": 24, "dam_width": 10,
            "model": "cg",
            "rho_heavy": 0.8, "rho_light": 0.4,
            "G": 0.9, "tau": 1.0, "g": 5e-5,
            "n_steps": 4, "output_interval": 4,
        }
        kwargs.update(overrides)
        return DamBreakConfig(**kwargs)

    def test_valid_config_does_not_raise(self) -> None:
        self._base().validate()

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"nx": 4}, "at least 16"),
            ({"ny": 4}, "at least 16"),
            ({"dam_width": 0}, "dam_width"),
            ({"dam_width": 40}, "dam_width"),
            ({"rho_heavy": 0.3, "rho_light": 0.5}, "rho_heavy"),
            ({"rho_heavy": 0.4, "rho_light": 0.4}, "rho_heavy"),
            ({"tau": 0.5}, "tau"),
            ({"tau": 0.3}, "tau"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        cfg = self._base(**overrides)
        with pytest.raises(ValueError, match=match):
            cfg.validate()

    def test_resolved_run_name_default(self) -> None:
        cfg = self._base()
        name = cfg.resolved_run_name()
        assert "dam" in name
        assert "cg" in name
        assert "nx40" in name

    def test_resolved_run_name_custom(self) -> None:
        cfg = self._base(run_name="my_run")
        assert cfg.resolved_run_name() == "my_run"

    def test_output_root_is_path(self) -> None:
        cfg = DamBreakConfig(output_root="some/dir")
        assert isinstance(cfg.output_root, Path)

    def test_device_lowercased(self) -> None:
        cfg = DamBreakConfig(device="CPU")
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# DamBreakConfig: all four model variants
# ---------------------------------------------------------------------------


class TestDamBreakConfigModels:
    @pytest.mark.parametrize("model", ["cg", "sc", "scmp", "fe"])
    def test_model_accepted(self, model: str) -> None:
        cfg = DamBreakConfig(nx=32, ny=20, dam_width=8, model=model)
        cfg.validate()


# ---------------------------------------------------------------------------
# run_dam_break – smoke tests for each model
# ---------------------------------------------------------------------------


class TestRunDamBreakSmoke:
    """Minimal smoke tests – tiny grids, few steps, checks file outputs."""

    def _cfg(self, tmp_path: Path, model: str, **kwargs: object) -> DamBreakConfig:
        base: dict = {
            "nx": 32, "ny": 20, "dam_width": 8,
            "model": model,
            "rho_heavy": 0.8, "rho_light": 0.4,
            "G": 0.9, "tau": 1.0, "g": 5e-5,
            "n_steps": 4, "output_interval": 4,
            "output_root": tmp_path,
            "run_name": f"smoke_{model}",
            "overwrite": True,
        }
        base.update(kwargs)
        return DamBreakConfig(**base)

    @pytest.mark.parametrize("model", ["cg", "sc", "scmp", "fe"])
    def test_smoke_run_creates_output_files(self, model: str, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path, model)
        run_dir = run_dam_break(cfg)
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "front_position.csv").exists()

    @pytest.mark.parametrize("model", ["cg", "sc", "scmp", "fe"])
    def test_smoke_run_metadata_contents(self, model: str, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path, model)
        run_dir = run_dam_break(cfg)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["config"]["model"] == model
        assert metadata["config"]["n_steps"] == 4
        assert len(metadata["diagnostics"]) >= 1

    @pytest.mark.parametrize("model", ["cg", "sc", "scmp", "fe"])
    def test_smoke_run_front_csv(self, model: str, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path, model)
        run_dir = run_dam_break(cfg)
        with (run_dir / "front_position.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 1
        assert "t_star" in rows[0]
        assert "X_star" in rows[0]

    @pytest.mark.parametrize("model", ["cg", "sc", "scmp", "fe"])
    def test_smoke_run_snapshot_created(self, model: str, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path, model)
        run_dir = run_dam_break(cfg)
        pngs = list(run_dir.glob("snapshot_*.png"))
        assert len(pngs) >= 1

    def test_run_returns_path(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path, "cg")
        result = run_dam_break(cfg)
        assert isinstance(result, Path)
        assert result.exists()

    def test_overwrite_false_raises_if_exists(self, tmp_path: Path) -> None:
        """Second run with overwrite=False should raise when directory exists."""
        cfg1 = self._cfg(tmp_path, "cg", overwrite=True)
        run_dam_break(cfg1)
        cfg2 = self._cfg(tmp_path, "cg", overwrite=False)
        with pytest.raises((FileExistsError, OSError)):
            run_dam_break(cfg2)

    def test_unknown_model_raises(self, tmp_path: Path) -> None:
        """An unsupported model string must raise ValueError at runtime."""
        cfg = DamBreakConfig(
            nx=32, ny=20, dam_width=8,
            model="unknown",  # type: ignore[arg-type]
            n_steps=2, output_interval=2,
            output_root=tmp_path,
            run_name="unknown_model_test",
            overwrite=True,
        )
        with pytest.raises(ValueError, match="Unknown model"):
            run_dam_break(cfg)


# ---------------------------------------------------------------------------
# Internal helpers (imported directly for unit testing)
# ---------------------------------------------------------------------------


class TestWallMask:
    def test_shape(self) -> None:
        from tensorlbm.dam_break import _wall_mask
        mask = _wall_mask(ny=20, nx=32, device=torch.device("cpu"))
        assert mask.shape == (20, 32)
        assert mask.dtype == torch.bool

    def test_borders_are_true(self) -> None:
        from tensorlbm.dam_break import _wall_mask
        mask = _wall_mask(ny=20, nx=32, device=torch.device("cpu"))
        assert mask[0, :].all(), "bottom row"
        assert mask[-1, :].all(), "top row"
        assert mask[:, 0].all(), "left column"
        assert mask[:, -1].all(), "right column"

    def test_interior_is_false(self) -> None:
        from tensorlbm.dam_break import _wall_mask
        mask = _wall_mask(ny=20, nx=32, device=torch.device("cpu"))
        assert not mask[1:-1, 1:-1].any()


class TestSmoothProfile:
    def test_shape(self) -> None:
        from tensorlbm.dam_break import _smooth_profile
        prof = _smooth_profile(nx=32, dam_width=10, width=3.0, device=torch.device("cpu"))
        assert prof.shape == (1, 32)

    def test_values_in_range(self) -> None:
        from tensorlbm.dam_break import _smooth_profile
        prof = _smooth_profile(nx=32, dam_width=10, width=3.0, device=torch.device("cpu"))
        assert (prof >= 0.0).all()
        assert (prof <= 1.0).all()

    def test_high_inside_dam(self) -> None:
        from tensorlbm.dam_break import _smooth_profile
        prof = _smooth_profile(nx=64, dam_width=20, width=2.0, device=torch.device("cpu"))
        # Well inside the dam the profile should be close to 1
        assert float(prof[0, 5].item()) > 0.9

    def test_low_outside_dam(self) -> None:
        from tensorlbm.dam_break import _smooth_profile
        prof = _smooth_profile(nx=64, dam_width=20, width=2.0, device=torch.device("cpu"))
        # Well outside the dam the profile should be close to 0
        assert float(prof[0, 50].item()) < 0.1


class TestFindFrontX:
    def test_returns_float(self) -> None:
        from tensorlbm.dam_break import _find_front_x
        ny, nx = 12, 32
        rho_heavy = torch.zeros((ny, nx))
        rho_heavy[:, :16] = 0.8
        rho_light = torch.ones((ny, nx)) * 0.4
        result = _find_front_x(rho_heavy, rho_light, torch.zeros((ny, nx), dtype=torch.bool))
        assert isinstance(result, float)

    def test_no_heavy_phase_returns_zero(self) -> None:
        from tensorlbm.dam_break import _find_front_x
        ny, nx = 12, 32
        rho_heavy = torch.zeros((ny, nx))
        rho_light = torch.ones((ny, nx)) * 0.8
        result = _find_front_x(rho_heavy, rho_light, torch.zeros((ny, nx), dtype=torch.bool))
        assert result == 0.0

    def test_full_heavy_phase_returns_last_column(self) -> None:
        from tensorlbm.dam_break import _find_front_x
        ny, nx = 12, 32
        rho_heavy = torch.ones((ny, nx)) * 0.8
        rho_light = torch.zeros((ny, nx))
        result = _find_front_x(rho_heavy, rho_light, torch.zeros((ny, nx), dtype=torch.bool))
        assert result == pytest.approx(nx - 1, abs=1)
