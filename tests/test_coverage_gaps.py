"""Tests targeting specific coverage gaps identified by the pytest-cov report.

Modules covered:
- unit_converter.LBMUnitConverter  – l_phys/u_phys/nu_phys validation errors and tau > 2.0 warning
- config_io                        – load_config_yaml (fully uncovered), load_config_json
                                     skip-unknown-key and non-dataclass branches, coercion failure
- ai/model._activation             – relu / gelu / invalid-name branches
                                     + wrong feature-count forward
- ai/dataset                       – strain_rate_tensor_2d shape error, extract_les_samples_2d mask
                                     mismatch, extract_les_samples_2d_multi empty,
                                     split bad fraction
- ai/train                         – _r2_score zero-variance, load dataset from Path,
                                     too-small dataset
- ai/database                      – get_model_record missing ID, _rows_to_dicts malformed JSON
- ai/pipeline                      – AIPipelineResult.to_dict, no-snapshot fallback in smoke test
- multiphase3d                     – collide_sc_single_component_3d
                                     (entire function + solid_mask branch)
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# unit_converter – missing validation errors and tau > 2.0 warning
# ---------------------------------------------------------------------------

class TestLBMUnitConverterAdditionalValidation:
    """Cover the three remaining ValueError guards in LBMUnitConverter.__init__."""

    def test_non_positive_l_phys_raises(self) -> None:
        from tensorlbm.unit_converter import LBMUnitConverter

        with pytest.raises(ValueError, match="l_phys must be positive"):
            LBMUnitConverter(re=100.0, l_phys=0.0, u_phys=1.0, nu_phys=0.01, nx=100)

    def test_negative_l_phys_raises(self) -> None:
        from tensorlbm.unit_converter import LBMUnitConverter

        with pytest.raises(ValueError, match="l_phys must be positive"):
            LBMUnitConverter(re=100.0, l_phys=-1.0, u_phys=1.0, nu_phys=0.01, nx=100)

    def test_non_positive_u_phys_raises(self) -> None:
        from tensorlbm.unit_converter import LBMUnitConverter

        with pytest.raises(ValueError, match="u_phys must be positive"):
            LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=0.0, nu_phys=0.01, nx=100)

    def test_negative_u_phys_raises(self) -> None:
        from tensorlbm.unit_converter import LBMUnitConverter

        with pytest.raises(ValueError, match="u_phys must be positive"):
            LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=-0.5, nu_phys=0.01, nx=100)

    def test_non_positive_nu_phys_raises(self) -> None:
        from tensorlbm.unit_converter import LBMUnitConverter

        with pytest.raises(ValueError, match="nu_phys must be positive"):
            LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.0, nx=100)

    def test_negative_nu_phys_raises(self) -> None:
        from tensorlbm.unit_converter import LBMUnitConverter

        with pytest.raises(ValueError, match="nu_phys must be positive"):
            LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=-1e-3, nx=100)

    def test_tau_greater_than_two_emits_warning(self) -> None:
        """tau > 2.0 should trigger a UserWarning about instability."""
        from tensorlbm.unit_converter import LBMUnitConverter

        # With re=1, nx=100, u_lb=0.05:
        #   nu_lb = 0.05 * 100 / 1 = 5.0
        #   tau   = 0.5 + 5.0 / (1/3) = 0.5 + 15 = 15.5  → > 2.0
        # Also triggers the re_check warning (re=1, u*l/nu = 0.01*1/0.01 = 1.0 ✓)
        # The warning message uses 'τ' (Unicode), so match on the instability phrase.
        with pytest.warns(UserWarning, match="unstable"):
            LBMUnitConverter(
                re=1.0,
                l_phys=1.0,
                u_phys=0.01,
                nu_phys=0.01,
                nx=100,
                u_lb=0.05,
                ma_warn=1.0,  # suppress Ma warning so only tau warning fires
            )


# ---------------------------------------------------------------------------
# config_io – load_config_yaml (entirely uncovered), load_config_json gaps
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _CfgSimple:
    nx: int = 32
    ny: int = 16
    u_in: float = 0.05
    label: str = "default"
    active: bool = True


class TestLoadConfigYamlWrapper:
    """load_config_yaml was entirely uncovered – exercise its key branches."""

    def test_happy_path_loads_field(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        from tensorlbm.config_io import load_config_yaml

        p = tmp_path / "cfg.yaml"
        p.write_text("nx: 64\nny: 32\n", encoding="utf-8")
        cfg = load_config_yaml(_CfgSimple, p)
        assert cfg.nx == 64
        assert cfg.ny == 32

    def test_yml_extension_accepted(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        from tensorlbm.config_io import load_config_yaml

        p = tmp_path / "cfg.yml"
        p.write_text("nx: 10\n", encoding="utf-8")
        cfg = load_config_yaml(_CfgSimple, p)
        assert cfg.nx == 10

    def test_wrong_extension_raises_value_error(self, tmp_path: Path) -> None:
        from tensorlbm.config_io import load_config_yaml

        p = tmp_path / "cfg.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="load_config_yaml expects a .yaml or .yml file"):
            load_config_yaml(_CfgSimple, p)

    def test_yaml_not_installed_raises_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate pyyaml being absent to cover the ImportError branch."""
        import sys

        from tensorlbm.config_io import load_config_yaml

        p = tmp_path / "cfg.yaml"
        p.write_text("nx: 8\n", encoding="utf-8")

        # Temporarily hide yaml from the import system.
        original = sys.modules.get("yaml")
        sys.modules["yaml"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="pyyaml is required"):
                load_config_yaml(_CfgSimple, p)
        finally:
            if original is None:
                sys.modules.pop("yaml", None)
            else:
                sys.modules["yaml"] = original


class TestLoadConfigJsonAdditionalBranches:
    """Cover the skip-unknown-key and non-dataclass branches of load_config_json."""

    def test_unknown_key_in_json_is_silently_skipped(self, tmp_path: Path) -> None:
        from tensorlbm.config_io import load_config_json

        p = tmp_path / "cfg.json"
        p.write_text(
            json.dumps({"nx": 48, "ny": 24, "unknown_key": "ignored"}),
            encoding="utf-8",
        )
        cfg = load_config_json(_CfgSimple, p)
        assert cfg.nx == 48
        assert cfg.ny == 24
        # unknown_key must not appear on the resulting config
        assert not hasattr(cfg, "unknown_key")

    def test_non_dataclass_class_uses_raw_kwargs(self, tmp_path: Path) -> None:
        from tensorlbm.config_io import load_config_json

        class _PlainConfig:
            def __init__(self, nx: int = 32, ny: int = 16) -> None:
                self.nx = nx
                self.ny = ny

        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"nx": 64, "ny": 32}), encoding="utf-8")
        cfg = load_config_json(_PlainConfig, p)
        assert cfg.nx == 64
        assert cfg.ny == 32

    def test_coercion_failure_leaves_original_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover the except (ValueError, AttributeError): pass in load_config."""
        pytest.importorskip("yaml")
        from tensorlbm import load_config

        # Supply "not_a_number" via env var for an int field; int() will raise
        # ValueError, the code catches it, and the raw string is kept.
        p = tmp_path / "cfg.yaml"
        p.write_text("", encoding="utf-8")
        monkeypatch.setenv("TENSORLBM_NX", "not_a_number")
        # The config will be constructed with nx="not_a_number" (no type
        # enforcement by dataclasses); the important thing is no exception is
        # raised from inside load_config itself.
        cfg = load_config(_CfgSimple, p, env_prefix="TENSORLBM")
        assert cfg.nx == "not_a_number"  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# ai/model – _activation branches (relu / gelu / invalid) + wrong feature count
# ---------------------------------------------------------------------------

class TestActivationBranches:
    def test_relu_activation_creates_relu_module(self) -> None:
        import torch.nn as nn

        from tensorlbm.ai.model import _activation

        act = _activation("relu")
        assert isinstance(act, nn.ReLU)

    def test_gelu_activation_creates_gelu_module(self) -> None:
        import torch.nn as nn

        from tensorlbm.ai.model import _activation

        act = _activation("gelu")
        assert isinstance(act, nn.GELU)

    def test_case_insensitive(self) -> None:
        import torch.nn as nn

        from tensorlbm.ai.model import _activation

        assert isinstance(_activation("ReLU"), nn.ReLU)
        assert isinstance(_activation("GELU"), nn.GELU)
        assert isinstance(_activation("Tanh"), nn.Tanh)

    def test_unsupported_activation_raises(self) -> None:
        from tensorlbm.ai.model import _activation

        with pytest.raises(ValueError, match="Unsupported activation"):
            _activation("sigmoid")

    def test_eddyviscosity_mlp_relu_activation_runs_forward(self) -> None:
        from tensorlbm.ai.model import EddyViscosityMLP, ModelArch

        arch = ModelArch(hidden_features=8, n_hidden_layers=1, activation="relu")
        model = EddyViscosityMLP(arch)
        x = torch.randn(4, 3)
        out = model(x)
        assert out.shape == (4, 1)
        assert (out >= 0).all()

    def test_eddyviscosity_mlp_wrong_feature_count_raises(self) -> None:
        from tensorlbm.ai.model import EddyViscosityMLP

        model = EddyViscosityMLP()  # default in_features=3
        x = torch.randn(4, 5)  # 5 features instead of 3
        with pytest.raises(ValueError, match="Expected input with 3 features"):
            model(x)


# ---------------------------------------------------------------------------
# ai/dataset – shape errors, mask mismatch, empty multi, invalid split fraction
# ---------------------------------------------------------------------------

class TestStrainRateTensor2dShapeError:
    def test_1d_input_raises(self) -> None:
        from tensorlbm.ai.dataset import strain_rate_tensor_2d

        ux = torch.ones(8)
        uy = torch.ones(8)
        with pytest.raises(ValueError, match="2-D tensors"):
            strain_rate_tensor_2d(ux, uy)

    def test_mismatched_shapes_raise(self) -> None:
        from tensorlbm.ai.dataset import strain_rate_tensor_2d

        ux = torch.ones(8, 8)
        uy = torch.ones(8, 16)
        with pytest.raises(ValueError, match="2-D tensors of equal shape"):
            strain_rate_tensor_2d(ux, uy)


class TestExtractLesSamples2dMaskMismatch:
    def test_mask_wrong_shape_raises(self) -> None:
        from tensorlbm.ai.dataset import extract_les_samples_2d

        ux = torch.ones(8, 8)
        uy = torch.ones(8, 8)
        mask = torch.zeros(4, 4, dtype=torch.bool)  # wrong shape
        with pytest.raises(ValueError, match="mask shape"):
            extract_les_samples_2d(ux, uy, mask=mask)


class TestExtractLesSamples2dMultiEmpty:
    def test_empty_snapshots_returns_zero_tensors(self) -> None:
        from tensorlbm.ai.dataset import extract_les_samples_2d_multi

        feats, target = extract_les_samples_2d_multi([])
        assert feats.shape == (0, 3)
        assert target.shape == (0, 1)


class TestEddyViscosityDatasetSplit:
    def _make_dataset(self, n: int = 100) -> object:
        from tensorlbm.ai.dataset import EddyViscosityDataset

        feats = torch.randn(n, 3)
        targets = torch.rand(n, 1)
        return EddyViscosityDataset(features=feats, targets=targets)

    def test_split_produces_correct_sizes(self) -> None:
        from tensorlbm.ai.dataset import EddyViscosityDataset

        ds = self._make_dataset(100)
        assert isinstance(ds, EddyViscosityDataset)
        train_ds, val_ds = ds.split(val_fraction=0.2)
        assert len(train_ds) + len(val_ds) == 100

    def test_split_invalid_fraction_raises(self) -> None:
        ds = self._make_dataset(50)
        with pytest.raises(ValueError, match="val_fraction"):
            ds.split(val_fraction=0.0)

    def test_split_fraction_above_one_raises(self) -> None:
        ds = self._make_dataset(50)
        with pytest.raises(ValueError, match="val_fraction"):
            ds.split(val_fraction=1.0)


# ---------------------------------------------------------------------------
# ai/train – _r2_score zero-variance, load dataset from Path, too-small dataset
# ---------------------------------------------------------------------------

class TestR2ScoreZeroVariance:
    def test_zero_variance_returns_zero(self) -> None:
        from tensorlbm.ai.train import _r2_score

        y_true = torch.full((20, 1), 0.5)
        y_pred = torch.randn(20, 1)
        score = _r2_score(y_pred, y_true)
        assert score == pytest.approx(0.0)


class TestTrainEddyViscosityModel:
    def test_too_small_dataset_raises(self, tmp_path: Path) -> None:
        from tensorlbm.ai.dataset import EddyViscosityDataset
        from tensorlbm.ai.train import train_eddy_viscosity_model

        ds = EddyViscosityDataset(
            features=torch.randn(3, 3),
            targets=torch.rand(3, 1),
        )
        with pytest.raises(ValueError, match="too small"):
            train_eddy_viscosity_model(ds, tmp_path / "model.pt")

    def test_load_from_path(self, tmp_path: Path) -> None:
        """Passing a Path (not a dataset) triggers the load_dataset_pt branch."""
        from tensorlbm.ai.dataset import EddyViscosityDataset, save_dataset_pt
        from tensorlbm.ai.train import TrainConfig, train_eddy_viscosity_model

        feats = torch.randn(20, 3)
        targets = torch.rand(20, 1)
        ds = EddyViscosityDataset(features=feats, targets=targets)
        ds_path = save_dataset_pt(ds, tmp_path / "ds.pt")

        cfg = TrainConfig(epochs=2, hidden_features=8, n_hidden_layers=1)
        result = train_eddy_viscosity_model(ds_path, tmp_path / "model.pt", config=cfg)
        assert "final_train_mse" in result
        assert (tmp_path / "model.pt").exists()


# ---------------------------------------------------------------------------
# ai/database – get_model_record missing ID, _rows_to_dicts malformed JSON
# ---------------------------------------------------------------------------

class TestGetModelRecordMissingId:
    def _make_conn(self) -> sqlite3.Connection:
        from tensorlbm.ai.database import _SCHEMA

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    def test_missing_model_id_returns_none(self) -> None:
        from tensorlbm.ai.database import get_model_record

        conn = self._make_conn()
        result = get_model_record(conn, model_id=999)
        assert result is None


class TestRowsToDictsMalformedJson:
    def test_invalid_json_in_config_key_becomes_empty_dict(self) -> None:
        from tensorlbm.ai.database import _rows_to_dicts

        # Simulate a sqlite3.Row-like dict with invalid JSON in a known JSON column.
        class _FakeRow(dict):
            """dict subclass that behaves like sqlite3.Row for _rows_to_dicts."""

        row = _FakeRow(
            config_json="this is not valid json {{{",
            metadata_json=None,
            arch_json='{"hidden_features": 16}',
            metrics_json=None,
            id=1,
        )
        results = _rows_to_dicts([row])
        assert len(results) == 1
        # The invalid JSON should produce an empty dict for the 'config' key.
        assert results[0].get("config") == {}
        # The valid JSON key should still be parsed correctly.
        assert results[0].get("arch") == {"hidden_features": 16}


# ---------------------------------------------------------------------------
# ai/pipeline – AIPipelineResult.to_dict and no-snapshot fallback
# ---------------------------------------------------------------------------

class TestAIPipelineResultToDict:
    def test_to_dict_returns_dict_with_string_paths(self, tmp_path: Path) -> None:
        from tensorlbm.ai.pipeline import AIPipelineResult

        result = AIPipelineResult(
            work_dir=tmp_path / "work",
            db_path=tmp_path / "db.sqlite",
            dataset_path=tmp_path / "ds.pt",
            model_path=tmp_path / "model.pt",
            run_id=1,
            dataset_id=2,
            model_id=3,
            n_samples=100,
            training={"final_loss": 0.001},
            validation={"stable": True},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        # Path fields must be strings after to_dict
        assert isinstance(d["work_dir"], str)
        assert isinstance(d["db_path"], str)
        assert isinstance(d["dataset_path"], str)
        assert isinstance(d["model_path"], str)
        assert d["n_samples"] == 100


class TestRunLesSmokeTestNoSnapshotFallback:
    def test_sample_every_larger_than_n_steps_still_returns_one_snapshot(self) -> None:
        """When no step satisfies the sampling criterion the fallback snapshot is used."""
        from tensorlbm.ai.pipeline import _run_les_smoke

        snapshots = _run_les_smoke(
            nx=8,
            ny=8,
            tau=0.8,
            c_s=0.1,
            n_steps=3,      # only 3 steps
            sample_every=10,  # would need step 9; never satisfied
            seed=0,
            device=torch.device("cpu"),
        )
        assert len(snapshots) == 1
        ux, uy = snapshots[0]
        assert ux.shape == (8, 8)


# ---------------------------------------------------------------------------
# multiphase3d – collide_sc_single_component_3d (full body + solid_mask branch)
# ---------------------------------------------------------------------------

class TestCollideSCSingleComponent3d:
    """The entire collide_sc_single_component_3d function was uncovered."""

    def _make_f(
        self, nz: int = 4, ny: int = 4, nx: int = 4
    ) -> torch.Tensor:
        from tensorlbm.d3q19 import equilibrium3d

        rho = torch.ones(nz, ny, nx)
        ux = torch.zeros(nz, ny, nx)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        return equilibrium3d(rho, ux, uy, uz)

    def test_output_shape_matches_input(self) -> None:
        from tensorlbm.multiphase3d import collide_sc_single_component_3d

        f = self._make_f()
        f_out = collide_sc_single_component_3d(f, G=-4.0, tau=1.0)
        assert f_out.shape == f.shape

    def test_output_is_finite(self) -> None:
        from tensorlbm.multiphase3d import collide_sc_single_component_3d

        f = self._make_f()
        f_out = collide_sc_single_component_3d(f, G=-4.0, tau=1.0)
        assert torch.isfinite(f_out).all()

    def test_with_solid_mask_frozen_cells_unchanged(self) -> None:
        """Solid cells must not be modified (solid_mask branch, lines 252-253)."""
        from tensorlbm.multiphase3d import collide_sc_single_component_3d

        nz, ny, nx = 4, 4, 4
        f = self._make_f(nz, ny, nx)
        solid_mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        solid_mask[2, 2, 2] = True  # mark one cell as solid

        f_out = collide_sc_single_component_3d(f, G=-4.0, tau=1.0, solid_mask=solid_mask)
        # The solid cell must be identical in input and output.
        assert torch.allclose(f_out[:, 2, 2, 2], f[:, 2, 2, 2])
        # Non-solid cells should be updated (at least in principle).
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_with_body_force(self) -> None:
        from tensorlbm.multiphase3d import collide_sc_single_component_3d

        f = self._make_f()
        f_out = collide_sc_single_component_3d(f, G=-4.0, tau=1.0, gz=-0.0001)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_custom_psi_fn(self) -> None:
        from tensorlbm.multiphase3d import collide_sc_single_component_3d, psi_linear

        f = self._make_f()
        f_out = collide_sc_single_component_3d(f, G=-4.0, tau=1.0, psi_fn=psi_linear)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()
