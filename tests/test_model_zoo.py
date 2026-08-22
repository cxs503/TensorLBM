"""Tests for :mod:`tensorlbm.zoo` (manifest-driven model zoo, C1).

Covers the CRUD loop (register -> list -> info -> load) with real artifacts
saved through the existing ``save_*`` conventions (eddy-viscosity MLP with a
short CPU training, FNO2d, drag surrogate), prediction consistency after the
zoo round-trip, the manifest schema (write + read side), fail-closed
behaviour (duplicate ids, missing files, bad schemas, unknown loaders),
overwrite/move semantics, and default-root resolution.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorlbm.ai.drag_surrogate import (
    DragNorm,
    FNODragArch,
    FNODragRegressor,
    save_drag_regressor,
)
from tensorlbm.ai.fno import FNO2d, FNO2dArch, save_fno2d
from tensorlbm.ai.model import EddyViscosityMLP, ModelArch, save_model
from tensorlbm.zoo import (
    SUGGESTED_LOADERS,
    ZOO_SCHEMA_VERSION,
    ModelZoo,
    ZooError,
    ZooManifestError,
    resolve_zoo_root,
)

_LOADER_MLP = SUGGESTED_LOADERS["eddy-viscosity"]
_LOADER_FNO = SUGGESTED_LOADERS["fno2d"]
_LOADER_DRAG = SUGGESTED_LOADERS["drag-surrogate"]

_TINY_FNO_ARCH = FNO2dArch(
    in_channels=2,
    out_channels=2,
    width=8,
    n_layers=2,
    modes_x=6,
    modes_y=6,
    mlp_hidden=16,
)

_TINY_DRAG_ARCH = FNODragArch(
    in_channels=3,
    width=4,
    n_layers=1,
    modes_y=3,
    modes_x=3,
    mlp_hidden=8,
)


@pytest.fixture
def zoo(tmp_path):
    return ModelZoo(tmp_path / "zoo")


def _train_tiny_eddy_mlp(
    tmp_path: Path, epochs: int = 15, seed: int = 0, name: str = "eddy_mlp.pt"
):
    """Train a small eddy-viscosity MLP on synthetic strain->nu data."""
    torch.manual_seed(seed)
    model = EddyViscosityMLP(ModelArch(in_features=3, hidden_features=16, n_hidden_layers=2))
    x = torch.randn(256, 3) * 0.5
    y = 0.05 + 0.1 * x.abs().sum(dim=1, keepdim=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    model.set_feature_stats(x.mean(dim=0), x.std(dim=0))
    model.eval()
    path = tmp_path / name
    save_model(model, path)
    return model, path


def _save_tiny_fno(tmp_path: Path) -> Path:
    torch.manual_seed(1)
    path = tmp_path / "tiny_fno.pt"
    save_fno2d(FNO2d(_TINY_FNO_ARCH), path)
    return path


def _save_tiny_drag(tmp_path: Path) -> Path:
    torch.manual_seed(2)
    model = FNODragRegressor(_TINY_DRAG_ARCH)
    norm = DragNorm(
        channel_mean=[0.1, 0.2, 0.3],
        channel_std=[1.0, 1.1, 1.2],
        target_mean=0.5,
        target_std=0.2,
    )
    path = tmp_path / "tiny_drag.pt"
    save_drag_regressor(model, norm, path)
    return path


_DATASET = {
    "path": "/data/campaigns/cyl_re_scan_v3",
    "split": "train=64/val=8/test=8 (point-level, seed 7)",
    "n_samples": 80,
}


# ---------------------------------------------------------------------------
# register -> list -> info -> load
# ---------------------------------------------------------------------------


class TestCrudLoop:
    def test_register_lists_and_infos(self, zoo, tmp_path):
        model, path = _train_tiny_eddy_mlp(tmp_path)
        info = zoo.register(
            path,
            "eddy-viscosity-mlp-v1",
            "eddy-viscosity",
            _LOADER_MLP,
            metrics={"test_mape": 4.2, "n_test": 8},
            dataset=_DATASET,
            notes="tiny synthetic run",
            code_sha="abc123def456",
        )
        assert info.model_id == "eddy-viscosity-mlp-v1"
        assert info.task == "eddy-viscosity"
        assert info.loader == _LOADER_MLP
        assert info.metrics["test_mape"] == pytest.approx(4.2)
        assert info.dataset == _DATASET
        assert info.code_sha == "abc123def456"
        assert info.artifact == "eddy_mlp.pt"
        assert datetime.fromisoformat(info.created_at)

        listed = zoo.list_models()
        assert [m.model_id for m in listed] == ["eddy-viscosity-mlp-v1"]
        assert zoo.info("eddy-viscosity-mlp-v1") == info

    def test_manifest_on_disk(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "eddy-a", "eddy-viscosity", _LOADER_MLP, dataset=_DATASET)
        manifest = json.loads(
            (tmp_path / "zoo" / "eddy-a" / "model.json").read_text(encoding="utf-8")
        )
        assert manifest["schema_version"] == ZOO_SCHEMA_VERSION
        assert manifest["model_id"] == "eddy-a"
        assert manifest["task"] == "eddy-viscosity"
        assert manifest["loader"] == _LOADER_MLP
        assert manifest["dataset"]["path"] == _DATASET["path"]
        assert manifest["dataset"]["split"] == _DATASET["split"]
        assert len(manifest["artifact_sha256"]) == 64
        assert (
            manifest["artifact_bytes"]
            == (tmp_path / "zoo" / "eddy-a" / "eddy_mlp.pt").stat().st_size
        )
        assert manifest["created_at"]

    def test_code_sha_autocaptured(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        info = zoo.register(path, "eddy-b", "eddy-viscosity", _LOADER_MLP)
        assert info.code_sha  # captured (or "unknown"), never empty

    def test_load_predicts_identically(self, zoo, tmp_path):
        model, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "eddy-rt", "eddy-viscosity", _LOADER_MLP)
        loaded = zoo.load("eddy-rt")
        assert isinstance(loaded, EddyViscosityMLP)
        assert not loaded.training  # eval mode
        x_test = torch.randn(64, 3) * 0.5
        with torch.no_grad():
            torch.testing.assert_close(loaded(x_test), model(x_test))

    def test_task_filter(self, zoo, tmp_path):
        _, mlp_path = _train_tiny_eddy_mlp(tmp_path)
        fno_path = _save_tiny_fno(tmp_path)
        zoo.register(mlp_path, "eddy-1", "eddy-viscosity", _LOADER_MLP)
        zoo.register(fno_path, "fno-1", "fno2d", _LOADER_FNO)
        assert [m.model_id for m in zoo.list_models("eddy-viscosity")] == ["eddy-1"]
        assert [m.model_id for m in zoo.list_models("fno2d")] == ["fno-1"]
        assert len(zoo.list_models()) == 2
        assert zoo.list_models("drag-surrogate") == []

    def test_missing_root_lists_empty(self, tmp_path):
        assert ModelZoo(tmp_path / "nope").list_models() == []


# ---------------------------------------------------------------------------
# Loader families (reuse of the existing save_* conventions)
# ---------------------------------------------------------------------------


class TestLoaderFamilies:
    def test_fno_roundtrip_and_companion_travels(self, zoo, tmp_path):
        torch.manual_seed(3)
        reference = FNO2d(_TINY_FNO_ARCH)
        src = tmp_path / "fno.pt"
        save_fno2d(reference, src)
        info = zoo.register(src, "fno-rt", "fno2d", _LOADER_FNO)
        # arch sidecar travels with the weights (load_fno2d needs it)
        assert info.artifact_companion == "fno.pt.json"
        assert (Path(info.entry_dir) / "fno.pt.json").is_file()
        model = zoo.load("fno-rt")
        for (k1, p1), (k2, p2) in zip(model.state_dict().items(), reference.state_dict().items()):
            assert k1 == k2
            assert torch.equal(p1, p2)
        x = torch.randn(1, 2, 16, 16)
        with torch.no_grad():
            assert model(x).shape == (1, 2, 16, 16)

    def test_drag_regressor_tuple_loader(self, zoo, tmp_path):
        src = _save_tiny_drag(tmp_path)
        zoo.register(src, "drag-rt", "drag-surrogate", _LOADER_DRAG)
        result = zoo.load("drag-rt")
        assert isinstance(result, tuple) and len(result) == 2
        model, norm = result
        assert isinstance(model, FNODragRegressor)
        assert not model.training
        assert norm.channel_mean == [0.1, 0.2, 0.3]
        assert norm.target_std == pytest.approx(0.2)
        with torch.no_grad():
            assert tuple(model(torch.randn(2, 3, 8, 8)).shape) == (2,)


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_duplicate_id_raises(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "dup", "eddy-viscosity", _LOADER_MLP)
        with pytest.raises(ZooError, match="already exists"):
            zoo.register(path, "dup", "eddy-viscosity", _LOADER_MLP)

    def test_missing_artifact_file_raises(self, zoo, tmp_path):
        with pytest.raises(FileNotFoundError):
            zoo.register(tmp_path / "nope.pt", "ghost", "eddy-viscosity", _LOADER_MLP)

    def test_bad_model_id_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        for bad in ("Upper", "with space", "slash/id", "..", "", " trailing"):
            with pytest.raises(ZooError, match="model_id"):
                zoo.register(path, bad, "eddy-viscosity", _LOADER_MLP)

    def test_bad_task_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ZooError, match="task"):
            zoo.register(path, "ok-id", "Drag Surrogate", _LOADER_MLP)

    def test_bad_loader_format_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        for bad in ("no-colon", "a:b:c", ":lead", "trail:", "mod:1attr"):
            with pytest.raises(ZooError, match="loader"):
                zoo.register(path, "ok-id", "eddy-viscosity", bad)

    def test_unknown_loader_module_raises_importerror(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ImportError, match="no_such_zoo_module"):
            zoo.register(path, "ok-id", "eddy-viscosity", "tensorlbm.no_such_zoo_module:load")

    def test_unknown_loader_attr_raises_importerror(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ImportError, match="load_nothing"):
            zoo.register(path, "ok-id", "eddy-viscosity", "tensorlbm.zoo:load_nothing")

    def test_nested_metrics_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ZooManifestError, match="scalar"):
            zoo.register(
                path,
                "ok-id",
                "eddy-viscosity",
                _LOADER_MLP,
                metrics={"bad": {"nested": 1}},
            )

    def test_nan_metrics_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ZooManifestError, match="NaN|finite"):
            zoo.register(
                path, "ok-id", "eddy-viscosity", _LOADER_MLP, metrics={"mape": float("nan")}
            )

    def test_inf_metrics_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ZooManifestError, match="finite"):
            zoo.register(
                path, "ok-id", "eddy-viscosity", _LOADER_MLP, metrics={"mape": float("inf")}
            )

    def test_unserialisable_dataset_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ZooManifestError, match="serialisable"):
            zoo.register(
                path,
                "ok-id",
                "eddy-viscosity",
                _LOADER_MLP,
                dataset={"path": Path("/data/x")},  # Path is not JSON
            )

    def test_bad_metadata_leaves_no_stray_entry(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        with pytest.raises(ZooManifestError, match="scalar"):
            zoo.register(
                path,
                "stray",
                "eddy-viscosity",
                _LOADER_MLP,
                metrics={"bad": [1, 2]},
            )
        assert not (tmp_path / "zoo" / "stray").exists()
        assert zoo.list_models() == []

    def test_unknown_id_raises_keyerror(self, zoo):
        with pytest.raises(KeyError):
            zoo.info("does-not-exist")
        with pytest.raises(KeyError):
            zoo.load("does-not-exist")

    def test_register_from_inside_entry_dir_rejected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "inside", "eddy-viscosity", _LOADER_MLP)
        inner = Path(zoo.info("inside").artifact_path)
        with pytest.raises(ZooError, match="inside the entry"):
            zoo.register(inner, "inside", "eddy-viscosity", _LOADER_MLP, overwrite=True)


class TestCorruptManifests:
    @pytest.fixture
    def entry_dir(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "victim", "eddy-viscosity", _LOADER_MLP)
        return tmp_path / "zoo" / "victim"

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda m: m.update(schema_version=99),
            lambda m: m.update(model_id="../escape"),
            lambda m: m.update(loader="not-a-loader"),
            lambda m: m.update(artifact="../../../etc/passwd"),
            lambda m: m.update(artifact_sha256="deadbeef"),
            lambda m: m.update(created_at="yesterday"),
            lambda m: m.update(unknown_key=1),
            lambda m: m.pop("task"),
        ],
    )
    def test_bad_schema_fail_closed(self, zoo, entry_dir, mutation):
        manifest_path = entry_dir / "model.json"
        manifest = json.loads(manifest_path.read_text())
        mutation(manifest)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ZooManifestError):
            zoo.list_models()
        with pytest.raises(ZooManifestError):
            zoo.info("victim")
        with pytest.raises(ZooManifestError):
            zoo.load("victim")

    def test_unparseable_manifest(self, zoo, entry_dir):
        (entry_dir / "model.json").write_text("{not json")
        with pytest.raises(ZooManifestError):
            zoo.list_models()

    def test_model_id_dir_mismatch_fail_closed(self, zoo, entry_dir):
        manifest_path = entry_dir / "model.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["model_id"] = "mismatched-id"
        manifest_path.write_text(json.dumps(manifest))
        # listing refuses to silently serve an entry under the wrong id
        with pytest.raises(ZooManifestError, match="mismatched-id"):
            zoo.list_models()

    def test_stray_dirs_ignored(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "real", "eddy-viscosity", _LOADER_MLP)
        (tmp_path / "zoo" / "stray").mkdir()
        (tmp_path / "zoo" / "loose.txt").write_text("x")
        assert [m.model_id for m in zoo.list_models()] == ["real"]


class TestValidate:
    def test_ok_entry_passes_all_checks(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "good", "eddy-viscosity", _LOADER_MLP, metrics={"mape": 5.0})
        report = zoo.validate("good")
        assert report.ok
        assert set(report.checks) == {
            "manifest_schema",
            "artifact_present",
            "integrity",
            "loader_importable",
            "model_loads",
        }
        assert all(report.checks.values())
        assert report.to_dict()["ok"] is True

    def test_missing_artifact_detected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "gone", "eddy-viscosity", _LOADER_MLP)
        Path(zoo.info("gone").artifact_path).unlink()
        assert not zoo.validate("gone").ok
        with pytest.raises(FileNotFoundError):
            zoo.load("gone")

    def test_missing_companion_detected(self, zoo, tmp_path):
        src = _save_tiny_fno(tmp_path)
        zoo.register(src, "nocomp", "fno2d", _LOADER_FNO)
        (tmp_path / "zoo" / "nocomp" / "tiny_fno.pt.json").unlink()
        report = zoo.validate("nocomp")
        assert not report.ok
        assert report.checks["artifact_present"] is False

    def test_tampered_artifact_detected(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "tampered", "eddy-viscosity", _LOADER_MLP)
        artifact = Path(zoo.info("tampered").artifact_path)
        artifact.write_bytes(artifact.read_bytes() + b"\x00tamper")
        report = zoo.validate("tampered")
        assert report.checks["integrity"] is False
        assert not report.ok

    def test_wrong_family_loader_detected(self, zoo, tmp_path):
        # Integrity intact, loader imports, but it cannot restore this file.
        src = _save_tiny_drag(tmp_path)
        zoo.register(src, "cross", "drag-surrogate", _LOADER_MLP)
        report = zoo.validate("cross")
        assert report.checks["integrity"] is True
        assert report.checks["model_loads"] is False
        assert not report.ok

    def test_unknown_id_raises(self, zoo):
        with pytest.raises(KeyError):
            zoo.validate("never-registered")


# ---------------------------------------------------------------------------
# Overwrite / move / root resolution
# ---------------------------------------------------------------------------


class TestOverwriteAndMove:
    def test_overwrite_replaces_entry(self, zoo, tmp_path):
        first_model, first = _train_tiny_eddy_mlp(tmp_path, seed=10, name="first.pt")
        second_model, second = _train_tiny_eddy_mlp(tmp_path, seed=20, name="second.pt")
        zoo.register(first, "entry", "eddy-viscosity", _LOADER_MLP, notes="first")
        info = zoo.register(
            second, "entry", "eddy-viscosity", _LOADER_MLP, overwrite=True, notes="second"
        )
        assert info.notes == "second"
        assert len(zoo.list_models()) == 1
        assert Path(info.artifact_path).name == "second.pt"
        # the entry now serves the *second* artifact's weights
        loaded = zoo.load("entry")
        assert torch.equal(next(iter(loaded.parameters())), next(iter(second_model.parameters())))
        assert not torch.equal(
            next(iter(loaded.parameters())), next(iter(first_model.parameters()))
        )

    def test_move_leaves_no_source(self, zoo, tmp_path):
        _, path = _train_tiny_eddy_mlp(tmp_path)
        zoo.register(path, "moved", "eddy-viscosity", _LOADER_MLP, move=True)
        assert not path.exists()
        assert not path.with_suffix(".pt.json").exists()
        assert isinstance(zoo.load("moved"), EddyViscosityMLP)


class TestRootResolution:
    def test_explicit_root_wins(self, tmp_path):
        assert resolve_zoo_root(tmp_path / "x") == tmp_path / "x"

    def test_env_var_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TENSORLBM_ZOO_ROOT", str(tmp_path / "envzoo"))
        assert resolve_zoo_root() == tmp_path / "envzoo"
        assert resolve_zoo_root(None) == tmp_path / "envzoo"

    def test_default_under_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TENSORLBM_ZOO_ROOT", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert resolve_zoo_root() == tmp_path / "home" / ".tensorlbm" / "zoo"

    def test_module_level_functions_use_env_root(self, monkeypatch, tmp_path):
        import tensorlbm.zoo as zoo_mod

        monkeypatch.setenv("TENSORLBM_ZOO_ROOT", str(tmp_path / "envzoo"))
        _, path = _train_tiny_eddy_mlp(tmp_path)
        info = zoo_mod.register(path, "mod-level", "eddy-viscosity", _LOADER_MLP)
        assert Path(info.entry_dir).parent == tmp_path / "envzoo"
        assert [m.model_id for m in zoo_mod.list_models()] == ["mod-level"]
        assert isinstance(zoo_mod.load("mod-level"), EddyViscosityMLP)
        assert zoo_mod.info("mod-level").model_id == "mod-level"
        assert zoo_mod.validate("mod-level").ok


class TestPublicSurface:
    def test_api_exports(self):
        from tensorlbm import api

        for name in (
            "ModelZoo",
            "ModelInfo",
            "ZooValidation",
            "resolve_zoo_root",
            "zoo_register",
            "zoo_load",
            "zoo_list_models",
            "zoo_info",
            "zoo_validate",
        ):
            assert hasattr(api, name), name

    def test_package_exports(self):
        import tensorlbm

        for name in ("ModelZoo", "zoo_register", "zoo_load", "zoo_list_models"):
            assert hasattr(tensorlbm, name), name
