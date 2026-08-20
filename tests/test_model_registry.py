"""Tests for :mod:`tensorlbm.ml.model_registry` (model asset layer).

Covers the CRUD loop (register -> list -> get -> load), the store directory
convention with ``meta.json`` sidecars, metadata completeness (task, metrics,
dataset product id, git sha, timestamps), filters, fail-closed validation,
lifecycle updates, and custom family loaders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.ai.fno import FNO2d, FNO2dArch, save_fno2d
from tensorlbm.ml.model_registry import (
    ModelAssetRegistry,
    register_family_loader,
)
from tensorlbm.ml.serving import FAMILY_FNO

_TINY_ARCH = FNO2dArch(
    in_channels=2, out_channels=2, width=8, n_layers=2,
    modes_x=6, modes_y=6, mlp_hidden=16,
)


@pytest.fixture
def store(tmp_path):
    reg = ModelAssetRegistry.open(tmp_path / "store")
    yield reg
    reg.close()


@pytest.fixture
def ckpt(tmp_path):
    """A real FNO2d checkpoint saved through ``save_fno2d`` (+ arch sidecar)."""
    path = tmp_path / "tiny_fno.pt"
    save_fno2d(FNO2d(_TINY_ARCH), path)
    return path


def _meta(**overrides):
    meta = {
        "task": "flow_super_resolution",
        "name": "test-fno",
        "family": FAMILY_FNO,
        "metrics": {"train_loss_final": 0.01, "n_train_samples": 8},
        "arch": {"width": 8, "n_layers": 2},
        "dataset_product_id": "flagship_sr:u",
        "training_job_id": "job_abc123def456",
        "tags": ("test", "superres"),
        "description": "unit-test asset",
    }
    meta.update(overrides)
    return meta


# ---------------------------------------------------------------------------
# register -> list -> get -> load
# ---------------------------------------------------------------------------

class TestCrudLoop:

    def test_register_returns_id_and_lists(self, store, ckpt):
        model_id = store.register(ckpt, _meta())
        assert model_id.startswith("mdl_")
        listed = store.list_models()
        assert [m.model_id for m in listed] == [model_id]

    def test_metadata_completeness(self, store, ckpt):
        model_id = store.register(ckpt, _meta())
        asset = store.get_model(model_id)
        assert asset is not None
        assert asset.task == "flow_super_resolution"
        assert asset.name == "test-fno"
        assert asset.family == FAMILY_FNO
        assert asset.framework == "torch"
        assert asset.metrics["train_loss_final"] == pytest.approx(0.01)
        assert asset.metrics["n_train_samples"] == 8
        assert asset.arch["width"] == 8
        assert asset.dataset_product_id == "flagship_sr:u"
        assert asset.training_job_id == "job_abc123def456"
        assert asset.tags == ("test", "superres")
        assert asset.description == "unit-test asset"
        assert asset.git_sha  # captured (or "unknown"), never empty
        assert asset.created_at and asset.updated_at
        assert asset.stage == "development"
        assert asset.serving_model_id is None

    def test_store_layout_convention(self, store, ckpt, tmp_path):
        model_id = store.register(ckpt, _meta())
        asset = store.get_model(model_id)
        dest = Path(asset.checkpoint_path)
        assert dest == (
            tmp_path / "store" / "flow_super_resolution" / model_id / "checkpoint.pt"
        )
        assert dest.is_file()
        # arch companion travels with the weights (load_fno2d needs it)
        assert dest.with_suffix(dest.suffix + ".json").is_file()
        # human-readable sidecar mirrors the index row
        sidecar = dest.parent / "meta.json"
        assert sidecar.is_file()
        data = json.loads(sidecar.read_text())
        assert data["model_id"] == model_id
        assert data["dataset_product_id"] == "flagship_sr:u"
        assert data["metrics"]["train_loss_final"] == pytest.approx(0.01)

    def test_load_model_roundtrip(self, store, tmp_path):
        reference = FNO2d(_TINY_ARCH)
        path = tmp_path / "roundtrip.pt"
        save_fno2d(reference, path)
        model_id = store.register(path, _meta())
        model = store.load_model(model_id)
        assert isinstance(model, FNO2d)
        assert not model.training  # eval mode
        for (k1, p1), (k2, p2) in zip(
            model.state_dict().items(), reference.state_dict().items()
        ):
            assert k1 == k2
            assert torch.equal(p1, p2)
        x = torch.randn(1, 2, 16, 16)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 16, 16)

    def test_register_by_reference(self, store, ckpt):
        model_id = store.register(ckpt, _meta(copy=False))
        asset = store.get_model(model_id)
        assert asset.artifact_relpath == ""
        assert Path(asset.checkpoint_path) == ckpt.resolve()
        assert isinstance(store.load_model(model_id), FNO2d)

    def test_index_persists_across_reopen(self, tmp_path, ckpt):
        reg = ModelAssetRegistry.open(tmp_path / "store")
        model_id = reg.register(ckpt, _meta())
        reg.close()
        reg2 = ModelAssetRegistry.open(tmp_path / "store")
        try:
            assert reg2.get_model(model_id) is not None
            assert isinstance(reg2.load_model(model_id), FNO2d)
        finally:
            reg2.close()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class TestFilters:

    def test_filter_by_task_name_dataset_tag(self, store, ckpt):
        a = store.register(ckpt, _meta(name="fno-a"))
        b = store.register(ckpt, _meta(
            name="mlp-b", task="turbulence_closure",
            family="eddy_viscosity_mlp", dataset_product_id="other:u",
            training_job_id="job_other999", tags=("closure",),
        ))
        assert [m.model_id for m in store.list_models(task="flow_super_resolution")] == [a]
        assert [m.model_id for m in store.list_models(task="turbulence_closure")] == [b]
        assert [m.model_id for m in store.list_models(name_contains="mlp")] == [b]
        assert [m.model_id for m in store.list_models(dataset_product_id="flagship_sr:u")] == [a]
        assert [m.model_id for m in store.list_models(tag="closure")] == [b]
        assert [m.model_id for m in store.list_models(training_job_id="job_abc123def456")] == [a]
        assert len(store.list_models(family=FAMILY_FNO)) == 1

    def test_limit(self, store, ckpt):
        for i in range(5):
            store.register(ckpt, _meta(name=f"m{i}"))
        assert len(store.list_models(limit=3)) == 3


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_task_raises(self, store, ckpt):
        meta = _meta()
        del meta["task"]
        with pytest.raises(ValueError, match="task"):
            store.register(ckpt, meta)

    def test_missing_name_raises(self, store, ckpt):
        meta = _meta()
        del meta["name"]
        with pytest.raises(ValueError, match="name"):
            store.register(ckpt, meta)

    def test_unknown_meta_key_raises(self, store, ckpt):
        with pytest.raises(ValueError, match="unsupported meta keys.*product_id"):
            store.register(ckpt, _meta(product_id="typo-of-dataset_product_id"))

    def test_missing_checkpoint_raises(self, store, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.register(tmp_path / "nope.pt", _meta())

    def test_duplicate_model_id_raises(self, store, ckpt):
        first = store.register(ckpt, _meta(model_id="mdl_fixed"))
        assert first == "mdl_fixed"
        with pytest.raises(ValueError, match="already exists"):
            store.register(ckpt, _meta(model_id="mdl_fixed"))

    def test_invalid_stage_raises(self, store, ckpt):
        with pytest.raises(ValueError, match="stage"):
            store.register(ckpt, _meta(stage="beta"))

    def test_get_unknown_returns_none(self, store):
        assert store.get_model("mdl_does_not_exist") is None

    def test_load_unknown_raises(self, store):
        with pytest.raises(KeyError):
            store.load_model("mdl_does_not_exist")

    def test_meta_not_mapping_raises(self, store, ckpt):
        with pytest.raises(TypeError):
            store.register(ckpt, ["not", "a", "mapping"])


# ---------------------------------------------------------------------------
# Lifecycle updates
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_record_metrics_merges(self, store, ckpt):
        model_id = store.register(ckpt, _meta())
        updated = store.record_metrics(model_id, {"val_relative_l2": 0.12})
        assert updated.metrics["train_loss_final"] == pytest.approx(0.01)
        assert updated.metrics["val_relative_l2"] == pytest.approx(0.12)

    def test_stage_transitions(self, store, ckpt):
        model_id = store.register(ckpt, _meta())
        store.set_stage(model_id, "staging")
        store.set_stage(model_id, "production")
        asset = store.get_model(model_id)
        assert asset.stage == "production"
        # production cannot be silently demoted
        with pytest.raises(ValueError, match="invalid stage transition"):
            store.set_stage(model_id, "development")

    def test_archive_hides_from_default_list(self, store, ckpt):
        keep = store.register(ckpt, _meta(name="keep"))
        gone = store.register(ckpt, _meta(name="gone"))
        store.archive(gone)
        ids = [m.model_id for m in store.list_models()]
        assert keep in ids and gone not in ids
        assert gone in [m.model_id for m in store.list_models(include_archived=True)]
        # archived is terminal
        with pytest.raises(ValueError, match="invalid stage transition"):
            store.set_stage(gone, "development")

    def test_link_serving_model(self, store, ckpt):
        model_id = store.register(ckpt, _meta())
        asset = store.link_serving_model(model_id, 7)
        assert asset.serving_model_id == 7
        with pytest.raises(TypeError):
            store.link_serving_model(model_id, "seven")

    def test_sidecar_stays_in_sync(self, store, ckpt, tmp_path):
        model_id = store.register(ckpt, _meta())
        store.record_metrics(model_id, {"extra": 1.0})
        asset = store.get_model(model_id)
        sidecar = Path(asset.checkpoint_path).parent / "meta.json"
        data = json.loads(sidecar.read_text())
        assert data["metrics"]["extra"] == 1.0


# ---------------------------------------------------------------------------
# Custom family loaders
# ---------------------------------------------------------------------------

class TestCustomLoader:

    def test_register_family_loader(self, store, ckpt):
        seen: list[str] = []

        def fake_loader(path: str):
            seen.append(path)
            return FNO2d(_TINY_ARCH)

        register_family_loader("fake_family", fake_loader)
        model_id = store.register(ckpt, _meta(family="fake_family"))
        model = store.load_model(model_id)
        assert isinstance(model, FNO2d)
        assert seen and Path(seen[0]).is_file()

    def test_register_family_loader_validation(self):
        with pytest.raises(ValueError):
            register_family_loader("  ", lambda p: None)
        with pytest.raises(TypeError):
            register_family_loader("ok_family", "not-callable")
