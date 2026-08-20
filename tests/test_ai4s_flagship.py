"""Tests for :mod:`tensorlbm.apps.ai4s_flagship` (end-to-end flagship demo lib).

Each stage of the closed loop is a small unit-tested function: solver data
production (public API), the provisional HDF5 writer/loader, super-resolution
dataset construction, train/val split, and inference post-processing.  One
micro end-to-end test runs the whole loop (CPU, seconds) and asserts every
link of the chain — data -> dataset -> job -> model asset -> serving ->
lineage — exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.apps.ai4s_flagship import (
    FlagshipConfig,
    add_stationary_roughness,
    build_super_resolution_dataset,
    load_snapshots_hdf5,
    prediction_error_metrics,
    produce_velocity_snapshots,
    run_flagship_demo,
    split_train_val,
    try_load_pilot_dataset,
    write_snapshots_hdf5,
)


# ---------------------------------------------------------------------------
# Stage 1: data
# ---------------------------------------------------------------------------

class TestDataProduction:

    def test_produce_velocity_snapshots_public_solver(self):
        snapshots = produce_velocity_snapshots(
            nx=16, ny=16, n_steps=4, sample_every=2, seed=3,
        )
        assert len(snapshots) == 2
        ux, uy = snapshots[0]
        assert ux.shape == (16, 16)
        assert uy.shape == (16, 16)
        assert torch.isfinite(ux).all()
        assert torch.isfinite(uy).all()
        # snapshots evolve (the solver actually stepped)
        assert not torch.allclose(snapshots[0][0], snapshots[1][0])

    def test_produce_velocity_snapshots_seed_varies_data(self):
        a = produce_velocity_snapshots(nx=12, ny=12, n_steps=2, sample_every=2, seed=1)
        b = produce_velocity_snapshots(nx=12, ny=12, n_steps=2, sample_every=2, seed=2)
        assert not torch.allclose(a[0][0], b[0][0])

    def test_hdf5_roundtrip(self, tmp_path):
        snapshots = produce_velocity_snapshots(
            nx=12, ny=12, n_steps=2, sample_every=2, seed=1,
        )
        path = write_snapshots_hdf5(
            tmp_path / "snap.h5", snapshots,
            attrs={"source": "test", "n_snapshots": len(snapshots)},
        )
        assert path.is_file()
        back, attrs = load_snapshots_hdf5(path)
        assert len(back) == len(snapshots)
        assert torch.allclose(back[0][0], snapshots[0][0])
        assert torch.allclose(back[-1][1], snapshots[-1][1])
        assert attrs["source"] == "test"
        assert attrs["n_snapshots"] == len(snapshots)

    def test_try_load_pilot_dataset_absent(self, tmp_path):
        assert try_load_pilot_dataset(tmp_path / "does_not_exist") is None
        assert try_load_pilot_dataset(None) is None

    def test_try_load_pilot_dataset_consumes_h5(self, tmp_path):
        snapshots = produce_velocity_snapshots(
            nx=12, ny=12, n_steps=2, sample_every=2, seed=9,
        )
        write_snapshots_hdf5(tmp_path / "pilot.h5", snapshots, attrs={"k": "v"})
        loaded = try_load_pilot_dataset(tmp_path)
        assert loaded is not None
        pilots, info = loaded
        assert len(pilots) == len(snapshots)
        assert info["dataset"] == "velocity"
        assert info["path"].endswith("pilot.h5")


class TestStationaryRoughness:

    @pytest.fixture
    def snapshots(self):
        return produce_velocity_snapshots(
            nx=24, ny=24, n_steps=4, sample_every=2, seed=1,
        )

    def test_pattern_is_stationary_and_deterministic(self, snapshots):
        rough = add_stationary_roughness(snapshots, amplitude=0.01)
        rough_again = add_stationary_roughness(snapshots, amplitude=0.01)
        # identical fixed delta on every snapshot ...
        delta_a = rough[0][0] - snapshots[0][0]
        delta_b = rough[1][0] - snapshots[1][0]
        assert torch.allclose(delta_a, delta_b)
        # uy receives the same fixed pattern scaled by 0.8
        assert torch.allclose(
            rough[0][1] - snapshots[0][1], 0.8 * delta_a, atol=1e-6,
        )
        # ... deterministic across calls, and actually perturbs the fields
        assert torch.allclose(rough[0][0], rough_again[0][0])
        assert not torch.allclose(rough[0][0], snapshots[0][0])
        assert float(delta_a.abs().max()) > 0.0

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            add_stationary_roughness([], amplitude=0.01)
        snaps = [(torch.zeros(4, 4), torch.zeros(4, 4))]
        with pytest.raises(ValueError):
            add_stationary_roughness(snaps, amplitude=-0.1)

    def test_sub_grid_content_hurts_bilinear(self, snapshots):
        # with the fixed fine-scale pattern, coarse->bilinear reconstruction
        # must degrade (this is what makes the task FNO-relevant)
        smooth = build_super_resolution_dataset(snapshots, factor=4)
        rough = build_super_resolution_dataset(
            add_stationary_roughness(snapshots, amplitude=0.014), factor=4,
        )
        err_smooth = prediction_error_metrics(
            smooth["inputs"][0], smooth["targets"][0],
        )
        err_rough = prediction_error_metrics(
            rough["inputs"][0], rough["targets"][0],
        )
        assert err_rough["relative_l2"] > err_smooth["relative_l2"]


# ---------------------------------------------------------------------------
# Stage 2: dataset
# ---------------------------------------------------------------------------

class TestDatasetConstruction:

    @pytest.fixture
    def snapshots(self):
        return produce_velocity_snapshots(
            nx=16, ny=16, n_steps=4, sample_every=2, seed=5,
        )

    def test_build_super_resolution_shapes(self, snapshots):
        ds = build_super_resolution_dataset(snapshots, factor=2)
        assert ds["inputs"].shape == (len(snapshots), 2, 16, 16)
        assert ds["targets"].shape == (len(snapshots), 2, 16, 16)
        assert ds["grid"] == (16, 16)
        assert ds["downsample_factor"] == 2
        assert ds["n_samples"] == len(snapshots)

    def test_input_is_coarsened_not_identical(self, snapshots):
        ds = build_super_resolution_dataset(snapshots, factor=2)
        assert not torch.allclose(ds["inputs"][0], ds["targets"][0])
        # coarsening removes small-scale content: input variance <= target's
        assert ds["inputs"][0].var() <= ds["targets"][0].var() + 1e-12

    def test_build_rejects_empty(self):
        with pytest.raises(ValueError):
            build_super_resolution_dataset([], factor=2)

    def test_split_train_val_deterministic_disjoint(self):
        n = 20
        ds = {
            "inputs": torch.arange(n * 8, dtype=torch.float32).reshape(n, 2, 2, 2),
            "targets": torch.arange(n * 8, dtype=torch.float32).reshape(n, 2, 2, 2),
            "n_samples": n,
        }
        train, val = split_train_val(ds, val_fraction=0.2, seed=0)
        assert train["n_samples"] + val["n_samples"] == n
        assert val["n_samples"] == 4
        assert set(train["indices"]).isdisjoint(val["indices"])
        assert sorted(train["indices"] + val["indices"]) == list(range(n))
        train2, val2 = split_train_val(ds, val_fraction=0.2, seed=0)
        assert train2["indices"] == train["indices"]
        assert val2["indices"] == val["indices"]

    def test_split_rejects_bad_fraction(self):
        ds = {"inputs": torch.zeros(4, 2, 2, 2), "targets": torch.zeros(4, 2, 2, 2), "n_samples": 4}
        with pytest.raises(ValueError):
            split_train_val(ds, val_fraction=1.5, seed=0)


# ---------------------------------------------------------------------------
# Stage 5 helper: inference post-processing
# ---------------------------------------------------------------------------

class TestPredictionErrors:

    def test_known_values(self):
        target = torch.ones(2, 4, 4)
        pred = torch.zeros(2, 4, 4)
        m = prediction_error_metrics(pred, target)
        assert m["mse"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(1.0)
        assert m["max_abs_error"] == pytest.approx(1.0)
        assert m["relative_l2"] == pytest.approx(1.0)

    def test_identical_fields_zero_error(self):
        t = torch.randn(2, 8, 8)
        m = prediction_error_metrics(t, t.clone())
        assert m["mse"] == 0.0
        assert m["relative_l2"] == 0.0

    def test_accepts_numpy_and_half_precision(self):
        t = torch.ones(2, 4, 4, dtype=torch.float16)
        m = prediction_error_metrics(t.numpy() * 2, t)
        assert m["mse"] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            prediction_error_metrics(torch.zeros(2, 4, 4), torch.zeros(2, 4, 5))


# ---------------------------------------------------------------------------
# The whole loop (micro configuration, CPU, seconds)
# ---------------------------------------------------------------------------

class TestMicroEndToEnd:

    def test_run_flagship_demo(self, tmp_path):
        cfg = FlagshipConfig(
            workdir=tmp_path / "run",
            device="cpu",
            pilot_dir=None,
            nx=16, ny=16, n_steps=4, sample_every=2, seeds=(1, 2),
            epochs=2, batch_size=2,
            arch=dict(width=8, n_layers=2, modes_x=6, modes_y=6, mlp_hidden=16),
        )
        report = run_flagship_demo(cfg)

        # data link
        assert report.data_source == "provisional_solver"
        assert Path(report.data_path).is_file()
        assert report.n_train + report.n_val == 4  # 2 seeds x 2 snapshots

        # training-job link
        assert report.job_id.startswith("job_")
        assert report.job_status == "completed"
        assert len(report.loss_history) == 2
        assert all(loss >= 0 for loss in report.loss_history)

        # model-asset link
        assert report.model_id.startswith("mdl_")
        assert Path(report.ckpt_path).is_file()
        store_sidecar = (
            Path(report.model_store)
            / "flow_super_resolution" / report.model_id / "meta.json"
        )
        assert store_sidecar.is_file()
        meta = json.loads(store_sidecar.read_text())
        assert meta["dataset_product_id"] == report.product_asset_id
        assert meta["training_job_id"] == report.job_id
        assert meta["git_sha"]

        # serving link: live held-out inference + asset/serving cross-check
        assert report.serving_model_id >= 1
        assert report.val_errors
        for e in report.val_errors:
            assert e["mse"] >= 0.0
            assert e["relative_l2"] >= 0.0
            assert e["serving_vs_asset_max_abs_diff"] == pytest.approx(0.0, abs=1e-6)

        # lineage link: every upstream node reachable from the serving asset
        chain = set(report.lineage_upstream)
        for node in (
            report.product_asset_id,
            report.dataset_asset_id,
            f"flagship_sr:job:{report.job_id}",
            report.model_id,
        ):
            assert node in chain, f"missing lineage node {node}"

        # report.json persisted
        data = json.loads((tmp_path / "run" / "report.json").read_text())
        assert data["model_id"] == report.model_id
        assert data["serving_model_id"] == report.serving_model_id

    def test_run_flagship_demo_uses_pilot_when_present(self, tmp_path):
        # a pilot-style h5 in the pilot dir must be picked up instead of the solver
        snapshots = produce_velocity_snapshots(
            nx=12, ny=12, n_steps=4, sample_every=2, seed=42,
        )
        write_snapshots_hdf5(tmp_path / "pilot_data.h5", snapshots, attrs={})
        cfg = FlagshipConfig(
            workdir=tmp_path / "run2",
            device="cpu",
            pilot_dir=tmp_path,
            nx=16, ny=16, epochs=1, batch_size=2,
            arch=dict(width=8, n_layers=2, modes_x=6, modes_y=6, mlp_hidden=16),
        )
        report = run_flagship_demo(cfg)
        assert report.data_source.startswith("pilot:")
        assert report.n_train + report.n_val == len(snapshots)
