from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm import CylinderFlowConfig, run_cylinder_flow
from tensorlbm.ai import (
    CylinderConditionSpec,
    CylinderMultimodalDataset,
    MultimodalCylinderArch,
    MultimodalCylinderForecaster,
    MultimodalDatasetManifest,
    MultimodalTrainConfig,
    TrajectoryValidationError,
    collate_cylinder_multimodal,
    evaluate_forecaster_vs_persistence,
    export_cylinder_run_to_r2_catalog,
    forecast_from_conditions,
    load_multimodal_forecaster,
    normalized_sdf_2d,
    rollout_trajectory_forecast,
    signed_distance_field_2d,
    split_cylinder_trajectories,
    train_multimodal_cylinder_forecaster,
    trim_trajectory_spinup,
    validate_cylinder_trajectory,
)
from tensorlbm.data import FieldDataCatalog, load_product, load_product_arrays
from tensorlbm.ml.training_job import TrainingJobRegistry


def _make_mini_run(root: Path, name: str, re_val: float = 30.0) -> Path:
    cfg = CylinderFlowConfig(
        nx=48,
        ny=24,
        u_in=0.06,
        re=re_val,
        radius=3.0,
        n_steps=45,
        output_interval=15,
        output_root=root,
        run_name=name,
        overwrite=True,
        save_field_npz=True,
    )
    return run_cylinder_flow(cfg)


@pytest.mark.unit
def test_signed_distance_field_2d_conventions() -> None:
    mask = np.zeros((16, 20), dtype=np.bool_)
    mask[6:10, 8:12] = True

    sdf = signed_distance_field_2d(mask)
    assert sdf.shape == (16, 20)
    assert sdf.dtype == np.float32
    assert np.all(sdf[mask] < 0.0)
    assert np.all(sdf[~mask] > 0.0)

    sdf_norm = normalized_sdf_2d(mask, clip_voxels=8.0)
    assert float(sdf_norm.min()) >= -1.0
    assert float(sdf_norm.max()) <= 1.0

    # Degenerate all-fluid and all-solid guards
    all_fluid = normalized_sdf_2d(np.zeros((8, 8), dtype=np.bool_), clip_voxels=8.0)
    all_solid = normalized_sdf_2d(np.ones((8, 8), dtype=np.bool_), clip_voxels=8.0)
    np.testing.assert_allclose(all_fluid, 1.0)
    np.testing.assert_allclose(all_solid, -1.0)


@pytest.mark.integration
def test_quality_gates_reject_corrupt_trajectories(tmp_path: Path) -> None:
    valid_dir = _make_mini_run(tmp_path / "base", "re30")
    traj = validate_cylinder_trajectory(valid_dir, fallback_code_revision="deadbeef" * 5)
    assert len(traj.frames) == 3

    # 1. Reject pickled object array in NPZ
    pickle_dir = tmp_path / "bad_pickle"
    shutil.copytree(valid_dir, pickle_dir)
    with np.load(pickle_dir / "fields_step_000015.npz", allow_pickle=False) as orig:
        pickle_payload = {k: orig[k].copy() for k in orig.files}
    pickle_payload["ux"] = np.array([{"x": 1}], dtype=object)
    np.savez(pickle_dir / "fields_step_000015.npz", **pickle_payload)
    with pytest.raises(TrajectoryValidationError, match="allow_pickle=False"):
        validate_cylinder_trajectory(pickle_dir, fallback_code_revision="deadbeef" * 5)

    # 2. Reject NaN in continuous fields
    nan_dir = tmp_path / "bad_nan"
    shutil.copytree(valid_dir, nan_dir)
    with np.load(nan_dir / "fields_step_000030.npz", allow_pickle=False) as orig:
        payload = {k: orig[k].copy() for k in orig.files}
    payload["ux"][0, 0] = np.nan
    np.savez_compressed(nan_dir / "fields_step_000030.npz", **payload)
    with pytest.raises(TrajectoryValidationError, match="non-finite"):
        validate_cylinder_trajectory(nan_dir, fallback_code_revision="deadbeef" * 5)

    # 3. Reject inconsistent solid_mask across snapshots
    mask_dir = tmp_path / "bad_mask"
    shutil.copytree(valid_dir, mask_dir)
    with np.load(mask_dir / "fields_step_000030.npz", allow_pickle=False) as orig:
        payload = {k: orig[k].copy() for k in orig.files}
    payload["solid_mask"][0, 0] = ~payload["solid_mask"][0, 0]
    np.savez_compressed(mask_dir / "fields_step_000030.npz", **payload)
    with pytest.raises(TrajectoryValidationError, match="solid_mask differs"):
        validate_cylinder_trajectory(mask_dir, fallback_code_revision="deadbeef" * 5)

    # 4. Reject missing visual PNG when require_visual_png=True
    no_png_dir = tmp_path / "bad_png"
    shutil.copytree(valid_dir, no_png_dir)
    (no_png_dir / "flow_step_000015.png").unlink()
    with pytest.raises(TrajectoryValidationError, match="flow_step_000015.png"):
        validate_cylinder_trajectory(
            no_png_dir,
            require_visual_png=True,
            fallback_code_revision="deadbeef" * 5,
        )
    # Allowed when visual frames are explicitly optional
    traj_no_png = validate_cylinder_trajectory(
        no_png_dir,
        require_visual_png=False,
        fallback_code_revision="deadbeef" * 5,
    )
    assert traj_no_png.frames[0].png_path is None


@pytest.mark.integration
def test_split_leak_guard_and_r2_catalog_bridge(tmp_path: Path) -> None:
    run_a = validate_cylinder_trajectory(
        _make_mini_run(tmp_path, "run_a", re_val=25.0),
        fallback_code_revision="a" * 40,
    )
    run_b = validate_cylinder_trajectory(
        _make_mini_run(tmp_path, "run_b", re_val=45.0),
        fallback_code_revision="a" * 40,
    )

    with pytest.raises(ValueError, match="Trajectory leakage"):
        split_cylinder_trajectories([run_a], val_runs=[run_a])

    manifest = split_cylinder_trajectories([run_a], val_runs=[run_b])
    manifest_file = manifest.save(tmp_path / "manifest.json")
    reloaded = MultimodalDatasetManifest.load(manifest_file)
    assert reloaded.fingerprint == manifest.fingerprint
    assert reloaded.train_run_ids == ("run_a",)
    assert reloaded.val_run_ids == ("run_b",)

    # Bridge run_a into FieldDataCatalog R2 products and read back arrays
    catalog = FieldDataCatalog.open(tmp_path / "catalog.sqlite")
    try:
        pids = export_cylinder_run_to_r2_catalog(
            run_a,
            catalog,
            h5_path=tmp_path / "r2" / "run_a.h5",
        )
        assert len(pids) == len(run_a.frames)
        product = load_product(catalog, pids[0])
        arrays = load_product_arrays(product)
        assert arrays["velocity"].shape == (24, 48, 2)
        assert arrays["rho"].shape == (24, 48)
        assert arrays["solid_mask"].shape == (24, 48)
    finally:
        catalog.close()


@pytest.mark.unit
def test_patch_forecaster_handles_160x64_and_zero_init_persistence() -> None:
    # 160 x 64 = 10,240 cells (> 4096 legacy token cap); 4x4 patching -> 640 tokens
    arch = MultimodalCylinderArch(patch_size=(4, 4), d_model=32, n_heads=4, n_layers=2, ffn_dim=64)
    model = MultimodalCylinderForecaster(arch)
    model.eval()

    bsz, ny, nx = 2, 64, 160
    field_t = torch.randn(bsz, 3, ny, nx)
    solid = torch.zeros(bsz, ny, nx, dtype=torch.bool)
    solid[:, 26:38, 34:46] = True
    field_t[:, 0] = field_t[:, 0].masked_fill(solid, 0.0)
    field_t[:, 1] = field_t[:, 1].masked_fill(solid, 0.0)
    geom = torch.stack([solid.float(), torch.ones_like(solid, dtype=torch.float32) * 0.5], dim=1)
    physics_cond = torch.zeros(bsz, 6)
    force_t = torch.tensor([[1.8, 0.02], [1.5, -0.01]])
    visual_t = torch.rand(bsz, 3, 32, 64)
    has_vis = torch.tensor([True, False])

    with torch.no_grad():
        out = model(
            field_t,
            geom,
            physics_cond,
            force_t=force_t,
            has_force=torch.ones(bsz, dtype=torch.bool),
            visual_t=visual_t,
            has_visual=has_vis,
        )

    assert out.field_next_norm.shape == (bsz, 3, ny, nx)
    assert out.force_next.shape == (bsz, 2)
    # Zero-initialized residual head starts at exact persistence on fluid and force
    torch.testing.assert_close(out.field_next_norm, field_t)
    torch.testing.assert_close(out.force_next, force_t)
    assert float(out.field_next_norm[:, :2].abs()[:, :, 26:38, 34:46].max().item()) == 0.0

    # Non-divisible grid dimensions (33 x 65) also pad and crop cleanly
    f_odd = torch.zeros(1, 3, 33, 65)
    g_odd = torch.zeros(1, 2, 33, 65)
    with torch.no_grad():
        out_odd = model(f_odd, g_odd, torch.zeros(1, 6))
    assert out_odd.field_next_norm.shape == (1, 3, 33, 65)


@pytest.mark.integration
def test_end_to_end_multimodal_training_and_persistence_evaluation(tmp_path: Path) -> None:
    train_dir_1 = _make_mini_run(tmp_path / "runs", "train_re25", re_val=25.0)
    train_dir_2 = _make_mini_run(tmp_path / "runs", "train_re45", re_val=45.0)
    test_dir = _make_mini_run(tmp_path / "runs", "test_re35", re_val=35.0)

    t1 = validate_cylinder_trajectory(train_dir_1, fallback_code_revision="b" * 40)
    t2 = validate_cylinder_trajectory(train_dir_2, fallback_code_revision="b" * 40)
    t_test = validate_cylinder_trajectory(test_dir, fallback_code_revision="b" * 40)

    manifest = split_cylinder_trajectories([t1, t2], test_runs=[t_test])
    train_ds = CylinderMultimodalDataset(
        [t1, t2],
        normalization=manifest.normalization,
        load_visual=True,
        image_size=(24, 48),
    )
    test_ds = CylinderMultimodalDataset(
        [t_test],
        normalization=manifest.normalization,
        load_visual=True,
        image_size=(24, 48),
    )

    arch = MultimodalCylinderArch(
        patch_size=(4, 4),
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
    )
    cfg = MultimodalTrainConfig(
        epochs=15,
        batch_size=4,
        learning_rate=3e-3,
        seed=42,
        device="cpu",
    )

    weights_path = tmp_path / "ckpt" / "forecaster.pt"
    registry = TrainingJobRegistry.open(tmp_path / "jobs.sqlite")
    try:
        res = train_multimodal_cylinder_forecaster(
            train_ds,
            arch=arch,
            config=cfg,
            manifest=manifest,
            job_registry=registry,
            save_weights_path=weights_path,
        )
        assert res.train_losses[-1] < res.train_losses[0]
        assert res.job_id is not None
        assert res.registered_model_id is not None
        job = registry.get_job(res.job_id)
        assert job is not None and job.status == "completed"
    finally:
        registry.close()

    reloaded = load_multimodal_forecaster(weights_path)
    mm_report = evaluate_forecaster_vs_persistence(reloaded, test_ds, use_visual=True)
    solver_report = evaluate_forecaster_vs_persistence(reloaded, test_ds, use_visual=False)

    assert mm_report.solid_velocity_max == 0.0
    assert solver_report.solid_velocity_max == 0.0
    assert mm_report.model_rel_l2_velocity < mm_report.persistence_rel_l2_velocity
    assert solver_report.model_rel_l2_velocity < solver_report.persistence_rel_l2_velocity

    rollout = rollout_trajectory_forecast(
        reloaded,
        t_test,
        normalization=manifest.normalization,
    )
    assert len(rollout) == 2
    assert rollout[0].model_rel_l2_velocity < rollout[0].persistence_rel_l2_velocity


@pytest.mark.unit
def test_stage3_spinup_trim_and_multiframe_window_dataset(tmp_path: Path) -> None:
    cfg = CylinderFlowConfig(
        nx=48,
        ny=24,
        u_in=0.06,
        re=35.0,
        radius=3.0,
        n_steps=75,
        output_interval=15,
        output_root=tmp_path,
        run_name="spinup_run",
        overwrite=True,
        save_field_npz=True,
    )
    run_dir = run_cylinder_flow(cfg)
    raw_traj = validate_cylinder_trajectory(run_dir)
    assert [f.step for f in raw_traj.frames] == [15, 30, 45, 60, 75]

    trimmed = trim_trajectory_spinup(raw_traj, min_step=30)
    assert [f.step for f in trimmed.frames] == [30, 45, 60, 75]

    with pytest.raises(TrajectoryValidationError, match="fewer than 2 frames"):
        trim_trajectory_spinup(raw_traj, min_step=75)

    manifest = split_cylinder_trajectories(
        [trimmed],
        min_step=30,
        history_steps=2,
        rollout_steps=2,
    )
    reloaded_manifest = MultimodalDatasetManifest.load(manifest.save(tmp_path / "m3.json"))
    assert reloaded_manifest.min_step == 30
    assert reloaded_manifest.history_steps == 2
    assert reloaded_manifest.rollout_steps == 2
    assert reloaded_manifest.fingerprint == manifest.fingerprint

    ds = CylinderMultimodalDataset(
        [trimmed],
        normalization=manifest.normalization,
        history_steps=2,
        rollout_steps=2,
    )
    # 4 frames [30, 45, 60, 75] with H=2, K=2 -> anchor at step 45 (hist=[30,45], targets=[60,75])
    assert len(ds) == 1
    batch = collate_cylinder_multimodal([ds[0]])
    assert batch["history_fields"].shape == (1, 2, 3, 24, 48)
    assert batch["history_forces"].shape == (1, 2, 2)
    assert batch["target_fields"].shape == (1, 2, 3, 24, 48)
    assert batch["target_forces"].shape == (1, 2, 2)
    assert batch["step_t"] == [45]
    assert batch["step_next"] == [60]


@pytest.mark.unit
def test_stage3_pushforward_and_variable_geometry_generalization(tmp_path: Path) -> None:
    def _gen(name: str, re_val: float, radius: float) -> Path:
        return run_cylinder_flow(
            CylinderFlowConfig(
                nx=48,
                ny=24,
                u_in=0.06,
                re=re_val,
                radius=radius,
                n_steps=75,
                output_interval=15,
                output_root=tmp_path,
                run_name=name,
                overwrite=True,
                save_field_npz=True,
            )
        )

    train_trajs = [
        validate_cylinder_trajectory(_gen("tr_re25_r3", 25.0, 3.0)),
        validate_cylinder_trajectory(_gen("tr_re55_r4", 55.0, 4.0)),
    ]
    test_traj = validate_cylinder_trajectory(_gen("te_re40_r3p5", 40.0, 3.5))

    manifest = split_cylinder_trajectories(
        train_trajs,
        test_runs=[test_traj],
        history_steps=2,
        rollout_steps=2,
    )
    train_ds = CylinderMultimodalDataset(
        train_trajs,
        normalization=manifest.normalization,
        history_steps=2,
        rollout_steps=2,
        load_visual=True,
    )
    test_ds = CylinderMultimodalDataset(
        [test_traj],
        normalization=manifest.normalization,
        history_steps=2,
        rollout_steps=1,
        load_visual=True,
    )

    arch = MultimodalCylinderArch(
        patch_size=(4, 4),
        history_steps=2,
        d_model=48,
        n_heads=4,
        n_layers=2,
        ffn_dim=96,
    )
    cfg = MultimodalTrainConfig(
        epochs=24,
        batch_size=4,
        learning_rate=3e-3,
        pushforward_steps=2,
        pushforward_decay=0.7,
        seed=7,
        device="cpu",
    )

    ckpt_path = tmp_path / "stage3_ckpt.pt"
    res = train_multimodal_cylinder_forecaster(
        train_ds,
        arch=arch,
        config=cfg,
        manifest=manifest,
        save_weights_path=ckpt_path,
    )
    assert res.train_losses[-1] < res.train_losses[0]

    reloaded = load_multimodal_forecaster(ckpt_path)
    assert reloaded.arch.history_steps == 2

    mm_eval = evaluate_forecaster_vs_persistence(reloaded, test_ds, use_visual=True)
    solver_eval = evaluate_forecaster_vs_persistence(reloaded, test_ds, use_visual=False)
    assert mm_eval.solid_velocity_max == 0.0
    assert solver_eval.solid_velocity_max == 0.0
    assert mm_eval.model_rel_l2_velocity < mm_eval.persistence_rel_l2_velocity
    assert solver_eval.model_rel_l2_velocity < solver_eval.persistence_rel_l2_velocity

    rollout = rollout_trajectory_forecast(
        reloaded,
        test_traj,
        start_index=1,
        normalization=manifest.normalization,
    )
    assert len(rollout) == 3
    assert rollout[0].model_rel_l2_velocity < rollout[0].persistence_rel_l2_velocity


@pytest.mark.unit
def test_scheme_a_pure_condition_to_trajectory_forecast(tmp_path: Path) -> None:
    def _gen_with_t0(name: str, re_val: float, radius: float) -> Path:
        return run_cylinder_flow(
            CylinderFlowConfig(
                nx=48,
                ny=24,
                u_in=0.06,
                re=re_val,
                radius=radius,
                n_steps=60,
                output_interval=15,
                output_root=tmp_path,
                run_name=name,
                overwrite=True,
                save_field_npz=True,
                save_initial_snapshot=True,
            )
        )

    train_trajs = [
        validate_cylinder_trajectory(_gen_with_t0("t0_re25_r3", 25.0, 3.0)),
        validate_cylinder_trajectory(_gen_with_t0("t0_re55_r4", 55.0, 4.0)),
    ]
    assert [f.step for f in train_trajs[0].frames] == [0, 15, 30, 45, 60]

    manifest = split_cylinder_trajectories(
        train_trajs,
        history_steps=2,
        rollout_steps=2,
    )
    train_ds = CylinderMultimodalDataset(
        train_trajs,
        normalization=manifest.normalization,
        history_steps=2,
        rollout_steps=2,
        load_visual=True,
    )
    arch = MultimodalCylinderArch(
        patch_size=(4, 4),
        history_steps=2,
        d_model=48,
        n_heads=4,
        n_layers=2,
        ffn_dim=96,
    )
    cfg = MultimodalTrainConfig(
        epochs=28,
        batch_size=4,
        learning_rate=3e-3,
        pushforward_steps=2,
        pushforward_decay=0.7,
        seed=11,
        device="cpu",
    )
    ckpt_path = tmp_path / "scheme_a.pt"
    train_multimodal_cylinder_forecaster(
        train_ds,
        arch=arch,
        config=cfg,
        manifest=manifest,
        save_weights_path=ckpt_path,
    )

    # Pure condition-only inference on unseen (Re=40, radius=3.5) with zero LBM solver calls
    cond = CylinderConditionSpec(
        nx=48,
        ny=24,
        u_in=0.06,
        re=40.0,
        radius=3.5,
        n_steps=60,
        output_interval=15,
    )
    ai_res = forecast_from_conditions(ckpt_path, cond, device="cpu")
    assert [s.step for s in ai_res.steps] == [0, 15, 30, 45, 60]

    ai_out_dir = ai_res.save_artifacts(tmp_path / "ai_predicted_run")
    validated_ai_traj = validate_cylinder_trajectory(ai_out_dir, require_visual_png=True)
    assert [f.step for f in validated_ai_traj.frames] == [0, 15, 30, 45, 60]

    # Verify against ground-truth LBM on the same unseen condition
    gt_traj = validate_cylinder_trajectory(_gen_with_t0("gt_re40_r3p5", 40.0, 3.5))
    assert np.array_equal(ai_res.solid_mask, gt_traj.solid_mask)
    fluid = (~ai_res.solid_mask).astype(np.float32)

    init_vel = np.stack([ai_res.steps[0].ux, ai_res.steps[0].uy], axis=0)
    for pred_step, gt_frame in zip(ai_res.steps[1:], gt_traj.frames[1:]):
        assert float(np.max(pred_step.speed[ai_res.solid_mask])) == 0.0
        pred_vel = np.stack([pred_step.ux, pred_step.uy], axis=0)
        true_vel = np.stack([gt_frame.ux, gt_frame.uy], axis=0)
        ai_err = np.sqrt(np.sum(((pred_vel - true_vel) ** 2) * fluid[None]))
        init_err = np.sqrt(np.sum(((init_vel - true_vel) ** 2) * fluid[None]))
        assert ai_err < init_err
