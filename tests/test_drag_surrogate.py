"""Drag-surrogate tests (synthetic campaign, CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.ai.drag_surrogate import (
    DragTrainConfig,
    FNODragArch,
    PlaneSampleSpec,
    build_drag_split,
    fit_norm,
    load_drag_regressor,
    load_exact_cd,
    load_exact_cd_per_point,
    power_law_fit,
    power_law_predict,
    predict_cd,
    regression_metrics,
    run_drag_surrogate_study,
    save_drag_regressor,
    train_drag_surrogate,
)

NZ, NY, NX = 8, 8, 16
STEPS = (500, 1000)
POWER_A, POWER_B = 20.0, -0.7
SPLIT_POINTS = {
    "train": ["p0000", "p0001", "p0002", "p0003"],
    "val": ["p0004"],
    "test": ["p0005", "p0006"],
}
RE_LEVELS = (50.0, 80.0, 120.0, 180.0, 270.0, 400.0, 600.0)


def _solid_mask() -> np.ndarray:
    """A streamwise block centred in the plane; identical for every point."""
    mask = np.zeros((NZ, NY, NX), dtype=np.int8)
    mask[:, 2:6, 4:8] = 1
    return mask


def _snapshot_fields(re: float, u_in: float) -> dict[str, np.ndarray]:
    """Deterministic wake whose amplitude follows the C_D power law in Re."""
    zz, yy, xx = np.mgrid[:NZ, :NY, :NX]
    amp = 0.02 * (re / 100.0) ** POWER_B
    r2 = (yy - NY / 2) ** 2 + (zz - NZ / 2) ** 2
    deficit = amp * np.exp(-r2 / 4.0) * np.exp(-np.clip(xx - 6, 0, None) / 8.0)
    ux = (u_in - deficit).astype(np.float32)
    uy = (0.01 * deficit * (yy - NY / 2)).astype(np.float32)
    uz = np.zeros_like(ux)
    rho = (1.0 + 0.05 * deficit / max(u_in, 1e-9)).astype(np.float32)
    return {"rho": rho, "ux": ux, "uy": uy, "uz": uz}


POINT_IDS = sorted({pid for pids in SPLIT_POINTS.values() for pid in pids})


def _write_campaign(fields_dir: Path, drag_dir: Path, *, with_sidecar: bool = True) -> None:
    import h5py

    mask = _solid_mask()
    s_proj = int((mask.max(axis=2) > 0).sum())  # projected (z, y) columns
    u_in = 0.1
    for point_id, re in zip(POINT_IDS, RE_LEVELS):
        (fields_dir / "points" / point_id).mkdir(parents=True, exist_ok=True)
        (drag_dir / "points" / point_id).mkdir(parents=True, exist_ok=True)
        with h5py.File(fields_dir / "points" / point_id / "fields.h5", "w") as f:
            for step in STEPS:
                fields = _snapshot_fields(re, u_in)
                grp = f.create_group(f"step_{step:06d}")
                for name, arr in fields.items():
                    grp.create_dataset(name, data=arr)
                grp.create_dataset("solid_mask", data=mask)
                grp.attrs.update({"re": re, "u_in": u_in, "nx": NX, "ny": NY, "nz": NZ})
        force = POWER_A * re**POWER_B * u_in**2 * s_proj / 2.0  # rho = 1
        (drag_dir / "points" / point_id / "status.json").write_text(
            json.dumps({"params": {"re": re}, "drag_final": force, "drag_mean_tail": force})
        )
        if with_sidecar:
            samples = [
                {
                    "step": 25 * (i + 1),
                    "force_x": force,
                    "force_y": 0.0,
                    "force_z": 0.0,
                    "force_abs": force,
                }
                for i in range(40)
            ]
            (drag_dir / "points" / point_id / "drag_history.json").write_text(
                json.dumps({"schema": "tensorlbm.drag-history/v1", "samples": samples})
            )
    (fields_dir / "dataset.json").write_text(json.dumps({"split_points": SPLIT_POINTS}))


@pytest.fixture()
def campaign(tmp_path: Path) -> tuple[Path, Path]:
    fields_dir = tmp_path / "fields"
    drag_dir = tmp_path / "drag"
    _write_campaign(fields_dir, drag_dir)
    return fields_dir, drag_dir


def _tiny_arch() -> FNODragArch:
    return FNODragArch(in_channels=3, width=8, n_layers=2, modes_y=4, modes_x=6, mlp_hidden=16)


def _tiny_config() -> DragTrainConfig:
    return DragTrainConfig(epochs=15, batch_size=4, lr=5e-3, patience=100, seed=0, device="cpu")


def test_load_exact_cd_matches_power_law(campaign) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    assert sorted(cd_by_re) == sorted(RE_LEVELS)
    for re, cd in cd_by_re.items():
        assert cd == pytest.approx(POWER_A * re**POWER_B, rel=1e-9)


def test_load_exact_cd_falls_back_to_status_tail(campaign) -> None:
    fields_dir, drag_dir = campaign
    for sidecar in (drag_dir / "points").glob("*/drag_history.json"):
        sidecar.unlink()
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    for re, cd in cd_by_re.items():
        assert cd == pytest.approx(POWER_A * re**POWER_B, rel=1e-9)


def test_build_drag_split_shapes_and_labels(campaign) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    spec = PlaneSampleSpec(steps=(500,))
    split = build_drag_split(fields_dir, cd_by_re, ["p0000", "p0003"], spec)
    assert split.x.shape == (2, 3, NY, NX)
    assert split.x.dtype == np.float32
    assert split.point_id == ["p0000", "p0003"]
    assert split.step == [500, 500]
    for re, cd in zip(split.re, split.cd):
        assert cd == pytest.approx(cd_by_re[float(re)], rel=1e-12)


def test_build_drag_split_multi_step(campaign) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    split = build_drag_split(fields_dir, cd_by_re, ["p0000"], PlaneSampleSpec(steps=STEPS))
    assert split.x.shape[0] == len(STEPS)
    assert split.step == list(STEPS)
    assert len(set(split.cd)) == 1


def test_missing_step_raises_with_available(campaign) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    with pytest.raises(KeyError, match="have"):
        build_drag_split(fields_dir, cd_by_re, ["p0000"], PlaneSampleSpec(steps=(123,)))


def test_norm_fitted_on_train_split_only(campaign) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    train = build_drag_split(
        fields_dir, cd_by_re, SPLIT_POINTS["train"], PlaneSampleSpec(steps=(500,))
    )
    norm = fit_norm(train, "log10")
    hand_mean = train.x.reshape(len(train), 3, -1).mean(axis=(0, 2))
    np.testing.assert_allclose(norm.channel_mean, hand_mean, rtol=1e-6)
    y = np.log10(train.cd)
    assert norm.target_mean == pytest.approx(float(y.mean()))
    # mutating a *different* split must not change train statistics
    test = build_drag_split(
        fields_dir, cd_by_re, SPLIT_POINTS["test"], PlaneSampleSpec(steps=(500,))
    )
    test.x = test.x * 3.0 + 1.0
    norm2 = fit_norm(train, "log10")
    assert norm2.channel_mean == norm.channel_mean


def test_per_point_join_when_re_repeats(campaign) -> None:
    """Multi-parameter campaigns repeat Re across points; labels join by pid."""
    fields_dir, drag_dir = campaign
    st_path = drag_dir / "points" / "p0001" / "status.json"
    status = json.loads(st_path.read_text())
    status["params"]["re"] = RE_LEVELS[0]  # duplicate p0000's Re, keep own label
    st_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="duplicate Re"):
        load_exact_cd(drag_dir, fields_dir)
    per_point = load_exact_cd_per_point(drag_dir, fields_dir)
    assert set(per_point) == set(POINT_IDS)
    assert per_point["p0000"] != per_point["p0001"]  # same Re, distinct labels
    split = build_drag_split(
        fields_dir,
        point_ids=["p0000", "p0001"],
        spec=PlaneSampleSpec(steps=(500,)),
        cd_by_point=per_point,
    )
    assert split.cd[0] == per_point["p0000"]
    assert split.cd[1] == per_point["p0001"]


def test_velocity_scale_divides_velocity_channels_only(campaign) -> None:
    """velocity_scale=True: ux/uy divided by u_in, rho untouched, labels equal."""
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    raw = build_drag_split(fields_dir, cd_by_re, POINT_IDS[:3], PlaneSampleSpec(steps=(500,)))
    scaled = build_drag_split(
        fields_dir, cd_by_re, POINT_IDS[:3], PlaneSampleSpec(steps=(500,), velocity_scale=True)
    )
    assert raw.u_in is not None and scaled.u_in is not None
    assert np.all(raw.u_in == pytest.approx(0.1))
    np.testing.assert_allclose(scaled.x[:, :2], raw.x[:, :2] / np.float32(0.1), rtol=1e-5)
    np.testing.assert_array_equal(scaled.x[:, 2:], raw.x[:, 2:])  # rho bitwise untouched
    np.testing.assert_array_equal(scaled.cd, raw.cd)
    np.testing.assert_array_equal(scaled.re, raw.re)


def test_power_law_fit_recovers_exponent() -> None:
    re = np.asarray(RE_LEVELS)
    cd = POWER_A * re**POWER_B
    log10_a, exponent = power_law_fit(re, cd)
    assert log10_a == pytest.approx(np.log10(POWER_A), abs=1e-9)
    assert exponent == pytest.approx(POWER_B, abs=1e-9)
    np.testing.assert_allclose(power_law_predict((log10_a, exponent), re), cd, rtol=1e-9)


def test_regression_metrics_sanity() -> None:
    y = np.asarray([1.0, 2.0, 3.0])
    perfect = regression_metrics(y, y)
    assert perfect["mae"] == 0.0 and perfect["rmse"] == 0.0
    assert perfect["r2"] == pytest.approx(1.0)
    mean_pred = regression_metrics(y, np.full_like(y, y.mean()))
    assert mean_pred["r2"] == pytest.approx(0.0, abs=1e-12)
    assert mean_pred["mape"] == pytest.approx(np.mean(np.abs(y - y.mean()) / y) * 100)


def test_train_loss_decreases_and_predicts(campaign) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    spec = PlaneSampleSpec(steps=(500,))
    train = build_drag_split(fields_dir, cd_by_re, SPLIT_POINTS["train"], spec)
    val = build_drag_split(fields_dir, cd_by_re, SPLIT_POINTS["val"], spec)
    result = train_drag_surrogate(train, val, _tiny_arch(), _tiny_config())
    assert result.history["train"][-1] < result.history["train"][0]
    pred = predict_cd(result.model, train, result.norm, device="cpu")
    assert pred.shape == (len(train),)
    assert np.all(pred > 0)


def test_save_load_roundtrip(campaign, tmp_path: Path) -> None:
    fields_dir, drag_dir = campaign
    cd_by_re = load_exact_cd(drag_dir, fields_dir)
    spec = PlaneSampleSpec(steps=(500,))
    train = build_drag_split(fields_dir, cd_by_re, SPLIT_POINTS["train"], spec)
    val = build_drag_split(fields_dir, cd_by_re, SPLIT_POINTS["val"], spec)
    result = train_drag_surrogate(train, val, _tiny_arch(), _tiny_config())
    path = save_drag_regressor(result.model, result.norm, tmp_path / "model.pt")
    assert path.is_file() and path.with_suffix(".pt.json").is_file()
    model2, norm2 = load_drag_regressor(path)
    test = build_drag_split(fields_dir, cd_by_re, SPLIT_POINTS["test"], spec)
    np.testing.assert_allclose(
        predict_cd(result.model, test, result.norm, device="cpu"),
        predict_cd(model2, test, norm2, device="cpu"),
        rtol=1e-6,
    )


def test_study_end_to_end(campaign, tmp_path: Path) -> None:
    fields_dir, drag_dir = campaign
    out = tmp_path / "study"
    summary = run_drag_surrogate_study(
        fields_dir,
        drag_dir,
        out,
        spec=PlaneSampleSpec(steps=(500,)),
        arch=_tiny_arch(),
        config=_tiny_config(),
    )
    assert summary["schema"] == "tensorlbm.drag-surrogate-study/v1"
    assert set(summary["metrics"]) == {"train", "val", "test"}
    for split_metrics in summary["metrics"].values():
        assert set(split_metrics) == {"fno", "power_law", "mean"}
        assert split_metrics["power_law"]["mape"] < 1e-6  # synthetic law is exact
    assert summary["metrics"]["test"]["fno"]["n"] == len(SPLIT_POINTS["test"])
    assert (out / "model.pt").is_file()
    assert (out / "metrics.json").is_file()
    rows = (out / "predictions.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + sum(len(v) for v in SPLIT_POINTS.values())
