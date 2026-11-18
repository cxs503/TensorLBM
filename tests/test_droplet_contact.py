import json
import sys

import numpy as np
import pytest
import torch

from tensorlbm.boundary.tank import TankStreamingPlan, stream_tank
from tensorlbm.phasefield.impact_tracking import ImpactTracker


@pytest.mark.parametrize("top", ["bounce_back", "pressure"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prepared_tank_bitwise_matches_direct_and_snapshots_geometry(top, dtype):
    rng = torch.Generator().manual_seed(33)
    f, g = torch.randn(2, 9, 8, 12, generator=rng, dtype=dtype).transpose(-1, -2)
    walls = torch.rand(12, 8, generator=rng) < 0.25
    expected = stream_tank(f, g, walls, top, 1.03)
    plan = TankStreamingPlan(walls, top, 1.03)
    walls.logical_not_()
    for actual, ref in zip(plan.stream(f, g), expected):
        assert torch.equal(actual, ref)


def scene():
    phi = np.full((12, 12), -1.0)
    phi[2:5, 4:8] = 1
    phi[8:] = 1
    return phi


def test_off_axis_bridge_and_interpolated_gap():
    field = scene()
    t = ImpactTracker(np.zeros_like(field, dtype=bool), (3, 6), (10, 6))
    field[5, 6] = -0.25
    m = t.observe(field, 0)
    assert m["centerline_gap"] == pytest.approx(7.5 - 4.8)
    assert t.observe(field, 3)["centerline_gap"] == pytest.approx(7.5 - 4.8)
    field[5:8, 4] = 1
    m = t.observe(field, 7)
    assert m["state"] == "first_bridge" and t.contact == [3, 7]
    assert m["pool_centerline_y"] == 1.5
    m = t.observe(field, 13)
    assert m["state"] == "after_first_bridge" and m["drop_centroid_y"] is None


@pytest.mark.parametrize("mode", ["diagonal", "wall", "disappearance", "initially_connected"])
def test_false_contact_is_excluded(mode):
    field = scene()
    solid = np.zeros_like(field, dtype=bool)
    if mode == "wall":
        solid[6, :] = True
    tracker = ImpactTracker(solid, (3, 6), (10, 6))
    if mode == "initially_connected":
        field[5:8, 4] = 1
    tracker.observe(field, 0)
    if mode == "diagonal":
        field[5:8, 3] = 1
    elif mode == "disappearance":
        field[2:5] = -1
    else:
        field[5:8, 4] = 1
    m = tracker.observe(field, 5)
    assert tracker.contact is None
    expected = {
        "diagonal": "separated",
        "wall": "separated",
        "disappearance": "tracking_lost",
        "initially_connected": "initially_connected",
    }
    assert m["state"] == expected[mode]


def test_required_contact_failure_keeps_evidence(tmp_path, monkeypatch):
    from examples import droplet_impact as impact

    monkeypatch.setattr(impact, "save_contour", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "impact",
            "--n-steps",
            "1",
            "--output-root",
            str(tmp_path),
            "--run-name",
            "short",
            "--require-contact",
        ],
    )
    with pytest.raises(RuntimeError, match="contact"):
        impact.main()
    meta = json.loads((tmp_path / "short/run_metadata.json").read_text())
    assert meta["health"]["passed"] and meta["status"] == "required_contact_unobserved"
    assert meta["impact"]["first_bridge_interval_open_closed"] is None
