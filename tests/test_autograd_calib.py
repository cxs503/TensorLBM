"""Tests for solver-in-the-loop closure calibration (autograd_calib)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tensorlbm.autograd_calib import (
    BoxCase,
    DragHistory,
    HullCase,
    bounded_drag,
    calibrate,
    cd_from_force,
    cs_power,
    drag_targets_from_sidecars,
    evaluate,
    load_drag_history,
    synthetic_targets,
    windowed_cd,
)
from tensorlbm.autograd_path import rollout
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_survey import projected_area

# Small but identifiability-safe: at tau <= 0.58 the windowed C_D responds
# ~13-17% across a 12x C_s range (measured 2026-08-23); at tau >= 0.65 the
# response collapses to 2-7% and the closure is NOT identifiable from drag.
BOX = BoxCase(nz=6, ny=8, nx=18, radius=2, u_in=0.20, steps=80, window_start=60)

# Reduced production-like hull: same builder/placement/tau relation as the
# suboff_n128 campaign case, scaled down for a CPU test (tau 0.60 at Re 30).
HULL = HullCase(nz=16, ny=16, nx=32, u_in=0.05, steps=60, window_start=40)


def test_box_case_validation() -> None:
    with pytest.raises(ValueError, match="window_start"):
        BoxCase(nz=6, ny=8, nx=14, steps=10, window_start=10)
    with pytest.raises(ValueError, match="inlet_method"):
        BoxCase(nz=6, ny=8, nx=14, inlet_method="bounce")
    with pytest.raises(ValueError, match="laterally"):
        BoxCase(nz=6, ny=8, nx=14, radius=4)


def test_mask_shape_and_tau() -> None:
    mask = BOX.make_mask()
    assert mask.shape == (6, 8, 18)
    assert mask.dtype == torch.bool
    assert 0 < int(mask.sum()) < 6 * 8 * 18
    # house relation: tau = 0.5 + 3*u_in*D/Re with D = 2r
    assert math.isclose(float(BOX.tau_of_re(40.0)), 0.5 + 3 * 0.20 * 4 / 40.0)


def test_bounded_drag_finite() -> None:
    cd = bounded_drag(BOX, re=6.0, cs=0.1)
    assert torch.isfinite(cd)
    assert cd.item() > 0.0


def test_bounded_drag_gradient_matches_fd() -> None:
    """Autograd dC_D/dC_s through the bounded rollout vs central differences."""
    cs = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    cd = bounded_drag(BOX, re=40.0, cs=cs)
    cd.backward()
    autograd = cs.grad.item()

    eps = 1e-5
    fd = (
        bounded_drag(BOX, re=40.0, cs=0.1 + eps).item()
        - bounded_drag(BOX, re=40.0, cs=0.1 - eps).item()
    ) / (2 * eps)
    assert autograd != 0.0
    assert abs(autograd - fd) / abs(fd) < 1e-6


def test_synthetic_targets_deterministic() -> None:
    closure = cs_power(0.1, -0.25, re_ref=40.0)
    a = synthetic_targets(BOX, (30.0, 48.0), closure)
    b = synthetic_targets(BOX, (30.0, 48.0), closure)
    assert [t.cd for t in a] == [t.cd for t in b]
    assert [t.re for t in a] == [30.0, 48.0]


def test_calibrate_rejects_empty_targets() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        calibrate([], BOX)


def test_calibrate_scalar_recovers_constant() -> None:
    """Verification mode: one constant C_s must come back from its own data."""
    truth = 0.12
    targets = synthetic_targets(BOX, (30.0, 48.0), cs_power(truth, 0.0))
    result = calibrate(targets, BOX, kind="scalar", cs0=0.05, iters=60, lr=0.03)
    assert result.loss_history[-1] < 0.05 * result.loss_history[0]
    assert abs(result.params["cs"] - truth) / truth < 0.05


def test_calibrate_power_tracks_re_dependence() -> None:
    """An Re-dependent truth: the power closure fits train Re and extrapolates.

    A single constant cannot fit a falling C_s(Re): the C_D-against-C_s slope
    varies ~8x across the sweep (steep at low Re, flat at high Re), so the
    best constant must compromise at the ends.  The two-parameter power
    closure trained on three Re reproduces them, stays close at a held-out
    fourth Re between the training points, and beats the constant at the
    train endpoints where the compromise hurts.
    """
    truth = cs_power(0.08, -1.2, re_ref=40.0)
    train_res = (30.0, 48.0, 70.0)
    train = synthetic_targets(BOX, train_res, truth)
    result = calibrate(train, BOX, kind="power", cs0=0.05, iters=110, lr=0.03)

    ev = evaluate(result, train, BOX)
    assert set(ev) == {f"{re:g}" for re in train_res}
    assert all(row["rel_err_pct"] < 3.0 for row in ev.values())

    def rel_err(closure, re: float) -> float:
        cd_true = bounded_drag(BOX, re=re, cs=truth(re)).item()
        cd_hat = bounded_drag(BOX, re=re, cs=closure(re)).item()
        return abs(cd_hat - cd_true) / cd_true

    assert rel_err(result.closure, 58.0) < 0.02

    # the constant closure, given the same data and budget, is worse at the
    # endpoints it had to compromise between (why B3 needs Re dependence)
    scalar = calibrate(train, BOX, kind="scalar", cs0=0.05, iters=110, lr=0.03)
    ends = (30.0, 70.0)
    scalar_err = max(rel_err(scalar.closure, re) for re in ends)
    power_err = max(rel_err(result.closure, re) for re in ends)
    assert scalar_err > power_err


# ---------------------------------------------------------------------------
# Real geometry: HullCase (production SUBOFF mask, free-stream sides)
# ---------------------------------------------------------------------------


def test_hull_case_validation() -> None:
    with pytest.raises(ValueError, match="window_start"):
        HullCase(nz=16, ny=16, nx=32, steps=10, window_start=10)
    with pytest.raises(ValueError, match="inlet_method"):
        HullCase(nz=16, ny=16, nx=32, inlet_method="bounce")
    with pytest.raises(ValueError, match="wall_method"):
        HullCase(nz=16, ny=16, nx=32, wall_method="sticky")
    with pytest.raises(ValueError, match="checkpoint_block"):
        HullCase(nz=16, ny=16, nx=32, checkpoint_block=0)
    with pytest.raises(ValueError, match="u_in"):
        HullCase(nz=16, ny=16, nx=32, u_in=0.0)


def test_hull_case_production_conventions() -> None:
    """HullCase reproduces the suboff_n128 case placement, tau and area.

    At the production resolution 64x64x128 the mask is the campaign hull
    itself: 4093 solid cells, 69 projected (y, z) columns — the exact
    ``S_proj`` the drag dataset is normalised with (a pi r_max^2 = 63.1
    convention would bias C_D by 9%).
    """
    n128 = HullCase(nz=64, ny=64, nx=128, u_in=0.1)
    assert n128.hull_length == 76.8  # 0.6 * nx, the production reference length
    # fp32 default dtype: agree with the production relation to fp32 rounding
    assert math.isclose(float(n128.tau_of_re(305.0)), 0.5 + 23.04 / 305.0, rel_tol=1e-6)
    mask = n128.make_mask()
    assert mask.shape == (64, 64, 128)
    assert int(mask.sum()) == 4093  # n_solid_cells in the dataset sidecars
    assert n128.ref_area(mask) == 69.0

    small = HullCase(nz=16, ny=16, nx=32, u_in=0.05, hull_length=19.2)
    assert math.isclose(float(small.tau_of_re(30.0)), 0.5 + 3 * 0.05 * 19.2 / 30.0, rel_tol=1e-6)


def test_hull_ref_area_matches_drag_survey_convention() -> None:
    mask = HULL.make_mask()
    # same reduction as the data-side tensorlbm.drag_survey.projected_area
    assert HULL.ref_area(mask) == float(projected_area(mask.cpu().numpy()))
    # and the sphere keeps its own pi r^2 convention
    assert math.isclose(BOX.ref_area(), math.pi * BOX.radius**2)


def test_bounded_drag_custom_mask_equals_default_sphere() -> None:
    """Passing the case's own mask explicitly is bit-for-bit the default path."""
    a = bounded_drag(BOX, re=40.0, cs=0.1)
    b = bounded_drag(BOX, re=40.0, cs=0.1, mask=BOX.make_mask())
    assert a.item() == b.item()


def test_hull_bounded_drag_gradient_matches_fd() -> None:
    """Gradients through the production-hull rollout (blocked checkpointing)."""
    case = HullCase(
        nz=16,
        ny=16,
        nx=32,
        u_in=0.05,
        steps=60,
        window_start=40,
        checkpoint=True,
        checkpoint_block=17,
        dtype=torch.float64,
    )
    mask = case.make_mask()
    cs = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    cd = bounded_drag(case, re=30.0, cs=cs, mask=mask)
    cd.backward()
    autograd = cs.grad.item()

    eps = 1e-4
    fd = (
        bounded_drag(case, re=30.0, cs=0.1 + eps, mask=mask).item()
        - bounded_drag(case, re=30.0, cs=0.1 - eps, mask=mask).item()
    ) / (2 * eps)
    assert abs(autograd - fd) / abs(fd) < 1e-6


def test_bounded_drag_blocked_checkpoint_matches_plain() -> None:
    """Block-level checkpointing: same windowed C_D and gradient (fp grouping)."""
    plain = HullCase(
        nz=16,
        ny=16,
        nx=32,
        u_in=0.05,
        steps=60,
        window_start=40,
        checkpoint=False,
        checkpoint_block=1,
        dtype=torch.float64,
    )
    blocked = HullCase(
        nz=16,
        ny=16,
        nx=32,
        u_in=0.05,
        steps=60,
        window_start=40,
        checkpoint=True,
        checkpoint_block=17,
        dtype=torch.float64,
    )
    mask = plain.make_mask()
    outs = []
    for case in (plain, blocked):
        cs = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        cd = bounded_drag(case, re=30.0, cs=cs, mask=mask)
        cd.backward()
        outs.append((cd.item(), cs.grad.item()))
    assert math.isclose(outs[0][0], outs[1][0], rel_tol=1e-12)
    assert math.isclose(outs[0][1], outs[1][1], rel_tol=1e-10)


def test_hull_freestream_walls_differ_from_periodic() -> None:
    """HullCase default closes the sides; periodic sides change the drag."""
    freestream = bounded_drag(HULL, re=30.0, cs=0.1)
    periodic = bounded_drag(
        HullCase(nz=16, ny=16, nx=32, u_in=0.05, steps=60, window_start=40, wall_method="periodic"),
        re=30.0,
        cs=0.1,
    )
    assert freestream.item() != periodic.item()


def test_hull_calibrate_smoke() -> None:
    """The calibration loop runs end to end on the production-geometry path."""
    truth = 0.12
    targets = synthetic_targets(HULL, (24.0, 30.0), cs_power(truth, 0.0))
    result = calibrate(targets, HULL, kind="scalar", cs0=0.05, iters=6, lr=0.03)
    assert result.loss_history[-1] < result.loss_history[0]
    ev = evaluate(result, targets, HULL)
    assert set(ev) == {"24", "30"}
    assert all(row["rel_err_pct"] < 15.0 for row in ev.values())


def test_rollout_probe_start() -> None:
    """probe_start collects exactly the tail probes (memory economy path)."""
    nz, ny, nx = 5, 6, 10
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float64)
    u = torch.zeros((3, nz, ny, nx), dtype=torch.float64)
    u[0] = 0.1
    f0 = equilibrium3d(rho0, u[0], u[1], u[2])
    _f, all_probes = rollout(f0, 6, 0.8, return_probes=True)
    _f, tail_probes = rollout(f0, 6, 0.8, return_probes=True, probe_start=4)
    assert len(tail_probes) == 2
    assert torch.equal(tail_probes[0], all_probes[4])
    assert torch.equal(tail_probes[1], all_probes[5])
    with pytest.raises(ValueError, match="probe_start"):
        rollout(f0, 6, 0.8, return_probes=True, probe_start=6)


# ---------------------------------------------------------------------------
# Real observations: drag_history sidecars -> DragTarget
# ---------------------------------------------------------------------------


def _write_sidecar(directory, re: float, force_tail: float, n_samples: int = 8) -> None:
    """A minimal scan_runner point directory: drag_history + status.json."""
    samples = [
        {
            "step": 25 * (i + 1),
            "force_x": force_tail + 1.0 / (i + 1),
            "force_y": 0.0,
            "force_z": 0.0,
            "force_abs": force_tail + 1.0 / (i + 1),
        }
        for i in range(n_samples)
    ]
    (directory / "drag_history.json").write_text(
        json.dumps(
            {
                "schema": "tensorlbm.drag-history/v1",
                "point_id": directory.name,
                "interval": 25,
                "samples": samples,
            }
        )
    )
    (directory / "status.json").write_text(
        json.dumps({"point_id": directory.name, "params": {"re": re}})
    )


def test_load_drag_history_and_windows(tmp_path) -> None:
    d = tmp_path / "p0000"
    d.mkdir()
    _write_sidecar(d, re=305.0, force_tail=1.8, n_samples=8)

    history = load_drag_history(d / "drag_history.json")
    assert len(history.steps) == 8
    assert history.steps[0] == 25 and history.steps[-1] == 200
    assert history.force.shape == (8, 3)

    # tail mean over the last 25% (2 samples, force 1.8+1/7 and 1.8+1/8)
    tail = history.force[-2:, 0].mean()
    assert math.isclose(cd_from_force(tail, 0.1, 69.0), 2.0 * tail / (0.01 * 69.0))
    # half-open windows; unknown windows are an error, not a silent nan
    assert windowed_cd(history, 25, 75, 0.1, 69.0) == cd_from_force(
        history.force[:2, 0].mean(), 0.1, 69.0
    )
    with pytest.raises(ValueError, match="no drag samples"):
        windowed_cd(history, 500, 600, 0.1, 69.0)


def test_load_drag_history_rejects_other_schema(tmp_path) -> None:
    p = tmp_path / "drag_history.json"
    p.write_text(json.dumps({"schema": "tensorlbm.drag-history/v2", "samples": []}))
    with pytest.raises(ValueError, match="schema"):
        load_drag_history(p)
    p.write_text(json.dumps({"schema": "tensorlbm.drag-history/v1", "samples": []}))
    with pytest.raises(ValueError, match="no samples"):
        load_drag_history(p)


def test_drag_history_validation() -> None:
    with pytest.raises(ValueError, match="DragHistory"):
        DragHistory(steps=np.zeros(3), force=np.zeros((2, 3)))


def test_drag_targets_from_sidecars(tmp_path) -> None:
    for name, re, f in (("p0", 305.0, 1.8), ("p1", 437.8, 1.4)):
        d = tmp_path / name
        d.mkdir()
        _write_sidecar(d, re=re, force_tail=f)

    targets = drag_targets_from_sidecars(
        sorted(tmp_path.glob("p*/drag_history.json")), u_in=0.1, ref_area=69.0
    )
    assert [t.re for t in targets] == [305.0, 437.8]
    for t, f in zip(targets, (1.8, 1.4)):
        expected_tail = f + (1.0 / 7 + 1.0 / 8) / 2  # last-2-sample mean
        assert math.isclose(t.cd, 2.0 * expected_tail / (0.01 * 69.0), rel_tol=1e-12)

    # explicit re_values bypass the sibling status.json ...
    explicit = drag_targets_from_sidecars(
        [tmp_path / "p0" / "drag_history.json"], u_in=0.1, ref_area=69.0, re_values=[9.0]
    )
    assert explicit[0].re == 9.0
    with pytest.raises(ValueError, match="re_values has"):
        drag_targets_from_sidecars(
            [tmp_path / "p0" / "drag_history.json"], u_in=0.1, ref_area=69.0, re_values=(1.0, 2.0)
        )
    with pytest.raises(ValueError, match="tail_frac"):
        drag_targets_from_sidecars(
            [tmp_path / "p0" / "drag_history.json"], u_in=0.1, ref_area=69.0, tail_frac=0.0
        )
    # ... but without them a missing status.json is an explicit error
    orphan = tmp_path / "lonely.json"
    orphan.write_text(
        json.dumps(
            {
                "schema": "tensorlbm.drag-history/v1",
                "samples": [
                    {"step": 25, "force_x": 1.0, "force_y": 0.0, "force_z": 0.0, "force_abs": 1.0}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="status.json"):
        drag_targets_from_sidecars([orphan], u_in=0.1, ref_area=69.0)
