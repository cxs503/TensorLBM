"""Tests for solver-in-the-loop closure calibration (autograd_calib)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.autograd_calib import (
    BoxCase,
    DragTarget,
    bounded_drag,
    calibrate,
    cs_power,
    evaluate,
    synthetic_targets,
)

# Small but identifiability-safe: at tau <= 0.58 the windowed C_D responds
# ~13-17% across a 12x C_s range (measured 2026-08-23); at tau >= 0.65 the
# response collapses to 2-7% and the closure is NOT identifiable from drag.
BOX = BoxCase(nz=6, ny=8, nx=18, radius=2, u_in=0.20, steps=80, window_start=60)


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
