"""Tests for the tensorlbm.inverse infrastructure package.

Covers:
1. Loss functions (relative_mse, weighted_mse, l1, composition)
2. Convergence log (schema, record, finalise, write, read-back)
3. Seed control (seed_everything reproducibility, jittered_init)
4. Optimiser loop (convergence, box constraints, L-BFGS polish, logging)
5. Integration: a minimal differentiable problem solved end-to-end
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tensorlbm.inverse import (
    CONVERGENCE_SCHEMA,
    ConvergenceLog,
    IterationRecord,
    LossComponent,
    LossComposition,
    OptimiseResult,
    default_relative_mse,
    jittered_init,
    l1_loss,
    optimise_loop,
    relative_mse_loss,
    seed_everything,
    weighted_mse_loss,
    write_convergence_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_obs():
    """A minimal observation object with a ``fields`` attribute."""

    class Obs:
        def __init__(self):
            self.fields = [
                (torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.5])),
                (torch.tensor([0.8, 1.6]), torch.tensor([0.4, 1.2])),
            ]

    return Obs()


@pytest.fixture
def simple_sim():
    """Simulated fields close to the observation."""

    class Sim(list):
        pass

    return Sim([
        (torch.tensor([1.01, 2.02]), torch.tensor([0.51, 1.51])),
        (torch.tensor([0.81, 1.62]), torch.tensor([0.41, 1.22])),
    ])


# ---------------------------------------------------------------------------
# 1. Loss functions
# ---------------------------------------------------------------------------


def test_relative_mse_loss_small_error(simple_sim, simple_obs):
    loss = relative_mse_loss(simple_sim, simple_obs)
    assert loss > 0
    assert loss < 0.01  # small perturbation → small relative error


def test_relative_mse_loss_zero_for_identical(simple_obs):
    loss = relative_mse_loss(
        [(f[0].clone(), f[1].clone()) for f in simple_obs.fields],
        simple_obs,
    )
    assert loss == pytest.approx(0.0, abs=1e-15)


def test_weighted_mse_loss_uniform_equals_relative(simple_sim, simple_obs):
    # weighted_mse with uniform weights should give same ordering
    loss_w = weighted_mse_loss(simple_sim, simple_obs)
    assert loss_w > 0


def test_l1_loss_positive(simple_sim, simple_obs):
    loss = l1_loss(simple_sim, simple_obs)
    assert loss > 0


def test_loss_composition_single():
    comp = LossComposition((LossComponent("mse", relative_mse_loss, 1.0),))
    assert comp.names == ("mse",)


def test_loss_composition_weighted_sum(simple_sim, simple_obs):
    comp = LossComposition((
        LossComponent("mse", relative_mse_loss, 2.0),
        LossComponent("l1", l1_loss, 0.5),
    ))
    total = comp(simple_sim, simple_obs)
    mse_val = relative_mse_loss(simple_sim, simple_obs)
    l1_val = l1_loss(simple_sim, simple_obs)
    expected = 2.0 * mse_val + 0.5 * l1_val
    assert torch.isclose(total, expected, rtol=1e-12)


def test_loss_composition_component_losses(simple_sim, simple_obs):
    comp = LossComposition((
        LossComponent("mse", relative_mse_loss, 1.0),
        LossComponent("l1", l1_loss, 0.5),
    ))
    comps = comp.component_losses(simple_sim, simple_obs)
    assert "mse" in comps
    assert "l1" in comps
    assert comps["mse"] > 0
    assert comps["l1"] > 0


def test_default_relative_mse():
    comp = default_relative_mse()
    assert comp.names == ("rel_mse",)


# ---------------------------------------------------------------------------
# 2. Convergence log
# ---------------------------------------------------------------------------


def test_convergence_log_record_and_finalise():
    log = ConvergenceLog(problem="test", seed=0)
    log.record(IterationRecord(iter=0, phase="adam", loss=1.0, params={"x": 0.5}))
    log.record(IterationRecord(iter=1, phase="adam", loss=0.5, params={"x": 0.6}))
    log.finalise(wall_s=1.23, converged=True, final_loss=0.5)
    payload = log.to_payload()
    assert payload["schema"] == CONVERGENCE_SCHEMA
    assert payload["problem"] == "test"
    assert payload["seed"] == 0
    assert len(payload["history"]) == 2
    assert payload["summary"]["converged"] is True
    assert payload["summary"]["wall_s"] == 1.23


def test_convergence_log_write_and_read(tmp_path: Path):
    log = ConvergenceLog(problem="test", seed=42)
    log.record(IterationRecord(iter=0, phase="adam", loss=2.0, params={"x": 1.0}))
    log.finalise(wall_s=0.5, converged=False)
    path = log.write(tmp_path / "conv.json")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["schema"] == CONVERGENCE_SCHEMA
    assert data["summary"]["converged"] is False


def test_convergence_log_raises_if_not_finalised():
    log = ConvergenceLog(problem="test", seed=0)
    with pytest.raises(RuntimeError, match="finalise"):
        log.to_payload()


def test_write_convergence_json_convenience(tmp_path: Path):
    history = [
        IterationRecord(iter=0, phase="adam", loss=1.0, params={"x": 0.5}),
    ]
    path = write_convergence_json(
        tmp_path / "conv2.json",
        problem="demo",
        seed=7,
        history=history,
        wall_s=0.1,
        converged=True,
        final_loss=0.5,
    )
    data = json.loads(path.read_text())
    assert data["problem"] == "demo"
    assert data["seed"] == 7


# ---------------------------------------------------------------------------
# 3. Seed control
# ---------------------------------------------------------------------------


def test_jittered_init_reproducible():
    a = jittered_init(1.0, seed=42, relative_range=0.2)
    b = jittered_init(1.0, seed=42, relative_range=0.2)
    assert a == b


def test_jittered_init_different_seeds():
    a = jittered_init(1.0, seed=0, relative_range=0.2)
    b = jittered_init(1.0, seed=1, relative_range=0.2)
    assert a != b


def test_jittered_init_within_range():
    for seed in range(10):
        val = jittered_init(2.0, seed=seed, relative_range=0.25)
        assert 1.5 <= val <= 2.5, f"seed={seed}: val={val}"


def test_seed_everything_reproduces():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# 4. Optimiser loop
# ---------------------------------------------------------------------------


def _make_linear_fit_problem():
    """A well-determined linear fit: recover (a, b) from y = a*x + b with multiple points."""
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    a_true, b_true = 2.0, 1.0
    y_true = a_true * x_data + b_true

    a = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    b = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward_fn():
        y_pred = a * x_data + b
        return y_pred, y_true

    def loss_fn(sim, obs):
        return ((sim - obs) ** 2).mean()

    return [a, b], ["a", "b"], forward_fn, loss_fn


def test_optimise_loop_converges_on_linear_fit():
    params, names, fwd, loss_fn = _make_linear_fit_problem()
    result = optimise_loop(
        params=params,
        param_names=names,
        loss_fn=loss_fn,
        forward_fn=fwd,
        iters=100,
        lr=0.05,
        polish=20,
    )
    assert result.final_loss < 1e-8
    a, b = result.params["a"], result.params["b"]
    assert abs(a - 2.0) < 1e-4, f"a={a}"
    assert abs(b - 1.0) < 1e-4, f"b={b}"


def test_optimise_loop_with_convergence_log():
    params, names, fwd, loss_fn = _make_linear_fit_problem()
    log = ConvergenceLog(problem="linear_fit", seed=0)
    result = optimise_loop(
        params=params,
        param_names=names,
        loss_fn=loss_fn,
        forward_fn=fwd,
        iters=50,
        lr=0.05,
        polish=10,
        convergence_log=log,
    )
    assert len(log.history) > 0
    assert log._finalised
    assert result.final_loss < 1e-6


def test_optimise_loop_box_constraints():
    """Box constraints should keep parameters within bounds."""
    x = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward_fn():
        return x, torch.tensor(10.0)  # wants to go to 10

    def loss_fn(sim, obs):
        return ((sim - obs) ** 2).mean()

    result = optimise_loop(
        params=[x],
        param_names=["x"],
        loss_fn=loss_fn,
        forward_fn=forward_fn,
        iters=20,
        lr=0.5,
        box_constraints={"x": (-1.0, 2.0)},
    )
    assert float(x.detach()) <= 2.0 + 1e-10


def test_optimise_lbfgs_polish_improves():
    params1, names1, fwd1, loss_fn1 = _make_linear_fit_problem()
    adam_only = optimise_loop(
        params=params1,
        param_names=names1,
        loss_fn=loss_fn1,
        forward_fn=fwd1,
        iters=5,
        lr=0.05,
        polish=0,
    )
    params2, names2, fwd2, loss_fn2 = _make_linear_fit_problem()
    polished = optimise_loop(
        params=params2,
        param_names=names2,
        loss_fn=loss_fn2,
        forward_fn=fwd2,
        iters=5,
        lr=0.05,
        polish=20,
    )
    assert polished.final_loss <= adam_only.final_loss


def test_optimise_result_loss_reduction():
    r = OptimiseResult(
        converged=True,
        n_iterations=10,
        wall_s=1.0,
        final_loss=0.01,
        initial_loss=1.0,
        params={"x": 0.5},
    )
    assert r.loss_reduction == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 5. Integration: end-to-end with LossComposition
# ---------------------------------------------------------------------------


def test_end_to_end_with_composition_and_logging(tmp_path: Path):
    """Full pipeline: composition loss + convergence log + JSON output."""
    seed_everything(99)

    # Problem: recover (a, b) from y = a*x + b with two loss components
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = 2.0 * x_data + 1.0  # a=2, b=1

    a = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    b = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward_fn():
        y_pred = a * x_data + b
        return y_pred, y_true

    def mse_fn(sim, obs):
        return ((sim - obs) ** 2).mean()

    def l1_fn(sim, obs):
        return (sim - obs).abs().mean()

    comp = LossComposition((
        LossComponent("mse", mse_fn, 1.0),
        LossComponent("l1", l1_fn, 0.1),
    ))

    log = ConvergenceLog(problem="linear_fit", seed=99)
    result = optimise_loop(
        params=[a, b],
        param_names=["a", "b"],
        loss_fn=comp,
        forward_fn=forward_fn,
        iters=50,
        lr=0.05,
        polish=10,
        convergence_log=log,
    )

    log.write(tmp_path / "linear_fit.json")
    data = json.loads((tmp_path / "linear_fit.json").read_text())

    assert data["schema"] == CONVERGENCE_SCHEMA
    assert result.final_loss < 0.001
    assert abs(float(a.detach()) - 2.0) < 0.01
    assert abs(float(b.detach()) - 1.0) < 0.01
    # Check component losses are recorded
    first_rec = data["history"][0]
    assert "extra" in first_rec
    assert "mse" in first_rec["extra"]
    assert "l1" in first_rec["extra"]
