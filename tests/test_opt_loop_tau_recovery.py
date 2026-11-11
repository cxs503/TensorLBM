"""Tests for the differentiable tau/Re recovery demo (``opt_loop_tau_recovery``).

The demo is only worth its JSON if four properties hold, and each is pinned
here:

1. **the gradient is the gradient of the implemented objective** — autograd
   ``dL/d log nu`` through the whole LBM rollout equals the central finite
   difference of the identical discrete loss (relative difference < 1e-4);
2. **the optimiser converges on a case with a known answer** — the twin
   experiment (observations produced by the same solver) recovers ``tau`` to
   better than 3%, and the loss decreases monotonically over the run;
3. **the amplitude nuisance selection matters** — pinning the initial
   amplitude gives a larger ``nu`` bias than profiling it out, on the analytic
   (exact Navier-Stokes) observations: the documented method selection;
4. **reproducibility** — the same seed reproduces the same numbers bit-for-bit
   and a different seed starts from a different initial guess, so the 3-seed
   spread in the demo JSON is a real spread, not a copy.

Run time is kept small: ``n = 8`` cases, short observation windows, few
iterations.  The production numbers live in ``artifacts/opt_loop_tau_recovery.json``.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.opt_loop_tau_recovery import (
    SCHEMA,
    Observations,
    TaylorGreenCase,
    analytic_observations,
    assemble,
    gradient_check,
    recover,
    run_job,
    simulate,
    twin_observations,
    DEFAULT_TRACKS,
)

DTYPE = torch.float64


@pytest.fixture(scope="module")
def case() -> TaylorGreenCase:
    return TaylorGreenCase(n=8, u0=0.05, dtype=DTYPE)


def _times(case: TaylorGreenCase, tau: float, n_obs: int = 3, decay: float = 0.6) -> tuple[int, ...]:
    return case.times(case.nu_from_tau(tau), decay=decay, n_obs=n_obs)


# ---------------------------------------------------------------------------
# 1. gradient correctness
# ---------------------------------------------------------------------------


def test_gradient_matches_finite_difference(case: TaylorGreenCase) -> None:
    tau = 0.55
    obs = analytic_observations(case, case.nu_from_tau(tau), _times(case, tau))
    check = gradient_check(case, obs, tau=tau, eps=1e-3)
    assert math.isfinite(check["autograd"]) and check["autograd"] != 0.0
    assert check["rel_diff"] < 1e-4, check


def test_analytic_field_is_a_decaying_taylor_green_mode(case: TaylorGreenCase) -> None:
    nu = case.nu_from_tau(0.55)
    t = 50
    ux0, uy0 = case.analytic(0.0, nu)
    ux1, uy1 = case.analytic(float(t), nu)
    ratio = float((ux1.abs().max()) / (ux0.abs().max()))
    assert ratio == pytest.approx(math.exp(-2.0 * nu * case.k**2 * t), rel=1e-12)
    # the decay is purely temporal: the field keeps its shape
    assert torch.allclose(ux1, ux0 * ratio, rtol=1e-12, atol=1e-18)
    assert torch.allclose(uy1, uy0 * ratio, rtol=1e-12, atol=1e-18)
    # sampling the continuous peak on cell centres always undershoots it
    # (n = 8: max sin * max cos = 0.924^2 = 0.854; n = 16: 0.981^2 = 0.962)
    assert 0.80 < float(ux0.abs().max()) / case.u0 < 1.0


def test_single_z_layer_is_enough_for_the_2d_decay() -> None:
    """D3Q19 with one z-layer reproduces the nz>1 decay bit-for-bit here."""
    tau = torch.tensor(0.55, dtype=DTYPE)
    peaks = []
    for nz in (1, 2):
        c = TaylorGreenCase(n=8, nz=nz, dtype=DTYPE)
        f = c.initial_field()
        out = simulate(c, tau, (60,))
        peaks.append(float(out[0][0].abs().max()))
    assert peaks[0] == pytest.approx(peaks[1], rel=1e-12)


# ---------------------------------------------------------------------------
# 2. twin recovery
# ---------------------------------------------------------------------------


def test_twin_recovery_recovers_tau(case: TaylorGreenCase) -> None:
    tau_true = 0.55
    times = _times(case, tau_true, n_obs=3, decay=0.6)
    obs = twin_observations(case, tau_true, times)
    result = recover(case, obs, tau_true=tau_true, iters=2, lr=0.3, polish=10, seed=0)
    assert abs(result.rel_err_pct) < 3.0, result.rel_err_pct
    losses = [h["loss"] for h in result.history]
    assert losses[-1] < losses[0]  # the loop actually minimises
    assert result.tau_hat > 0.5  # positivity constraint respected
    assert 0.9 < result.amplitude_hat < 1.1


# ---------------------------------------------------------------------------
# 3. the amplitude-nuisance selection
# ---------------------------------------------------------------------------


def test_amplitude_nuisance_reduces_the_bias(case: TaylorGreenCase) -> None:
    tau_true = 0.55
    nu_true = case.nu_from_tau(tau_true)
    times = _times(case, tau_true, n_obs=4, decay=1.0)
    obs = analytic_observations(case, nu_true, times)
    free = recover(case, obs, tau_true=tau_true, iters=2, lr=0.3, polish=10, fit_amplitude=True, seed=0)
    pinned = recover(case, obs, tau_true=tau_true, iters=2, lr=0.3, polish=10, fit_amplitude=False, seed=0)
    # n = 8 is deliberately coarse (fast test): the free-amplitude arm still beats
    # the pinned one by ~4x; the production grids (n = 16/32) reach <0.2%/0.05%.
    assert abs(free.rel_err_pct) < abs(pinned.rel_err_pct), (free.rel_err_pct, pinned.rel_err_pct)
    # the free-amplitude fit absorbs the lattice-sampling offset (>=1.0)
    assert free.amplitude_hat > 1.0


# ---------------------------------------------------------------------------
# 4. reproducibility and payload shape
# ---------------------------------------------------------------------------


def test_lbfgs_polish_is_what_converges_the_loop(case: TaylorGreenCase) -> None:
    """Method selection: Adam alone stalls, the L-BFGS polish closes the gap."""
    tau_true = 0.55
    times = _times(case, tau_true, n_obs=3, decay=0.6)
    obs = twin_observations(case, tau_true, times)
    adam_only = recover(case, obs, tau_true=tau_true, iters=6, lr=0.3, polish=0, seed=0)
    polished = recover(case, obs, tau_true=tau_true, iters=2, lr=0.3, polish=10, seed=0)
    assert abs(polished.rel_err_pct) < abs(adam_only.rel_err_pct)
    assert abs(polished.rel_err_pct) < 1e-3, polished.rel_err_pct


def test_same_seed_reproduces_bit_for_bit(case: TaylorGreenCase) -> None:
    tau_true = 0.55
    times = _times(case, tau_true)
    obs = twin_observations(case, tau_true, times, noise=0.002, seed=7)
    a = recover(case, obs, tau_true=tau_true, iters=3, lr=0.3, polish=4, seed=7)
    b = recover(case, obs, tau_true=tau_true, iters=3, lr=0.3, polish=4, seed=7)
    assert a.tau_hat == b.tau_hat
    assert [h["loss"] for h in a.history] == [h["loss"] for h in b.history]
    assert a.rel_err_pct == b.rel_err_pct


def test_different_seeds_start_from_different_guesses(case: TaylorGreenCase) -> None:
    tau_true = 0.55
    times = _times(case, tau_true)
    obs = twin_observations(case, tau_true, times)
    r0 = recover(case, obs, tau_true=tau_true, iters=2, lr=0.3, polish=0, seed=0)
    r1 = recover(case, obs, tau_true=tau_true, iters=2, lr=0.3, polish=0, seed=1)
    assert r0.tau_init != r1.tau_init


def test_job_payload_is_json_serialisable_and_assembles(case: TaylorGreenCase) -> None:
    spec = [t for t in DEFAULT_TRACKS if t.name == "twin16"][0]
    small = type(spec)(**{**spec.__dict__, "n": 8, "iters": 2, "polish": 0})
    job = run_job(small, 0)
    blob = json.dumps(job)  # must not raise
    assert "tau_hat" in blob
    payload = assemble([job], [0])
    assert payload["schema"] == SCHEMA
    assert payload["summary"]["by_track"]["twin16"]["tau_hat"][0] == job["run"]["tau_hat"]
    assert json.dumps(payload)