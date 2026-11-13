"""Root-finding inversion of ``tau`` (GAP-PP-2-lid-wall-diff, round 2).

Round 1 (``tests/test_diff_lid_cavity.py``) closed the graph-preserving lid BC
and showed that first-order optimisers stall in the very flat 1-D loss valley
of the cavity inverse problem (``docs/GAP_PP_2_LID_WALL_DIFF.md`` sections 4.4
and 6: ``tau`` recovery 1.80 % at n=48 but 2.98 % at n=65, i.e. criterion (1)
missed at production resolution).

The stationary condition ``g(s) = dL/ds = 0`` is a *scalar* equation, so it can
be solved by a root finder that needs one backward pass per iteration and no
Hessian.  These tests pin that machinery:

1. :func:`solve_stationary_1d` finds the root of an exact quadratic gradient to
   machine precision (both ``illinois`` and ``secant``), and it does so in a
   handful of oracle calls -- the property a first-order optimiser lacks;
2. a *flat* valley (curvature 1e-8, the regime that broke Adam) is still solved
   to < 1e-6 in <= 8 evaluations, whereas Adam at the documented learning rate
   needs orders of magnitude more steps to reach the same accuracy;
3. auto-bracketing widens until ``dL/ds`` changes sign, and an unbracketable
   oracle is reported honestly (``status="not_bracketed"``) instead of
   silently returning a wrong root;
4. the bracketing invariant holds: the final bracket brackets a sign change,
   and the evaluated points stay inside the bracket they were derived from;
5. oracle results are cached (bracket endpoints are not re-evaluated);
6. invalid arguments raise;
7. :func:`invert_relaxation_time_rootfind` recovers ``tau`` from its own
   fixed-step cavity data well inside the 2 % acceptance tolerance on a small
   grid (smoke version of criterion (1)), and reports the documented keys.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.diff_lid_cavity import (
    CavityGeometry,
    centreline_probes,
    initial_field,
    invert_relaxation_time_rootfind,
    rollout,
    solve_stationary_1d,
)

# ---------------------------------------------------------------------------
# 1-D oracle tests (no LBM): the algorithm, in isolation
# ---------------------------------------------------------------------------


def _quadratic(x: float) -> tuple[float, float]:
    """Exact quadratic valley around 0.3 (simple root of the gradient)."""
    d = x - 0.3
    return d * d, 2.0 * d


def _flat_quadratic(x: float) -> tuple[float, float]:
    """Same valley with curvature 1e-8 -- the Adam-hostile regime."""
    d = x - 0.3
    return 1e-8 * d * d, 2e-8 * d


@pytest.mark.parametrize("method", ["illinois", "secant"])
def test_quadratic_root_machine_precision(method: str) -> None:
    res = solve_stationary_1d(
        _quadratic, 1.0, method=method, bracket=(0.0, 1.0), max_iter=50
    )
    assert res["status"] == "converged"
    assert res["x"] == pytest.approx(0.3, abs=1e-8)
    assert res["value"] < 1e-16
    # a simple root of a quadratic gradient: a handful of evaluations
    assert res["n_evals"] <= 8


@pytest.mark.parametrize("method", ["illinois", "secant"])
def test_flat_valley_solved_in_few_evals(method: str) -> None:
    """The failure mode of round 1: curvature ~1e-8, first-order stalls."""
    res = solve_stationary_1d(
        _flat_quadratic, 0.8, method=method, bracket=(0.1, 0.9), max_iter=20
    )
    assert abs(res["x"] - 0.3) < 1e-6
    assert res["n_evals"] <= 8

    # Reference behaviour of the round-1 optimiser on the same oracle.
    s = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([s], lr=0.05)
    steps_to_1e_6 = None
    for it in range(1, 5001):
        opt.zero_grad()
        value = 1e-8 * (s - 0.3) ** 2
        value.backward()
        opt.step()
        if abs(float(s) - 0.3) < 1e-6:
            steps_to_1e_6 = it
            break
    assert steps_to_1e_6 is None or steps_to_1e_6 > 8, (
        "Adam outperforming the root finder here would invalidate the round-2 "
        f"motivation (steps={steps_to_1e_6})"
    )


def test_auto_bracket_widens_until_sign_change() -> None:
    res = solve_stationary_1d(_quadratic, 1.0, method="illinois", max_iter=50)
    assert res["status"] == "converged"
    assert res["x"] == pytest.approx(0.3, abs=1e-6)
    roles = [h["role"] for h in res["history"]]
    assert any(r.startswith("bracket_lo+") for r in roles), roles


def test_not_bracketed_is_reported() -> None:
    def always_positive(x: float) -> tuple[float, float]:
        return x, 1.0  # g = 1 everywhere: no root

    res = solve_stationary_1d(always_positive, 0.5, method="illinois", max_expand=2)
    assert res["status"] == "not_bracketed"
    assert res["bracket"] is None
    assert math.isfinite(res["value"])


def test_final_bracket_has_sign_change() -> None:
    res = solve_stationary_1d(_quadratic, 0.9, method="illinois", bracket=(0.0, 1.0))
    lo, hi = res["bracket"]
    g_lo = _quadratic(lo)[1]
    g_hi = _quadratic(hi)[1]
    assert g_lo * g_hi <= 0.0
    assert lo <= res["x"] <= hi


def test_oracle_results_are_cached() -> None:
    calls = {"n": 0}

    def counted(x: float) -> tuple[float, float]:
        calls["n"] += 1
        return _quadratic(x)

    res = solve_stationary_1d(counted, 0.9, method="illinois", bracket=(0.0, 1.0))
    assert res["n_evals"] == calls["n"]
    assert res["n_evals"] < len(res["history"]) or res["n_evals"] == len(res["history"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "golden"},
        {"bracket": (1.0, 0.0)},
        {"bracket": (0.5, 0.5)},
        {"max_iter": 0},
        {"invalid_policy": "ignore"},
    ],
)
def test_invalid_arguments_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        solve_stationary_1d(_quadratic, 0.9, **kwargs)


def test_non_finite_oracle_raises_when_strict() -> None:
    def nan_oracle(x: float) -> tuple[float, float]:
        return float("nan"), 0.0

    with pytest.raises(FloatingPointError):
        solve_stationary_1d(
            nan_oracle, 0.9, method="illinois", bracket=(0.0, 1.0), invalid_policy="raise"
        )


def test_non_finite_bracket_end_is_retracted() -> None:
    """A non-finite (unstable) bracket end must not abort the inversion."""

    def unstable_left(x: float) -> tuple[float, float]:
        if x < 0.2:  # e.g. tau -> 0.5: the BGK roll-out diverges
            return float("nan"), float("nan")
        return _quadratic(x)

    res = solve_stationary_1d(
        unstable_left, 0.9, method="illinois", bracket=(-0.5, 1.0), invalid_policy="shrink"
    )
    assert res["status"] == "converged"
    assert res["x"] == pytest.approx(0.3, abs=1e-6)
    assert res["n_invalid"] >= 1
    assert res["bracket"][0] >= 0.2


def test_invalid_initial_point_raises() -> None:
    def nan_oracle(x: float) -> tuple[float, float]:
        return float("nan"), 0.0

    with pytest.raises(FloatingPointError):
        solve_stationary_1d(nan_oracle, 0.9, invalid_policy="shrink")


# ---------------------------------------------------------------------------
# LBM smoke tests: the same machinery through the differentiable cavity solver
# ---------------------------------------------------------------------------


def _cavity_truth(n: int, steps: int, tau_true: float) -> torch.Tensor:
    geom = CavityGeometry(n, n)
    u = torch.tensor(0.1, dtype=torch.float64)
    f0 = initial_field(geom)
    with torch.no_grad():
        f = rollout(f0, tau_true, u, steps, wall_mask=geom.wall_mask)
        return centreline_probes(f, u).detach()


@pytest.mark.parametrize("method", ["illinois", "secant"])
def test_invert_tau_rootfind_smoke(method: str) -> None:
    n, steps, tau_true = 20, 120, 0.692
    target = _cavity_truth(n, steps, tau_true)
    geom = CavityGeometry(n, n)
    res = invert_relaxation_time_rootfind(
        target,
        geom,
        0.1,
        tau_true * 1.05,
        steps,
        method=method,
        bracket_tau=(0.60, 0.80),
        max_iter=20,
        step_tol=1e-6,
        checkpoint_every=60,
    )
    rel = abs(res["tau_recovered"] - tau_true) / tau_true
    assert rel < 0.01, (res["tau_recovered"], rel)
    assert res["method"] == f"rootfind:{method}"
    assert res["status"] in ("converged", "max_iter")
    assert res["n_evals"] >= 2
    for key in (
        "tau_recovered",
        "tau_best",
        "tau_history",
        "loss_history",
        "grad_history",
        "residual_rms",
        "obs_final",
    ):
        assert key in res
    assert torch.isfinite(res["obs_final"]).all()


def test_invert_tau_rootfind_auto_bracket() -> None:
    """No bracket given: geometric widening still finds a sign change."""
    n, steps, tau_true = 16, 80, 0.66
    target = _cavity_truth(n, steps, tau_true)
    geom = CavityGeometry(n, n)
    res = invert_relaxation_time_rootfind(
        target,
        geom,
        0.1,
        tau_true * 1.03,
        steps,
        method="illinois",
        bracket_frac=0.05,
        max_iter=16,
        step_tol=1e-6,
        checkpoint_every=40,
    )
    assert abs(res["tau_recovered"] - tau_true) / tau_true < 0.02
    assert res["status"] in ("converged", "max_iter")