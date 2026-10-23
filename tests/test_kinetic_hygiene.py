"""Unit tests for the backend-agnostic kinetic-hygiene probes.

The period-2 conviction convention (p15): mean absolute one-/two-step
differences of the per-substep global x-momentum series, their ratio and
the odd/even mean difference.  The tests pin the three calibration regimes
(alternation, white noise, smooth trend) with deterministic inputs.
"""

from __future__ import annotations

import random

import pytest
import torch

from tensorlbm.d3q19 import C
from tensorlbm.kinetic_hygiene import global_x_momentum, period2_mx_probe


def _manual_momentum(f: torch.Tensor, mask: torch.Tensor | None) -> float:
    total = 0.0
    for q in range(19):
        cx = float(C[q][0])
        if cx == 0.0:
            continue
        part = f[q][mask].sum() if mask is not None else f[q].sum()
        total += cx * float(part.item())
    return total


def test_global_x_momentum_matches_manual_sum() -> None:
    generator = torch.Generator().manual_seed(7)
    f = torch.rand((19, 6, 5, 4), generator=generator)
    mask = torch.rand((6, 5, 4), generator=generator) > 0.5
    # fp32 reduction-order tolerance only: the structural claim (sign
    # convention + masking) is exact
    assert global_x_momentum(f) == pytest.approx(_manual_momentum(f, None), rel=1e-4)
    assert global_x_momentum(f, mask) == pytest.approx(_manual_momentum(f, mask), rel=1e-4)
    # an all-True mask is the unmasked sum
    assert global_x_momentum(f, torch.ones_like(mask)) == pytest.approx(
        global_x_momentum(f), rel=1e-4
    )
    # uniform equilibrium state with u_x = 0.05: Mx = rho * u * N_cells
    from tensorlbm.d3q19 import equilibrium3d

    rho = torch.ones((3, 3, 3))
    ux = torch.full_like(rho, 0.05)
    zero = torch.zeros_like(rho)
    feq = equilibrium3d(rho, ux, zero, zero)
    assert global_x_momentum(feq) == pytest.approx(0.05 * 27.0, rel=1e-5)


def test_period2_pure_alternation() -> None:
    series = [1.0, -1.0] * 500
    probe = period2_mx_probe(series)
    assert probe["samples"] == 1000
    assert probe["mean_abs_d2"] == pytest.approx(0.0, abs=1e-12)
    assert probe["d2_over_d1"] == pytest.approx(0.0, abs=1e-12)
    assert probe["odd_even_mean_difference"] == pytest.approx(2.0)
    # the series mean is exactly zero, so the normalised variant is the
    # documented guard value (None), not a division by zero
    assert probe["odd_even_mean_difference_over_abs_mean"] is None


def test_period2_smooth_trend() -> None:
    series = [0.01 * i for i in range(2000)]
    probe = period2_mx_probe(series)
    assert probe["d2_over_d1"] == pytest.approx(2.0, rel=1e-6)
    assert probe["odd_even_mean_difference"] == pytest.approx(0.01, abs=1e-9)


def test_period2_iid_white_noise_ratio_is_one() -> None:
    rng = random.Random(0)
    series = [rng.gauss(0.0, 1.0) for _ in range(4000)]
    probe = period2_mx_probe(series)
    # iid samples: one- and two-step differences share the same variance
    assert 0.90 < probe["d2_over_d1"] < 1.10


def test_period2_random_walk_ratio_near_sqrt2() -> None:
    rng = random.Random(0)
    steps = [rng.gauss(0.0, 1.0) for _ in range(4000)]
    series = [0.0]
    for step in steps:
        series.append(series[-1] + step)
    probe = period2_mx_probe(series)
    # increment-dominated series: Var(d2) = 2 Var(d1)
    assert 1.30 < probe["d2_over_d1"] < 1.50


def test_period2_short_series_returns_none_fields() -> None:
    probe = period2_mx_probe([3.0, 4.0])
    assert probe["samples"] == 2
    assert probe["mean_abs_d1"] is None
    assert probe["d2_over_d1"] is None
    assert probe["odd_even_mean_difference"] is None
    assert probe["reading"]
    assert period2_mx_probe([])["samples"] == 0


def test_period2_alternation_onto_trend_is_detected() -> None:
    series = [0.001 * i + (0.5 if i % 2 else -0.5) for i in range(1000)]
    probe = period2_mx_probe(series)
    assert probe["d2_over_d1"] < 0.1  # alternation dominates both differences
    # odd/even mean gap = alternation amplitude + one drift step
    assert probe["odd_even_mean_difference"] == pytest.approx(1.001, abs=1e-9)
