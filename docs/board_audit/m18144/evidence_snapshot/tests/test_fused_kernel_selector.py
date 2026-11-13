"""Tests for the fused-kernel-type selector (XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR).

These tests assert *invariants*, not just that code runs:

* granularity equivalence -- the fused single-pass kernel must reproduce
  ``stream(collide_bgk(f))`` to round-off (the whole point of the selector:
  it is a performance knob, so any physics change is a defect);
* the alternative measurement order (``collide(stream(f))``, lbmpy's pull
  kernel) is reachable and stays distinct from the textbook order, so the
  documented semantics cannot silently drift;
* conservation of mass and momentum of the fused kernel;
* the selector rejects an unknown ``kernel_type`` exactly like
  ``lbmpy.creationfunctions.create_lb_update_rule`` does;
* the shear-wave physics benchmark stays far inside the 15 % criterion;
* the OSS cross-check against lbmpy is within round-off (skipped when the OSS
  stack is absent -- reported as not-run by the benchmark, never as passed).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.fused_kernel_selector import (
    KERNEL_TYPES,
    collide_stream_bgk,
    resolve_kernel_type,
    step,
)
from tensorlbm.solver import collide_bgk, stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xval" / "fused_kernel_type_selector"))

torch.set_num_threads(2)


def _random_field(n: int = 24, seed: int = 3, amp: float = 0.05) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = torch.ones((n, n), dtype=torch.float64)
    ux = amp * torch.randn((n, n), dtype=torch.float64, generator=g)
    uy = amp * torch.randn((n, n), dtype=torch.float64, generator=g)
    return equilibrium(rho, ux, uy)


def test_kernel_types_mirror_lbmpy():
    assert KERNEL_TYPES == ("default_stream_collide", "collide_only", "stream_pull_only")
    assert resolve_kernel_type("collide_only") == "collide_only"
    with pytest.raises(ValueError):
        resolve_kernel_type("not_a_kernel_type")


def test_fused_step_equals_separate_collide_then_stream():
    """``default_stream_collide`` == ``stream(collide_bgk(...))`` to round-off."""
    f = _random_field()
    got = step(f, 0.8, "default_stream_collide")
    ref = stream(collide_bgk(f, 0.8))
    assert torch.allclose(got, ref, atol=1e-14, rtol=0.0)

    # and it stays equal over a chain (no slow drift / no boundary leakage)
    a, b = f.clone(), f.clone()
    for _ in range(50):
        a = step(a, 0.8, "default_stream_collide")
        b = stream(collide_bgk(b, 0.8))
    assert float((a - b).abs().max()) < 1e-13


def test_collide_only_and_stream_pull_only_are_their_building_blocks():
    f = _random_field()
    assert torch.allclose(step(f, 0.8, "collide_only"), collide_bgk(f, 0.8), atol=0.0, rtol=0.0)
    assert torch.allclose(step(f, 0.8, "stream_pull_only"), stream(f), atol=0.0, rtol=0.0)


def test_alternative_moment_order_is_distinct_and_lbmpy_compatible():
    """``moments="post_stream"`` reproduces ``collide_bgk(stream(f))``, not S∘C."""
    f = _random_field()
    post = collide_stream_bgk(f, 0.8, moments="post_stream")
    assert torch.allclose(post, collide_bgk(stream(f), 0.8), atol=1e-14, rtol=0.0)
    pre = collide_stream_bgk(f, 0.8)
    # the two orders are genuinely different operators (this is why the order
    # has to be an explicit argument); on random fields the gap is O(10 %).
    assert float((post - pre).abs().max()) > 1e-3
    with pytest.raises(ValueError):
        collide_stream_bgk(f, 0.8, moments="whatever")


def test_fused_kernel_conserves_mass_and_momentum():
    f = _random_field()
    g = collide_stream_bgk(f, 0.8)
    # streaming permutes values between cells: total mass and total momentum
    # (periodic box) must be conserved to round-off.
    assert float(g.sum()) == pytest.approx(float(f.sum()), rel=1e-14)
    rho_a, ux_a, uy_a = macroscopic(f)
    rho_b, ux_b, uy_b = macroscopic(g)
    np_a = float((rho_a * ux_a).sum()), float((rho_a * uy_a).sum())
    np_b = float((rho_b * ux_b).sum()), float((rho_b * uy_b).sum())
    assert np_a == pytest.approx(np_b, rel=1e-12, abs=1e-14)


def test_rejects_bad_input():
    f = _random_field(8)
    with pytest.raises(ValueError):
        collide_stream_bgk(f[0], 0.8)  # wrong rank
    with pytest.raises(Exception):
        collide_stream_bgk(f, 0.4)  # tau <= 0.5 rejected by the shared contract


@pytest.mark.parametrize("n,tau,steps", [(32, 0.8, 200), (64, 0.6, 200)])
def test_shear_wave_decay_matches_analytic_within_15pct(n: int, tau: float, steps: int):
    """Physics benchmark: fused kernel vs the analytic shear-mode decay."""
    k = 2.0 * math.pi / n
    y = torch.arange(n, dtype=torch.float64)
    mode = torch.sin(k * y).unsqueeze(1)
    u0 = 1e-4
    rho = torch.ones((n, n), dtype=torch.float64)
    ux = u0 * mode.expand(n, n).clone()
    f0 = equilibrium(rho, ux, torch.zeros((n, n), dtype=torch.float64))
    amp0 = float((macroscopic(f0)[1] * mode).mean())

    f = f0
    for _ in range(steps):
        f = step(f, tau, "default_stream_collide")
    amp = float((macroscopic(f)[1] * mode).mean())

    nu = (tau - 0.5) / 3.0
    ref = amp0 * math.exp(-nu * k * k * steps)
    err = abs(amp - ref) / ref
    assert err < 0.15, f"shear-wave amplitude error {err:.3%} exceeds the 15 % criterion"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lbmpy") is None,
    reason="OSS lbmpy/pystencils not installed -- cross-check NOT RUN",
)
def test_cross_check_against_lbmpy_is_round_off():
    """tensorlbm fused(post_stream) vs lbmpy's compiled fused kernel."""
    import lbmpy_reference as oss

    f = _random_field(n=16)
    ref = torch.tensor(oss.run_lbmpy_steps(f.numpy(), 0.8, 1))
    got = collide_stream_bgk(f, 0.8, moments="post_stream")
    assert float((got - ref).abs().max()) < 1e-12

    ref50 = torch.tensor(oss.run_lbmpy_steps(f.numpy(), 0.8, 50))
    got50 = f.clone()
    for _ in range(50):
        got50 = collide_stream_bgk(got50, 0.8, moments="post_stream")
    assert float((got50 - ref50).abs().max()) < 1e-12