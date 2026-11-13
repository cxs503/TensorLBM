"""Tests for the AA-pattern stepper and its even/odd timestep tracker.

The anchor is ``XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER``: a single-field
(in-place) streaming scheme whose kernels ``e``/``o`` move the two velocity
halves, plus the tracker that knows whether the buffer is in a complete or a
half-streamed state.  Every claim made in ``src/tensorlbm/aa_pattern.py`` has a
test here; the numbers behind the write-up live in
``benchmarks/xval_aa_pattern_even_odd.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.aa_pattern import (
    HALF_A,
    HALF_B,
    AAPatternParityError,
    AAPatternStepper,
    ab_step,
    ab_stream,
    equilibrium_from_velocity,
)
from tensorlbm.d2q9 import macroscopic

TOL_BITWISE = 0.0  # streaming is a permutation: float64 must round-trip exactly


def _random_state(ny: int = 16, nx: int = 24, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 1e-3 * torch.randn((ny, nx), generator=g, dtype=torch.float64)
    ux = 1e-2 * torch.randn((ny, nx), generator=g, dtype=torch.float64)
    uy = 1e-2 * torch.randn((ny, nx), generator=g, dtype=torch.float64)
    return equilibrium_from_velocity(rho, ux, uy)


def _tg_state(n: int = 32, u0: float = 0.05, modes: int = 1) -> tuple[torch.Tensor, float, float]:
    k = modes * 2.0 * math.pi / n
    ii, jj = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="xy"
    )
    x = ii + 0.5
    y = jj + 0.5
    ux = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy = u0 * torch.sin(k * x) * torch.cos(k * y)
    return equilibrium_from_velocity(torch.ones((n, n), dtype=torch.float64), ux, uy), k, u0


# ------------------------------------------------------------------ structure
def test_velocity_halves_partition_the_direction_set():
    assert set(HALF_A) | set(HALF_B) == set(range(1, 9))
    assert not (set(HALF_A) & set(HALF_B))
    assert len(HALF_A) == len(HALF_B) == 4


def test_single_persistent_field_is_the_aa_storage_claim():
    f0 = _random_state()
    st = AAPatternStepper(f0, 0.7)
    report = st.parity_report()
    assert report["n_persistent_fields"] == 1
    assert report["nbytes_persistent"] == 9 * 16 * 24 * 8
    # the AB reference needs a second field of the same size
    assert report["nbytes_persistent"] < 2 * 9 * 16 * 24 * 8


# ------------------------------------------------------------------ kernel map
def test_one_kernel_pair_equals_ab_streaming():
    f0 = _random_state()
    st = AAPatternStepper(f0, 0.7)
    st.kernel_even()
    st.kernel_odd()
    ref = ab_stream(f0)
    assert float((st.f - ref).abs().max()) == TOL_BITWISE
    assert st.phase == "complete"


def test_kernel_pair_is_exact_on_odd_grid_sizes():
    """No assumption on ny/nx parity: the split is by velocity half, not cells."""
    f0 = _random_state(ny=17, nx=23)
    st = AAPatternStepper(f0, 0.7)
    st.kernel_even()
    st.kernel_odd()
    assert float((st.f - ab_stream(f0)).abs().max()) == TOL_BITWISE


def test_kernel_order_is_immaterial():
    f0 = _random_state()
    a = AAPatternStepper(f0, 0.7, kernel_order=("even", "odd"))
    b = AAPatternStepper(f0, 0.7, kernel_order=("odd", "even"))
    for _ in range(20):
        a.step()
        b.step()
    assert float((a.f - b.f).abs().max()) == TOL_BITWISE


# --------------------------------------------------------------- equivalence
def test_multi_step_bitwise_equivalence_with_ab_reference():
    f0 = _random_state()
    st = AAPatternStepper(f0, 0.7)
    f = f0.clone()
    for _ in range(100):
        st.step()
        f = ab_step(f, 0.7)
    assert float((st.f - f).abs().max()) == TOL_BITWISE
    assert st.time == 100


def test_probe_does_not_perturb_the_run():
    """Taking the half-window snapshot must leave the trajectory untouched."""
    f0 = _random_state()
    plain = AAPatternStepper(f0, 0.7)
    probed = AAPatternStepper(f0, 0.7)
    for step in range(30):
        plain.step()
        if step == 12:
            snap = probed.probe_half_streamed()
            assert snap["phase"] == "half"
            assert snap["kernels_since_complete"] == 1
            assert snap["time"] == 12
        else:
            probed.step()
    assert float((plain.f - probed.f).abs().max()) == TOL_BITWISE
    assert plain.time == probed.time == 30


# ---------------------------------------------------------------- invariants
def test_mass_and_momentum_are_conserved():
    f0 = _random_state()
    st = AAPatternStepper(f0, 0.7)
    m0 = float(f0.sum())
    rho0, ux0, uy0 = macroscopic(f0)
    p0 = (float((rho0 * ux0).sum()), float((rho0 * uy0).sum()))
    for _ in range(200):
        st.step()
    rho, ux, uy = st.macroscopic()
    assert abs(float(rho.sum()) - m0) / abs(m0) < 1e-13
    assert abs(float((rho * ux).sum()) - p0[0]) < 1e-13
    assert abs(float((rho * uy).sum()) - p0[1]) < 1e-13


# ------------------------------------------------------------------- tracker
def test_half_streamed_state_is_refused():
    st = AAPatternStepper(_random_state(), 0.7)
    st.collide()
    st.kernel_even()
    assert st.phase == "half"
    assert st.kernels_since_complete == 1
    with pytest.raises(AAPatternParityError):
        st.macroscopic()
    with pytest.raises(AAPatternParityError):
        st.collide()
    with pytest.raises(AAPatternParityError):
        st.step()
    st.kernel_odd()
    assert st.phase == "complete"
    rho, _, _ = st.macroscopic()  # allowed again
    assert torch.isfinite(rho).all()
    report = st.parity_report()
    assert report["parity_violations"] == 3
    assert report["parity_checks"] == 5  # macroscopic, collide, step, macroscopic


def test_allow_partial_read_is_the_control_arm():
    f0 = _random_state()
    st = AAPatternStepper(f0, 0.7)
    st.collide()
    st.kernel_even()
    rho, ux, uy = st.macroscopic(allow_partial=True)
    assert torch.isfinite(ux).all() and torch.isfinite(uy).all()
    assert st.parity_report()["parity_violations"] == 1
    # the half state is not a valid time level: it differs from the complete one
    st.kernel_odd()
    rho_c, ux_c, uy_c = st.macroscopic()
    rel = float(torch.sqrt(((ux - ux_c) ** 2 + (uy - uy_c) ** 2).mean())
                / torch.sqrt((ux_c**2 + uy_c**2).mean()))
    assert rel > 1e-3


def test_timestep_parity_alternates():
    st = AAPatternStepper(_random_state(), 0.7)
    seen = []
    for _ in range(4):
        seen.append(st.timestep_parity)
        st.step()
    assert seen == ["even", "odd", "even", "odd"]
    report = st.parity_report()
    assert report["even_kernel_calls"] == 4 and report["odd_kernel_calls"] == 4
    # the leading kernel alternates with the timestep parity, so the *pairs*
    # are (even, odd) then (odd, even) -- both halves move once per timestep
    tail = report["kernel_log_tail"]
    assert len(tail) == 4
    assert set(tail) == {"even", "odd"}
    assert tail[0] == "even" and tail[2] == "odd"


def test_parity_blind_arm_is_detectable():
    """Losing the even/odd alternation is a real, measurable defect."""
    f0 = _random_state()
    good = AAPatternStepper(f0, 0.7)
    bad = AAPatternStepper(f0, 0.7)
    f = f0.clone()
    for _ in range(20):
        good.step()
        bad.parity_blind_step()
        f = ab_step(f, 0.7)
    assert float((good.f - f).abs().max()) == TOL_BITWISE
    # the broken arm still looks "complete" to a slot-counting check
    assert bad.phase == "complete"
    rho_b, ux_b, uy_b = bad.macroscopic()
    rho_f, ux_f, uy_f = macroscopic(f)
    rel = float(torch.sqrt(((ux_b - ux_f) ** 2 + (uy_b - uy_f) ** 2).mean())
                / torch.sqrt((ux_f**2 + uy_f**2).mean()))
    assert rel > 0.1, rel


# ------------------------------------------------------------- physics anchor
def test_taylor_green_decay_matches_closed_form():
    """Small end-to-end acceptance: gamma_K = 4 nu k^2 within 5 %."""
    n, u0, re, steps = 32, 0.05, 100.0, 400
    f0, k, _ = _tg_state(n, u0)
    nu = u0 * n / re
    tau = 3.0 * nu + 0.5
    st = AAPatternStepper(f0, tau)
    hist = []
    for step in range(steps + 1):
        if step % 20 == 0:
            rho, ux, uy = st.macroscopic()
            hist.append((step, float(0.5 * (ux * ux + uy * uy).mean())))
        if step == steps:
            break
        st.step()
    import numpy as np

    t = np.array([h[0] for h in hist[4:]], dtype=np.float64)
    val = np.array([h[1] for h in hist[4:]], dtype=np.float64)
    gamma = -float(np.polyfit(t, np.log(val), 1)[0])
    gamma_exact = 4.0 * nu * k * k
    assert abs(gamma / gamma_exact - 1.0) < 0.05, (gamma, gamma_exact)


# ------------------------------------------------------------------ contract
def test_input_contract():
    f0 = _random_state()
    with pytest.raises(ValueError):
        AAPatternStepper(f0, 0.5)  # tau must be > 0.5
    with pytest.raises(ValueError):
        AAPatternStepper(f0[0], 0.7)  # shape must be (9, ny, nx)
    with pytest.raises(ValueError):
        AAPatternStepper(f0, 0.7, kernel_order=("even", "even"))
    with pytest.raises(ValueError):
        AAPatternStepper(f0, 0.7, timestep_parity="middle")


def test_stepper_does_not_alias_the_caller_buffer():
    f0 = _random_state()
    snapshot = f0.clone()
    st = AAPatternStepper(f0, 0.7)
    st.step()
    assert float((f0 - snapshot).abs().max()) == 0.0