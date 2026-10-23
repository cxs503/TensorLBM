"""B3 stage 5: differentiable D3Q27 collision families (cumulant / MRT).

Covers the tensor-safe rate path in :func:`tensorlbm.d3q27.collide_mrt27`,
the autograd-clean cumulant rewrite
:func:`tensorlbm.autograd_calib.collide_cumulant27_diffable` (bitwise equal
to :func:`tensorlbm.cumulant.collide_cumulant_d3q27`), the D3Q27 rollout /
:func:`press_profile27` observable, the finite-difference identifiability
probe :func:`rate_fd_response27` and the differentiable
:func:`calibrate_collision27` on a synthetic self-consistency target.
"""

import math

import pytest
import torch

from tensorlbm.autograd_calib import (
    CUMULANT27_RATES,
    MRT27_RATES,
    BoxCase,
    calibrate_collision27,
    collide_cumulant27_diffable,
    press_profile27,
    rate_fd_response27,
    rollout27,
)
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import collide_mrt27, equilibrium27

BOX = BoxCase(nz=20, ny=20, nx=32, radius=3.5, u_in=0.12, steps=60, window_start=40)
RE = 100.0


def _state() -> torch.Tensor:
    torch.manual_seed(0)
    nz, ny, nx = 6, 7, 8
    rho = torch.ones((nz, ny, nx)) + 0.01 * torch.randn(nz, ny, nx)
    u = 0.01 * torch.randn(3, nz, ny, nx)
    f = equilibrium27(rho, u[0], u[1], u[2])
    return f + 0.02 * torch.randn_like(f)


def test_mrt27_tensor_rates_bitwise_equal_float_path():
    f = _state()
    kw = {"s_e": 1.19, "s_eps": 1.4, "s_q": 1.2}
    ref = collide_mrt27(f, 0.7, **kw)
    tens = collide_mrt27(f, 0.7, **{k: torch.tensor(v) for k, v in kw.items()})
    assert torch.equal(ref, tens)
    tau_t = torch.tensor(0.7)
    assert torch.equal(ref, collide_mrt27(f, tau_t, **kw))


def test_mrt27_rate_gradients_flow():
    f = _state()
    rates = {k: torch.tensor(v, requires_grad=True) for k, v in MRT27_RATES.items()}
    out = collide_mrt27(f, 0.7, s_e=rates["s_e"], s_eps=rates["s_eps"], s_q=rates["s_q"])
    # Random-direction projection: conserved projections (density, momentum)
    # cannot see relaxation rates at all (probe-degeneracy note,
    # docs/closure_calibration.md).
    w = torch.randn(27)
    (w.view(-1, 1, 1, 1) * out).sum().backward()
    for k, r in rates.items():
        assert r.grad is not None
        assert torch.isfinite(r.grad)
        assert float(r.grad.abs()) > 0.0, k


def test_cumulant27_diffable_bitwise_equal_float_rates():
    f = _state()
    ref = collide_cumulant_d3q27(f, 0.7, omega_b=1.3, omega_odd=0.8, omega_even=1.1)
    new = collide_cumulant27_diffable(f, 0.7, omega_b=1.3, omega_odd=0.8, omega_even=1.1)
    assert torch.equal(ref, new)


def test_cumulant27_diffable_rate_gradients_flow():
    f = _state()
    rates = {k: torch.tensor(v, requires_grad=True) for k, v in CUMULANT27_RATES.items()}
    out = collide_cumulant27_diffable(
        f,
        0.7,
        omega_b=rates["omega_b"],
        omega_odd=rates["omega_odd"],
        omega_even=rates["omega_even"],
    )
    w = torch.randn(27)
    (w.view(-1, 1, 1, 1) * out).sum().backward()
    for k, r in rates.items():
        assert r.grad is not None
        assert torch.isfinite(r.grad)
        assert float(r.grad.abs()) > 0.0, k


def test_rollout27_end_to_end_gradient_through_rates():
    """The full step chain (collide -> stream -> faces -> bounce-back) is
    autograd-clean: a rate gradient reaches the probe observables."""
    from tensorlbm.autograd_calib import _rate27_collide

    torch.manual_seed(3)
    nz, ny, nx = 8, 9, 12
    mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
    mask[2:6, 3:6, 4:8] = True
    rho0 = torch.ones((nz, ny, nx))
    u0 = torch.zeros((3, nz, ny, nx))
    u0[0] = 0.1
    f = equilibrium27(rho0, u0[0], u0[1], u0[2])
    rates = {k: torch.tensor(v, requires_grad=True) for k, v in CUMULANT27_RATES.items()}
    collide = _rate27_collide(torch.tensor(0.7), rates, "cumulant")
    f, probes = rollout27(f, 3, torch.tensor(0.7), mask, collide, 0.1, return_probes=True)
    rho = torch.stack([p.sum(dim=0) for p in probes])  # (3, nz, ny, nx)
    w = torch.randn_like(rho)
    (w * rho).sum().backward()
    for k, r in rates.items():
        assert r.grad is not None
        assert torch.isfinite(r.grad)


def test_press_profile27_shape_and_determinism():
    p1, cd1 = press_profile27(BOX, RE, bins=16)
    p2, cd2 = press_profile27(BOX, RE, bins=16)
    assert p1.shape == (16,)
    assert p1.dtype == torch.float64
    assert torch.isfinite(p1).all()
    assert cd1 > 0.0
    assert torch.equal(p1, p2)
    assert cd1 == pytest.approx(cd2, rel=1e-12)
    assert float(p1[0]) == 0.0
    assert int((p1 != 0).sum()) >= 4


def test_press_profile27_reads_rates_and_family():
    p_lo, _ = press_profile27(BOX, RE, dict(CUMULANT27_RATES, omega_b=0.7), bins=16)
    p_hi, _ = press_profile27(BOX, RE, dict(CUMULANT27_RATES, omega_b=1.6), bins=16)
    d = torch.linalg.vector_norm(p_hi - p_lo) / torch.linalg.vector_norm(p_lo)
    assert float(d) > 0.01
    p_mrt, _ = press_profile27(BOX, RE, family="mrt", bins=16)
    d_fam = torch.linalg.vector_norm(p_mrt - p_lo) / torch.linalg.vector_norm(p_lo)
    assert float(d_fam) > 0.01


def test_rate_fd_response27_keys_and_sanity():
    for family, expect in (("cumulant", set(CUMULANT27_RATES)), ("mrt", set(MRT27_RATES))):
        out = rate_fd_response27(BOX, RE, family=family, frac=0.2, bins=16)
        assert set(out) == expect, family
        for name, row in out.items():
            assert row["press_per_efold"] > 0.0, (family, name)
            assert row["cd_per_efold"] >= 0.0, (family, name)


def test_calibrate_collision27_recovers_perturbed_rate():
    truth = dict(CUMULANT27_RATES, omega_b=1.5)
    target, _ = press_profile27(BOX, RE, truth, bins=16)
    res = calibrate_collision27(
        {RE: target}, BOX, family="cumulant", iters=8, lr=0.05, block=5, bins=16
    )
    assert res.family == "cumulant"
    assert res.loss_history[-1] < res.loss_history[0]
    assert res.rates["omega_b"] > CUMULANT27_RATES["omega_b"]
    ev = res.eval[f"{RE:g}"]
    assert ev["press_after"] < ev["press_before"]


def test_calibrate_collision27_rejects_bad_family_and_block():
    target = torch.zeros(16)
    with pytest.raises(ValueError, match="family"):
        calibrate_collision27({RE: target}, BOX, family="bgk", bins=16)
    with pytest.raises(ValueError, match="divisible"):
        calibrate_collision27({RE: target}, BOX, block=7, bins=16)


def test_calibrate_collision27_nan_guard_keeps_finite_rates():
    """A diverging rollout (NaN loss) must not poison the rates: the guard
    reverts to the family defaults and stops with a finite loss history."""
    nan_target = torch.full((16,), float("nan"))
    res = calibrate_collision27({RE: nan_target}, BOX, family="cumulant", iters=3, block=5, bins=16)
    assert res.rates == CUMULANT27_RATES
    assert res.loss_history == []
    ev = res.eval[f"{RE:g}"]
    # the press gaps are NaN only because the target itself is NaN;
    # the simulation side (C_D) stays finite
    assert math.isnan(ev["press_before"]) and math.isnan(ev["press_after"])
    assert math.isfinite(ev["cd_before"]) and math.isfinite(ev["cd_after"])
