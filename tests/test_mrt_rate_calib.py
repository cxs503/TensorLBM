"""B3 stage 4: MRT moment-rate calibration (press-profile observable).

Covers the tensor-safe rate path in :func:`tensorlbm.solver3d.collide_mrt3d`,
the no-grad :func:`press_profile` observable, the finite-difference
identifiability probe and the differentiable :func:`calibrate_mrt_rates`
on a synthetic self-consistency target.
"""

import pytest
import torch

from tensorlbm.autograd_calib import (
    DEFAULT_MRT_RATES,
    BoxCase,
    calibrate_mrt_rates,
    press_profile,
    rate_fd_response,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.solver3d import collide_mrt3d

BOX = BoxCase(nz=20, ny=20, nx=32, radius=3.5, u_in=0.12, steps=120, window_start=80)
RE = 100.0


def test_tensor_rates_bitwise_equal_float_path():
    torch.manual_seed(0)
    nz, ny, nx = 6, 7, 8
    rho = torch.ones((nz, ny, nx)) + 0.01 * torch.randn(nz, ny, nx)
    u = 0.01 * torch.randn(3, nz, ny, nx)
    f = equilibrium3d(rho, u[0], u[1], u[2])
    tau = 0.7
    kw = {"s_e": 1.19, "s_eps": 1.4, "s_q": 1.2}
    ref = collide_mrt3d(f, tau, **kw)
    tens = collide_mrt3d(f, tau, **{k: torch.tensor(v) for k, v in kw.items()})
    assert torch.equal(ref, tens)


def test_rate_gradients_flow():
    torch.manual_seed(0)
    nz, ny, nx = 6, 7, 8
    rho = torch.ones((nz, ny, nx)) + 0.01 * torch.randn(nz, ny, nx)
    u = 0.01 * torch.randn(3, nz, ny, nx)
    f = equilibrium3d(rho, u[0], u[1], u[2])
    rates = {k: torch.tensor(v, requires_grad=True) for k, v in DEFAULT_MRT_RATES.items()}
    collide_mrt3d(f, 0.7, s_e=rates["s_e"], s_eps=rates["s_eps"], s_q=rates["s_q"]).sum().backward()
    for k, r in rates.items():
        assert r.grad is not None
        assert torch.isfinite(r.grad)
        assert float(r.grad.abs()) > 0.0, k


def test_press_profile_shape_and_determinism():
    p1, cd1 = press_profile(BOX, RE, bins=16)
    p2, cd2 = press_profile(BOX, RE, bins=16)
    assert p1.shape == (16,)
    assert torch.isfinite(p1).all()
    assert cd1 > 0.0
    assert torch.equal(p1, p2)
    assert cd1 == pytest.approx(cd2, rel=1e-12)
    # empty bins (upstream of the obstacle and beyond the wake tail) stay zero
    assert float(p1[0]) == 0.0
    assert float(p1[-1]) == 0.0
    # the shell lives somewhere mid-domain: some bins non-zero
    assert int((p1 != 0).sum()) >= 4


def test_press_profile_reads_rates():
    lo = dict(DEFAULT_MRT_RATES, s_e=0.8)
    hi = dict(DEFAULT_MRT_RATES, s_e=1.6)
    p_lo, _ = press_profile(BOX, RE, lo)
    p_hi, _ = press_profile(BOX, RE, hi)
    d = torch.linalg.vector_norm(p_hi - p_lo) / torch.linalg.vector_norm(p_lo)
    assert float(d) > 0.01


def test_rate_fd_response_keys_and_sanity():
    out = rate_fd_response(BOX, RE, frac=0.2)
    assert set(out) == {"s_e", "s_eps", "s_q"}
    for name, row in out.items():
        assert row["press_per_efold"] > 0.0, name
        assert row["cd_per_efold"] >= 0.0, name


def test_calibrate_mrt_rates_recovers_perturbed_rate():
    truth = dict(DEFAULT_MRT_RATES, s_e=1.45)
    target, _ = press_profile(BOX, RE, truth)
    res = calibrate_mrt_rates({RE: target}, BOX, iters=8, lr=0.05, block=5, bounds=(0.5, 2.0))
    assert res.loss_history[-1] < res.loss_history[0]
    assert res.rates["s_e"] > DEFAULT_MRT_RATES["s_e"]
    ev = res.eval[f"{RE:g}"]
    assert ev["press_after"] < ev["press_before"]


def test_window_must_divide_by_block():
    target = torch.zeros(32)
    with pytest.raises(ValueError, match="divisible"):
        calibrate_mrt_rates({RE: target}, BOX, block=7)
