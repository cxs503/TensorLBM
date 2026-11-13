"""Tests for the differentiable graph-unroll / BPTT LES parameter path.

Invariants asserted (not merely "code runs"):

1. the conv-as-streaming operator equals a roll-based gather (shift = +c);
2. the LES closure degenerates to plain BGK for ``Sc = 0`` (cross-checked
   against the pre-existing ``solver.collide_bgk``);
3. the forward operator reproduces the vendored OSS reference arm to float32
   round-off (the XVAL parity claim);
4. the BPTT gradient equals a central finite difference of the same unrolled
   loss (the claim that the retained graph is differentiated correctly);
5. BPTT reduces the calibration loss on a short twin experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "xval" / "XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER"))

from tensorlbm import les_bptt as L  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_bgk_les  # noqa: E402

torch.set_num_threads(1)
oss_arm = pytest.importorskip("oss_arm")


def test_conv_stream_equals_roll_gather():
    torch.manual_seed(0)
    f = torch.rand(1, 9, 8, 10, dtype=torch.float64)
    rolled = torch.stack(
        [torch.roll(f[:, q], shifts=(int(c[1]), int(c[0])), dims=(1, 2))
         for q, c in enumerate(L.D2Q9_C)],
        dim=1,
    )
    assert torch.allclose(L.stream(f), rolled, atol=1e-14)


def test_les_zero_recovers_bgk():
    f0, b = L.taylor_green_state(n=16, u0=0.1, k=2)
    out_les, _, _ = collide_bgk_les(f0, tau0=0.56, smagorinsky_sc=0.0, boundary=b)
    ref = collide_bgk(f0[0], 0.56).unsqueeze(0)
    assert torch.allclose(out_les, ref, atol=1e-12)


def test_forward_parity_with_oss_reference():
    n, steps, tau0, sc, u0 = 16, 3, 0.56, 0.17, 0.1
    f0, b = L.taylor_green_state(n=n, u0=u0, k=2)
    hist_t = L.unroll(f0, steps, tau0=tau0, sc=sc, boundary=b)
    f_o = f0.numpy().transpose(0, 2, 3, 1).astype("float32")
    b_o = b.numpy().transpose(0, 2, 3, 1).astype("float32")
    hist_o = oss_arm.unroll_sc(f_o, steps, tau0, sc, b_o)
    ft = hist_t[-1].f[0].permute(1, 2, 0).numpy()
    fo = hist_o[-1]["f"][0]
    rel = float(abs(ft - fo).max() / max(abs(fo).max(), 1e-30))
    assert rel < 1e-5, rel


def test_bptt_gradient_matches_finite_difference():
    f0, b = L.taylor_green_state(n=16, u0=0.2, k=2)
    steps, tau0 = 12, 0.56
    target = [s.vel.detach() for s in L.unroll(f0, steps, tau0=tau0, sc=0.51, boundary=b)]
    sc = torch.tensor(0.17, dtype=f0.dtype, requires_grad=True)
    loss = L.velocity_mse_loss(L.unroll(f0, steps, tau0=tau0, sc=sc, boundary=b), target)
    loss.backward()
    grad_ad = float(sc.grad)
    eps = 1e-3

    def loss_at(val):
        return float(L.velocity_mse_loss(
            L.unroll(f0, steps, tau0=tau0, sc=torch.tensor(val, dtype=f0.dtype), boundary=b),
            target))

    fd = (loss_at(0.17 + eps) - loss_at(0.17 - eps)) / (2 * eps)
    assert abs(fd - grad_ad) / abs(fd) < 1e-2


def test_bptt_calibration_reduces_loss():
    f0, b = L.taylor_green_state(n=16, u0=0.2, k=2)
    steps, tau0, sc_true = 15, 0.56, 0.17
    target = [s.vel.detach() for s in L.unroll(f0, steps, tau0=tau0, sc=sc_true, boundary=b)]
    hist = L.calibrate_les_parameter(target, f0, tau0=tau0, sc_init=0.05, boundary=b,
                                    iters=60, lr=0.2)
    assert hist["loss"] < hist["loss_history"][0]
    assert abs(hist["sc"] - sc_true) < 0.5 * sc_true


def test_unroll_history_is_graph_connected():
    f0, b = L.taylor_green_state(n=8, u0=0.1, k=1)
    sc = torch.tensor(0.17, dtype=f0.dtype, requires_grad=True)
    hist = L.unroll(f0, 5, tau0=0.56, sc=sc, boundary=b)
    assert all(h.f.grad_fn is not None for h in hist)
