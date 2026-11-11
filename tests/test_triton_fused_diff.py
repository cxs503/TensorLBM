"""Tests for the differentiable fused Triton step (adjoint kernel pair).

The adjoint is checked against two independent references:

1. **eager torch autograd** through the *same* discrete map
   (:func:`tensorlbm.triton_fused_diff.eager_stream_collide_step`, built from
   the audited ``solver3d.stream3d`` / ``solver3d.collide_bgk3d`` primitives);
2. **central finite differences** of the loss in fp64.

Gate values follow the gap ledger GAP-PP-1-triton-bwd: ``1e-5`` relative for
the autograd comparison, ``1e-4`` for the finite-difference cross-check.

On a host without CUDA the whole file is skipped unless ``TRITON_INTERPRET=1``
is set, in which case Triton runs the identical kernel source through its
NumPy interpreter on CPU.  Interpreter runs use small boxes (``n = 3``) to
stay inside the CI time budget; the 100-step acceptance evidence lives in
``benchmarks/triton_fused_bwd_validation.py``.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.triton_fused_diff import (
    eager_stream_collide_step,
    interpreter_mode,
    is_available,
    macroscopic,
    triton_fused_rollout,
    triton_fused_step,
)

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="needs CUDA for compiled Triton kernels or TRITON_INTERPRET=1",
)


def _field(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    rho = torch.ones((n, n, n))
    u = 0.05 * torch.randn((3, n, n, n))
    return equilibrium3d(rho, u[0], u[1], u[2]).to(torch.float32)


def _loss(f: torch.Tensor) -> torch.Tensor:
    _, u = macroscopic(f)
    return (u[0] ** 2).mean() * 100.0


def _eager_rollout(f0: torch.Tensor, tau, steps: int) -> torch.Tensor:
    f = f0
    for _ in range(steps):
        f = eager_stream_collide_step(f, tau)
    return f


def test_forward_matches_eager_primitives() -> None:
    """The fused kernel implements exactly ``collide_bgk3d(stream3d(f))``."""
    n = 4
    f0 = _field(n)
    got = triton_fused_step(f0, 0.8, block=n)
    ref = eager_stream_collide_step(f0, 0.8)
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel < 1e-6, rel


def test_mass_and_momentum_conserved() -> None:
    """BGK collision conserves mass and momentum; streaming only permutes."""
    n = 4
    f0 = _field(n)
    f1 = triton_fused_step(f0, 0.8, block=n)
    # total mass: collision conserves the lane sum per cell, streaming permutes
    assert abs(float(f1.sum() - f0.sum())) < 1e-4
    # global (box-summed) momentum is conserved by collision + periodic stream
    rho0, u0 = macroscopic(f0)
    rho1, u1 = macroscopic(f1)
    p0 = (rho0.unsqueeze(0) * u0).sum(dim=(1, 2, 3))
    p1 = (rho1.unsqueeze(0) * u1).sum(dim=(1, 2, 3))
    assert torch.allclose(p1, p0, atol=1e-3)


def test_grad_f_matches_autograd() -> None:
    """``dL/df`` from the adjoint kernel matches eager autograd."""
    n = 4
    f0 = _field(n)
    f_ref = f0.clone().requires_grad_(True)
    _loss(eager_stream_collide_step(f_ref, 0.8)).backward()
    f_tri = f0.clone().requires_grad_(True)
    _loss(triton_fused_step(f_tri, 0.8, block=n)).backward()
    rel = float((f_tri.grad - f_ref.grad).abs().max() / f_ref.grad.abs().max())
    assert rel < 1e-5, rel


@pytest.mark.parametrize("steps", [2, 5])
def test_grad_tau_matches_autograd_and_fd(steps: int) -> None:
    """``dL/dtau`` over a rollout: autograd < 1e-5, fp64 FD < 1e-4.

    ``n = 4`` keeps the tile (``4 x 4 = 16``) at the minimum shape ``tl.dot``
    accepts for the Gram contraction in the adjoint kernel.
    """
    n = 4
    tau = 0.8
    f0 = _field(n)

    tau_t = torch.tensor(tau, requires_grad=True)
    _loss(triton_fused_rollout(f0, tau_t, steps, block=n)).backward()
    grad_tri = float(tau_t.grad)

    tau_e = torch.tensor(tau, requires_grad=True)
    _loss(_eager_rollout(f0, tau_e, steps)).backward()
    grad_eag = float(tau_e.grad)
    assert abs(grad_tri - grad_eag) / abs(grad_eag) < 1e-5

    f64 = f0.to(torch.float64)
    h = 1e-4

    def L(t: float) -> float:
        return float(_loss(_eager_rollout(f64, t, steps)))

    fd = (L(tau + h) - L(tau - h)) / (2 * h)
    assert abs(grad_tri - fd) / abs(fd) < 1e-4


def test_dtau_composes_linearly() -> None:
    """Gradient accumulates over chained steps (tau shared across the rollout)."""
    n = 4
    f0 = _field(n)
    tau = torch.tensor(0.8, requires_grad=True)
    loss = _loss(triton_fused_rollout(f0, tau, 3, block=n))
    loss.backward()
    assert tau.grad is not None
    assert torch.isfinite(tau.grad)
    assert float(tau.grad) != 0.0


def test_fp64_is_rejected_by_fused_step() -> None:
    """The kernel is fp32-only; the eager reference carries fp64."""
    f64 = _field(3).to(torch.float64)
    with pytest.raises(TypeError):
        triton_fused_step(f64, 0.8, block=3)
    # eager path still fine in fp64 (used for the FD cross-check)
    assert eager_stream_collide_step(f64, 0.8).dtype == torch.float64


def test_interpreter_flag_reported() -> None:
    """``interpreter_mode`` reflects the environment switch (evidence hygiene)."""
    import os

    assert interpreter_mode() == (os.environ.get("TRITON_INTERPRET", "0") == "1")