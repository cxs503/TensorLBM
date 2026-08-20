"""Autograd tests for the differentiable reference (eager) path.

The eager solver (``tensorlbm.solver`` / ``tensorlbm.solver3d``:
gather/roll streaming + tensorised collision) is the *differentiable
reference path* of TensorLBM — the production path (Triton fused kernels)
is not differentiable.  These tests pin down that contract:

1. one collide->stream step admits finite gradients w.r.t. both the
   distribution ``f`` and a tensor relaxation time ``tau`` (BGK family,
   MRT, TRT — 2-D and 3-D, gather and roll streaming);
2. gradients through many steps match finite differences of the same
   discrete objective (float noise documented per-dtype tolerances);
3. SGD on ``tau`` drives a shear-wave inverse problem to the target
   viscosity with monotonically decreasing loss — the "solver in the
   loop" pattern used by ``examples/differentiable_lbm.py``.

See ``docs/differentiable_path.md`` for the full positioning (relation
to ``adjoint.py``'s frozen-field surrogate and to the Triton path).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver import (
    collide_bgk,
    collide_bgk_fused,
    collide_bgk_matmul,
    collide_mrt,
    collide_trt,
    stream,
)
from tensorlbm.solver3d import (
    collide_bgk3d,
    collide_mrt3d,
    stream3d,
    stream3d_roll,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Helpers: decaying shear wave (same physics as benchmarks/verified/shear_wave_decay)
# ---------------------------------------------------------------------------


def shear_wave_f0(
    n: int,
    amplitude: float | torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """D2Q9 equilibrium initialisation of u = a*sin(ky), v = 0 (periodic)."""
    y, _x = torch.meshgrid(
        torch.arange(n, dtype=dtype, device=device),
        torch.arange(n, dtype=dtype, device=device),
        indexing="ij",
    )
    k = TWO_PI / n
    ux0 = amplitude * torch.sin(k * y)
    zeros = torch.zeros_like(ux0)
    return equilibrium(torch.ones_like(ux0), ux0, zeros)


def nonequilibrium_f(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 7,
) -> torch.Tensor:
    """Deterministic off-equilibrium field so tau-gradients are non-trivial.

    An exact equilibrium is a fixed point of BGK/MRT collision, making every
    tau-derivative identically zero; superposing non-equilibrium noise makes
    the relaxation rate observable.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(shape, generator=gen, dtype=torch.float64).to(dtype) - 0.5
    if len(shape) == 3:  # (9, ny, nx)
        rho = torch.ones(1, *shape[1:], dtype=dtype, device=device)
        base = equilibrium(
            rho.squeeze(0),
            torch.zeros(*shape[1:], dtype=dtype, device=device),
            torch.zeros(*shape[1:], dtype=dtype, device=device),
        )
    else:  # (19, nz, ny, nx)
        base = equilibrium3d(
            torch.ones(*shape[1:], dtype=dtype, device=device),
            torch.zeros(*shape[1:], dtype=dtype, device=device),
            torch.zeros(*shape[1:], dtype=dtype, device=device),
            torch.zeros(*shape[1:], dtype=dtype, device=device),
        )
    return base + 0.05 * noise.to(device)


def rollout2d(f0: torch.Tensor, tau: float | torch.Tensor, steps: int) -> torch.Tensor:
    """Plain eager stream->collide loop (no torch.compile, autograd-safe)."""
    f = f0
    for _ in range(steps):
        f = collide_bgk(stream(f), tau)
    return f


# ---------------------------------------------------------------------------
# 1. Single-step gradients: f and tensor tau (2-D operators)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collide",
    [collide_bgk, collide_bgk_fused, collide_bgk_matmul, collide_mrt, collide_trt],
    ids=["bgk", "bgk_fused", "bgk_matmul", "mrt", "trt"],
)
def test_collide_step_gradients_f_and_tau(collide) -> None:
    """dLoss/df and dLoss/dtau exist, are finite and non-zero for every 2-D collision operator."""
    f = nonequilibrium_f((9, 20, 24), torch.float64, DEVICE).requires_grad_(True)
    tau = torch.tensor(0.8, dtype=torch.float64, device=DEVICE, requires_grad=True)

    out = collide(stream(f), tau)
    loss = out.pow(2).mean()
    g_f, g_tau = torch.autograd.grad(loss, [f, tau], allow_unused=True)

    assert g_f is not None and g_tau is not None
    assert torch.isfinite(g_f).all() and torch.isfinite(g_tau).all()
    assert g_f.abs().max() > 0.0
    assert g_tau.abs() > 0.0


def test_mrt_float_tau_unchanged_by_graph_fix() -> None:
    """The graph-preserving s_vec construction is value-identical for float tau."""
    f = nonequilibrium_f((9, 20, 24), torch.float64, DEVICE)
    tau_scalar, tau_tensor = 0.8, torch.tensor(0.8, dtype=torch.float64, device=DEVICE)
    assert torch.equal(collide_mrt(f, tau_scalar), collide_mrt(f, tau_tensor))


# ---------------------------------------------------------------------------
# 2. 3-D chains: gather and roll streaming both carry gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream3d_fn", [stream3d, stream3d_roll], ids=["gather", "roll"])
def test_3d_bgk_rollout_gradients(stream3d_fn) -> None:
    """5-step 3-D BGK chain: dLoss/df0 and dLoss/dtau finite, dLoss/dtau non-zero."""
    f0 = nonequilibrium_f((19, 10, 10, 12), torch.float32, DEVICE).requires_grad_(True)
    tau = torch.tensor(0.9, dtype=torch.float32, device=DEVICE, requires_grad=True)

    f = f0
    for _ in range(5):
        f = collide_bgk3d(stream3d_fn(f), tau)
    _rho, ux, _uy, _uz = macroscopic3d(f)
    loss = ux.pow(2).mean()
    g_f0, g_tau = torch.autograd.grad(loss, [f0, tau], allow_unused=True)

    assert g_f0 is not None and g_tau is not None
    assert torch.isfinite(g_f0).all() and torch.isfinite(g_tau).all()
    assert g_tau.abs() > 0.0


def test_3d_mrt_tau_gradient_flows() -> None:
    """Regression: MRT relaxation vector used to silently detach tensor tau.

    ``collide_mrt3d`` built ``s_vec`` via ``torch.tensor([...])`` which
    converts a graph-connected 1/tau to a plain scalar.  The graph-preserving
    construction must keep dLoss/dtau alive through a multi-step chain.
    """
    f0 = nonequilibrium_f((19, 10, 10, 12), torch.float32, DEVICE).requires_grad_(True)
    tau = torch.tensor(0.9, dtype=torch.float32, device=DEVICE, requires_grad=True)

    f = f0
    for _ in range(5):
        f = collide_mrt3d(stream3d(f), tau)
    loss = f.pow(2).mean()
    g_f0, g_tau = torch.autograd.grad(loss, [f0, tau], allow_unused=True)

    assert g_f0 is not None and g_tau is not None
    assert torch.isfinite(g_f0).all() and torch.isfinite(g_tau).all()
    assert g_tau.abs() > 0.0


# ---------------------------------------------------------------------------
# 3. Finite-difference cross-checks (physics target: analytic shear-wave decay)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "eps", "rtol"),
    [
        (torch.float64, 1e-4, 1e-4),
        # float32 forward accumulates rounding noise over the rollout; the
        # finite-difference quotient therefore only agrees to ~1%.
        (torch.float32, 1e-2, 2e-2),
    ],
    ids=["float64", "float32"],
)
def test_tau_gradient_matches_finite_difference(dtype, eps, rtol) -> None:
    """dLoss/dtau (autograd) == central finite difference of the same discrete loss."""
    n, steps, tau0, u0 = 32, 12, 0.8, 0.05
    k = TWO_PI / n
    nu = (tau0 - 0.5) / 3.0
    # analytic decaying shear wave (BGK bias O(k^4) is irrelevant: FD uses
    # the same discrete objective)
    y, _x = torch.meshgrid(
        torch.arange(n, dtype=dtype, device=DEVICE),
        torch.arange(n, dtype=dtype, device=DEVICE),
        indexing="ij",
    )
    ux_target = u0 * torch.sin(k * y) * math.exp(-nu * k * k * steps)

    def loss_at(tau_val: float) -> float:
        tau = torch.tensor(tau_val, dtype=dtype, device=DEVICE)
        f = shear_wave_f0(n, u0, dtype, DEVICE)
        for _ in range(steps):
            f = collide_bgk(stream(f), tau)
        _rho, ux, _uy = macroscopic(f)
        return float(((ux - ux_target) ** 2).mean())

    tau = torch.tensor(tau0, dtype=dtype, device=DEVICE, requires_grad=True)
    f = shear_wave_f0(n, u0, dtype, DEVICE)
    for _ in range(steps):
        f = collide_bgk(stream(f), tau)
    _rho, ux, _uy = macroscopic(f)
    loss = ((ux - ux_target) ** 2).mean()
    (g_ad,) = torch.autograd.grad(loss, tau)

    fd = (loss_at(tau0 + eps) - loss_at(tau0 - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < rtol


def test_initial_amplitude_gradient_matches_fd() -> None:
    """Gradient w.r.t. a scalar initial-condition parameter (perturbation amplitude)."""
    n, steps = 32, 10
    with torch.no_grad():
        target = macroscopic(
            rollout2d(shear_wave_f0(n, 0.05, torch.float64, DEVICE), 0.9, steps)
        )[1]

    def loss_at(a_val: float) -> float:
        a = torch.tensor(a_val, dtype=torch.float64, device=DEVICE)
        ux = macroscopic(rollout2d(shear_wave_f0(n, a, torch.float64, DEVICE), 0.9, steps))[1]
        return float(((ux - target) ** 2).mean())

    a = torch.tensor(0.03, dtype=torch.float64, device=DEVICE, requires_grad=True)
    ux = macroscopic(rollout2d(shear_wave_f0(n, a, torch.float64, DEVICE), 0.9, steps))[1]
    loss = ((ux - target) ** 2).mean()
    (g_ad,) = torch.autograd.grad(loss, a)

    eps = 1e-5
    fd = (loss_at(0.03 + eps) - loss_at(0.03 - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6


def test_initial_field_gradient_matches_fd_entries() -> None:
    """Element-wise dLoss/df0 agrees with per-entry central differences."""
    n, steps = 24, 8
    # Target from a *different* amplitude so the evaluated f0 does not sit
    # exactly at the loss minimum (where every gradient is identically 0).
    with torch.no_grad():
        target = macroscopic(
            rollout2d(shear_wave_f0(n, 0.06, torch.float64, DEVICE), 0.8, steps)
        )[1]

    f0 = shear_wave_f0(n, 0.05, torch.float64, DEVICE).requires_grad_(True)
    ux = macroscopic(rollout2d(f0, 0.8, steps))[1]
    loss = ((ux - target) ** 2).mean()
    (g_ad,) = torch.autograd.grad(loss, f0)

    eps = 1e-6
    for q, i, j in [(0, 5, 7), (3, 12, 3), (7, 20, 15), (8, 2, 21)]:
        f_plus, f_minus = f0.detach().clone(), f0.detach().clone()
        f_plus[q, i, j] += eps
        f_minus[q, i, j] -= eps
        l_plus = float(
            ((macroscopic(rollout2d(f_plus, 0.8, steps))[1] - target) ** 2).mean()
        )
        l_minus = float(
            ((macroscopic(rollout2d(f_minus, 0.8, steps))[1] - target) ** 2).mean()
        )
        fd = (l_plus - l_minus) / (2.0 * eps)
        denom = max(abs(float(g_ad[q, i, j])), abs(fd), 1e-30)
        assert abs(float(g_ad[q, i, j]) - fd) / denom < 1e-6, (q, i, j)


# ---------------------------------------------------------------------------
# 4. End-to-end inverse problem: SGD on tau through the solver
# ---------------------------------------------------------------------------


def test_sgd_tau_recovery_monotone_decrease() -> None:
    """Recover tau*=0.9 from a target velocity field by SGD through 30 solver steps.

    Learning rate 1000 keeps every iteration monotonically decreasing (the
    loss is ~quadratic in tau near the recoverable optimum; tau is clipped
    to the physical range (0.5, 5] exactly like XLB's differentiable-lbm
    example clips its parameters).
    """
    n, steps, tau_star, lr, iters = 32, 30, 0.9, 1000.0, 40
    f0 = shear_wave_f0(n, 0.05, torch.float64, DEVICE)
    with torch.no_grad():
        target = macroscopic(rollout2d(f0, tau_star, steps))[1]

    tau = torch.tensor(0.55, dtype=torch.float64, device=DEVICE, requires_grad=True)
    losses: list[float] = []
    for _ in range(iters):
        ux = macroscopic(rollout2d(f0, tau, steps))[1]
        loss = ((ux - target) ** 2).mean()
        (g,) = torch.autograd.grad(loss, tau)
        with torch.no_grad():
            tau -= lr * g
            tau.clamp_(0.5005, 5.0)
        losses.append(loss.item())

    assert all(losses[i + 1] <= losses[i] + 1e-12 for i in range(len(losses) - 1)), losses
    assert losses[-1] < 1e-3 * losses[0]
    assert abs(tau.detach().item() - tau_star) < 1e-4


# ---------------------------------------------------------------------------
# 5. Gradient checkpointing: same gradients, bounded activation memory
# ---------------------------------------------------------------------------


def test_checkpointed_steps_match_plain_autograd() -> None:
    """Per-step torch.utils.checkpoint reproduces plain-autograd gradients.

    This is the multi-step strategy quantified in
    ``examples/differentiable_lbm.py`` (activation memory vs steps).
    """
    n, steps = 24, 10
    tau0 = 0.8
    with torch.no_grad():
        target = macroscopic(
            rollout2d(shear_wave_f0(n, 0.05, torch.float64, DEVICE), 0.9, steps)
        )[1]

    def loss_and_grad(use_checkpoint: bool):
        f0 = shear_wave_f0(n, 0.05, torch.float64, DEVICE).requires_grad_(True)
        tau = torch.tensor(tau0, dtype=torch.float64, device=DEVICE, requires_grad=True)

        def one_step(f, t):
            return collide_bgk(stream(f), t)

        f = f0
        for _ in range(steps):
            if use_checkpoint:
                f = torch.utils.checkpoint.checkpoint(
                    one_step, f, tau, use_reentrant=False
                )
            else:
                f = one_step(f, tau)
        ux = macroscopic(f)[1]
        loss = ((ux - target) ** 2).mean()
        g_f0, g_tau = torch.autograd.grad(loss, [f0, tau])
        return loss.detach(), g_f0, g_tau

    loss_plain, g_f0_plain, g_tau_plain = loss_and_grad(False)
    loss_ckpt, g_f0_ckpt, g_tau_ckpt = loss_and_grad(True)

    assert torch.allclose(loss_plain, loss_ckpt, rtol=1e-12)
    assert torch.allclose(g_f0_plain, g_f0_ckpt, rtol=1e-10, atol=1e-14)
    assert torch.allclose(g_tau_plain, g_tau_ckpt, rtol=1e-10)
