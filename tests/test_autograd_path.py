"""Tests for the packaged differentiable step chain (``tensorlbm.autograd_path``).

``tests/test_autograd.py`` established that the individual eager operators
admit gradients; this file pins down the *composition contract* of
``docs/differentiable_path.md``'s solver-in-the-loop path:

1. ``differentiable_step`` reproduces the manual
   collide-fluid -> stream -> bounce-back composition exactly;
2. gradients through masked rollouts (obstacle present) match central
   finite differences of the same discrete objective, per dtype;
3. the momentum-exchange force probe is differentiable (FD cross-check);
4. solver-in-the-loop identification converges: a scalar ``tau`` (BGK) and a
   Smagorinsky ``C_s`` are recovered from a target observable by gradient
   descent through the solver;
5. per-step checkpointing returns identical gradients.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch

from tensorlbm.autograd_path import differentiable_step, obstacle_force, rollout
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_bgk3d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TWO_PI = 2.0 * math.pi

# Small 3-D case: periodic box, centred sphere obstacle, decaying shear flow.
NZ, NY, NX = 10, 12, 16
RADIUS = 2.5


def make_mask(device) -> torch.Tensor:
    return sphere_mask(NX, NY, NZ, cx=NX / 2, cy=NY / 2, cz=NZ / 2, radius=RADIUS, device=device)


def shear_flow_f0(
    amplitude: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """D3Q19 equilibrium initialisation u = a*(sin(2*pi*y/ny) + 0.3*cos(2*pi*z/nz)) x-hat."""
    zz, yy, _xx = torch.meshgrid(
        torch.arange(NZ, dtype=dtype, device=device),
        torch.arange(NY, dtype=dtype, device=device),
        torch.arange(NX, dtype=dtype, device=device),
        indexing="ij",
    )
    ux = amplitude * (torch.sin(TWO_PI * yy / NY) + 0.3 * torch.cos(TWO_PI * zz / NZ))
    zeros = torch.zeros_like(ux)
    return equilibrium3d(torch.ones_like(ux), ux, zeros, zeros)


def nonequilibrium_f(
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 11,
) -> torch.Tensor:
    """Equilibrium shear flow plus deterministic off-equilibrium noise (tau signal at step 1)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((19, NZ, NY, NX), generator=gen, dtype=torch.float64).to(dtype) - 0.5
    return shear_flow_f0(0.05, dtype, device) + 0.05 * noise.to(device)


def ux_fluid(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Streamwise velocity on the fluid cells (solid cells hold reflected populations)."""
    _rho, ux, _uy, _uz = macroscopic3d(f)
    return ux[~mask]


def field_loss(f: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((ux_fluid(f, mask) - target) ** 2).mean()


# ---------------------------------------------------------------------------
# 1. Value contract: the packaged step == the manual composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_mask", [True, False], ids=["obstacle", "periodic"])
def test_step_equals_manual_composition(with_mask) -> None:
    """differentiable_step == where-skip-solid collision -> stream3d -> where bounce-back."""
    dtype, device = torch.float64, DEVICE
    f = nonequilibrium_f(dtype, device)
    tau = 0.8
    mask = make_mask(device) if with_mask else None

    out = differentiable_step(f, tau, mask)

    f_col = collide_bgk3d(f, tau)
    if with_mask:
        f_col = torch.where(mask.unsqueeze(0), f, f_col)
    f_str = stream3d(f_col)
    manual = (
        f_str
        if mask is None
        else torch.where(mask.unsqueeze(0), f_str[OPPOSITE.to(f_str.device)], f_str)
    )
    assert torch.equal(out, manual)
    # the periodic chain keeps the audited collide->stream order of tests/test_autograd.py
    if not with_mask:
        assert torch.equal(out, stream3d(collide_bgk3d(f, tau)))


def test_solid_cells_skip_collision_and_bounce_back() -> None:
    """Post-step solid content is exactly the reflection of what streamed in."""
    dtype, device = torch.float64, DEVICE
    f = nonequilibrium_f(dtype, device)
    mask = make_mask(device)
    out, probe = differentiable_step(f, 0.8, mask, return_probe=True)
    opp = OPPOSITE.to(device)
    assert torch.equal(out[:, mask], probe[opp][:, mask])
    assert torch.equal(out[:, ~mask], probe[:, ~mask])


# ---------------------------------------------------------------------------
# 2. Gradient existence through the masked chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
def test_single_step_gradients_f_and_tau(dtype) -> None:
    """dLoss/df and dLoss/dtau exist, finite, non-zero with the obstacle in the chain."""
    f = nonequilibrium_f(dtype, DEVICE).requires_grad_(True)
    tau = torch.tensor(0.8, dtype=dtype, device=DEVICE, requires_grad=True)
    mask = make_mask(DEVICE)

    loss = differentiable_step(f, tau, mask).pow(2).mean()
    g_f, g_tau = torch.autograd.grad(loss, [f, tau])

    assert torch.isfinite(g_f).all() and torch.isfinite(g_tau).all()
    assert g_f.abs().max() > 0.0
    # the obstacle surface is in the graph: solid-cell gradients flow (bounce-back branch)
    assert g_f[:, mask].abs().sum() > 0.0
    assert g_tau.abs() > 0.0


# ---------------------------------------------------------------------------
# 3. Finite-difference cross-checks through masked rollouts
# ---------------------------------------------------------------------------


def _rollout_loss_tau(
    tau_val: float,
    f0: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
) -> float:
    tau = torch.tensor(tau_val, dtype=dtype, device=DEVICE)
    f = rollout(f0, steps, tau, mask)
    return float(field_loss(f, target, mask))


@pytest.mark.parametrize(
    ("dtype", "eps", "rtol"),
    [
        (torch.float64, 1e-5, 1e-6),
        # fp32 rollout rounding limits the FD quotient; measured agreement at
        # eps=5e-3 is ~2.2e-4 (see docs/differentiable_path.md).  eps must stay
        # large enough that the loss difference exceeds the fp32 noise floor.
        (torch.float32, 5e-3, 1e-3),
    ],
    ids=["float64", "float32"],
)
def test_tau_gradient_matches_finite_difference(dtype, eps, rtol) -> None:
    """dLoss/dtau through a 12-step masked rollout == central FD of the same loss."""
    steps, tau_star, tau_eval = 12, 0.85, 0.7
    mask = make_mask(DEVICE)
    f0 = shear_flow_f0(0.08, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(rollout(f0, steps, tau_star, mask), mask)

    tau = torch.tensor(tau_eval, dtype=dtype, device=DEVICE, requires_grad=True)
    f = rollout(f0, steps, tau, mask)
    loss = field_loss(f, target, mask)
    (g_ad,) = torch.autograd.grad(loss, tau)

    fd = (
        _rollout_loss_tau(tau_eval + eps, f0, target, mask, steps, dtype)
        - _rollout_loss_tau(tau_eval - eps, f0, target, mask, steps, dtype)
    ) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < rtol


def test_f0_entry_gradients_match_fd() -> None:
    """Element-wise dLoss/df0 vs per-entry central differences, incl. obstacle-adjacent cells."""
    steps, tau0 = 8, 0.8
    dtype = torch.float64
    mask = make_mask(DEVICE)
    f0 = shear_flow_f0(0.05, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(rollout(f0, steps, 0.95, mask), mask)  # different tau: off-optimum

    f0 = f0.requires_grad_(True)
    loss = field_loss(rollout(f0, steps, tau0, mask), target, mask)
    (g_ad,) = torch.autograd.grad(loss, f0)

    # (q, iz, iy, ix): far fluid cell, obstacle-adjacent fluid cell, inside the sphere
    cz, cy, cx = NZ / 2, NY / 2, NX / 2
    entries = [
        (0, 2, 3, 4),
        (5, round(cz) + 3, round(cy), round(cx)),
        (2, round(cz), round(cy), round(cx)),
    ]
    eps = 1e-6
    for q, iz, iy, ix in entries:
        f_plus, f_minus = f0.detach().clone(), f0.detach().clone()
        f_plus[q, iz, iy, ix] += eps
        f_minus[q, iz, iy, ix] -= eps
        l_plus = float(field_loss(rollout(f_plus, steps, tau0, mask), target, mask))
        l_minus = float(field_loss(rollout(f_minus, steps, tau0, mask), target, mask))
        fd = (l_plus - l_minus) / (2.0 * eps)
        denom = max(abs(float(g_ad[q, iz, iy, ix])), abs(fd), 1e-30)
        assert abs(float(g_ad[q, iz, iy, ix]) - fd) / denom < 1e-6, (q, iz, iy, ix)


def test_obstacle_force_probe_gradient_matches_fd() -> None:
    """Drag observable: d(sum_k F_x)/dtau through the solver == central FD."""
    steps, tau_eval = 8, 0.8
    dtype, eps = torch.float64, 1e-5
    mask = make_mask(DEVICE)
    f0 = shear_flow_f0(0.08, dtype, DEVICE)

    def drag_sum(tau_val: float) -> float:
        tau = torch.tensor(tau_val, dtype=dtype, device=DEVICE)
        _f, probes = rollout(f0, steps, tau, mask, return_probes=True)
        return float(sum(obstacle_force(p, mask)[0] for p in probes))

    tau = torch.tensor(tau_eval, dtype=dtype, device=DEVICE, requires_grad=True)
    _f, probes = rollout(f0, steps, tau, mask, return_probes=True)
    forces = [obstacle_force(p, mask) for p in probes]
    loss = sum(f_vec[0] for f_vec in forces)
    (g_ad,) = torch.autograd.grad(loss, tau)

    assert all(torch.isfinite(f_vec).all() for f_vec in forces)
    assert forces[-1][0].abs() > 0.0  # the obstacle absorbs x-momentum
    fd = (drag_sum(tau_eval + eps) - drag_sum(tau_eval - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6


# ---------------------------------------------------------------------------
# 4. Solver-in-the-loop identification (the XLB-paradigm demo)
# ---------------------------------------------------------------------------


def test_tau_recovery_with_obstacle() -> None:
    """Recover tau* = 0.85 from the final velocity field by Adam through 15 solver steps."""
    steps, tau_star = 15, 0.85
    dtype = torch.float64
    mask = make_mask(DEVICE)
    f0 = shear_flow_f0(0.08, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(rollout(f0, steps, tau_star, mask), mask)

    tau = torch.tensor(0.6, dtype=dtype, device=DEVICE, requires_grad=True)
    optim = torch.optim.Adam([tau], lr=0.02)
    loss0 = None
    for it in range(120):
        loss = field_loss(rollout(f0, steps, tau, mask), target, mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            tau.clamp_(0.5005, 5.0)
        if loss0 is None:
            loss0 = loss.detach().item()
    assert math.isfinite(loss.detach().item())
    assert loss.detach().item() < 1e-2 * loss0
    assert abs(tau.detach().item() - tau_star) < 5e-3


def test_cs_recovery_smagorinsky() -> None:
    """Identify the Smagorinsky constant C_s* = 0.12 through the solver (BGK-Smagorinsky slot)."""
    steps, tau0, cs_star = 15, 0.55, 0.12
    dtype = torch.float64
    mask = make_mask(DEVICE)
    f0 = shear_flow_f0(0.1, dtype, DEVICE)

    with torch.no_grad():
        target = ux_fluid(
            rollout(
                f0,
                steps,
                tau0,
                mask,
                collide=functools.partial(collide_smagorinsky_bgk3d, C_s=torch.tensor(cs_star)),
            ),
            mask,
        )

    cs = torch.tensor(0.03, dtype=dtype, device=DEVICE, requires_grad=True)
    optim = torch.optim.Adam([cs], lr=0.01)
    loss0 = None
    for _ in range(80):
        loss = field_loss(
            rollout(
                f0, steps, tau0, mask, collide=functools.partial(collide_smagorinsky_bgk3d, C_s=cs)
            ),
            target,
            mask,
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            cs.clamp_(0.0, 0.5)
        if loss0 is None:
            loss0 = loss.detach().item()
    assert math.isfinite(loss.detach().item())
    assert loss.detach().item() < 0.2 * loss0
    assert abs(cs.detach().item() - cs_star) < 0.03


# ---------------------------------------------------------------------------
# 5. Checkpointed rollout: identical gradients, per-step rematerialisation
# ---------------------------------------------------------------------------


def test_rollout_checkpoint_gradients_equal() -> None:
    """checkpoint=True reproduces plain-autograd gradients (fields, tau and drag probes)."""
    steps = 10
    dtype = torch.float64
    mask = make_mask(DEVICE)
    with torch.no_grad():
        target = ux_fluid(rollout(shear_flow_f0(0.08, dtype, DEVICE), steps, 0.9, mask), mask)

    def loss_and_grads(use_checkpoint: bool):
        f0 = shear_flow_f0(0.08, dtype, DEVICE).requires_grad_(True)
        tau = torch.tensor(0.7, dtype=dtype, device=DEVICE, requires_grad=True)
        f, probes = rollout(
            f0,
            steps,
            tau,
            mask,
            checkpoint=use_checkpoint,
            return_probes=True,
        )
        loss = field_loss(f, target, mask) + sum(obstacle_force(p, mask)[0] for p in probes)
        g_f0, g_tau = torch.autograd.grad(loss, [f0, tau])
        return loss.detach(), g_f0, g_tau

    loss_plain, g_f0_plain, g_tau_plain = loss_and_grads(False)
    loss_ckpt, g_f0_ckpt, g_tau_ckpt = loss_and_grads(True)

    assert torch.allclose(loss_plain, loss_ckpt, rtol=1e-12)
    assert torch.allclose(g_f0_plain, g_f0_ckpt, rtol=1e-10, atol=1e-14)
    assert torch.allclose(g_tau_plain, g_tau_ckpt, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. CUDA parity (only when a device is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_parity_masked_rollout() -> None:
    """Same fp32 masked rollout and tau-gradient on CPU and CUDA."""
    dtype = torch.float32
    steps = 10
    mask_cuda = make_mask(torch.device("cuda"))
    mask_cpu = make_mask(torch.device("cpu"))

    f0_cpu = shear_flow_f0(0.08, dtype, torch.device("cpu"))
    tau_cpu = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    target = ux_fluid(rollout(f0_cpu, steps, tau_cpu, mask_cpu), mask_cpu).detach()
    loss_cpu = field_loss(rollout(f0_cpu, steps, tau_cpu, mask_cpu), target, mask_cpu)
    (g_cpu,) = torch.autograd.grad(loss_cpu, tau_cpu)

    f0_cuda = f0_cpu.to("cuda")
    tau_cuda = torch.tensor(0.7, dtype=dtype, device="cuda", requires_grad=True)
    target_cuda = target.to("cuda")
    loss_cuda = field_loss(rollout(f0_cuda, steps, tau_cuda, mask_cuda), target_cuda, mask_cuda)
    (g_cuda,) = torch.autograd.grad(loss_cuda, tau_cuda)

    assert torch.allclose(loss_cpu, loss_cuda.cpu(), rtol=1e-5, atol=1e-7)
    assert torch.allclose(g_cpu, g_cuda.cpu(), rtol=1e-3, atol=1e-8)
