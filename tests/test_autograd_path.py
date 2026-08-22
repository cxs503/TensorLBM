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
5. per-step checkpointing returns identical gradients;
6. the bounded-domain extension (velocity inlet + zero-gradient outlet)
   applies the boundary planes exactly where claimed, keeps gradients
   flowing *through the boundary overwrites* back to ``tau`` / ``C_s`` /
   ``f0`` / ``u_in`` (FD cross-checks), is checkpoint- and CUDA-consistent,
   and drives a physical wake with a stabilised drag;
7. the A6++ full bounded box (convective outlet + free-slip / free-stream
   lateral walls) keeps the default path bit-for-bit unchanged, applies the
   lateral closures exactly where claimed, passes the FD cross-checks
   through *all six* faces — including the learnable Courant number and the
   convective outlet's own time recursion — and beats the periodic-sides
   baseline quantitatively on wall-normal pollution;
8. the A6+++ per-face lateral control (``WallSpec.overrides`` with face keys
   "-y"/"+y"/"-z"/"+z") reproduces the shared-spec operator bit-for-bit when
   every face resolves to the same closure, applies each face's own method
   exactly where claimed (mixed free-slip / free-stream / periodic, on an
   asymmetric field, edge last-write-wins), keeps the freestream override
   velocity in the autograd graph (FD cross-check) and round-trips through
   ``to_dict`` / ``from_dict`` while loading pre-A6+++ payloads unchanged.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch

from tensorlbm.autograd_path import (
    InletSpec,
    OutletSpec,
    WallSpec,
    differentiable_step,
    obstacle_force,
    rollout,
)
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
        diff = abs(float(g_ad[q, iz, iy, ix]) - fd)
        denom = max(abs(float(g_ad[q, iz, iy, ix])), abs(fd), 1e-30)
        # the corner-line entry carries a ~1e-8 gradient — below the fp64 FD
        # noise floor of the loss difference quotient — so accept an absolute
        # agreement of 1e-12 there (measured: |ad - fd| ~ 1e-14)
        assert diff / denom < 1e-6 or diff < 1e-12, (q, iz, iy, ix)


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


# ---------------------------------------------------------------------------
# 7. Bounded domain: differentiable inlet / zero-gradient outlet
# ---------------------------------------------------------------------------

# Small bounded case: sphere obstacle downstream of the inlet plane, uniform
# inflow (the SUBOFF production phase order: x=0 velocity inlet, x=nx-1
# zero-gradient outlet, lateral planes periodic in this first version).
BNZ, BNY, BNX = 8, 10, 20
BCX, BRADIUS = 6.0, 2.0
U_IN = 0.08
UNK_OUT = [2, 8, 10, 12, 14]  # c_x = -1: overwritten by the outlet copy
UNK_IN = [1, 7, 9, 11, 13]  # c_x = +1: reconstructed by the inlet closure


def make_bounded_mask(device) -> torch.Tensor:
    return sphere_mask(BNX, BNY, BNZ, cx=BCX, cy=BNY / 2, cz=BNZ / 2, radius=BRADIUS, device=device)


def uniform_flow_f0(
    u: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 23,
) -> torch.Tensor:
    """Uniform equilibrium inflow plus deterministic off-equilibrium noise."""
    ones_f = torch.ones(BNZ, BNY, BNX, dtype=dtype, device=device)
    zeros_f = torch.zeros_like(ones_f)
    f0 = equilibrium3d(ones_f, u * ones_f, zeros_f, zeros_f)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((19, BNZ, BNY, BNX), generator=gen, dtype=torch.float64).to(dtype) - 0.5
    return f0 + 0.02 * noise.to(device)


@pytest.mark.parametrize("method", ["equilibrium", "zouhe"])
def test_bounded_step_value_contract(method) -> None:
    """Planes after the step: inlet closure, zero-gradient outlet, interior streamed."""
    dtype, device = torch.float64, DEVICE
    mask = make_bounded_mask(device)
    f = uniform_flow_f0(U_IN, dtype, device)
    inlet = InletSpec(ux=U_IN, method=method)
    out = differentiable_step(f, 0.55, mask, inlet=inlet, outlet=OutletSpec())

    f_str = stream3d(torch.where(mask.unsqueeze(0), f, collide_bgk3d(f, 0.55)))
    # fluid interior planes (1 .. nx-2) are untouched by the boundary pass
    assert torch.equal(
        out[:, :, :, 1:-1][:, ~mask[:, :, 1:-1]], f_str[:, :, :, 1:-1][:, ~mask[:, :, 1:-1]]
    )
    # outlet: unknown outgoing directions copied from x=nx-2, known ones streamed
    assert torch.equal(out[UNK_OUT, ..., -1], f_str[UNK_OUT, ..., -2])
    known = [q for q in range(19) if q not in UNK_OUT]
    assert torch.equal(out[known, ..., -1], f_str[known, ..., -1])

    ones2 = torch.ones(BNZ, BNY, dtype=dtype, device=device)
    zeros2 = torch.zeros_like(ones2)
    rho_p, ux_p, _uy_p, _uz_p = macroscopic3d(out[..., :1])
    if method == "equilibrium":
        exp = equilibrium3d(ones2, U_IN * ones2, zeros2, zeros2)[:, 0].unsqueeze(-1)
        assert torch.equal(out[..., :1], exp)
    else:
        rho_zh = (
            f_str[(0, 3, 4, 5, 6, 15, 16, 17, 18), ..., 0].sum(dim=0)
            + 2.0 * f_str[UNK_OUT, ..., 0].sum(dim=0)
        ) / (1.0 - U_IN)
        # the Zou/He closure pins the normal velocity and the plane density
        assert torch.allclose(ux_p[..., 0], torch.full_like(ux_p[..., 0], U_IN), atol=1e-12)
        assert torch.allclose(rho_p[..., 0], rho_zh, atol=1e-12)
        feq = equilibrium3d(rho_zh, U_IN * ones2, zeros2, zeros2)
        for q, q_opp in zip(UNK_IN, UNK_OUT, strict=True):
            cand = feq[q] + f_str[q_opp, ..., 0] - feq[q_opp]
            assert torch.allclose(out[q, ..., 0], cand, atol=1e-14)


@pytest.mark.parametrize("method", ["equilibrium", "zouhe"])
def test_bounded_tau_gradient_matches_finite_difference(method) -> None:
    """dLoss/dtau through a bounded rollout == central FD (gradient crosses the BCs)."""
    steps, tau_eval, eps = 10, 0.7, 1e-5
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet = InletSpec(ux=U_IN, method=method)
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(rollout(f0, steps, 0.85, mask, inlet=inlet, outlet=OutletSpec()), mask)

    def loss_of(tau_val: float) -> float:
        tau = torch.tensor(tau_val, dtype=dtype, device=DEVICE)
        return float(
            field_loss(
                rollout(f0, steps, tau, mask, inlet=inlet, outlet=OutletSpec()), target, mask
            )
        )

    tau = torch.tensor(tau_eval, dtype=dtype, device=DEVICE, requires_grad=True)
    loss = field_loss(rollout(f0, steps, tau, mask, inlet=inlet, outlet=OutletSpec()), target, mask)
    (g_ad,) = torch.autograd.grad(loss, tau)

    fd = (loss_of(tau_eval + eps) - loss_of(tau_eval - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


def test_bounded_cs_gradient_matches_finite_difference() -> None:
    """d(accumulated drag)/dC_s through the bounded solver == central FD."""
    steps, tau0, cs_eval, eps = 8, 0.55, 0.1, 1e-5
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet = InletSpec(ux=U_IN)
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)

    def drag_sum(cs_val: float) -> float:
        cs = torch.tensor(cs_val, dtype=dtype, device=DEVICE)
        _f, probes = rollout(
            f0,
            steps,
            tau0,
            mask,
            collide=functools.partial(collide_smagorinsky_bgk3d, C_s=cs),
            inlet=inlet,
            outlet=OutletSpec(),
            return_probes=True,
        )
        return float(sum(obstacle_force(p, mask)[0] for p in probes))

    cs = torch.tensor(cs_eval, dtype=dtype, device=DEVICE, requires_grad=True)
    _f, probes = rollout(
        f0,
        steps,
        tau0,
        mask,
        collide=functools.partial(collide_smagorinsky_bgk3d, C_s=cs),
        inlet=inlet,
        outlet=OutletSpec(),
        return_probes=True,
    )
    loss = sum(obstacle_force(p, mask)[0] for p in probes)
    (g_ad,) = torch.autograd.grad(loss, cs)

    fd = (drag_sum(cs_eval + eps) - drag_sum(cs_eval - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


@pytest.mark.parametrize("method", ["equilibrium", "zouhe"])
def test_bounded_f0_boundary_entry_gradients_match_fd(method) -> None:
    """Element-wise dLoss/df0 on and next to the boundary planes vs central FD.

    The entries on the inlet/outlet planes are overwritten after streaming,
    so their gradient must flow through the collision phase that precedes
    the overwrite (and, for Zou/He, through the closure itself).
    """
    steps, tau0 = 8, 0.7
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet = InletSpec(ux=U_IN, method=method)
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(rollout(f0, steps, 0.9, mask, inlet=inlet, outlet=OutletSpec()), mask)

    f0 = f0.requires_grad_(True)
    loss = field_loss(
        rollout(f0, steps, tau0, mask, inlet=inlet, outlet=OutletSpec()), target, mask
    )
    (g_ad,) = torch.autograd.grad(loss, f0)

    cz, cy = round(BNZ / 2), round(BNY / 2)
    entries = [
        (0, 2, 3, 4),  # far interior
        (1, cz, cy, 0),  # incoming direction, inlet plane (overwritten by the closure)
        (2, cz, cy, 0),  # outgoing direction, inlet plane
        (2, cz, cy, BNX - 1),  # unknown direction, outlet plane (overwritten by the copy)
        (7, cz, cy, BNX - 2),  # source plane of the zero-gradient copy
    ]
    eps = 1e-6
    for q, iz, iy, ix in entries:
        f_plus, f_minus = f0.detach().clone(), f0.detach().clone()
        f_plus[q, iz, iy, ix] += eps
        f_minus[q, iz, iy, ix] -= eps
        l_plus = float(
            field_loss(
                rollout(f_plus, steps, tau0, mask, inlet=inlet, outlet=OutletSpec()), target, mask
            )
        )
        l_minus = float(
            field_loss(
                rollout(f_minus, steps, tau0, mask, inlet=inlet, outlet=OutletSpec()), target, mask
            )
        )
        fd = (l_plus - l_minus) / (2.0 * eps)
        diff = abs(float(g_ad[q, iz, iy, ix]) - fd)
        denom = max(abs(float(g_ad[q, iz, iy, ix])), abs(fd), 1e-30)
        # the corner-line entry carries a ~1e-8 gradient — below the fp64 FD
        # noise floor of the loss difference quotient — so accept an absolute
        # agreement of 1e-12 there (measured: |ad - fd| ~ 1e-14)
        assert diff / denom < 1e-6 or diff < 1e-12, (q, iz, iy, ix)


@pytest.mark.parametrize("method", ["equilibrium", "zouhe"])
def test_bounded_uin_gradient_matches_finite_difference(method) -> None:
    """The learnable inlet velocity u_in: autograd gradient vs central FD."""
    steps, tau0, eps = 10, 0.7, 1e-5
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(f0, steps, 0.85, mask, inlet=InletSpec(ux=U_IN), outlet=OutletSpec()), mask
        )

    def loss_of(u_val: float) -> float:
        return float(
            field_loss(
                rollout(
                    f0,
                    steps,
                    tau0,
                    mask,
                    inlet=InletSpec(ux=u_val, method=method),
                    outlet=OutletSpec(),
                ),
                target,
                mask,
            )
        )

    u_in = torch.tensor(U_IN, dtype=dtype, device=DEVICE, requires_grad=True)
    loss = field_loss(
        rollout(
            f0, steps, tau0, mask, inlet=InletSpec(ux=u_in, method=method), outlet=OutletSpec()
        ),
        target,
        mask,
    )
    (g_ad,) = torch.autograd.grad(loss, u_in)

    fd = (loss_of(U_IN + eps) - loss_of(U_IN - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


def test_bounded_rollout_checkpoint_gradients_equal() -> None:
    """checkpoint=True reproduces plain gradients with both boundary conditions active."""
    steps = 8
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(
                uniform_flow_f0(U_IN, dtype, DEVICE),
                steps,
                0.9,
                mask,
                inlet=InletSpec(ux=U_IN),
                outlet=OutletSpec(),
            ),
            mask,
        )

    def loss_and_grads(use_checkpoint: bool):
        f0 = uniform_flow_f0(U_IN, dtype, DEVICE).requires_grad_(True)
        tau = torch.tensor(0.7, dtype=dtype, device=DEVICE, requires_grad=True)
        u_in = torch.tensor(U_IN, dtype=dtype, device=DEVICE, requires_grad=True)
        inlet = InletSpec(ux=u_in, method="zouhe")
        f, probes = rollout(
            f0,
            steps,
            tau,
            mask,
            checkpoint=use_checkpoint,
            inlet=inlet,
            outlet=OutletSpec(),
            return_probes=True,
        )
        loss = field_loss(f, target, mask) + sum(obstacle_force(p, mask)[0] for p in probes)
        g_f0, g_tau, g_u = torch.autograd.grad(loss, [f0, tau, u_in])
        return loss.detach(), g_f0, g_tau, g_u

    loss_plain, g_f0_p, g_tau_p, g_u_p = loss_and_grads(False)
    loss_ckpt, g_f0_c, g_tau_c, g_u_c = loss_and_grads(True)

    assert torch.allclose(loss_plain, loss_ckpt, rtol=1e-12)
    assert torch.allclose(g_f0_p, g_f0_c, rtol=1e-10, atol=1e-14)
    assert torch.allclose(g_tau_p, g_tau_c, rtol=1e-10)
    assert torch.allclose(g_u_p, g_u_c, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_bounded_cuda_parity() -> None:
    """Same fp32 bounded rollout (Zou/He inlet + outlet) and gradients on CPU/CUDA."""
    dtype, steps = torch.float32, 10
    mask_cpu = make_bounded_mask(torch.device("cpu"))
    mask_cuda = make_bounded_mask(torch.device("cuda"))

    def run(f0: torch.Tensor, mask: torch.Tensor, tau: torch.Tensor, u_in: torch.Tensor):
        inlet = InletSpec(ux=u_in, method="zouhe")
        f, probes = rollout(
            f0, steps, tau, mask, inlet=inlet, outlet=OutletSpec(), return_probes=True
        )
        loss = ((macroscopic3d(f)[1][~mask]) ** 2).mean() + sum(
            obstacle_force(p, mask)[0] for p in probes
        )
        g_tau, g_u = torch.autograd.grad(loss, [tau, u_in])
        return loss.detach(), g_tau, g_u

    f0_cpu = uniform_flow_f0(U_IN, dtype, torch.device("cpu"))
    tau_cpu = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    u_cpu = torch.tensor(U_IN, dtype=dtype, requires_grad=True)
    loss_c, g_tau_c, g_u_c = run(f0_cpu, mask_cpu, tau_cpu, u_cpu)

    tau_cuda = torch.tensor(0.7, dtype=dtype, device="cuda", requires_grad=True)
    u_cuda = torch.tensor(U_IN, dtype=dtype, device="cuda", requires_grad=True)
    loss_g, g_tau_g, g_u_g = run(f0_cpu.to("cuda"), mask_cuda, tau_cuda, u_cuda)

    assert torch.allclose(loss_c, loss_g.cpu(), rtol=1e-5, atol=1e-7)
    assert torch.allclose(g_tau_c, g_tau_g.cpu(), rtol=1e-3, atol=1e-8)
    assert torch.allclose(g_u_c, g_u_g.cpu(), rtol=1e-3, atol=1e-8)


def test_bounded_wake_physics() -> None:
    """Inlet-driven bounded campaign: sustained flow, wake deficit, stable drag."""
    steps, tau = 300, 0.55
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    fluid = ~mask
    ones_f = torch.ones(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    zeros_f = torch.zeros_like(ones_f)
    f = equilibrium3d(ones_f, U_IN * ones_f, zeros_f, zeros_f)
    inlet, outlet = InletSpec(ux=U_IN), OutletSpec()

    drags = []
    for _ in range(steps):
        f, probe = differentiable_step(f, tau, mask, return_probe=True, inlet=inlet, outlet=outlet)
        drags.append(obstacle_force(probe, mask)[0].detach())
    rho, ux, _uy, _uz = macroscopic3d(f)

    assert torch.isfinite(f).all()
    assert 0.85 < float(rho.min()) < float(rho.max()) < 1.15
    # the equilibrium inlet pins the inflow velocity exactly
    assert torch.allclose(ux[..., 0], torch.full_like(ux[..., 0], U_IN), atol=1e-10)
    # the drive sustains the flow (an undriven periodic box would decay)
    assert float(ux[fluid].mean()) > 0.9 * U_IN
    # momentum-exchange drag: positive, window-average converged
    w_early = float(torch.stack(drags[50:100]).mean())
    w_late = float(torch.stack(drags[250:300]).mean())
    assert w_late > 0.0
    assert abs(w_late - w_early) < 0.05 * w_late
    # wake: centre-line momentum deficit just behind the sphere, recovering downstream
    # (measured profile: near ~0.13*u_in, recovering to ~0.66*u_in before the outlet)
    cz, cy = round(BNZ / 2), round(BNY / 2)
    back = round(BCX + BRADIUS)
    near_wake = float(ux[cz, cy, back + 1 : back + 3].mean())
    far_wake = float(ux[cz, cy, -4:-1].mean())
    assert near_wake < 0.4 * U_IN
    assert far_wake > 0.6 * U_IN


def test_bounded_tau_recovery_from_drag() -> None:
    """Campaign-style identification: recover tau from measured accumulated drag."""
    steps, tau_star = 12, 0.85
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    ones_f = torch.ones(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    zeros_f = torch.zeros_like(ones_f)
    f0 = equilibrium3d(ones_f, U_IN * ones_f, zeros_f, zeros_f)
    inlet, outlet = InletSpec(ux=U_IN), OutletSpec()

    with torch.no_grad():
        _f, probes = rollout(
            f0, steps, tau_star, mask, inlet=inlet, outlet=outlet, return_probes=True
        )
        drag_true = sum(obstacle_force(p, mask)[0] for p in probes)

    tau = torch.tensor(0.65, dtype=dtype, device=DEVICE, requires_grad=True)
    optim = torch.optim.Adam([tau], lr=0.05)
    loss0 = None
    for it in range(100):
        # cosine learning-rate decay as in examples/solver_in_the_loop.py
        for group in optim.param_groups:
            group["lr"] = 0.05 * 0.5 * (1.0 + math.cos(math.pi * it / 100))
        _f, probes = rollout(f0, steps, tau, mask, inlet=inlet, outlet=outlet, return_probes=True)
        loss = (sum(obstacle_force(p, mask)[0] for p in probes) - drag_true) ** 2
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


# ---------------------------------------------------------------------------
# 8. A6++ bounded box: convective outlet + free-slip / free-stream walls
# ---------------------------------------------------------------------------

# Independent (hand-derived) specular-reflection contracts for the lateral
# planes: FLIP_*[q] is the index of direction q with one transverse velocity
# component negated; the unknown sets are the directions that wrap around the
# domain on each plane after streaming.
FLIP_Y = (0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15)
FLIP_Z = (0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16)
UNK_Y0 = (3, 7, 10, 15, 17)  # c_y = +1: wrapped at y = 0
UNK_Y1 = (4, 8, 9, 16, 18)  # c_y = -1: wrapped at y = ny - 1
UNK_Z0 = (5, 11, 14, 15, 18)  # c_z = +1: wrapped at z = 0
UNK_Z1 = (6, 12, 13, 16, 17)  # c_z = -1: wrapped at z = nz - 1


def test_full_box_default_bitwise_unchanged() -> None:
    """All-new kwargs at their defaults: bit-for-bit the original periodic chain."""
    steps, tau = 5, 0.8
    dtype = torch.float64
    mask = make_mask(DEVICE)
    f0 = nonequilibrium_f(dtype, DEVICE)

    f = f0.clone()
    opp = OPPOSITE.to(DEVICE)
    for _ in range(steps):
        f_col = torch.where(mask.unsqueeze(0), f, collide_bgk3d(f, tau))
        f_str = stream3d(f_col)
        f = torch.where(mask.unsqueeze(0), f_str[opp], f_str)
    base = rollout(f0, steps, tau, mask)

    assert torch.equal(base, f)
    # explicit default-None kwargs and the periodic WallSpec are no-ops
    assert torch.equal(rollout(f0, steps, tau, mask, walls=None, inlet=None, outlet=None), f)
    assert torch.equal(rollout(f0, steps, tau, mask, walls=WallSpec()), f)
    assert torch.equal(rollout(f0, steps, tau, mask, walls=WallSpec(method="periodic")), f)


def test_free_slip_wall_value_contract() -> None:
    """Specular reflection: unknown wall populations take the mirror partner.

    Face interiors (off the corner lines) are exactly the mirror swap; the
    wall-normal velocity cancels pairwise to machine precision; interior
    cells are untouched.
    """
    dtype, device = torch.float64, DEVICE
    f = nonequilibrium_f(dtype, device)
    out = differentiable_step(f, 0.7, None, walls=WallSpec(method="free-slip"))

    f_str = stream3d(collide_bgk3d(f, 0.7))
    # interior untouched in every direction
    assert torch.equal(out[:, 1:-1, 1:-1, :], f_str[:, 1:-1, 1:-1, :])
    # y = 0 / y = ny - 1 faces (columns off the z corner lines)
    for q in range(19):
        expected = f_str[FLIP_Y[q], :, 0, :] if q in UNK_Y0 else f_str[q, :, 0, :]
        assert torch.equal(out[q, 1:-1, 0, :], expected[1:-1, :]), q
        expected = f_str[FLIP_Y[q], :, -1, :] if q in UNK_Y1 else f_str[q, :, -1, :]
        assert torch.equal(out[q, 1:-1, -1, :], expected[1:-1, :]), q
    # z = 0 / z = nz - 1 faces (rows off the y corner lines)
    for q in range(19):
        expected = f_str[FLIP_Z[q], 0, :, :] if q in UNK_Z0 else f_str[q, 0, :, :]
        assert torch.equal(out[q, 0, 1:-1, :], expected[1:-1, :]), q
        expected = f_str[FLIP_Z[q], -1, :, :] if q in UNK_Z1 else f_str[q, -1, :, :]
        assert torch.equal(out[q, -1, 1:-1, :], expected[1:-1, :]), q

    _rho, _ux, uy, uz = macroscopic3d(out)
    assert float(uy[1:-1, 0, :].abs().max()) < 1e-14
    assert float(uy[1:-1, -1, :].abs().max()) < 1e-14
    assert float(uz[0, 1:-1, :].abs().max()) < 1e-14
    assert float(uz[-1, 1:-1, :].abs().max()) < 1e-14


def test_freestream_wall_value_contract() -> None:
    """Freestream walls: whole faces reset to f_eq(rho0, u_inf), interior kept."""
    dtype, device = torch.float64, DEVICE
    f = nonequilibrium_f(dtype, device)
    walls = WallSpec(method="freestream", ux=U_IN)
    out = differentiable_step(f, 0.7, None, walls=walls)

    feq = equilibrium3d(
        torch.tensor(1.0, dtype=dtype, device=device),
        torch.tensor(U_IN, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
        device,
    )
    assert torch.equal(out[:, :, :1, :], feq.expand(19, NZ, 1, NX))
    assert torch.equal(out[:, :, -1:, :], feq.expand(19, NZ, 1, NX))
    assert torch.equal(out[:, :1, :, :], feq.expand(19, 1, NY, NX))
    assert torch.equal(out[:, -1:, :, :], feq.expand(19, 1, NY, NX))
    f_str = stream3d(collide_bgk3d(f, 0.7))
    assert torch.equal(out[:, 1:-1, 1:-1, :], f_str[:, 1:-1, 1:-1, :])
    _rho, ux, _uy, _uz = macroscopic3d(out)
    assert torch.allclose(ux[:, :1, :], torch.full_like(ux[:, :1, :], U_IN), atol=1e-12)


def test_convective_outlet_value_contract() -> None:
    """Upwind recursion on the unknown outlet populations, chained over steps.

    f_out^{n+1} = f_out^n + U_c (f_{out-1}^n - f_out^n) with f_out^n the
    previous post-boundary outlet face (seeded from the initial condition);
    the known directions keep their streamed values.
    """
    dtype, device = torch.float64, DEVICE
    u_c, tau, steps = U_IN, 0.7, 3
    f0 = uniform_flow_f0(U_IN, dtype, device)
    outlet = OutletSpec(method="convective", u_conv=u_c)
    f_end, probes = rollout(f0, steps, tau, None, outlet=outlet, return_probes=True)

    # independent manual replay of the same discrete recursion
    f = f0.clone()
    face = f0[..., -1:]  # seed: the initial condition's outlet plane
    for _ in range(steps):
        f_str = stream3d(collide_bgk3d(f, tau))
        candidate = face + u_c * (f_str[..., -2:-1] - face)
        plane_new = torch.where(_unk_out_selector(device), candidate, f_str[..., -1:])
        f = torch.cat([f_str[..., :-1], plane_new], dim=-1)
        face = f[..., -1:]
    assert torch.equal(f_end, f)

    # bookkeeping on the first probe: unknowns follow the seeded recursion,
    # known directions are the plain streamed values
    f_str0 = stream3d(collide_bgk3d(f0, tau))
    seed = f0[..., -1:]
    expected_unk = seed[UNK_OUT] + u_c * (f_str0[UNK_OUT, ..., -2:-1] - seed[UNK_OUT])
    assert torch.allclose(probes[0][UNK_OUT, ..., -1:], expected_unk, atol=1e-15)
    known = [q for q in range(19) if q not in UNK_OUT]
    assert torch.equal(probes[0][known, ..., -1:], f_str0[known, ..., -1:])
    # the recursion is active: convective differs from the plain copy outlet
    assert not torch.equal(f_end, rollout(f0, steps, tau, None, outlet=OutletSpec()))


def _unk_out_selector(device: torch.device) -> torch.Tensor:
    sel = torch.zeros((19, 1, 1, 1), dtype=torch.bool, device=device)
    sel[UNK_OUT, 0, 0, 0] = True
    return sel


def test_convective_outlet_single_step_seeding() -> None:
    """outlet_prev=None seeds the recursion from the step input's outlet plane."""
    dtype, device = torch.float64, DEVICE
    f0 = uniform_flow_f0(U_IN, dtype, device)
    outlet = OutletSpec(method="convective", u_conv=U_IN)
    single = differentiable_step(f0, 0.7, None, outlet=outlet)
    first, probe1 = differentiable_step(
        f0, 0.7, None, return_probe=True, outlet=outlet, outlet_prev=f0[..., -1:]
    )
    assert torch.equal(single, first)
    second = differentiable_step(first, 0.7, None, outlet=outlet, outlet_prev=probe1[..., -1:])
    assert torch.equal(second, rollout(f0, 2, 0.7, None, outlet=outlet))


@pytest.mark.parametrize("method", ["equilibrium", "zouhe"])
def test_bounded_box_tau_gradient_matches_finite_difference(method) -> None:
    """dLoss/dtau through the full bounded box (inlet + convective outlet +
    free-slip walls) == central FD of the same discrete loss."""
    steps, tau_eval, eps = 10, 0.7, 1e-5
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet = InletSpec(ux=U_IN, method=method)
    outlet, walls = OutletSpec(method="convective"), WallSpec(method="free-slip")
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(f0, steps, 0.85, mask, inlet=inlet, outlet=outlet, walls=walls), mask
        )

    def loss_of(tau_val: float) -> float:
        tau = torch.tensor(tau_val, dtype=dtype, device=DEVICE)
        return float(
            field_loss(
                rollout(f0, steps, tau, mask, inlet=inlet, outlet=outlet, walls=walls),
                target,
                mask,
            )
        )

    tau = torch.tensor(tau_eval, dtype=dtype, device=DEVICE, requires_grad=True)
    loss = field_loss(
        rollout(f0, steps, tau, mask, inlet=inlet, outlet=outlet, walls=walls), target, mask
    )
    (g_ad,) = torch.autograd.grad(loss, tau)

    fd = (loss_of(tau_eval + eps) - loss_of(tau_eval - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


def test_bounded_box_uin_gradient_matches_finite_difference() -> None:
    """dLoss/du_in with the convective speed derived from the inlet
    (U_c = u_in): the gradient flows through the inlet closure and the
    outlet recursion simultaneously."""
    steps, tau0, eps = 10, 0.7, 1e-5
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    walls = WallSpec(method="free-slip")
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(
                f0,
                steps,
                0.85,
                mask,
                inlet=InletSpec(ux=U_IN, method="zouhe"),
                outlet=OutletSpec(method="convective"),
                walls=walls,
            ),
            mask,
        )

    def loss_of(u_val: float) -> float:
        return float(
            field_loss(
                rollout(
                    f0,
                    steps,
                    tau0,
                    mask,
                    inlet=InletSpec(ux=u_val, method="zouhe"),
                    outlet=OutletSpec(method="convective"),
                    walls=walls,
                ),
                target,
                mask,
            )
        )

    u_in = torch.tensor(U_IN, dtype=dtype, device=DEVICE, requires_grad=True)
    loss = field_loss(
        rollout(
            f0,
            steps,
            tau0,
            mask,
            inlet=InletSpec(ux=u_in, method="zouhe"),
            outlet=OutletSpec(method="convective"),
            walls=walls,
        ),
        target,
        mask,
    )
    (g_ad,) = torch.autograd.grad(loss, u_in)

    fd = (loss_of(U_IN + eps) - loss_of(U_IN - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


def test_convective_speed_gradient_matches_finite_difference() -> None:
    """The learnable Courant number: dLoss/dU_c through the outlet recursion == FD."""
    steps, tau0, eps = 10, 0.7, 1e-5
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet, walls = InletSpec(ux=U_IN, method="zouhe"), WallSpec(method="free-slip")
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(
                f0, steps, 0.85, mask, inlet=inlet, outlet=OutletSpec(method="copy"), walls=walls
            ),
            mask,
        )

    def loss_of(u_c_val: float) -> float:
        outlet = OutletSpec(method="convective", u_conv=u_c_val)
        return float(
            field_loss(
                rollout(f0, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=walls),
                target,
                mask,
            )
        )

    u_c = torch.tensor(U_IN, dtype=dtype, device=DEVICE, requires_grad=True)
    outlet = OutletSpec(method="convective", u_conv=u_c)
    loss = field_loss(
        rollout(f0, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=walls), target, mask
    )
    (g_ad,) = torch.autograd.grad(loss, u_c)

    fd = (loss_of(U_IN + eps) - loss_of(U_IN - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


def test_bounded_box_f0_boundary_entry_gradients_match_fd() -> None:
    """Element-wise dLoss/df0 through the full box: wall-plane unknowns, the
    convective recursion seed (initial outlet face) and the interior."""
    steps, tau0 = 8, 0.7
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet, walls = InletSpec(ux=U_IN, method="zouhe"), WallSpec(method="free-slip")
    outlet = OutletSpec(method="convective")
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(f0, steps, 0.9, mask, inlet=inlet, outlet=outlet, walls=walls), mask
        )

    f0 = f0.requires_grad_(True)
    loss = field_loss(
        rollout(f0, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=walls), target, mask
    )
    (g_ad,) = torch.autograd.grad(loss, f0)

    cz, cy = round(BNZ / 2), round(BNY / 2)
    entries = [
        (0, 2, 3, 4),  # far interior
        (3, cz, 0, 4),  # unknown (mirrored) direction on the y = 0 wall plane
        (5, 0, cy, 4),  # unknown (mirrored) direction on the z = 0 wall plane
        (16, 0, 0, 6),  # corner-line direction, doubly unknown (y and z)
        (2, cz, cy, BNX - 1),  # outlet-plane unknown: seeds the convective recursion
    ]
    eps = 1e-6
    for q, iz, iy, ix in entries:
        f_plus, f_minus = f0.detach().clone(), f0.detach().clone()
        f_plus[q, iz, iy, ix] += eps
        f_minus[q, iz, iy, ix] -= eps
        l_plus = float(
            field_loss(
                rollout(f_plus, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=walls),
                target,
                mask,
            )
        )
        l_minus = float(
            field_loss(
                rollout(f_minus, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=walls),
                target,
                mask,
            )
        )
        fd = (l_plus - l_minus) / (2.0 * eps)
        diff = abs(float(g_ad[q, iz, iy, ix]) - fd)
        denom = max(abs(float(g_ad[q, iz, iy, ix])), abs(fd), 1e-30)
        # the corner-line entry carries a ~1e-8 gradient — below the fp64 FD
        # noise floor of the loss difference quotient — so accept an absolute
        # agreement of 1e-12 there (measured: |ad - fd| ~ 1e-14)
        assert diff / denom < 1e-6 or diff < 1e-12, (q, iz, iy, ix)


def test_bounded_box_checkpoint_gradients_equal() -> None:
    """checkpoint=True reproduces plain gradients with the full bounded box
    active, convective outlet history chaining included."""
    steps = 8
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    walls = WallSpec(method="free-slip")
    with torch.no_grad():
        target = ux_fluid(
            rollout(
                uniform_flow_f0(U_IN, dtype, DEVICE),
                steps,
                0.9,
                mask,
                inlet=InletSpec(ux=U_IN, method="zouhe"),
                outlet=OutletSpec(method="convective"),
                walls=walls,
            ),
            mask,
        )

    def loss_and_grads(use_checkpoint: bool):
        f0 = uniform_flow_f0(U_IN, dtype, DEVICE).requires_grad_(True)
        tau = torch.tensor(0.7, dtype=dtype, device=DEVICE, requires_grad=True)
        u_in = torch.tensor(U_IN, dtype=dtype, device=DEVICE, requires_grad=True)
        inlet = InletSpec(ux=u_in, method="zouhe")
        f, probes = rollout(
            f0,
            steps,
            tau,
            mask,
            checkpoint=use_checkpoint,
            inlet=inlet,
            outlet=OutletSpec(method="convective"),
            walls=walls,
            return_probes=True,
        )
        loss = field_loss(f, target, mask) + sum(obstacle_force(p, mask)[0] for p in probes)
        g_f0, g_tau, g_u = torch.autograd.grad(loss, [f0, tau, u_in])
        return loss.detach(), g_f0, g_tau, g_u

    loss_plain, g_f0_p, g_tau_p, g_u_p = loss_and_grads(False)
    loss_ckpt, g_f0_c, g_tau_c, g_u_c = loss_and_grads(True)

    assert torch.allclose(loss_plain, loss_ckpt, rtol=1e-12)
    assert torch.allclose(g_f0_p, g_f0_c, rtol=1e-10, atol=1e-14)
    assert torch.allclose(g_tau_p, g_tau_c, rtol=1e-10)
    assert torch.allclose(g_u_p, g_u_c, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_bounded_box_cuda_parity() -> None:
    """Same fp32 full-box rollout (zouhe inlet + convective outlet + free-slip
    walls) and gradients on CPU and CUDA."""
    dtype, steps = torch.float32, 10
    mask_cpu = make_bounded_mask(torch.device("cpu"))
    mask_cuda = make_bounded_mask(torch.device("cuda"))

    def run(f0: torch.Tensor, mask: torch.Tensor, tau: torch.Tensor, u_in: torch.Tensor):
        walls = WallSpec(method="free-slip")
        f, probes = rollout(
            f0,
            steps,
            tau,
            mask,
            inlet=InletSpec(ux=u_in, method="zouhe"),
            outlet=OutletSpec(method="convective"),
            walls=walls,
            return_probes=True,
        )
        loss = ((macroscopic3d(f)[1][~mask]) ** 2).mean() + sum(
            obstacle_force(p, mask)[0] for p in probes
        )
        g_tau, g_u = torch.autograd.grad(loss, [tau, u_in])
        return loss.detach(), g_tau, g_u

    f0_cpu = uniform_flow_f0(U_IN, dtype, torch.device("cpu"))
    tau_cpu = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    u_cpu = torch.tensor(U_IN, dtype=dtype, requires_grad=True)
    loss_c, g_tau_c, g_u_c = run(f0_cpu, mask_cpu, tau_cpu, u_cpu)

    tau_cuda = torch.tensor(0.7, dtype=dtype, device="cuda", requires_grad=True)
    u_cuda = torch.tensor(U_IN, dtype=dtype, device="cuda", requires_grad=True)
    loss_g, g_tau_g, g_u_g = run(f0_cpu.to("cuda"), mask_cuda, tau_cuda, u_cuda)

    assert torch.allclose(loss_c, loss_g.cpu(), rtol=1e-5, atol=1e-7)
    assert torch.allclose(g_tau_c, g_tau_g.cpu(), rtol=1e-3, atol=1e-8)
    assert torch.allclose(g_u_c, g_u_g.cpu(), rtol=1e-3, atol=1e-8)


def test_wall_and_outlet_spec_validation() -> None:
    """Malformed specs and invalid Courant numbers fail loudly."""
    with pytest.raises(ValueError, match="outlet method"):
        OutletSpec(method="bogus")
    with pytest.raises(ValueError, match="wall method"):
        WallSpec(method="bogus")
    f0 = uniform_flow_f0(U_IN, torch.float64, DEVICE)
    with pytest.raises(ValueError, match="convective speed"):
        differentiable_step(f0, 0.7, None, outlet=OutletSpec(method="convective"))
    with pytest.raises(ValueError, match="Courant"):
        differentiable_step(f0, 0.7, None, outlet=OutletSpec(method="convective", u_conv=1.5))
    with pytest.raises(ValueError, match="Courant"):
        differentiable_step(f0, 0.7, None, outlet=OutletSpec(method="convective", u_conv=0.0))
    with pytest.raises(ValueError, match="outlet_prev"):
        differentiable_step(
            f0,
            0.7,
            None,
            outlet=OutletSpec(method="convective", u_conv=U_IN),
            outlet_prev=f0,
        )


def _run_sphere_box(walls: WallSpec | None, steps: int = 300):
    """Drive the bounded sphere campaign with a given lateral closure."""
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    ones_f = torch.ones(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    zeros_f = torch.zeros_like(ones_f)
    f = equilibrium3d(ones_f, U_IN * ones_f, zeros_f, zeros_f)
    inlet, outlet = InletSpec(ux=U_IN, method="zouhe"), OutletSpec(method="convective")
    drags = []
    for _ in range(steps):
        f, probe = differentiable_step(
            f, 0.55, mask, return_probe=True, inlet=inlet, outlet=outlet, walls=walls
        )
        drags.append(obstacle_force(probe, mask)[0].detach())
    rho, ux, uy, uz = macroscopic3d(f)
    return f, rho, ux, uy, uz, mask, torch.stack(drags)


def test_bounded_box_sphere_physics() -> None:
    """300-step fp64 sphere campaign in the full bounded box: finite fields,
    machine-zero wall-normal velocity on the lateral faces, and far less
    side pollution than the periodic-sides baseline."""
    f_s, rho_s, ux_s, uy_s, uz_s, mask, drags = _run_sphere_box(WallSpec(method="free-slip"))
    fluid = ~mask

    assert torch.isfinite(f_s).all()
    assert 0.85 < float(rho_s.min()) < float(rho_s.max()) < 1.15
    # the drive sustains the flow
    assert float(ux_s[fluid].mean()) > 0.9 * U_IN
    # free-slip faces: the reflected pairs cancel, wall-normal velocity ~ 0
    assert float(uy_s[1:-1, 0, :].abs().max()) < 1e-12
    assert float(uy_s[1:-1, -1, :].abs().max()) < 1e-12
    assert float(uz_s[0, 1:-1, :].abs().max()) < 1e-12
    assert float(uz_s[-1, 1:-1, :].abs().max()) < 1e-12
    # drag stays positive and settles (window-average drift below 10%)
    w_early = float(drags[50:100].mean())
    w_late = float(drags[250:300].mean())
    assert w_late > 0.0
    assert abs(w_late - w_early) < 0.1 * w_late

    # quantitative side-pollution comparison vs the periodic-sides baseline
    f_p, _rho_p, _ux_p, uy_p, uz_p, _m, _d = _run_sphere_box(None)
    assert torch.isfinite(f_p).all()
    metric_slip = float(
        uy_s[:, 0, 1:-1].abs().mean()
        + uy_s[:, -1, 1:-1].abs().mean()
        + uz_s[0, :, 1:-1].abs().mean()
        + uz_s[-1, :, 1:-1].abs().mean()
    )
    metric_periodic = float(
        uy_p[:, 0, 1:-1].abs().mean()
        + uy_p[:, -1, 1:-1].abs().mean()
        + uz_p[0, :, 1:-1].abs().mean()
        + uz_p[-1, :, 1:-1].abs().mean()
    )
    # measured: ~2e-17 (slip) vs ~7e-3 (periodic) — the periodic wrap injects
    # cross-domain normal flow at the glued planes, free-slip reflects it
    assert metric_periodic > 1e-3
    assert metric_slip < 1e-10 * metric_periodic

# ---------------------------------------------------------------------------
# 9. A6+++ per-face lateral walls: WallSpec.overrides with face keys
# ---------------------------------------------------------------------------

FACE_KEYS = ("-y", "+y", "-z", "+z")


def asymmetric_flow_f0(
    amplitude: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 37,
) -> torch.Tensor:
    """Asymmetric equilibrium field u = (u0 + a*sin(2pi z/nz + 0.7),
    a*cos(2pi z/nz + 0.3), a*sin(2pi y/ny + 1.1)) plus deterministic noise.

    Every component is O(a) at every lateral face (so the free-slip normal
    cancellation and the free-stream reset act on a non-trivial field) and
    the phase offsets break the z / y mirror symmetries of the box, so no
    symmetric cancellation can fake a passing face assertion.
    """
    zz, yy, _xx = torch.meshgrid(
        torch.arange(NZ, dtype=dtype, device=device),
        torch.arange(NY, dtype=dtype, device=device),
        torch.arange(NX, dtype=dtype, device=device),
        indexing="ij",
    )
    ux = 0.06 + amplitude * torch.sin(TWO_PI * zz / NZ + 0.7)
    uy = amplitude * torch.cos(TWO_PI * zz / NZ + 0.3)
    uz = amplitude * torch.sin(TWO_PI * yy / NY + 1.1)
    f = equilibrium3d(torch.ones_like(ux), ux, uy, uz)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((19, NZ, NY, NX), generator=gen, dtype=torch.float64).to(dtype) - 0.5
    return f + 0.03 * noise.to(device)


def _manual_face_close(
    f: torch.Tensor, key: str, spec: WallSpec, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Independent single-face closure using the hand-derived test tables.

    Replays the documented policy only: the face is read from the current
    chain state, closed by its own spec (mirror swap / equilibrium reset /
    periodic no-op), and the result is re-assembled with torch.cat.
    """
    if spec.method == "periodic":
        return f
    y_axis = key in ("-y", "+y")
    at_start = key in ("-y", "-z")
    dim = 2 if y_axis else 1
    if y_axis:
        plane = f[:, :, :1] if at_start else f[:, :, -1:]
    else:
        plane = f[:, :1] if at_start else f[:, -1:]
    if spec.method == "free-slip":
        flip = FLIP_Y if y_axis else FLIP_Z
        unk = {"-y": UNK_Y0, "+y": UNK_Y1, "-z": UNK_Z0, "+z": UNK_Z1}[key]
        plane_new = plane.clone()
        for q in unk:
            plane_new[q] = plane[flip[q]]
    else:  # "freestream"
        feq = equilibrium3d(
            torch.tensor(float(spec.rho0), dtype=dtype, device=device),
            torch.tensor(float(spec.ux), dtype=dtype, device=device),
            torch.tensor(float(spec.uy), dtype=dtype, device=device),
            torch.tensor(float(spec.uz), dtype=dtype, device=device),
            device,
        )
        face_shape = list(f.shape)
        face_shape[dim] = 1
        plane_new = feq.expand(*face_shape)
    if at_start:
        interior = f[:, 1:] if dim == 1 else f[:, :, 1:]
        return torch.cat([plane_new, interior], dim=dim)
    interior = f[:, :-1] if dim == 1 else f[:, :, :-1]
    return torch.cat([interior, plane_new], dim=dim)


def _manual_walls(
    f_str: torch.Tensor, walls: WallSpec, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Close the four faces in the documented order (y0, y1, z0, z1)."""
    f = f_str
    for key in FACE_KEYS:
        spec = walls if walls.overrides is None else walls.overrides.get(key, walls)
        f = _manual_face_close(f, key, spec, dtype, device)
    return f


def _perface_walls(u_fs: float | torch.Tensor) -> WallSpec:
    """The mixed-spec layout of section 9: free-slip default, periodic -z,
    free-stream +z with a (learnable) far-field speed."""
    return WallSpec(
        method="free-slip",
        overrides={
            "-z": WallSpec(method="periodic"),
            "+z": WallSpec(method="freestream", rho0=1.03, ux=u_fs, uy=-0.02),
        },
    )


def test_perface_default_path_bitwise_unchanged() -> None:
    """overrides=None / absent / empty: bit-for-bit the shared-spec operator."""
    dtype = torch.float64
    f0 = nonequilibrium_f(dtype, DEVICE)

    # the shared-spec result is pinned by section 8; here: no-ops are exact
    plain = differentiable_step(f0, 0.8, None)
    assert torch.equal(differentiable_step(f0, 0.8, None, walls=WallSpec()), plain)
    assert torch.equal(
        differentiable_step(f0, 0.8, None, walls=WallSpec(overrides={})), plain
    )
    # all faces overridden to periodic: the wrap survives bit-for-bit
    all_periodic = WallSpec(overrides={key: WallSpec() for key in FACE_KEYS})
    assert torch.equal(differentiable_step(f0, 0.8, None, walls=all_periodic), plain)
    # and a fully periodic box with one face overridden stays a no-op
    assert torch.equal(
        differentiable_step(f0, 0.8, None, walls=WallSpec(overrides={"+y": WallSpec()})),
        plain,
    )


def test_perface_uniform_overrides_equal_shared_spec() -> None:
    """Explicit per-face replicas of one closure == the shared spec operator
    (the per-face loop and the batched path agree bit-for-bit)."""
    dtype, device = torch.float64, DEVICE
    f = nonequilibrium_f(dtype, device)
    closures = [
        ("free-slip", {}),
        ("freestream", {"rho0": 1.02, "ux": 0.05, "uy": 0.01}),
    ]
    for method, fields in closures:
        shared = WallSpec(method=method, **fields)
        out_shared = differentiable_step(f, 0.7, None, walls=shared)
        replicas = {key: WallSpec(method=method, **fields) for key in FACE_KEYS}
        # all four faces listed, three listed (one uses the default), empty dict
        for overrides in (replicas, {k: v for k, v in replicas.items() if k != "-z"}, {}):
            walls = WallSpec(method=method, **fields, overrides=overrides)
            assert torch.equal(differentiable_step(f, 0.7, None, walls=walls), out_shared)


def test_perface_wall_value_contract() -> None:
    """Mixed methods applied exactly where claimed, edge last-write-wins."""
    dtype, device = torch.float64, DEVICE
    f = nonequilibrium_f(dtype, device)
    walls = _perface_walls(0.1)
    out = differentiable_step(f, 0.7, None, walls=walls)

    f_str = stream3d(collide_bgk3d(f, 0.7))
    manual = _manual_walls(f_str, walls, dtype, device)
    # the full post-boundary state matches the independent face-by-face replay
    assert torch.equal(out, manual)
    # interior planes untouched in every direction
    assert torch.equal(out[:, 1:-1, 1:-1, :], f_str[:, 1:-1, 1:-1, :])
    # the periodic -z face keeps the wrap: untouched off the y-edge lines
    # (those carry the earlier y closures — last write wins there)
    assert torch.equal(out[:, 0, 1:-1, :], f_str[:, 0, 1:-1, :])
    # the free-stream +z face is fully reset *including* its edge lines (the
    # z closure runs last: last write wins over the y closures there)
    feq = equilibrium3d(
        torch.tensor(1.03, dtype=dtype, device=device),
        torch.tensor(0.1, dtype=dtype, device=device),
        torch.tensor(-0.02, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
        device,
    )
    assert torch.equal(out[:, -1:, :, :], feq.expand(19, 1, NY, NX))

    # freestream default with free-slip y overrides (the complementary mix)
    walls2 = WallSpec(
        method="freestream",
        rho0=1.01,
        ux=0.04,
        overrides={"-y": WallSpec(method="free-slip"), "+y": WallSpec(method="free-slip")},
    )
    out2 = differentiable_step(f, 0.7, None, walls=walls2)
    assert torch.equal(out2, _manual_walls(f_str, walls2, dtype, device))


def test_perface_wall_asymmetric_physics() -> None:
    """Asymmetric field, one method per face, each asserted on its own."""
    dtype, device = torch.float64, DEVICE
    f = asymmetric_flow_f0(0.05, dtype, device)
    tau = 0.7
    walls = _perface_walls(0.1)
    out = differentiable_step(f, tau, None, walls=walls)
    f_str = stream3d(collide_bgk3d(f, tau))
    plain = differentiable_step(f, tau, None)  # fully periodic reference

    rho, ux, uy, uz = macroscopic3d(out)

    # free-slip faces (-y, +y): wall-normal velocity machine zero, tangential
    # momentum retained at its O(amplitude) level (not wiped by the closure)
    assert float(uy[1:-1, 0, :].abs().max()) < 1e-14
    assert float(uy[1:-1, -1, :].abs().max()) < 1e-14
    assert float(uz[1:-1, 0, :].abs().mean()) > 1e-3
    assert float(ux[1:-1, -1, :].abs().mean()) > 1e-3
    for q in range(19):  # exact mirror replay on both faces, off the z edges
        expected = f_str[FLIP_Y[q], :, 0, :] if q in UNK_Y0 else f_str[q, :, 0, :]
        assert torch.equal(out[q, 1:-1, 0, :], expected[1:-1, :]), q
        expected = f_str[FLIP_Y[q], :, -1, :] if q in UNK_Y1 else f_str[q, :, -1, :]
        assert torch.equal(out[q, 1:-1, -1, :], expected[1:-1, :]), q

    # free-stream face (+z): the whole plane sits at its own far field
    rho_f, ux_f, uy_f, uz_f = macroscopic3d(out[:, -1:, :, :])
    assert torch.allclose(rho_f, torch.full_like(rho_f, 1.03), atol=1e-12)
    assert torch.allclose(ux_f, torch.full_like(ux_f, 0.1), atol=1e-12)
    assert torch.allclose(uy_f, torch.full_like(uy_f, -0.02), atol=1e-12)
    assert torch.allclose(uz_f, torch.full_like(uz_f, 0.0), atol=1e-12)

    # periodic face (-z): keeps the wrap — off the y-edge lines it matches the
    # fully periodic chain exactly (the two ends of the axis stay glued)
    assert torch.equal(out[:, 0, 1:-1, :], plain[:, 0, 1:-1, :])
    assert torch.equal(out[:, 0, 1:-1, :], f_str[:, 0, 1:-1, :])


def test_perface_wall_uinf_gradient_matches_finite_difference() -> None:
    """Learnable free-stream speed on the +z override: dLoss/du_inf == FD
    through the overrides path (gradient crosses the face reset)."""
    steps, tau0, eps, u_fs_val = 10, 0.7, 1e-5, 0.06
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    inlet, outlet = InletSpec(ux=U_IN), OutletSpec(method="convective")
    f0 = uniform_flow_f0(U_IN, dtype, DEVICE)
    with torch.no_grad():
        target = ux_fluid(
            rollout(
                f0, steps, 0.85, mask, inlet=inlet, outlet=outlet, walls=_perface_walls(u_fs_val)
            ),
            mask,
        )

    def loss_of(u_val: float) -> float:
        return float(
            field_loss(
                rollout(
                    f0, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=_perface_walls(u_val)
                ),
                target,
                mask,
            )
        )

    u_fs = torch.tensor(u_fs_val, dtype=dtype, device=DEVICE, requires_grad=True)
    loss = field_loss(
        rollout(f0, steps, tau0, mask, inlet=inlet, outlet=outlet, walls=_perface_walls(u_fs)),
        target,
        mask,
    )
    (g_ad,) = torch.autograd.grad(loss, u_fs)

    fd = (loss_of(u_fs_val + eps) - loss_of(u_fs_val - eps)) / (2.0 * eps)
    denom = max(abs(float(g_ad)), abs(fd), 1e-30)
    assert abs(float(g_ad) - fd) / denom < 1e-6
    assert float(g_ad) != 0.0


def test_perface_wall_checkpoint_gradients_equal() -> None:
    """checkpoint=True reproduces plain gradients with mixed per-face walls."""
    steps = 8
    dtype = torch.float64
    mask = make_bounded_mask(DEVICE)
    walls = _perface_walls(0.06)
    with torch.no_grad():
        target = ux_fluid(
            rollout(
                uniform_flow_f0(U_IN, dtype, DEVICE),
                steps,
                0.9,
                mask,
                inlet=InletSpec(ux=U_IN),
                outlet=OutletSpec(method="convective"),
                walls=walls,
            ),
            mask,
        )

    def loss_and_grads(use_checkpoint: bool):
        f0 = uniform_flow_f0(U_IN, dtype, DEVICE).requires_grad_(True)
        tau = torch.tensor(0.7, dtype=dtype, device=DEVICE, requires_grad=True)
        f, probes = rollout(
            f0,
            steps,
            tau,
            mask,
            checkpoint=use_checkpoint,
            inlet=InletSpec(ux=U_IN),
            outlet=OutletSpec(method="convective"),
            walls=walls,
            return_probes=True,
        )
        loss = field_loss(f, target, mask) + sum(obstacle_force(p, mask)[0] for p in probes)
        g_f0, g_tau = torch.autograd.grad(loss, [f0, tau])
        return loss.detach(), g_f0, g_tau

    loss_plain, g_f0_p, g_tau_p = loss_and_grads(False)
    loss_ckpt, g_f0_c, g_tau_c = loss_and_grads(True)

    assert torch.allclose(loss_plain, loss_ckpt, rtol=1e-12)
    assert torch.allclose(g_f0_p, g_f0_c, rtol=1e-10, atol=1e-14)
    assert torch.allclose(g_tau_p, g_tau_c, rtol=1e-10)


def test_wallspec_to_dict_from_dict_roundtrip() -> None:
    """to_dict/from_dict: exact roundtrip incl. nested per-face overrides."""
    spec = WallSpec(
        method="free-slip",
        overrides={
            "+z": WallSpec(method="freestream", rho0=1.03, ux=0.1, uy=-0.02),
            "-z": WallSpec(method="periodic"),
        },
    )
    payload = spec.to_dict()
    assert set(payload) == {"method", "rho0", "ux", "uy", "uz", "overrides"}
    assert payload["overrides"]["+z"] == {
        "method": "freestream",
        "rho0": 1.03,
        "ux": 0.1,
        "uy": -0.02,
        "uz": 0.0,
    }
    assert WallSpec.from_dict(payload) == spec

    # specs without overrides serialise to the pre-A6+++ payload shape
    plain = WallSpec(method="freestream", rho0=1.02, ux=0.05)
    assert plain.to_dict() == {
        "method": "freestream",
        "rho0": 1.02,
        "ux": 0.05,
        "uy": 0.0,
        "uz": 0.0,
    }
    assert WallSpec.from_dict(plain.to_dict()) == plain

    # tensor fields flatten to numeric values (the graph is not serialisable)
    t = WallSpec(method="freestream", ux=torch.tensor(0.07, dtype=torch.float64))
    payload_t = t.to_dict()
    assert isinstance(payload_t["ux"], float) and payload_t["ux"] == 0.07
    assert WallSpec.from_dict(payload_t) == WallSpec(method="freestream", ux=0.07)


def test_wallspec_old_payload_without_overrides_loads() -> None:
    """Pre-A6+++ payloads (no overrides key) and partial payloads load."""
    old = {"method": "freestream", "rho0": 1.02, "ux": 0.05, "uy": 0.0, "uz": 0.0}
    spec = WallSpec.from_dict(old)
    assert spec == WallSpec(method="freestream", rho0=1.02, ux=0.05)
    assert spec.overrides is None
    # missing numeric fields fall back to the dataclass defaults
    assert WallSpec.from_dict({"method": "free-slip"}) == WallSpec(method="free-slip")
    # unknown extra keys are ignored (forward compatibility)
    assert WallSpec.from_dict({"method": "free-slip", "engine": "bgk"}) == WallSpec(
        method="free-slip"
    )


def test_perface_wall_spec_validation() -> None:
    """Malformed per-face specs fail loudly; valid no-ops construct."""
    with pytest.raises(ValueError, match="overrides keys"):
        WallSpec(method="free-slip", overrides={"x+": WallSpec()})
    with pytest.raises(ValueError, match="overrides keys"):
        WallSpec(method="free-slip", overrides={"-y": WallSpec(), "top": WallSpec()})
    with pytest.raises(ValueError, match="must be a WallSpec"):
        WallSpec(method="free-slip", overrides={"+y": "free-slip"})
    with pytest.raises(ValueError, match="nested overrides"):
        WallSpec(method="free-slip", overrides={"+y": WallSpec(overrides={"-y": WallSpec()})})
    # valid constructions: empty mapping and periodic override specs
    assert WallSpec(method="free-slip", overrides={}).overrides == {}
    assert WallSpec(overrides={"+z": WallSpec(method="periodic")}).overrides is not None
