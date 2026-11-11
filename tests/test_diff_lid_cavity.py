"""Contract tests for the graph-preserving moving-lid BC (GAP-PP-2).

Pinned properties (each one is a claim the module's docstring makes):

1. the out-of-place lid closure is *value-identical* to the production
   ``lid_driven_cavity.zou_he_moving_lid`` (no physics reimplementation);
2. a uniform Galilean state ``f^eq(1, (U, 0))`` is an exact fixed point of the
   lid closure, and the lid plane carries ``j_x = rho U``, ``j_y = 0``;
3. the closure is out of place (input untouched, no ``CopySlices`` node) and
   passes ``torch.autograd.gradcheck`` in float64;
4. autograd gradients through ``d(loss)/d(tau)`` and ``d(loss)/d(u_lid)`` agree
   with central finite differences on the same fixed-step map (<= 1e-4 rel);
5. the closure survives ``torch.vmap`` over the lid velocity -- the in-place
   production version does not (that is the concrete defect it replaces);
6. ``rollout`` gradient checkpointing is value- and gradient-identical to the
   eager rollout;
7. ``collide_bgk`` keeps a 0-dim tensor tau connected (regression for the silent
   ``float(tau)`` detach that blocked solver-in-the-loop tau inversion);
8. the inverse machinery recovers tau from its own fixed-step data on a small
   grid (smoke version of the acceptance criterion, tolerance 2 %).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.diff_lid_cavity import (
    CavityGeometry,
    cavity_wall_mask,
    centreline_probes,
    initial_field,
    invert_relaxation_time,
    moving_lid_bc,
    moving_lid_values,
    rollout,
    step,
    tau_from_reynolds,
)
from tensorlbm.lid_driven_cavity import zou_he_moving_lid
from tensorlbm.solver import collide_bgk

DT = torch.float64


def _random_state(ny: int = 6, nx: int = 7, u_lid: float = 0.1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    feq = equilibrium(
        torch.ones(ny, nx, dtype=DT),
        torch.full((ny, nx), u_lid, dtype=DT),
        torch.zeros(ny, nx, dtype=DT),
    )
    return feq + 0.01 * torch.randn(feq.shape, generator=gen, dtype=DT)


# ---------------------------------------------------------------------------
# 1. equivalence with the production closure
# ---------------------------------------------------------------------------


def test_moving_lid_values_match_production_zou_he() -> None:
    f = _random_state()
    for u_lid in (0.0, 0.05, 0.1, -0.03):
        new = moving_lid_bc(f, u_lid)
        ref = zou_he_moving_lid(f, u_lid)
        assert torch.equal(new, ref)


def test_moving_lid_values_accepts_profile() -> None:
    f = _random_state(nx=8)
    profile = torch.linspace(-0.05, 0.05, 8, dtype=DT)
    plane = moving_lid_values(f[:, -1, :], profile)
    assert plane.shape == (9, 8)
    assert torch.isfinite(plane).all()


# ---------------------------------------------------------------------------
# 2. exact invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("u_lid", [0.0, 0.05, 0.1, 0.2])
def test_galilean_state_is_fixed_point(u_lid: float) -> None:
    rho = torch.ones(5, 6, dtype=DT)
    ux = torch.full_like(rho, u_lid)
    uy = torch.zeros_like(rho)
    feq = equilibrium(rho, ux, uy)
    assert torch.allclose(moving_lid_bc(feq, u_lid), feq, atol=1e-14, rtol=0)


@pytest.mark.parametrize("u_lid", [0.05, 0.1, -0.08])
def test_lid_plane_carries_prescribed_momentum(u_lid: float) -> None:
    f = moving_lid_bc(_random_state(), u_lid)
    rho, ux, uy = macroscopic(f)
    rho_lid, ux_lid, uy_lid = rho[-1], ux[-1], uy[-1]
    assert torch.allclose(rho_lid * ux_lid, rho_lid * u_lid, atol=1e-13, rtol=0)
    assert torch.allclose(uy_lid, torch.zeros_like(uy_lid), atol=1e-13, rtol=0)


# ---------------------------------------------------------------------------
# 3. out-of-place + gradcheck
# ---------------------------------------------------------------------------


def test_lid_bc_is_out_of_place() -> None:
    f = _random_state().requires_grad_(True)
    before = f.detach().clone()
    u = torch.tensor(0.1, dtype=DT, requires_grad=True)
    out = moving_lid_bc(f, u)
    assert torch.equal(f.detach(), before), "input field must not be mutated"
    assert out.grad_fn is not None
    chain = str(out.grad_fn)
    assert "CopySlices" not in chain and "CopyBackwards" not in chain


def test_lid_bc_gradcheck() -> None:
    f = _random_state().requires_grad_(True)
    u = torch.tensor(0.1, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda ff, uu: moving_lid_bc(ff, uu), (f, u), eps=1e-6, atol=1e-9
    )


def test_gradcheck_moving_lid_values() -> None:
    f_top = _random_state(ny=1)[:, 0, :].requires_grad_(True)
    u = torch.tensor(-0.07, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda ff, uu: moving_lid_values(ff, uu), (f_top, u), eps=1e-6, atol=1e-9
    )


# ---------------------------------------------------------------------------
# 4. autograd vs finite differences on the fixed-step map
# ---------------------------------------------------------------------------


def _observable(n: int, steps: int):
    geom = CavityGeometry(n, n, dtype=DT)
    f0 = initial_field(geom)
    mask = geom.wall_mask

    def run(tau: torch.Tensor, u_lid: torch.Tensor) -> torch.Tensor:
        f = rollout(f0, tau, u_lid, steps, wall_mask=mask)
        return (centreline_probes(f, u_lid) ** 2).sum()

    return run


@pytest.mark.parametrize("name", ["tau", "u_lid"])
def test_autograd_matches_finite_difference(name: str) -> None:
    n, steps, h = 24, 200, 1e-4
    run = _observable(n, steps)
    tau0 = tau_from_reynolds(100.0, 0.1, n - 1)
    u0 = 0.1
    if name == "tau":
        p = torch.tensor(tau0, dtype=DT, requires_grad=True)
        (ad,) = torch.autograd.grad(run(p, torch.tensor(u0, dtype=DT)), p)
        with torch.no_grad():
            jp = float(run(torch.tensor(tau0 + h, dtype=DT), torch.tensor(u0, dtype=DT)))
            jm = float(run(torch.tensor(tau0 - h, dtype=DT), torch.tensor(u0, dtype=DT)))
    else:
        p = torch.tensor(u0, dtype=DT, requires_grad=True)
        (ad,) = torch.autograd.grad(run(torch.tensor(tau0, dtype=DT), p), p)
        with torch.no_grad():
            jp = float(run(torch.tensor(tau0, dtype=DT), torch.tensor(u0 + h, dtype=DT)))
            jm = float(run(torch.tensor(tau0, dtype=DT), torch.tensor(u0 - h, dtype=DT)))
    fd = (jp - jm) / (2.0 * h)
    assert abs(float(ad) - fd) / abs(fd) < 1e-4, (name, float(ad), fd)


# ---------------------------------------------------------------------------
# 5. vmap (batched lid velocities) -- in-place production closure cannot do this
# ---------------------------------------------------------------------------


def test_lid_bc_is_vmap_able() -> None:
    f = _random_state()
    us = torch.tensor([0.05, 0.1, 0.15], dtype=DT)
    out = torch.vmap(lambda uu: moving_lid_bc(f, uu))(us)
    assert out.shape == (3, *f.shape)
    for k, uu in enumerate(us):
        assert torch.equal(out[k], moving_lid_bc(f, float(uu)))


def test_production_inplace_closure_fails_under_vmap() -> None:
    """Documents the defect the out-of-place closure removes."""
    f = _random_state()
    with pytest.raises(RuntimeError):
        torch.vmap(lambda uu: zou_he_moving_lid(f, uu))(torch.tensor([0.05, 0.1], dtype=DT))


# ---------------------------------------------------------------------------
# 6. checkpointed rollout == eager rollout (values and gradients)
# ---------------------------------------------------------------------------


def test_checkpointed_rollout_matches_eager() -> None:
    geom = CavityGeometry(20, 20, dtype=DT)
    f0 = initial_field(geom)
    kwargs = dict(wall_mask=geom.wall_mask)
    tau_a = torch.tensor(0.66, dtype=DT, requires_grad=True)
    tau_b = torch.tensor(0.66, dtype=DT, requires_grad=True)
    fa = rollout(f0, tau_a, 0.1, 60, checkpoint_every=0, **kwargs)
    fb = rollout(f0, tau_b, 0.1, 60, checkpoint_every=20, **kwargs)
    assert torch.equal(fa, fb)
    (ga,) = torch.autograd.grad((fa**2).sum(), tau_a)
    (gb,) = torch.autograd.grad((fb**2).sum(), tau_b)
    assert torch.allclose(ga, gb, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. regression: collide_bgk must not detach a 0-dim tensor tau
# ---------------------------------------------------------------------------


def test_collide_bgk_keeps_tensor_tau_in_graph() -> None:
    f = _random_state()
    tau = torch.tensor(0.66, dtype=DT, requires_grad=True)
    out = collide_bgk(f, tau)
    assert out.requires_grad
    (g,) = torch.autograd.grad(out.pow(2).sum(), tau)
    assert g is not None and torch.isfinite(g) and g.abs() > 0.0
    # value-identical to the float path
    assert torch.equal(out.detach(), collide_bgk(f, 0.66))


def test_collide_bgk_still_rejects_illegal_float_tau() -> None:
    """Plain scalars keep the range check; tensor tau stays graph-connected."""
    f = _random_state()
    with pytest.raises(ValueError):
        collide_bgk(f, 0.4)
    # tensor tau is passed through untouched (documented non-plain-scalar policy)
    assert collide_bgk(f, torch.tensor(0.7, dtype=DT)).requires_grad is False


# ---------------------------------------------------------------------------
# 8. inverse smoke test (small grid, short chain)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_invert_recovers_tau_smoke() -> None:
    n, steps, iters = 24, 500, 10
    geom = CavityGeometry(n, n, dtype=DT)
    tau_true = tau_from_reynolds(100.0, 0.1, n - 1)
    u = torch.as_tensor(0.1, dtype=DT)
    with torch.no_grad():
        f_truth = rollout(initial_field(geom), tau_true, u, steps, wall_mask=geom.wall_mask)
        target = centreline_probes(f_truth, u)
    res = invert_relaxation_time(
        target, geom, 0.1, tau_true * 0.92, steps, iters=iters, lr=0.08, checkpoint_every=100
    )
    rel = abs(res["tau_recovered"] - tau_true) / tau_true
    assert rel < 0.02, (res["tau_recovered"], tau_true, rel)
    assert math.isfinite(res["loss_history"][-1])


def test_tau_from_reynolds_and_mask() -> None:
    tau = tau_from_reynolds(100.0, 0.1, 64)
    assert math.isclose(tau, 0.692, rel_tol=1e-12)
    mask = cavity_wall_mask(9, 7)
    assert mask[0].all() and mask[:, 0].all() and mask[:, -1].all()
    # the lid row interior is handled by the moving-wall closure; its two
    # corner nodes stay in the stationary mask (corner bounce-back, as in the
    # production solver) and are then overwritten by the Zou/He closure.
    assert not mask[-1, 1:-1].any()


def test_step_keeps_graph_through_lid() -> None:
    geom = CavityGeometry(16, 16, dtype=DT)
    f0 = initial_field(geom)
    tau = torch.tensor(0.66, dtype=DT, requires_grad=True)
    u = torch.tensor(0.1, dtype=DT, requires_grad=True)
    f = step(f0, tau, u, geom.wall_mask)
    loss = f.pow(2).sum()
    g_tau, g_u = torch.autograd.grad(loss, [tau, u])
    assert torch.isfinite(g_tau) and torch.isfinite(g_u)