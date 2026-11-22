"""DE-X-COL2-REG: regularized (RR) collision in the free-surface solver.

The RR operator projects the non-equilibrium part back onto the equilibrium
manifold through its second-order stress only (Ladd 1994): every higher-order
non-equilibrium moment is discarded.  These tests pin the operator-level
contract of the ``collision="rr"`` branch of :func:`free_surface_step` and of
the underlying :func:`collide_regularized_stress_d3q19` kernel.

The problem-specific stability window (how far below the BGK floor the
drop-impact run survives) is established by the SDAA full-window runs, not
here; see ``runs/COL2-REG-*``.
"""

from __future__ import annotations

import torch

from tensorlbm.d3q19 import C, equilibrium3d, macroscopic3d
from tensorlbm.collision_d3q19_advanced import (
    collide_regularized_stress_d3q19,
    second_order_stress_d3q19,
)
from tensorlbm.free_surface_lbm import free_surface_step, init_flags_from_fill, init_mass_from_fill


def _lattice_views(shape):
    cx, cy, cz = (C[:, k].view(19, 1, 1, 1).to(torch.float64) for k in range(3))
    return cx, cy, cz


def _uniform_feq(rho, ux, uy, uz, shape=(6, 6, 6)):
    r = torch.full(shape, rho, dtype=torch.float64)
    u = torch.full(shape, ux, dtype=torch.float64)
    v = torch.full(shape, uy, dtype=torch.float64)
    w = torch.full(shape, uz, dtype=torch.float64)
    return equilibrium3d(r, u, v, w)


def test_rr_kernel_is_idempotent_on_equilibrium() -> None:
    """An equilibrium distribution is a fixed point of the RR collision."""
    f = _uniform_feq(1.0, 0.05, -0.02, 0.03)
    feq = equilibrium3d(*macroscopic3d(f))
    out = collide_regularized_stress_d3q19(f, feq, tau=0.7)
    assert torch.allclose(out, feq, atol=1e-12)


def test_rr_kernel_conserves_mass_and_momentum() -> None:
    """RR conserves the conserved moments of a valid (zero-mass) neq state."""
    torch.manual_seed(0)
    shape = (5, 5, 5)
    r = 1.0 + 0.05 * torch.randn(shape)
    u = 0.1 * torch.randn(shape)
    v = 0.1 * torch.randn(shape)
    w = 0.1 * torch.randn(shape)
    feq = equilibrium3d(r, u, v, w)
    # A valid non-equilibrium part: the second-order stress reconstruction
    # itself, which carries zero mass and zero momentum by lattice isotropy.
    from tensorlbm.collision_d3q19_advanced import reconstruct_second_order_stress_d3q19
    pi = [0.02 * torch.ones(shape), -0.01 * torch.ones(shape), 0.015 * torch.ones(shape),
          0.008 * torch.ones(shape), 0.0 * torch.ones(shape), 0.0 * torch.ones(shape)]
    f_neq = reconstruct_second_order_stress_d3q19(*pi)
    # The kernel's lattice constants are float32 (CG contract), so the
    # isotropy sums close to ~1e-9, not exactly zero.
    assert torch.allclose(f_neq.sum(0), torch.zeros_like(f_neq.sum(0)), atol=1e-6)
    f = feq + f_neq
    out = collide_regularized_stress_d3q19(f, feq, tau=0.8)
    cx, _, _ = _lattice_views(shape)
    assert torch.allclose(f.sum(0), out.sum(0), atol=1e-6)
    assert torch.allclose((f * cx).sum(0), (out * cx).sum(0), atol=1e-6)


def test_rr_relaxes_stress_at_rate_one_over_tau() -> None:
    """The surviving second-order stress is scaled by (1 - 1/tau)."""
    shape = (4, 4, 4)
    feq = _uniform_feq(1.0, 0.0, 0.0, 0.0, shape=shape)
    cx, cy, cz = _lattice_views(shape)
    f = feq.clone()
    f += 0.02 * (cx * cx - 1.0 / 3.0) * torch.ones(shape)
    tau = 0.9
    out = collide_regularized_stress_d3q19(f, feq, tau=tau)
    pi0 = second_order_stress_d3q19(f - feq)
    pi1 = second_order_stress_d3q19(out - feq)
    for a, b in zip(pi0, pi1):
        assert torch.allclose(b, (1.0 - 1.0 / tau) * a, atol=1e-12)


def test_rr_discards_higher_order_nonequilibrium() -> None:
    """A zero-stress (third-order) perturbation is projected away entirely."""
    shape = (4, 4, 4)
    feq = _uniform_feq(1.0, 0.0, 0.0, 0.0, shape=shape)
    cx, cy, cz = _lattice_views(shape)
    f = feq + 0.05 * (cx * cy * cz) * torch.ones(shape)
    # The perturbation carries no second-order stress ...
    for comp in second_order_stress_d3q19(f - feq):
        assert torch.allclose(comp, torch.zeros_like(comp), atol=1e-12)
    # ... so the RR collision removes it completely (projects to the manifold).
    out = collide_regularized_stress_d3q19(f, feq, tau=0.6)
    assert torch.allclose(out, feq, atol=1e-12)


def test_free_surface_step_rr_branch_runs_and_stays_finite() -> None:
    """The solver's ``collision="rr"`` branch steps a small field without NaN."""
    nz, ny, nx = 12, 12, 12
    z = torch.arange(nz).view(nz, 1, 1)
    level = 8.5
    fill = (level - z + 0.5).clamp(0, 1)  # flat pool, gas above
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[[0, -1]] = True
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    rho = 1.0 + 3.0 * (0.02 * (level - z).clamp_min(0))
    uz = -0.05 * (z > level + 0.5)  # a gentle downward push above the pool
    f = equilibrium3d(rho * (flags != 1), torch.zeros_like(fill), torch.zeros_like(fill), uz)
    mass = init_mass_from_fill(fill, flags)
    for _ in range(30):
        f, fill, flags, mass, _ = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=0.55, gz=-0.02,
            surface_tension=0.01, rho_gas=1.0, collision="rr",
        )
    assert torch.isfinite(f).all()
    assert torch.isfinite(mass).all()
