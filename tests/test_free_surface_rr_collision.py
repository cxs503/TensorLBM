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
from tensorlbm.free_surface_lbm import (
    GAS,
    SOLID,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)


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


# ---------------------------------------------------------------------------
# DE-X-RELX3-RRIFACE: interface-local RR (INTERFACE cells only, bulk BGK)
# ---------------------------------------------------------------------------


def _mixed_rr_iface(f, feq, tau, iface_mask):
    """The solver's rr_iface mixing expression (mirrors free_surface_step)."""
    from tensorlbm.collision_d3q19_advanced import collide_regularized_stress_d3q19

    f_bgk = f - (f - feq) / tau
    f_rr = collide_regularized_stress_d3q19(f, feq, tau)
    return torch.where(iface_mask.unsqueeze(0), f_rr, f_bgk)


def test_rr_iface_mixing_is_local_to_interface_cells() -> None:
    """Bulk cells stay bit-identical to BGK; interface cells take the RR value."""
    torch.manual_seed(1)
    shape = (6, 6, 6)
    r = 1.0 + 0.05 * torch.randn(shape)
    u = 0.1 * torch.randn(shape)
    v = 0.1 * torch.randn(shape)
    w = 0.1 * torch.randn(shape)
    feq = equilibrium3d(r, u, v, w)
    # A valid non-equilibrium state with both second- and third-order content:
    # the stress reconstruction is massless/momentumless by lattice isotropy,
    # and cx*cy*cz is odd in every coordinate (zero mass/momentum/stress).
    from tensorlbm.collision_d3q19_advanced import reconstruct_second_order_stress_d3q19

    cx, cy, cz = _lattice_views(shape)
    ones = torch.ones(shape)
    f_neq = reconstruct_second_order_stress_d3q19(
        0.02 * ones, -0.01 * ones, 0.015 * ones, 0.008 * ones, 0.0 * ones, 0.0 * ones
    )
    f = feq + f_neq + 0.03 * (cx * cy * cz)
    iface_mask = torch.zeros(shape, dtype=torch.bool)
    iface_mask[2:4, :, :] = True  # a slab of "interface" cells
    out = _mixed_rr_iface(f, feq, 0.7, iface_mask)
    f_bgk = f - (f - feq) / 0.7
    f_rr = _rr_iface_rr_part(f, feq, 0.7)
    bulk = ~iface_mask
    # Locality: the where() selects f_bgk exactly at every non-interface cell.
    assert torch.equal(out[:, bulk], f_bgk[:, bulk])
    # Interface cells carry the RR projection, which differs from BGK there.
    assert torch.equal(out[:, iface_mask], f_rr[:, iface_mask])
    assert not torch.equal(out[:, iface_mask], f_bgk[:, iface_mask])


def _rr_iface_rr_part(f, feq, tau):
    from tensorlbm.collision_d3q19_advanced import collide_regularized_stress_d3q19

    return collide_regularized_stress_d3q19(f, feq, tau)


def test_rr_iface_mixed_collision_conserves_moments() -> None:
    """The mask-mixed collision conserves mass and momentum (lattice isotropy)."""
    torch.manual_seed(2)
    shape = (5, 5, 5)
    r = 1.0 + 0.05 * torch.randn(shape)
    u = 0.1 * torch.randn(shape)
    v = 0.1 * torch.randn(shape)
    w = 0.1 * torch.randn(shape)
    feq = equilibrium3d(r, u, v, w)
    from tensorlbm.collision_d3q19_advanced import reconstruct_second_order_stress_d3q19

    cx, cy, cz = _lattice_views(shape)
    ones = torch.ones(shape)
    f_neq = reconstruct_second_order_stress_d3q19(
        0.02 * ones, -0.01 * ones, 0.015 * ones, 0.008 * ones, 0.0 * ones, 0.0 * ones
    )
    f = feq + f_neq + 0.03 * (cx * cy * cz)
    iface_mask = torch.zeros(shape, dtype=torch.bool)
    iface_mask[2:4, :, :] = True
    out = _mixed_rr_iface(f, feq, 0.8, iface_mask)
    assert torch.allclose(f.sum(0), out.sum(0), atol=1e-6)
    assert torch.allclose((f * cx).sum(0), (out * cx).sum(0), atol=1e-6)
    assert torch.allclose((f * cy).sum(0), (out * cy).sum(0), atol=1e-6)
    assert torch.allclose((f * cz).sum(0), (out * cz).sum(0), atol=1e-6)


def _flat_pool_field(nz=12, ny=12, nx=12, level=8.5):
    z = torch.arange(nz).view(nz, 1, 1)
    fill = (level - z + 0.5).clamp(0, 1)  # flat pool, gas above
    fill = fill.expand(nz, ny, nx).contiguous()
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[[0, -1]] = True
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    rho = (1.0 + 3.0 * (0.02 * (level - z).clamp_min(0))).expand(nz, ny, nx)
    uz = (-0.05 * (z > level + 0.5)).expand(nz, ny, nx)
    fluid = (flags != GAS) & (flags != SOLID)  # liquid + interface carry density
    f = equilibrium3d(rho * fluid, torch.zeros_like(fill), torch.zeros_like(fill), uz)
    mass = init_mass_from_fill(fill, flags)
    return f, fill, flags, solid, mass


def test_free_surface_step_rr_iface_runs_finite_and_diverges() -> None:
    """The solver's rr_iface branch steps a small field: finite, and distinct
    from both the plain BGK and the whole-field RR arms."""
    from tensorlbm.free_surface_lbm import total_liquid_inventory

    f0, fill0, flags0, solid, mass0 = _flat_pool_field()
    inv0 = total_liquid_inventory(f0, fill0, flags0).item()
    finals = {}
    for mode in ("bgk", "rr", "rr_iface"):
        f, fill, flags, mass = f0.clone(), fill0.clone(), flags0.clone(), mass0.clone()
        for _ in range(30):
            f, fill, flags, mass, _ = free_surface_step(
                f, fill, flags, solid, mass=mass, tau=0.55, gz=-0.02,
                surface_tension=0.01, rho_gas=1.0, collision=mode,
            )
        assert torch.isfinite(f).all()
        assert torch.isfinite(mass).all()
        finals[mode] = (f, fill, flags)
    # The interface-local arm must not collapse onto either neighbour arm.
    assert not torch.equal(finals["rr_iface"][0], finals["bgk"][0])
    assert not torch.equal(finals["rr_iface"][0], finals["rr"][0])
    # Bulk locality in action: the interface slab is thin, so the liquid
    # inventory of the interface-local arm drifts far less than the
    # whole-field RR arm (which rewrites every cell's higher-order modes).
    def drift(mode):
        f, fill, flags = finals[mode]
        return abs(total_liquid_inventory(f, fill, flags).item() / inv0 - 1)

    assert drift("rr_iface") < drift("rr")


def test_free_surface_step_default_collision_is_bgk() -> None:
    """Default-off contract: with no arm switches the solver path is plain BGK,
    bit-identical to an explicit collision='bgk' call."""
    f0, fill0, flags0, solid, mass0 = _flat_pool_field()
    outs = []
    for kwargs in ({}, {"collision": "bgk"}):
        f, fill, flags, mass, _ = f0.clone(), fill0.clone(), flags0.clone(), mass0.clone(), None
        for _ in range(10):
            f, fill, flags, mass, _ = free_surface_step(
                f, fill, flags, solid, mass=mass, tau=0.55, gz=-0.02,
                surface_tension=0.01, rho_gas=1.0, **kwargs,
            )
        outs.append(f)
    assert torch.equal(outs[0], outs[1])
