"""Formula-level diagnostics for the Körner free-surface mass update.

These deliberately freeze topology/conversion: they test the population-link
accounting independently of a wave-validation claim.
"""
from __future__ import annotations

import torch

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    _stream19_roll,
    free_surface_step,
)


def _source_flags(flags: torch.Tensor) -> torch.Tensor:
    """flag(x-c_q), matching `_stream19_roll` pull streaming exactly."""
    return torch.stack([
        flags.roll((int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), (0, 1, 2))
        for q in range(19)
    ])


def test_frozen_all_interface_link_exchange_is_pairwise_antisymmetric() -> None:
    """Every q link x<-x-cq cancels its qbar counterpart at x-cq."""
    nz, ny, nx = 3, 4, 5
    flags = torch.full((nz, ny, nx), INTERFACE, dtype=torch.int8)
    # Small nonuniform velocity makes the individual link transfers nonzero,
    # while keeping the mass strictly inside (0, 1), hence no conversion.
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    rho = torch.ones((nz, ny, nx))
    ux = 0.015 * torch.sin(2.0 * torch.pi * x / nx).expand_as(rho)
    zero = torch.zeros_like(rho)
    f_pre = equilibrium3d(rho, ux, zero, zero)
    f_streamed = _stream19_roll(f_pre)

    src_flags = _source_flags(flags)
    # Pull q at x is f_q^*(x-c_q).  The other end of that physical link is
    # the local outgoing f_bar(q)^*(x), not f_bar(q)^*(x-c_q).
    link = 0.5 * (f_streamed - f_pre[OPPOSITE])
    link = torch.where((flags == INTERFACE).unsqueeze(0) & (src_flags == INTERFACE), link, torch.zeros_like(link))

    # The same physical link is represented by (q,x) and (qbar,x-cq).
    paired = torch.stack([
        link[int(OPPOSITE[q])].roll(
            (int(C[q, 2]), int(C[q, 1]), int(C[q, 0])), (0, 1, 2)
        )
        for q in range(19)
    ])
    assert float(link.abs().max()) > 1.0e-6
    assert torch.allclose(link + paired, torch.zeros_like(link), atol=2.0e-8, rtol=0.0)
    assert abs(float(link.sum())) < 2.0e-7


def test_implementation_exchange_matches_pull_link_formula_at_liquid_interface() -> None:
    """A liquid/interface link must use f_q^*(x-cq)-f_barq^*(x)."""
    nz, ny, nx = 3, 4, 6
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int8)
    # Periodic-x topology: I | G | I | L | L | L.  The x=0 interface
    # separates the liquid tail from the periodic gas head.
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3:] = LIQUID
    solid = torch.zeros_like(flags, dtype=torch.bool)
    fill = torch.zeros((nz, ny, nx))
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0
    mass = fill.clone()
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    rho = torch.where(flags == GAS, torch.zeros_like(fill), torch.ones_like(fill))
    ux = 0.015 * torch.sin(2.0 * torch.pi * x / nx).expand_as(rho)
    f_pre = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
    f_streamed = _stream19_roll(f_pre)
    src_flags = _source_flags(flags)
    expected_delta = torch.where(
        (flags == INTERFACE).unsqueeze(0) & (src_flags == LIQUID),
        f_streamed - f_pre[OPPOSITE], torch.zeros_like(f_streamed),
    ).sum(0)
    ledger: dict[str, float] = {}

    free_surface_step(f_pre, fill, flags, solid, mass=mass, tau=1.0, mass_ledger=ledger)
    assert abs(ledger["exchange"] - (float(mass.sum()) + float(expected_delta.sum()))) < 2.0e-6


def test_implementation_has_no_global_exchange_source_on_frozen_interface() -> None:
    """Exercise free_surface_step's actual exchange stencil without ABB/flags changing."""
    nz, ny, nx = 3, 4, 5
    flags = torch.full((nz, ny, nx), INTERFACE, dtype=torch.int8)
    solid = torch.zeros_like(flags, dtype=torch.bool)
    fill = torch.full((nz, ny, nx), 0.5)
    mass = fill.clone()
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    rho = torch.ones_like(fill)
    ux = 0.015 * torch.sin(2.0 * torch.pi * x / nx).expand_as(rho)
    f = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
    ledger: dict[str, float] = {}

    _, _, out_flags, out_mass, _ = free_surface_step(
        f, fill, flags, solid, mass=mass, tau=1.0, mass_ledger=ledger,
    )
    assert torch.equal(out_flags, flags)  # frozen topology premise
    assert abs(ledger["exchange"] - ledger["start"]) < 2.0e-6
    assert abs(float(out_mass.sum()) - float(mass.sum())) < 2.0e-6


def test_gas_link_uses_korner_anti_bounce_back_not_simple_bounce_back() -> None:
    """Körner ABB is feq_q(rho_g,u_g)+feq_qbar(rho_g,u_g)-f_qbar, not f_qbar."""
    nz, ny, nx = 3, 3, 5
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int8)
    # I | G | I | L | I: every D3Q19 liquid/gas link is interface-covered.
    flags[:, :, 0] = INTERFACE
    flags[:, :, 2] = INTERFACE
    flags[:, :, 3] = LIQUID
    flags[:, :, 4] = INTERFACE
    solid = torch.zeros_like(flags, dtype=torch.bool)
    fill = torch.zeros((nz, ny, nx))
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0
    mass = fill.clone()
    rho = torch.where(flags == GAS, torch.full_like(fill, 0.001), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    f = equilibrium3d(rho, zero, zero, zero)

    out, _, _, _, _ = free_surface_step(
        f, fill, flags, solid, mass=mass, tau=1.0, rho_gas=0.001,
    )
    # q=+x at x=2 pulls from the GAS source x=1.  After streaming its
    # opposite is the +x-neighbour's qbar population, here w_q*rho_liquid.
    q, z, y, x0 = 1, 1, 1, 2
    streamed = _stream19_roll(torch.where((flags != GAS).unsqueeze(0), f, torch.zeros_like(f)))
    f_opp = streamed[int(OPPOSITE[q]), z, y, x0]
    feq_g = equilibrium3d(torch.full_like(fill, 0.001), zero, zero, zero)
    expected_abb = feq_g[q, z, y, x0] + feq_g[int(OPPOSITE[q]), z, y, x0] - f_opp

    # This is intentionally a red test for the present simple-copy boundary.
    assert torch.allclose(out[q, z, y, x0], expected_abb, atol=1.0e-7, rtol=0.0)
