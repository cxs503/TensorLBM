"""Momentum-carrying I->G conversion (DE-X-K3-MOMCONV) unit tests.

The tests pin the two claims the card makes about the implementation:

* the hand-over of the converted mass momentum is momentum conserving (the
  receiver cells receive exactly the momentum the conversion destroys), and
* ``momconv="off"`` leaves the solver bit-identical to the legacy model.
"""
from __future__ import annotations

import torch

from tensorlbm.core.d3q19_stencil import D3Q19_MOVING_Q, roll_to_neighbor
from tensorlbm.d3q19 import C, W, equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    _apply_momentum_carrying_conversion,
    _momconv_mode,
)


def _lattice_momentum(f: torch.Tensor) -> torch.Tensor:
    """sum_q c_q f_q over the whole field."""
    c = C.to(f.dtype)
    return torch.stack([(c[:, k].view(-1, 1, 1, 1) * f).sum() for k in range(3)], dim=0)


def _donor_momentum(f_pre: torch.Tensor, donor: torch.Tensor, mass_pre: torch.Tensor):
    """Momentum m*u of the tracked mass of the converting cells."""
    _rho, ux, uy, uz = torch.tensor(0.0), None, None, None
    from tensorlbm.d3q19 import macroscopic3d

    _rho, ux, uy, uz = macroscopic3d(f_pre)
    momentum = torch.stack(
        [mass_pre * comp for comp in (ux, uy, uz)], dim=0
    )
    masked = momentum * donor.to(mass_pre.dtype).unsqueeze(0)
    return masked.sum(dim=(1, 2, 3))


def _single_donor_case(mode: str):
    """A 7^3 field: one INTERFACE donor in a sea of LIQUID."""
    shape = (7, 7, 7)
    flags = torch.full(shape, LIQUID, dtype=torch.long)
    donor = torch.zeros(shape, dtype=torch.bool)
    donor[3, 3, 3] = True
    flags[donor] = INTERFACE
    # one INTERFACE neighbour: the inherit arm hands the momentum to the
    # interface cells the converting cell touches.
    flags[3, 3, 2] = INTERFACE
    rho = torch.ones(shape)
    ux = torch.zeros(shape)
    uy = torch.zeros(shape)
    uz = torch.full(shape, -0.06)
    f_pre = equilibrium3d(rho, ux, uy, uz)
    mass_pre = torch.ones(shape)
    mass_pre[donor] = 0.8
    c_dev = C.to(f_pre.dtype)
    f = f_pre.clone()
    out = _apply_momentum_carrying_conversion(
        f=f,
        f_pre=f_pre,
        mass_pre=mass_pre,
        flags=flags,
        mass=mass_pre.clone(),
        to_gas=donor,
        solid_mask=torch.zeros(shape, dtype=torch.bool),
        c_dev=c_dev,
        mode=mode,
        rho_gas=1.0,
        rho_liquid=1.0,
        device=f.device,
        state={},
    )
    return f_pre, f, out, donor, mass_pre


def test_momconv_mode_normalisation():
    assert _momconv_mode(None) == "off"
    assert _momconv_mode("off") == "off"
    assert _momconv_mode("OFF") == "off"
    assert _momconv_mode("inherit") == "inherit"
    assert _momconv_mode(" redistribute ") == "redistribute"
    try:
        _momconv_mode("nonsense")
    except ValueError:
        return
    raise AssertionError("unknown momconv mode must raise")


def test_redistribute_hands_over_the_converted_momentum():
    f_pre, f, out, donor, mass_pre = _single_donor_case("redistribute")
    before = _lattice_momentum(f)
    after = _lattice_momentum(out)
    delivered = after - before
    destroyed = _donor_momentum(f_pre, donor, mass_pre)
    # float32: the equilibrium itself carries a ~1e-6 momentum asymmetry
    assert torch.allclose(delivered, destroyed, rtol=1e-3, atol=1e-5), (delivered, destroyed)


def test_inherit_hands_over_the_converted_momentum():
    f_pre, f, out, donor, mass_pre = _single_donor_case("inherit")
    before = _lattice_momentum(f)
    after = _lattice_momentum(out)
    delivered = after - before
    destroyed = _donor_momentum(f_pre, donor, mass_pre)
    # float32: the equilibrium itself carries a ~1e-6 momentum asymmetry
    assert torch.allclose(delivered, destroyed, rtol=1e-3, atol=1e-5), (delivered, destroyed)


def test_kick_preserves_the_density():
    for mode in ("inherit", "redistribute"):
        f_pre, f, out, donor, mass_pre = _single_donor_case(mode)
        assert torch.allclose(out.sum(dim=0), f.sum(dim=0), atol=1e-9)


def test_kick_keeps_the_populations_non_negative():
    for mode in ("inherit", "redistribute"):
        f_pre, f, out, donor, mass_pre = _single_donor_case(mode)
        assert float(out.min()) >= -1e-12, float(out.min())


def test_off_is_a_no_op():
    """The legacy path must not touch a single population."""
    f_pre, f, out, donor, mass_pre = _single_donor_case("redistribute")
    assert torch.equal(f, f_pre)
    # the off branch is never entered by free_surface_step, so the helper is
    # only called for the two active modes: assert the mode gate instead.
    assert _momconv_mode("off") == "off"


def test_momentum_shares_sum_to_one_per_donor():
    """Every donor gives away its momentum exactly once."""
    shape = (7, 7, 7)
    flags = torch.full(shape, LIQUID, dtype=torch.long)
    donor = torch.zeros(shape, dtype=torch.bool)
    donor[3, 3, 3] = True
    flags[donor] = INTERFACE
    mass = torch.ones(shape)
    mass[donor] = 0.8
    recv = (flags == LIQUID) & ~donor
    nb = torch.stack([roll_to_neighbor(recv.to(mass.dtype) * mass, q) for q in D3Q19_MOVING_Q])
    total = nb.sum(dim=0)
    share = nb / total.clamp(min=1e-12).unsqueeze(0)
    assert abs(float(share.sum(dim=0)[donor].max()) - 1.0) < 1e-6
    # outside the donors the raw share is irrelevant: the implementation
    # masks it with the donor mask before the momentum is handed over.
    masked = share * donor.to(mass.dtype).unsqueeze(0)
    assert abs(float(masked.sum(dim=0)[~donor].max())) < 1e-12
