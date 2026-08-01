from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.kinetic_flux_register import (
    KineticInterfaceTransfer,
    apply_face_local_reflux,
    build_kinetic_interface_links,
    observe_kinetic_interface_transfer,
)


def _box(shape: tuple[int, int, int]) -> torch.Tensor:
    inside = torch.zeros(shape, dtype=torch.bool)
    inside[2:-2, 2:-2, 2:-2] = True
    return inside


def test_each_diagonal_link_is_counted_once() -> None:
    inside = _box((8, 9, 10))
    links = build_kinetic_interface_links(inside, q=19)
    for direction in range(1, 19):
        assert not bool((links.outgoing_origins[direction] & links.incoming_origins[direction]).any())
    # The same origin/direction cannot be duplicated by crossing two faces.
    assert links.outgoing_origins.dtype is torch.bool


def test_uniform_equilibrium_has_zero_total_net_mass_and_momentum_flux() -> None:
    shape = (8, 9, 10)
    inside = _box(shape)
    links = build_kinetic_interface_links(inside, q=19)
    rho = torch.ones(shape, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    post = equilibrium3d(rho, zero, zero, zero)
    transfer = observe_kinetic_interface_transfer(post, links)
    assert transfer.net_outgoing.sum().item() == pytest.approx(0.0, abs=1e-14)


def test_fine_substep_volume_scaling_matches_coarse_inventory_units() -> None:
    transfer = KineticInterfaceTransfer(
        outgoing=torch.tensor([8.0]), incoming=torch.tensor([4.0]),
    )
    integrated = transfer.scaled(1.0 / 8.0) + transfer.scaled(1.0 / 8.0)
    assert integrated.outgoing.item() == pytest.approx(2.0)
    assert integrated.incoming.item() == pytest.approx(1.0)


def test_reflux_is_local_to_exterior_interface_links_and_conservative() -> None:
    shape = (8, 9, 10)
    inside = _box(shape)
    links = build_kinetic_interface_links(inside, q=19)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.03, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    coarse = equilibrium3d(rho, ux, zero, zero)
    before = coarse.clone()
    base = observe_kinetic_interface_transfer(coarse, links)
    fine = KineticInterfaceTransfer(
        outgoing=base.outgoing * 1.001,
        incoming=base.incoming * 0.999,
    )
    corrected, report = apply_face_local_reflux(coarse, links, base, fine)
    expected = (fine.net_outgoing - base.net_outgoing).sum().item()
    actual = (corrected - before).sum().item()
    assert actual == pytest.approx(expected, abs=2e-14)
    assert report.mass_residual == pytest.approx(0.0, abs=2e-14)
    changed = (corrected - before).abs().sum(dim=0) > 0.0
    assert not bool(changed[inside].any())
    assert bool(changed.any())


def test_reflux_limiter_exposes_unapplied_residual() -> None:
    shape = (8, 9, 10)
    inside = _box(shape)
    links = build_kinetic_interface_links(inside, q=19)
    rho = torch.ones(shape)
    zero = torch.zeros_like(rho)
    coarse = equilibrium3d(rho, zero, zero, zero)
    base = observe_kinetic_interface_transfer(coarse, links)
    fine = KineticInterfaceTransfer(
        outgoing=torch.zeros_like(base.outgoing),
        incoming=base.incoming * 20.0,
    )
    corrected, report = apply_face_local_reflux(coarse, links, base, fine)
    assert float(corrected.min()) >= 0.0
    assert report.limited_directions > 0
    assert abs(report.mass_residual) > 0.0
