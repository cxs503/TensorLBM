"""Invariants of the volumetric-solid closures (``tensorlbm.volume_solid``).

The tests are written as *identities*, not as smoke runs: in the binary limit
both closures must reproduce the repository half-way bounce-back bit for bit,
the total solid volume must survive the mask smoothing, and the reconstructed
interface distance must match the analytic distance of the disc that generated
the fractions.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries import bounce_back_cells, compute_obstacle_forces
from tensorlbm.solver import stream
from tensorlbm.volume_solid import (
    blur_mask,
    interface_bounce_back,
    mask_scale_to_sigma,
    partial_bounce_back,
    partial_bounce_back_force,
    signed_distance_from_fraction,
    vof_circle,
)


def _state(ny: int, nx: int, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((9, ny, nx), generator=gen, dtype=torch.float64) + 0.1


def _binary_disc(ny: int, nx: int, cy: float, cx: float, r: float) -> torch.Tensor:
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1) - cy
    x = torch.arange(nx, dtype=torch.float64).view(1, nx) - cx
    return ((y * y + x * x) <= r * r).to(torch.float64)


def test_vof_volume_matches_analytic_disc():
    m = vof_circle(64, 48, 31.3, 23.7, 8.4)
    analytic = math.pi * 8.4**2
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert abs(float(m.sum()) - analytic) / analytic < 2e-3


def test_blur_preserves_solid_volume():
    m = vof_circle(64, 48, 31.3, 23.7, 8.4)
    blurred = blur_mask(m, mask_scale_to_sigma(0.5, 1.0))
    assert abs(float(blurred.sum()) - float(m.sum())) / float(m.sum()) < 1e-6
    # the *interface band* must actually become diffuse (the deep interior of
    # a resolvable body stays full, which is the point)
    band_sharp = ((m > 0.02) & (m < 0.98)).sum()
    band_soft = ((blurred > 0.02) & (blurred < 0.98)).sum()
    assert int(band_soft) > 2 * int(band_sharp)
    assert mask_scale_to_sigma(1.0) == 0.0


def test_interface_reconstruction_error_below_half_cell():
    ny = nx = 64
    cy, cx, r = 31.7, 31.2, 10.3
    m = vof_circle(nx, ny, cx, cy, r)
    phi = signed_distance_from_fraction(m)
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, nx)
    exact = r - torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    band = exact.abs() < 1.0
    err = (phi - exact).abs()[band]
    assert float(err.max()) < 0.6
    assert float(err.median()) < 0.15


@pytest.mark.parametrize("sigma", [0.0, 0.5])
def test_partial_bounce_back_is_halfway_bb_in_binary_limit(sigma):
    ny, nx = 24, 32
    m = _binary_disc(ny, nx, 12.0, 16.0, 5.0)
    m = blur_mask(m, sigma) if sigma > 0 else m
    solid = m >= 0.5
    f = _state(ny, nx)
    expected = bounce_back_cells(f, solid)
    got = partial_bounce_back(f, torch.where(solid, torch.ones_like(m), torch.zeros_like(m)))
    assert torch.equal(got, expected)


def test_interface_bounce_back_is_halfway_bb_in_binary_limit():
    """Binary mask => q = 1/2 => the closure is the fluid-side half-way rule.

    The reference is written out by hand (no shared code with the closure):
    for every fluid node whose neighbour in direction ``i`` is solid, the
    incoming population in direction ``j = opposite(i)`` is the post-collision
    ``f_i`` of the fluid node itself.
    """
    ny, nx = 24, 32
    # a plane (face-aligned) wall: every crossing sits at exactly q = 1/2, so
    # the closure must reproduce the half-way rule cell by cell.  (On a
    # staircase circle the corners legitimately carry q != 1/2, which is the
    # sub-cell geometry the closure is *for*.)
    m = torch.zeros(ny, nx, dtype=torch.float64)
    m[:10, :] = 1.0
    solid = m >= 0.5
    phi = signed_distance_from_fraction(m)
    f_post = _state(ny, nx)
    streamed = stream(f_post)
    got = interface_bounce_back(streamed, f_post, phi)
    opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    cs = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    ref = streamed.clone()
    for i in range(1, 9):
        cy, cx = cs[i][1], cs[i][0]
        solid_b = torch.roll(solid, (-cy, -cx), (0, 1))  # solid at x + c_i
        crossing = (~solid) & solid_b
        ref[opp[i]] = torch.where(crossing, f_post[i], ref[opp[i]])
    fluid = ~solid
    assert torch.equal(got[:, fluid], ref[:, fluid])


def test_momentum_exchange_identity_with_hard_wall_estimator():
    ny, nx = 24, 32
    solid = _binary_disc(ny, nx, 12.0, 16.0, 5.0) >= 0.5
    fluid = ~solid
    ring = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )
    f = _state(ny, nx)
    fx_ref, fy_ref = compute_obstacle_forces(f, ring)
    m = solid.to(torch.float64) * ring.to(torch.float64)
    fx, fy = partial_bounce_back_force(f, m)
    assert abs(float(fx) - float(fx_ref)) < 1e-12
    assert abs(float(fy) - float(fy_ref)) < 1e-12


def test_partial_bounce_back_conserves_mass_and_charges_momentum():
    ny, nx = 24, 32
    m = vof_circle(nx, ny, 16.3, 12.2, 5.1)
    f = _state(ny, nx)
    out = partial_bounce_back(f, m)
    assert abs(float(out.sum() - f.sum())) < 1e-10
    fx, fy = partial_bounce_back_force(f, m)
    # momentum lost by the fluid equals the force handed to the solid
    c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]).to(torch.float64)
    before = (f.sum(dim=(1, 2))[:, None] * c).sum(dim=0)
    after = (out.sum(dim=(1, 2))[:, None] * c).sum(dim=0)
    assert abs(float(after[0] - before[0]) + float(fx)) < 1e-10
    assert abs(float(after[1] - before[1]) + float(fy)) < 1e-10


def test_soft_mask_slip_is_what_the_interface_closure_removes():
    """A diffuse mask must not move the reconstructed interface.

    ``mask_scale`` only sharpens the volume representation; the interface
    distance recovered from the fractions has to stay put, which is precisely
    what the amplitude closure is blind to (its reflectivity follows ``m``).
    """
    ny = nx = 64
    cy, cx, r = 31.7, 31.2, 10.3
    m = vof_circle(nx, ny, cx, cy, r)
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, nx)
    exact = r - torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    band = exact.abs() < 1.0
    for scale in (1.0, 0.75, 0.5):
        blurred = blur_mask(m, mask_scale_to_sigma(scale, 1.0))
        phi = signed_distance_from_fraction(blurred)
        err = (phi - exact).abs()[band]
        assert float(err.median()) < 0.2, scale
        # and the interface position itself (the zero crossing) is preserved
        assert abs(float((blurred - 0.5).sum()) - float((m - 0.5).sum())) / float(band.sum()) < 0.05