"""Cross-validation tests for :mod:`tensorlbm.warp_shift_streaming`.

Anchor: XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT (upstream
``PorousMediaSimulation/openLBMPM`` commit ``3d84189``, kernel
``ShanChen2D/AccelerateGPU2D.py:1260`` ``calStreamingGPU`` +
``:2733`` ``exchangeDistrGlobalGPU``).

The tests pin the two properties the transplant must have:

* K1 semantic identity -- the vectorised :func:`warp_shift_stream` (non-periodic
  switch setting) reproduces the line-by-line transcription
  :func:`oss_literal_stream` *bit for bit*, i.e. it is a transcription of the
  upstream rule and not a re-derivation;
* K2 physics neutrality -- with the two domain wraps closed
  (``x_periodic = y_periodic = True``) the operator equals the in-repo periodic
  pull streaming of the same field bit for bit, and a Taylor-Green vortex run
  with it reproduces the analytic ``exp(-2 nu k^2 t)`` decay within the owner
  15 % band.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.warp_shift_streaming import (
    bgk_stream_step,
    kinetic_energy,
    oss_literal_stream,
    pull_stream,
    taylor_green_field,
    warp_shift_stream,
)

TILES = (8, 16, 32)


def _fp32_forbidden() -> None:
    torch.set_default_dtype(torch.float64)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("tile", TILES)
def test_vectorised_matches_oss_literal(tile: int) -> None:
    """K1: vectorised form == literal OSS transcription, bit-exact."""
    torch.manual_seed(1234 + tile)
    nx, ny = 96, 40
    f = torch.rand(9, ny, nx, dtype=torch.float64)
    got = warp_shift_stream(f, tile=tile, x_periodic=False, y_periodic=False)
    ref = oss_literal_stream(f, tile=tile)
    assert torch.equal(got, ref), (got - ref).abs().max().item()


@pytest.mark.parametrize("tile", TILES)
def test_periodic_matches_pull_roll(tile: int) -> None:
    """K2: periodic warp-shift == in-repo periodic pull streaming, bit-exact."""
    torch.manual_seed(99 + tile)
    f = torch.rand(9, 32, 128, dtype=torch.float64)
    got = warp_shift_stream(f, tile=tile, x_periodic=True, y_periodic=True)
    ref = pull_stream(f)
    assert torch.equal(got, ref), (got - ref).abs().max().item()


def test_verbatim_transplant_breaks_domain_edges() -> None:
    """The upstream kernel is interior-consistent only: edges keep the stash.

    Documents the defect that forces the ``*_periodic`` fix in the transplant:
    with the upstream edge behaviour the domain edge columns/rows carry a
    population that never arrived, so the field differs from the periodic
    operator and mass is not conserved.
    """
    f = taylor_green_field(64, u0=0.04)
    verbatim = warp_shift_stream(f, tile=32, x_periodic=False, y_periodic=False)
    periodic = warp_shift_stream(f, tile=32, x_periodic=True, y_periodic=True)
    # the interior is identical ...
    assert torch.equal(verbatim[:, 1:-1, 1:-1], periodic[:, 1:-1, 1:-1])
    # ... and the edges are not
    edge = (verbatim - periodic)[:, [0, -1], :].abs().max().item()
    assert edge > 0.0
    assert abs(verbatim.sum().item() - f.sum().item()) > 0.0
    assert abs(periodic.sum().item() - f.sum().item()) < 1e-12


@pytest.mark.parametrize("tile", TILES)
def test_mass_conserved_by_periodic_step(tile: int) -> None:
    f = taylor_green_field(64, u0=0.03)
    m0 = f.sum().item()
    g = f
    for _ in range(10):
        g = bgk_stream_step(g, tau=0.9, tile=tile)
    assert abs(g.sum().item() - m0) / abs(m0) < 1e-12


def test_tile_width_does_not_change_periodic_result() -> None:
    """Tile width is a performance knob only once the wraps are closed."""
    f = taylor_green_field(128, u0=0.04)
    base = warp_shift_stream(f, tile=8, x_periodic=True, y_periodic=True)
    for tile in (16, 32, 64, 128):
        got = warp_shift_stream(f, tile=tile, x_periodic=True, y_periodic=True)
        assert torch.equal(got, base), tile


def test_taylor_green_decay_matches_analytic() -> None:
    """K3: BGK + warp-shift streaming reproduces the analytic TG decay."""
    nx = 64
    u0, tau = 0.04, 0.8
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / nx
    f = taylor_green_field(nx, u0=u0)
    e0 = kinetic_energy(f).item()
    steps = 400
    for _ in range(steps):
        f = bgk_stream_step(f, tau=tau, tile=32)
    e1 = kinetic_energy(f).item()
    got = -math.log(e1 / e0) / (2.0 * steps)
    ref = 2.0 * nu * k * k
    assert abs(got - ref) / ref < 0.15, (got, ref)


def test_bad_shape_and_tile_rejected() -> None:
    with pytest.raises(ValueError):
        warp_shift_stream(torch.zeros(9, 8, 30), tile=16)
    with pytest.raises(ValueError):
        warp_shift_stream(torch.zeros(8, 8, 32), tile=16)