"""Smoke test for the README quick-start snippet (English README.md).

The README "Minimal Python API" block is the first code a new user copies.
It once contained ``f.unsqueeze(0).expand(9,-1,-1)``, which no longer matches
the ``stream``/``collide_bgk`` shape contract (both operate on a ``[q, ny, nx]``
distribution tensor).  This test re-executes the snippet verbatim (with a
smaller grid/step count for CI speed) and pins the shape contract so the
documented example cannot silently rot again.

All CPU; expected runtime a few seconds.
"""

import torch

from tensorlbm import collide_bgk, equilibrium, macroscopic, stream


def test_readme_quickstart_snippet_runs():
    """The README snippet must run and stay finite."""
    ny, nx = 32, 64
    f = equilibrium(torch.ones(ny, nx), torch.zeros(ny, nx), torch.zeros(ny, nx))
    for _ in range(50):
        f = collide_bgk(f, tau=0.6)
        f = stream(f)
    assert torch.isfinite(f).all()


def test_readme_quickstart_resting_fluid_stays_at_rest():
    """A fluid initially at rest with no body force must remain at rest.

    Mass conservation (rho == 1) and zero velocity are physical invariants of
    the BGK + periodic-streaming pair; catching a regression here also catches
    wrong tensor layouts that happen to run without raising.
    """
    ny, nx = 32, 64
    f = equilibrium(torch.ones(ny, nx), torch.zeros(ny, nx), torch.zeros(ny, nx))
    for _ in range(50):
        f = collide_bgk(f, tau=0.6)
        f = stream(f)
    rho, ux, uy = macroscopic(f)
    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-5)
    assert float(ux.abs().max()) < 1e-6
    assert float(uy.abs().max()) < 1e-6


def test_stream_and_collide_shape_contract():
    """stream/collide_bgk accept [q, ny, nx] without a leading batch axis.

    This is the contract the broken README snippet violated: ``stream`` reads
    ``f.shape[1]``/``f.shape[2]`` as ny/nx, so a batch-first layout silently
    computes a wrong (or out-of-bounds) streaming pattern.
    """
    ny, nx = 16, 16
    f = equilibrium(torch.ones(ny, nx), torch.zeros(ny, nx), torch.zeros(ny, nx))
    assert f.shape == (9, ny, nx)
    out = stream(f)
    assert out.shape == (9, ny, nx)
    out = collide_bgk(f, tau=0.6)
    assert out.shape == (9, ny, nx)
