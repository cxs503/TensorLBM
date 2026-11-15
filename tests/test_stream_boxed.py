#!/usr/bin/env python3
"""Tests for :mod:`tensorlbm.stream_boxed` and the bounce-back map used by the
sessile-drop contact-angle cross-validation.

Why these tests exist
---------------------
``docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE-R2-CONVENTION.md`` could not
accept the end-to-end sessile-drop arm because the drop lost ~28 % of its
positive-phase area and a second blob condensed on the far wall.  Root cause
(round 3): the driver hard-coded ``OPP = (0, 3, 2, 1, 4, 7, 8, 5, 6)``, which
maps the two *vertical* D2Q9 directions onto themselves, so the prescribed
bounce-back was not a bounce-back.  These tests pin the invariant:

    reflecting a population must map every direction onto its geometric
    opposite, ``c[OPP[k]] == -c[k]``.

They also cover the sealed (non-periodic) streaming operator that removes the
periodic-seam conduit between the two walls of a closed cavity.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.amr_composite2d import stream_walls
from tensorlbm.solver import C, OPPOSITE_2D, stream
from tensorlbm.stream_boxed import seam_leak_fraction, stream_boxed

OPP_CANONICAL = tuple(int(v) for v in OPPOSITE_2D)
OPP_LEGACY_BUG = (0, 3, 2, 1, 4, 7, 8, 5, 6)


def test_canonical_opposite_map_is_geometric():
    """``OPP[k]`` must be the direction opposite to ``k`` (``c[OPP[k]] == -c[k]``)."""
    for k in range(9):
        assert (C[OPP_CANONICAL[k]] == -C[k]).all(), f"direction {k} broken"


def test_legacy_map_is_not_a_bounce_back():
    """The map used by rounds 1-2 is provably not an opposite map."""
    broken = [k for k in range(9) if not bool((C[OPP_LEGACY_BUG[k]] == -C[k]).all())]
    assert broken == [2, 4], broken


def test_wetting_driver_uses_canonical_map():
    """Regression: the shipped sessile-drop driver must not re-introduce the bug."""
    import inspect

    from tensorlbm import cg_wetting_geometric

    src = inspect.getsource(cg_wetting_geometric.run_sessile_drop_contact_angle)
    assert "OPPOSITE_2D" in src
    assert "OPP = (0, 3, 2, 1, 4, 7, 8, 5, 6)" not in src


def test_boxed_matches_reference_stream_walls():
    """``stream_boxed`` must reproduce the merged numpy half-way bounce-back."""
    ny, nx = 16, 14
    g = torch.Generator().manual_seed(7)
    f = torch.rand(9, ny, nx, dtype=torch.float64, generator=g)
    fluid = torch.ones(ny, nx, dtype=torch.bool)
    fluid[0, :] = False
    fluid[ny - 1, :] = False
    fluid[5, 4:7] = False
    solid = ~fluid
    got = stream_boxed(f, solid).numpy()
    ref, _ = stream_walls(f.numpy(), fluid.numpy())
    assert abs(got[:, fluid.numpy()] - ref[:, fluid.numpy()]).max() < 1e-12


def test_boxed_conserves_fluid_mass():
    ny, nx = 24, 20
    g = torch.Generator().manual_seed(11)
    f = torch.rand(9, ny, nx, dtype=torch.float64, generator=g)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[10, 8:12] = True
    fluid = ~solid
    m0 = float(f[:, fluid].sum())
    for _ in range(400):
        f = stream_boxed(f, solid)
    assert abs(float(f[:, fluid].sum()) - m0) < 1e-9


def test_boxed_has_no_seam_leak():
    """No population may reach a wall through the periodic seam."""
    ny, nx = 24, 20
    g = torch.Generator().manual_seed(13)
    f = torch.rand(9, ny, nx, dtype=torch.float64, generator=g)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    # the top rows carry a completely different state; the interior (and hence
    # the bottom wall) must not see it
    f2 = f.clone()
    f2[:, ny - 1, :] += 5.0
    a = stream_boxed(f, solid)
    b = stream_boxed(f2, solid)
    assert torch.equal(a[:, : ny - 2], b[:, : ny - 2])
    # ... whereas the stock periodic gather *does* move it across the seam, and
    # the stock bounce-back then stores that far-side data in the bottom wall row
    st_a = torch.where(solid.unsqueeze(0), stream(f)[list(OPP_CANONICAL)], stream(f))
    st_b = torch.where(solid.unsqueeze(0), stream(f2)[list(OPP_CANONICAL)], stream(f2))
    assert not torch.equal(st_a[:, 0], st_b[:, 0])
    assert seam_leak_fraction(f, solid) > 0.0


def test_boxed_static_fluid_is_unchanged():
    """A uniform, at-rest state must be a fixed point of streaming."""
    ny, nx = 12, 10
    # any c -> -c symmetric distribution is a fixed point of streaming
    f = torch.full((9, ny, nx), 1.0 / 9.0, dtype=torch.float64)
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    out = stream_boxed(f, solid)
    assert torch.allclose(out[:, ~solid], f[:, ~solid], atol=1e-14)


@pytest.mark.slow
def test_sessile_drop_holds_prescribed_angle():
    """End-to-end: with the canonical map the drop keeps its 60 deg cap.

    This is the acceptance-level check for
    ``XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE``: the commanded angle must be
    recovered within 15 % (owner standard), and --- the round-2 failure mode ---
    no phase may condense on the far wall.
    """
    from tensorlbm.cg_wetting_geometric import run_sessile_drop_contact_angle

    res = run_sessile_drop_contact_angle(theta_deg=60.0, steps=300, ny=72, nx=80,
                                         radius=16.0, sample_every=150, wetting=True)
    assert abs(res["err_pct_cap"]) <= 15.0, res
    assert abs(res["err_pct_circle"]) <= 15.0, res
    assert float(res.get("blob_area", 0.0)) == 0.0, res