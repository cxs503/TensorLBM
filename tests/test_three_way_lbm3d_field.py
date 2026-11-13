"""Round-2 (G5-07) interface tests: the transfer driven by a *real* D3Q19 field.

The field itself is a tiny D=6, 8-step run — this file tests the hand-off chain
(per-link ledger -> facet traction -> Q1 hex nodes) plus the added-mass guard and
the driven-response ledger, not the field accuracy (that is gated in
``scripts/three_way_g507_field_gates.py``).
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tensorlbm.three_way.interface3d import (
    DTYPE,
    HexGrid3D,
    added_mass_ratio,
    march_one_dof,
    staggered_spectral_radius,
)
from tensorlbm.three_way.lbm3d_field import (
    growth_rate,
    march_driven,
    run_sphere_field,
    transfer_gates,
)


@pytest.fixture(scope="module")
def field():
    return run_sphere_field(D=6, steps=8, sample=4, surface_levels=[(8, 16)])


def test_distribution_is_the_library_ledger(field):
    """The traction the interface consumes must be the verified force caliber."""
    assert field.closure["n_links"] > 100
    assert field.closure["rel_diff"] < 1e-4


def test_bin_conserves_resultant_exactly(field):
    load = field.load
    dist = torch.tensor(field.closure["distribution_total"], dtype=DTYPE)
    rel = float(torch.linalg.norm(load.forces.sum(0) - dist)) / float(torch.linalg.norm(dist))
    assert rel < 1e-10
    assert load.coverage > 0.5
    assert float(load.forces.sum(0)[0]) > 0.0


def test_transfer_invariants_on_real_traction(field):
    load, R, c = field.load, field.radius, field.centre
    grid = HexGrid3D(
        origin=(c[0] - R - 1.0, c[1] - R - 1.0, c[2] - R - 1.0),
        extent=(2 * R + 2.0,) * 3,
        n_cells=(6, 6, 6),
    )
    m_ref = [c[0], c[1], c[2] - 2.0 * R]

    def vel(nodes):
        r = nodes - torch.tensor(m_ref, dtype=DTYPE)
        w = torch.tensor([0.3, -0.7, 1.1], dtype=DTYPE)
        return torch.cross(w.expand_as(r), r, dim=1) + 1e-3 * r * r  # non-affine term

    out = transfer_gates(load, grid, origin=c, moment_origin=m_ref, velocity=vel)
    assert out["resultant_rel_err"] < 1e-10
    assert out["moment_rel_err"] < 1e-10
    assert out["work_rel_err"] < 1e-10


def test_frame_negative_control_is_detected(field):
    """A one-cell frame error must be visible — otherwise the gate has no teeth."""
    load, R, c = field.load, field.radius, field.centre
    grid = HexGrid3D(
        origin=(c[0] - R - 1.0, c[1] - R - 1.0, c[2] - R - 1.0),
        extent=(2 * R + 2.0,) * 3,
        n_cells=(6, 6, 6),
    )
    m_ref = torch.tensor([c[0], c[1], c[2] - 2.0 * R], dtype=DTYPE)
    good = grid.distribute_surface(load.surface, load.traction)
    m_good = torch.linalg.cross(grid.nodes - m_ref, good, dim=1).sum(0)
    shifted = dataclasses.replace(load.surface, z=load.surface.z + 1.0)
    bad = grid.distribute_surface(shifted, load.traction)
    m_bad = torch.linalg.cross(grid.nodes - m_ref, bad, dim=1).sum(0)
    rel = float(torch.linalg.norm(m_bad - m_good)) / float(torch.linalg.norm(m_good))
    assert rel > 1e-2


def test_added_mass_guard_on_real_geometry(field):
    vol = 4.0 / 3.0 * math.pi * field.radius**3
    mu_ice = added_mass_ratio(917.0 * vol, 1000.0, vol, 0.5)
    mu_light = added_mass_ratio(300.0 * vol, 1000.0, vol, 0.5)
    assert mu_ice < 1.0 < mu_light
    lam = staggered_spectral_radius(mu_light, 0.02)
    grew = growth_rate(march_one_dof(mu_light, 0.02, 300, 1.0))
    assert abs(grew - math.log(lam)) / abs(math.log(lam)) < 5e-3


def test_driven_ledger_closes_and_converges():
    f = [1.0] + [0.4 + 0.6 * math.exp(-s / 15.0) for s in range(1, 60)]
    coarse = march_driven(0.3, 0.02, f)
    fine = march_driven(0.3, 0.01, f)
    assert coarse["closure_max"] < 5e-3
    assert fine["closure_max"] < coarse["closure_max"]
    assert fine["w_scale"] > 0.0
