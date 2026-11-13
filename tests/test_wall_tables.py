"""Tests for :mod:`tensorlbm.wall_tables` (MF-LBM indirect-addressing wall tables)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import C, W_EXACT64
from tensorlbm.wall_tables import (
    MF_LBM_ISO8_COEFFS,
    MF_LBM_N_SMOOTH_ITERATIONS,
    build_wall_tables,
    dense_footprint_bytes,
    dense_la_weight_extrapolate,
    iso8_wall_normal,
    la_weight_extrapolate,
    table_footprint_bytes,
)


def _channel(nz: int = 10, ny: int = 8, nx: int = 8) -> torch.Tensor:
    """Plane channel: solid rows at both ends of axis 0."""
    walls = torch.zeros(nz, ny, nx, dtype=torch.bool)
    walls[0] = True
    walls[-1] = True
    return walls


def _sphere(n: int = 32, radius: float | None = None) -> torch.Tensor:
    centre = (n - 1) / 2.0
    radius = n / 4.0 if radius is None else radius
    z, y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    return (z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 < radius * radius


# --------------------------------------------------------------------------- #
# tagging / build rules
# --------------------------------------------------------------------------- #
def test_channel_tagging_matches_oss_rules():
    walls = _channel()
    tables = build_wall_tables(walls, periodic_axes=(False, True, True))
    nz, ny, nx = walls.shape
    # every wall cell has a fluid neighbour (both end layers), one fluid layer
    # next to each wall is a fluid-boundary layer
    assert tables.n_solid_boundary == 2 * ny * nx
    assert tables.n_fluid_boundary == 2 * ny * nx
    # la_weight of a wall cell in the plane channel: 5 of the 18 directions
    # point into the fluid -> 1/18 + 4 * 1/36 = 1/6
    assert torch.allclose(
        tables.solid_boundary.la_weight,
        torch.full((tables.n_solid_boundary,), 1.0 / 6.0, dtype=torch.float64),
        atol=1e-15,
    )
    assert set(tables.solid_boundary.n_fluid.tolist()) == {5}
    # fluid-boundary layer sits exactly one cell off each wall
    ijk = tables.fluid_boundary.ijk
    assert set(ijk[:, 0].tolist()) == {1, nz - 2}


def test_la_weight_equals_sum_of_listed_weights():
    walls = _sphere(24)
    tables = build_wall_tables(walls, periodic_axes=(True, True, True))
    rows = tables.solid_boundary.neighbor_rows()
    recomputed = torch.zeros(tables.n_solid_boundary, dtype=torch.float64)
    for row in range(tables.n_solid_boundary):
        for d in rows[row].tolist():
            if d < 0:
                continue
            recomputed[row] += float(W_EXACT64[d])
    assert torch.allclose(recomputed, tables.solid_boundary.la_weight, atol=1e-15)
    assert int(tables.solid_boundary.n_fluid.sum()) == int(tables.solid_boundary.nbr_dir.numel())
    # the CSR row view and the condensing agree
    assert (rows >= 0).sum() == tables.solid_boundary.n_fluid.sum()


def test_link_table_geometry_consistent():
    walls = _sphere(16)
    shape = walls.shape
    tables = build_wall_tables(walls)
    links = tables.links
    assert len(links) > 0
    assert torch.allclose(links.q, torch.full_like(links.q, 0.5))
    flat = torch.arange(int(torch.tensor(shape).prod())).view(*shape)
    for idx in range(len(links)):
        fluid_flat = int(links.fluid_flat[idx])
        solid_flat = int(links.solid_flat[idx])
        d = int(links.direction[idx])
        i, j, k = (
            fluid_flat // (shape[1] * shape[2]),
            (fluid_flat // shape[2]) % shape[1],
            fluid_flat % shape[2],
        )
        assert not bool(walls[i, j, k])
        step = C[d].tolist()
        assert int(flat[i + step[0], j + step[1], k + step[2]]) == solid_flat
        assert bool(walls[i + step[0], j + step[1], k + step[2]])
    # one link per (fluid cell, direction) pair whose neighbour is a *solid cell
    # inside the domain* (walls on a non-periodic face have no in-domain solid
    # partner and are counted in meta instead)
    dense_count = 0
    for d in range(1, C.shape[0]):
        step = C[d].tolist()
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if bool(walls[i, j, k]):
                        continue
                    i2, j2, k2 = i + step[0], j + step[1], k + step[2]
                    if 0 <= i2 < shape[0] and 0 <= j2 < shape[1] and 0 <= k2 < shape[2]:
                        if bool(walls[i2, j2, k2]):
                            dense_count += 1
                    else:
                        assert tables.meta["n_links_outside_domain"] > 0
    assert dense_count == len(links)


# --------------------------------------------------------------------------- #
# la_weight continuation operator
# --------------------------------------------------------------------------- #
def test_continuation_matches_bruteforce_dense_scan():
    walls = _sphere(20)
    tables = build_wall_tables(walls, periodic_axes=(True, True, True))
    torch.manual_seed(7)
    field = torch.randn(walls.shape, dtype=torch.float64)
    got = la_weight_extrapolate(field, tables)
    ref = dense_la_weight_extrapolate(field, walls)
    assert torch.allclose(got, ref, atol=1e-13, rtol=0)
    # fluid cells are untouched
    assert torch.equal(got[~walls], field[~walls])
    # wall cells without a fluid neighbour keep their value
    sb = tables.solid_boundary.flat
    untouched = torch.ones(walls.numel(), dtype=torch.bool)
    untouched[sb] = False
    assert torch.equal(got.view(-1)[untouched], field.view(-1)[untouched])


def test_continuation_of_linear_field_hits_weight_centroid():
    """``phi[x] = sum_n w_n phi[x + c_n] / la_weight`` is exact for linear fields.

    The operator returns the field *at the weight centroid* of the fluid
    neighbours, not at the wall node itself -- this test pins that semantics.
    """
    walls = _sphere(20)
    tables = build_wall_tables(walls, periodic_axes=(True, True, True))
    a = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    b = 0.25
    z, y, x = torch.meshgrid(
        torch.arange(walls.shape[0], dtype=torch.float64),
        torch.arange(walls.shape[1], dtype=torch.float64),
        torch.arange(walls.shape[2], dtype=torch.float64),
        indexing="ij",
    )
    field = a[0] * z + a[1] * y + a[2] * x + b
    got = la_weight_extrapolate(field, tables)
    sb = tables.solid_boundary
    rows = sb.neighbor_rows()
    max_err = 0.0
    for row in range(tables.n_solid_boundary):
        i, j, k = (int(v) for v in sb.ijk[row].tolist())
        weight = 0.0
        moment = torch.zeros(3, dtype=torch.float64)
        for d in rows[row].tolist():
            if d < 0:
                continue
            w = float(W_EXACT64[d])
            step = C[d]
            weight += w
            moment += w * (torch.tensor([i, j, k], dtype=torch.float64) + step.to(torch.float64))
        centroid = moment / weight
        exact = float(a[0] * centroid[0] + a[1] * centroid[1] + a[2] * centroid[2] + b)
        max_err = max(max_err, abs(float(got[i, j, k]) - exact))
    assert max_err < 1e-12


# --------------------------------------------------------------------------- #
# ISO8 wall normal
# --------------------------------------------------------------------------- #
def test_iso8_plane_wall_normal_is_axis_aligned():
    walls = torch.zeros(20, 12, 12, dtype=torch.bool)
    walls[0] = True
    normal = iso8_wall_normal(walls, periodic_axes=(False, True, True))
    got = normal[1, 6, 6]
    assert torch.allclose(got, torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)


def test_iso8_literal_oss_expression_is_anisotropic():
    """Pin the OSS-literal deviation (documented OSS finding, not a regression)."""
    walls = torch.zeros(20, 12, 12, dtype=torch.bool)
    walls[0] = True
    literal = iso8_wall_normal(walls, periodic_axes=(False, True, True), variant="oss_literal")[1, 6, 6]
    projected = iso8_wall_normal(walls, periodic_axes=(False, True, True))[1, 6, 6]
    assert torch.allclose(projected, torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    # the literal expression leaks 40% of the longitudinal component into the
    # two transverse components -> the normal is rotated ~26 deg off-axis
    assert abs(float(literal[1])) > 0.3 and abs(float(literal[2])) > 0.3
    assert abs(float(literal[0])) < 0.9


def test_iso8_sphere_normal_accuracy():
    walls = _sphere(32)
    tables = build_wall_tables(walls, periodic_axes=(True, True, True))
    normals = tables.fluid_boundary.normal
    ijk = tables.fluid_boundary.ijk.to(torch.float64) - (31.0 / 2.0)
    radial = ijk / ijk.norm(dim=1, keepdim=True)
    # normals point along increasing solid fraction: inward for a solid blob
    cos = -(normals * radial).sum(dim=1)
    assert float(cos.mean()) > 0.99
    assert float(cos.min()) > 0.98


def test_smoothing_iteration_count_is_the_oss_constant():
    assert MF_LBM_N_SMOOTH_ITERATIONS == 4
    assert len(MF_LBM_ISO8_COEFFS) == 7
    # unit-normalisation of the projected stencil: 2 * sum_s c_s sum_o o_x^2
    shells_x = [
        sum(o[0] ** 2 for o in shell if o[0] != 0) for shell in
        (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)),
            ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)),
            ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
            ((0, 1, 2), (0, 1, -2), (0, 2, 1), (0, 2, -1), (1, 0, 2), (1, 0, -2),
             (1, 2, 0), (1, -2, 0), (2, 0, 1), (2, 0, -1), (2, 1, 0), (2, -1, 0)),
            ((1, 1, 2), (1, 1, -2), (1, -1, 2), (1, -1, -2), (1, 2, 1), (1, 2, -1),
             (1, -2, 1), (1, -2, -1), (2, 1, 1), (2, 1, -1), (2, -1, 1), (2, -1, -1)),
            ((0, 2, 2), (0, 2, -2), (2, 0, 2), (2, 0, -2), (2, 2, 0), (2, -2, 0)),
        )
    ]
    assert shells_x == [1, 4, 4, 4, 20, 24, 16]
    # the OSS coefficient set is exactly the projected-gradient set: the
    # stencil gain is 1 with no remainder
    gain = 2.0 * sum(c * n for c, n in zip(MF_LBM_ISO8_COEFFS, shells_x))
    assert abs(gain - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# footprint / guards
# --------------------------------------------------------------------------- #
def test_footprint_is_below_dense_and_below_oss_rows():
    walls = _sphere(40, radius=12.0)
    tables = build_wall_tables(walls, periodic_axes=(True, True, True))
    fp = table_footprint_bytes(tables)
    dense = dense_footprint_bytes(walls.shape)
    assert fp["bytes_total_implemented"] < fp["bytes_total_mf_lbm_equivalent"]
    assert fp["bytes_total_implemented"] < dense["bytes_mask_plus_q"]
    assert fp["n_solid_boundary"] == tables.n_solid_boundary


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        build_wall_tables(torch.zeros(4, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        build_wall_tables(
            torch.zeros(4, 4, 4, dtype=torch.bool),
            q_field=torch.zeros(4, 4, dtype=torch.float32),
        )
    with pytest.raises(ValueError):
        iso8_wall_normal(torch.zeros(4, 4, 4, dtype=torch.bool), variant="nope")
    empty = build_wall_tables(torch.zeros(4, 4, 4, dtype=torch.bool), periodic_axes=(True, True, True))
    assert empty.n_solid_boundary == 0 and empty.n_links == 0
    with pytest.raises(ValueError):
        la_weight_extrapolate(torch.zeros(3, 4, 4, dtype=torch.float64), empty)


def test_q_field_is_used_when_supplied():
    walls = _channel()
    q = torch.full(walls.shape, 0.25, dtype=torch.float32)
    tables = build_wall_tables(walls, periodic_axes=(False, True, True), q_field=q)
    assert torch.allclose(tables.links.q, torch.full((tables.n_links,), 0.25))
    assert tables.meta["q_source"] == "supplied_q_field"


def test_all_solid_and_all_fluid_are_degenerate_but_safe():
    solid = torch.ones(5, 5, 5, dtype=torch.bool)
    t_solid = build_wall_tables(solid, periodic_axes=(True, True, True))
    assert t_solid.n_solid_boundary == 0 and t_solid.n_links == 0
    fluid = torch.zeros(5, 5, 5, dtype=torch.bool)
    t_fluid = build_wall_tables(fluid, periodic_axes=(True, True, True))
    assert t_fluid.n_solid_boundary == 0 and t_fluid.n_fluid_boundary == 0
    field = torch.zeros(5, 5, 5, dtype=torch.float64)
    out = la_weight_extrapolate(field, t_fluid)
    assert torch.equal(out, field)