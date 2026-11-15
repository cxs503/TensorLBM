"""The cavity acceptance must score against the *bound* Ghia Table II.

Regression tests for the harness defect found on 2026-11-16 (mission m22379):
``benchmarks/xval/aa_fused_stream_collide_ordering_audit.py`` carried its own
hand-typed copy of Ghia et al. (1982).  The Table I copy (``u``) was right, the
Table II copy (``v``) was **scrambled** -- the values were attached to the wrong
``x/L`` stations -- so the ``v``-profile error of the lid-driven cavity could not
fall below ~4.5 % on *any* grid.  That plateau was read as a solver defect in
``docs/xval_aa_pattern_fused_stream_collide_acceptance.md`` ("``L2 u_y`` 4.8 %")
and blocked the ``< 3 %`` gate for the anchor.

What is pinned here
-------------------
* the metric uses the provenance-bearing table
  (``tensorlbm.verification.lid_driven_cavity.reference_data.GHIA_RE100``,
  ``provenance="BOUND"``), not a re-typed copy;
* that table satisfies physical invariants a real Table II must satisfy, and the
  retired copy violates one of them -- so the test has teeth in both directions;
* an **exact** Ghia ``v`` profile scores ~0 against the bound table but ~4.4 %
  against the retired copy: the retired table made ``< 3 %`` unreachable even for
  a perfect solver;
* the centre-line sampling sits on ``x/H = y/H = 1/2`` exactly (the old code
  sampled the cell row ``L // 2``, i.e. ``y/H = 1/2 + 1/(2L)``).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "benchmarks", "xval"))

import aa_fused_stream_collide_ordering_audit as A  # noqa: E402
import aa_fused_cavity_acceptance as ACC  # noqa: E402
from tensorlbm.verification.lid_driven_cavity.reference_data import (  # noqa: E402
    GHIA_RE100,
)

U = 0.1


# --------------------------------------------------------------------------
# 1. single source of truth
# --------------------------------------------------------------------------
def test_u_table_is_the_bound_reference():
    ref = np.column_stack([GHIA_RE100.y, GHIA_RE100.u])
    assert np.array_equal(A.GHIA_U, ref)


def test_v_table_is_the_bound_reference():
    ref = np.column_stack([GHIA_RE100.x, GHIA_RE100.v])
    assert np.array_equal(A.GHIA_V, ref)


def test_reference_is_flagged_bound():
    assert GHIA_RE100.provenance == "BOUND"
    assert len(GHIA_RE100.x) == 17 and len(GHIA_RE100.y) == 17


def test_retired_table_is_really_different():
    """Guards against silently dropping the old copy from the record."""
    assert A.GHIA_V_RETIRED_SCRAMBLED.shape[1] == 2
    assert A.GHIA_V_RETIRED_SCRAMBLED.shape[0] == A.GHIA_V.shape[0] - 1  # no 0.8047
    assert not np.array_equal(A.GHIA_V_RETIRED_SCRAMBLED, A.GHIA_V)
    assert A.GHIA_V_RETIRED_SCRAMBLED[:, 1].max() > 0.0   # had +5e-2 at x=0.9766


# --------------------------------------------------------------------------
# 2. physical invariants of Table II
# --------------------------------------------------------------------------
def test_table2_vanishes_at_the_side_walls():
    x, v = A.GHIA_V[:, 0], A.GHIA_V[:, 1]
    assert v[x == 0.0][0] == 0.0 and v[x == 1.0][0] == 0.0


def test_table2_sign_follows_the_vortex_direction():
    """Lid moves in +x: the fluid rises on the left half (v > 0) and descends on
    the right half (v < 0) at mid height."""
    x, v = A.GHIA_V[:, 0], A.GHIA_V[:, 1]
    left = v[(x > 0.0) & (x < 0.5)]
    right = v[(x > 0.5) & (x < 1.0)]
    assert (left > 0.0).all()
    assert (right < 0.0).all()


def test_retired_table_violates_the_sign_rule_near_the_wall():
    """The retired copy claims v > 0 one cell away from the right wall, where the
    descending jet must approach zero from below.  This is the defect, not taste."""
    x, v = A.GHIA_V_RETIRED_SCRAMBLED[:, 0], A.GHIA_V_RETIRED_SCRAMBLED[:, 1]
    assert v[x == 0.9766][0] > 0.0
    right = v[(x > 0.5) & (x < 1.0)]
    assert not (right < 0.0).all()


def test_table1_endpoints_and_sign_change():
    y, u = A.GHIA_U[:, 0], A.GHIA_U[:, 1]
    assert u[y == 0.0][0] == 0.0 and u[y == 1.0][0] == 1.0
    assert np.count_nonzero(np.diff(u > 0.0)) == 1


# --------------------------------------------------------------------------
# 3. the retired table made < 3 % unreachable even for an exact solution
# --------------------------------------------------------------------------
def _exact_profiles(L: int) -> tuple:
    ys = (np.arange(L) + 0.5) / L
    u = np.interp(ys, A.GHIA_U[:, 0], A.GHIA_U[:, 1]) * U
    v = np.interp(ys, A.GHIA_V[:, 0], A.GHIA_V[:, 1]) * U
    return u, v, ys


def test_exact_v_profile_pays_only_the_wall_endpoint_floor():
    """An exact Ghia profile fed back through the metric must score *at* the
    reported floor and nothing more (the floor itself is the ``x = 0, 1``
    endpoint treatment, which the metric clamps to the last interior node)."""
    _, v, ys = _exact_profiles(64)
    err = ACC.profile_error(v, ys, A.GHIA_V, U)
    assert err == pytest.approx(ACC.metric_floor(64, U)["floor_v"], abs=1e-15)
    # the round trip table -> cell centres -> stations is piecewise linear, so a
    # 1e-2 kink between neighbouring stations survives at the 1e-3 level
    assert ACC.profile_error(v, ys, A.GHIA_V, U, interior=True) < 1e-2


def test_exact_v_profile_fails_against_the_retired_table():
    """The old harness scored a *perfect* solver at ~4.4 %: the metric could not
    distinguish a correct kernel from a broken one on that axis."""
    _, v, ys = _exact_profiles(64)
    err = ACC.profile_error(v, ys, A.GHIA_V_RETIRED_SCRAMBLED, U)
    assert err > 0.03, err
    assert 0.03 < err < 0.08


def test_exact_u_profile_only_pays_the_wall_extrapolation_floor():
    u, _, ys = _exact_profiles(64)
    err = ACC.profile_error(u, ys, A.GHIA_U, U)
    assert err == pytest.approx(ACC.metric_floor(64, U)["floor_u"], abs=1e-15)


# --------------------------------------------------------------------------
# 4. centre-line sampling is exact
# --------------------------------------------------------------------------
@pytest.mark.parametrize("L", [16, 32, 33, 64, 128])
def test_centre_line_weights_land_on_the_mid_plane(L):
    j, w = A._centre_line_weights(L)
    x_line = (j + 0.5) / L * (1.0 - w) + (j + 1.5) / L * w
    assert x_line == pytest.approx(0.5, abs=1e-15)


def test_sampling_uses_the_mid_plane_not_a_cell_row():
    """A field linear in ``y`` is sampled exactly at ``y/H = 1/2`` (value 0),
    whereas the retired row (``L // 2`` of the *padded* array) picks up a
    spurious half-cell offset ``1/(2L)``."""
    L, wt, slope = 32, 1, 4.0
    n = L + 2 * wt
    y = (torch.arange(n) - wt + 0.5) / L
    uy = (slope * (y - 0.5)).view(n, 1).repeat(1, n)     # depends on y only (axis 0)
    ux = torch.zeros(n, n)
    _, uyc, ys = A._sample_profiles_from_moments(ux, uy, L, wt)
    assert np.abs(uyc).max() < 1e-15
    # what the retired row would have reported
    retired = float(uy[L // 2, 0])
    assert abs(retired) == pytest.approx(slope * (1.0 / (2 * L)), abs=1e-12)


def test_sampling_of_u_is_exact_on_the_vertical_centre_line():
    L, wt, slope = 32, 1, 4.0
    n = L + 2 * wt
    x = (torch.arange(n) - wt + 0.5) / L
    ux = (slope * (x - 0.5)).view(1, n).repeat(n, 1)     # depends on x only (axis 1)
    uy = torch.zeros(n, n)
    uxc, _, ys = A._sample_profiles_from_moments(ux, uy, L, wt)
    assert np.abs(uxc).max() < 1e-15


# --------------------------------------------------------------------------
# 5. the acceptance driver inherits the same tables
# --------------------------------------------------------------------------
def test_acceptance_driver_imports_the_same_tables():
    assert ACC.GHIA_U is A.GHIA_U
    assert ACC.GHIA_V is A.GHIA_V