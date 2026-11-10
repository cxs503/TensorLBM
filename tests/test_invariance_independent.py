"""Tests for the independent invariance implementation (mission P170W2B-N04).

The tests are deliberately fast (a few seconds each); the production matrix -
four orientations, three resolutions, two Reynolds numbers - is run by
``tensorlbm.invariance.independent.runner`` on the compute cluster and recorded
in ``artifacts/invariance/independent/``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.invariance.independent import cases, galilean, geometry, lattice, runner


# --- geometry ----------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 14.036, 26.565, 45.0])
def test_tilted_channel_sdf_is_exact_and_resolved_area_matches_width(theta):
    """The discretised fluid fraction must reproduce H/d - the geometry gate."""
    L, width = 128, 24.0
    m, k = cases.RATIONAL_TILTS[theta]
    spacing = L / math.hypot(m, k)
    geom = geometry.TiltedChannelArray(theta, width, spacing)
    phi = geometry.sample(geom, L, L)
    # the resolved fluid area reproduces the channel width to within one lattice
    # unit (the half-cell bias of a staircase-quantised wall); a larger error
    # would mean the geometry, not the flow, is being measured
    width_resolved = (phi <= 0).mean() * spacing
    assert abs(width_resolved - width) <= 1.0

    # signed distance is analytic: sample a line and compare with a direct formula
    x = np.array([0.3, 12.7, 63.1, 100.9])
    y = np.array([7.9, 51.2, 90.4, 3.3])
    n_ = geom.normal
    u = n_[0] * x + n_[1] * y
    s = np.mod(u + 0.5 * spacing, spacing) - 0.5 * spacing
    assert np.allclose(geom.sdf(x, y), np.abs(s) - 0.5 * width)


def test_ray_distance_is_exact_for_a_plane_wall():
    """The sub-grid wall distance must be exact for a planar wall (no staircase)."""
    width, spacing = 20.0, 40.0
    geom = geometry.TiltedChannelArray(26.565, width, spacing)
    gx, gy = np.meshgrid(np.arange(spacing, dtype=float), np.arange(spacing, dtype=float))
    x = gx.ravel()
    y = gy.ravel()
    cx = np.full(x.size, 1.0)
    cy = np.full(x.size, 1.0)
    q = geometry.ray_distance(geom.sdf, x, y, cx, cy)
    checked = 0
    for xi, yi, qi in zip(x, y, q):
        # the bracketing precondition of the bisection: fluid here, solid there
        if not (geom.sdf(xi, yi) < 0.0 < geom.sdf(xi + 1.0, yi + 1.0)):
            continue
        checked += 1
        hit = geom.sdf(xi + qi, yi + qi)
        assert abs(hit) < 1e-9
        assert 0.0 <= qi <= 1.0 + 1e-12
    assert checked >= 5


def test_circle_frame_rotation_is_an_isometry():
    frame_a = geometry.Frame(0.0, origin=(10.0, 20.0))
    frame_b = geometry.Frame(37.0, origin=(10.0, 20.0))
    circle_a = geometry.Circle(6.0, frame_a)
    circle_b = geometry.Circle(6.0, frame_b)
    x = np.array([10.0, 15.0, 3.0])
    y = np.array([20.0, 26.0, 17.0])
    # the same physical body sampled in two frames gives the same SDF at the
    # physical points; the frames differ by a rigid rotation of the lattice
    pts = np.stack([x - 10.0, y - 20.0], axis=1)
    rot = np.array([[math.cos(math.radians(37.0)), -math.sin(math.radians(37.0))],
                    [math.sin(math.radians(37.0)), math.cos(math.radians(37.0))]])
    back = pts @ rot.T
    assert np.allclose(
        circle_b.sdf(back[:, 0] + 10.0, back[:, 1] + 20.0), circle_a.sdf(x, y)
    )


# --- kernel ------------------------------------------------------------------
def test_uniform_flow_is_an_exact_fixed_point():
    """A uniform flow must not decay: otherwise 'Galilean invariance' is moot."""
    res = galilean.uniform_flow_fixed_point(n=24, tau=0.6, boost=(0.05, -0.02), steps=150)
    assert res["velocity_drift_relative"] < 1e-12
    assert res["density_drift_relative"] < 1e-12


def test_equilibrium_moments_and_mass_conservation():
    n = 16
    rng = np.random.default_rng(7)
    rho = 1.0 + 0.01 * rng.standard_normal((n, n))
    ux = 0.01 * rng.standard_normal((n, n))
    uy = 0.01 * rng.standard_normal((n, n))
    f = lattice.equilibrium(rho, ux, uy)
    rho_f, ux_f, uy_f = lattice.macroscopic(f)
    assert np.allclose(rho_f, rho, atol=1e-13)
    assert np.allclose(ux_f, ux, atol=1e-13)
    assert np.allclose(uy_f, uy, atol=1e-13)

    sim = lattice.Simulation(n, n, 0.7)
    sim.f = f
    m0 = sim.f.sum()
    for _ in range(50):
        sim.step()
    assert abs(sim.f.sum() - m0) / m0 < 1e-14


def test_bfl_interpolated_bounce_back_holds_a_rest_fluid():
    """A wall at rest must not inject momentum: static fluid stays static."""
    L = 24
    geom = geometry.TiltedChannelArray(26.565, 8.0, L / math.sqrt(5.0))
    phi = geometry.sample(geom, L, L)
    links = lattice.build_wall_links(phi, geometry.make_ray_distance(geom.sdf))
    sim = lattice.Simulation(L, L, 0.6)
    sim.add_wall_links(links)
    for _ in range(200):
        sim.step()
    _, ux, uy = lattice.macroscopic(sim.f)
    assert np.abs(ux).max() < 1e-12 and np.abs(uy).max() < 1e-12


# --- physics: the declared laws ---------------------------------------------
def test_taylor_green_decay_matches_theory_and_ma2_law():
    res = cases.taylor_green_boost(
        n=24, nu=0.02, amplitude=0.05, mach_list=(0.0, 0.05, 0.10),
        steps=600, sample_every=20, discard=100,
    )
    exact = res["cases"][0]["decay_exponent_exact"]
    for row in res["cases"]:
        assert abs(row["decay_exponent"] / exact - 1.0) < 0.02
    law = res["law"]
    assert abs(law["free_exponent"] - 2.0) < 0.3
    assert law["free_r2"] > 0.99
    # the strict statement: the boosted vortex must *translate*, not deform
    for row in res["cases"][1:]:
        assert row["drift_error_rel"] < 0.02


def test_poiseuille_peak_velocity_and_orientation_invariance_small():
    """Two tilts on a coarse grid: the run is well posed and orientation-bounded.

    The tolerance here is *deliberately loose*: at W = 9 lattice units the wall
    discretisation error of the inclined channel is of order 10% (measured; see
    ``docs/INVARIANCE_INDEPENDENT_VERIFICATION.md`` section 6), so this test only
    guards against regressions/crashes.  The quantitative criterion (3% spread at
    the finest resolution) belongs to the cluster matrix, not to a unit test.
    """
    peaks = {}
    for theta in (0.0, 45.0):
        r = cases.inclined_channel(
            theta, L=48, width=9.0, nu=0.1, u_peak_target=0.02,
            max_steps=2500, tol=1e-7,
        )
        assert abs(r["u_peak_error"]) < 0.15
        peaks[theta] = r["u_peak"]
    spread = abs(peaks[45.0] - peaks[0.0]) / peaks[0.0]
    assert spread < 0.15


# --- report machinery --------------------------------------------------------
def test_aggregate_computes_spreads_and_orders_without_simulation():
    fake = []
    for i, L in enumerate((32, 64, 128)):
        width = 0.25 * L
        for theta in runner.CHANNEL_ANGLES:
            err = 0.02 * (width / 32.0) ** -2.0
            fake.append(
                {
                    "_case": "channel",
                    "_job": f"c{theta}{L}",
                    "theta_requested_deg": theta,
                    "theta_used_deg": theta,
                    "L": L,
                    "width": width,
                    "u_peak": 0.02 * (1.0 - err + 1e-3 * math.sin(theta)),
                    "flux": 0.02 * 0.25 * L * (1.0 - err),
                    "u_peak_error": -err,
                    "flux_error": -err,
                    "residual_velocity_change": 1e-11,
                    "converged": True,
                    "steps": 100,
                }
            )
    block = runner._channel_invariance(fake)
    assert block["finest_level"] == 128
    for entry in block["orders"].values():
        assert abs(entry["observed_order"] - 2.0) < 0.05
    assert block["finest_spread_rel"] < 0.01