"""Tests for the P158W3-N01 certification layer (``tensorlbm.rotating_certification``).

Each test pins one of the measurements the benchmark relies on, and each one is
fast (short integrations on small grids).  The negative result of the node --
that a half-way bounce-back on a sub-grid wall is *first* order, and that the
three obvious fixes do not help -- is itself regression-guarded here, so nobody
can quietly turn it back into a claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.rotating_certification import (
    ChannelPoiseuilleSubgrid,
    CouetteAnnulusSubgrid,
    circular_couette,
    couette_torque_coefficient,
    ekman_spiral_angle,
    fit_couette_AB,
    gci,
    invert_effective_radii,
    mass_drift,
    rotation_rate_sweep,
    torque_from_profile,
)
from tensorlbm.rotating_interior import rotating_channel_analytic

NU = 0.05
TAU = 3.0 * NU + 0.5


# --------------------------------------------------------------------------- #
# gci                                                                          #
# --------------------------------------------------------------------------- #
def test_gci_recovers_an_exact_second_order_sequence():
    """For f(h) = f_ex + C h^2 the observed order must be 2 and the Richardson
    extrapolation must return f_ex to round-off."""
    f_ex = 1.23456789
    C = 0.7
    r = 2.0
    vals = [f_ex + C * (1.0 / k**2) for k in (1, 2, 4)]  # h = 1, 1/2, 1/4
    out = gci(vals, r)
    assert out["monotone"]
    assert out["order"] == pytest.approx(2.0, abs=1e-10)
    assert out["extrapolated"] == pytest.approx(f_ex, abs=1e-12)
    assert 0.0 < out["gci_fine_pct"] < 5.0


def test_gci_flags_oscillatory_triplets():
    """An oscillatory error triplet must not be turned into a fake order."""
    out = gci([1.0, 1.1, 0.97], 1.5)
    assert not out["monotone"]
    assert math.isnan(out["order"])


# --------------------------------------------------------------------------- #
# analytic anchors                                                            #
# --------------------------------------------------------------------------- #
def test_circular_couette_reduces_to_rigid_body_and_zero_torque():
    c = circular_couette(np.array([12.0, 18.0, 24.0]), 12.0, 24.0, 0.003, 0.003)
    assert c["B"] == pytest.approx(0.0, abs=1e-18)
    assert c["A"] == pytest.approx(0.003)
    eff = invert_effective_radii(c["A"], c["B"], 0.003, 0.003)
    assert not eff["valid"]  # no shear => no wall-position information


def test_effective_radius_inversion_is_exact_for_the_nominal_geometry():
    ri, ro, wi, wo = 12.0, 24.0, 0.004, -0.002
    c = circular_couette(np.array([ri, ro]), ri, ro, wi, wo)
    eff = invert_effective_radii(c["A"], c["B"], wi, wo)
    assert eff["ri_eff"] == pytest.approx(ri, rel=1e-12)
    assert eff["ro_eff"] == pytest.approx(ro, rel=1e-12)


def test_torque_identity_is_r_independent_and_matches_analytic():
    """T(r) from the exact profile must be constant and equal -4 pi rho nu L B."""
    ri, ro, wi, wo = 12.0, 24.0, 0.004, 0.0
    r = np.linspace(ri + 1.0, ro - 1.0, 41)
    c = circular_couette(r, ri, ro, wi, wo)
    out = torque_from_profile(r, c["u_theta"], NU)
    T_ana = -4.0 * math.pi * NU * c["B"]
    assert out["T_mean_mid"] == pytest.approx(T_ana, rel=2e-3)
    assert out["spread"] < 1e-3
    assert couette_torque_coefficient(T_ana, NU) == pytest.approx(-2.0 * c["B"] / NU, rel=1e-12)  # T = -4 pi mu L B


# --------------------------------------------------------------------------- #
# sub-grid wall machinery + the documented negative result                     #
# --------------------------------------------------------------------------- #
def _plane(scheme, y_b, y_t, ny=24, nx=3, G=1e-5, steps=8000):
    sim = ChannelPoiseuilleSubgrid((ny, nx), TAU, G, y_b, y_t, wall_scheme=scheme)
    sim.run_until_steady(tol=1e-11, max_steps=steps, check_every=100)
    p = sim.profile()
    an = sim.analytic(p["y"])
    return float(np.max(np.abs(p["ux"] - an)) / max(abs(an).max(), 1e-30))


def test_plane_channel_aligned_wall_reproduces_poiseuille():
    err = _plane("halfway", 6.5, 17.5)
    assert err < 1.5e-2


def test_subgrid_wall_is_first_order_not_second_and_not_rescued():
    """The documented negative result: moving the wall off the half link costs
    O(h) accuracy, and none of the shipped/attempted stencils removes it.

    The assertion is deliberately two-sided -- it fails if someone "fixes" the
    wall treatment without updating the benchmark record (accuracy recovered by
    more than an order of magnitude) *and* if the error grows beyond the
    measured first-order level."""
    err_aligned = _plane("halfway", 6.5, 17.5)
    err_subgrid = _plane("halfway", 6.3, 17.7)
    assert err_subgrid > 5.0 * err_aligned
    assert err_subgrid < 2.0e-1


def test_subgrid_wall_error_scales_like_h():
    """The same 0.2 h wall offset at two resolutions: the error must *halve*,
    i.e. first order -- that is the quantitative form of the negative result."""
    coarse = _plane("halfway", 6.2, 17.6, ny=24, G=1e-5)
    fine = _plane("halfway", 12.4, 35.2, ny=48, G=1e-5 * (11.0 / 22.8) ** 2)
    ratio = coarse / fine
    assert 1.6 < ratio < 2.6, (coarse, fine)


def test_moving_wall_drive_has_the_sign_of_the_wall_velocity():
    """A wall moving in +x must drag the channel towards +x (sign guard)."""
    sim = ChannelPoiseuilleSubgrid((24, 3), TAU, 0.0, 6.5, 17.5, u_top=1e-2)
    sim.run_until_steady(tol=1e-11, max_steps=8000, check_every=100)
    p = sim.profile()
    assert p["ux"].mean() > 0.0
    err = float(np.max(np.abs(p["ux"] - p["ux"][-1] * (p["y"] - 6.5) / 11.0))
                / max(abs(p["ux"]).max(), 1e-30))
    assert err < 5e-2


# --------------------------------------------------------------------------- #
# annulus solver                                                              #
# --------------------------------------------------------------------------- #
def test_azimuthal_projection_convention():
    """u_theta must be ``(r x u)_z / r`` (counter-clockwise positive)."""
    sim = CouetteAnnulusSubgrid((40, 40), TAU, 8.0, 16.0, omega_i=0.0)
    yy, xx = sim.yy, sim.xx
    om = 0.01
    sim.initialize_from_velocity(-om * sim.dy, om * sim.dx)  # rigid body, CCW
    prof = sim.azimuthal_profile()
    rb = np.asarray(prof["r_bin"])
    tb = np.asarray(prof["u_theta"])
    ok = np.isfinite(tb) & (rb > 9.0) & (rb < 15.0)
    # floor(d) shells: the mean radius of a shell differs from k+0.5 by O(1/k),
    # so the tolerance guards the *sign* (a flip would give ~2, not 0.5 %)
    rel = np.abs(tb[ok] - om * rb[ok]) / (om * rb[ok])
    assert rel.max() < 2e-2


def test_co_rotating_annulus_keeps_the_rigid_body_state():
    """Both cylinders at the same angular velocity: the exact state is
    rigid-body rotation, so the solver must not drift away from it."""
    om = 0.002
    sim = CouetteAnnulusSubgrid((40, 40), TAU, 8.0, 16.0, omega_i=om, omega_o=om)
    r = np.maximum(sim.rad, 1e-12)
    c = circular_couette(r, 8.0, 16.0, om, om)
    sim.initialize_from_velocity(-c["u_theta"] * sim.dy / r, c["u_theta"] * sim.dx / r)
    sim.step(400)
    prof = sim.azimuthal_profile()
    rb = np.asarray(prof["r_bin"])
    tb = np.asarray(prof["u_theta"])
    ok = np.isfinite(tb) & (rb > 9.0) & (rb < 15.0)
    err = np.max(np.abs(tb[ok] - om * rb[ok])) / (om * 15.0)
    assert err < 2e-2
    tor = torque_from_profile(rb[ok], tb[ok], NU)
    # shell-averaged staircase profile: the finite-difference torque does not
    # vanish exactly, it is small *relative to the differential-case torque*
    assert abs(tor["T_mean_mid"]) < 5e-3 * 4.0 * math.pi * NU * 0.768


def test_inner_driven_annulus_matches_the_exact_couette_profile():
    """Short integration, chained from the analytic state: the solver must stay
    on the exact solution (no spurious radial flow, no frame drift)."""
    ri, ro, om = 8.0, 16.0, 0.002
    sim = CouetteAnnulusSubgrid((40, 40), TAU, ri, ro, omega_i=om, omega_o=0.0)
    r = np.maximum(sim.rad, 1e-12)
    c = circular_couette(r, ri, ro, om, 0.0)
    sim.initialize_from_velocity(-c["u_theta"] * sim.dy / r, c["u_theta"] * sim.dx / r)
    sim.step(800)
    prof = sim.azimuthal_profile()
    rb = np.asarray(prof["r_bin"])
    tb = np.asarray(prof["u_theta"])
    ub = np.asarray(prof["u_rad"])
    ok = np.isfinite(tb) & (rb > ri + 1.5) & (rb < ro - 1.5)
    ana = circular_couette(rb[ok], ri, ro, om, 0.0)["u_theta"]
    err = float(np.max(np.abs(tb[ok] - ana)) / np.max(np.abs(ana)))
    assert err < 1.0e-1  # ri = 8 grid: the 8-node gap is coarsely resolved
    assert float(np.nanmax(np.abs(ub[ok])) / np.max(np.abs(ana))) < 5e-3


def test_mass_drift_helper():
    rho = np.ones((4, 4))
    assert mass_drift(rho) == pytest.approx(0.0)
    rho[0, 0] += 0.5
    assert mass_drift(rho) == pytest.approx(0.5 / 16.0)


# --------------------------------------------------------------------------- #
# Ekman spiral angle                                                          #
# --------------------------------------------------------------------------- #
def test_ekman_spiral_angle_is_45_degrees_at_the_wall():
    """The classical Ekman layer: the *wall* shear is 45 deg from the geostrophic
    velocity.  Sampled at 0.1 node spacing so that the finite measurement
    distance is negligible (in the benchmark, delta = 6..16 nodes is used and the
    angle is extrapolated to 1/delta -> 0 instead)."""
    h, Om, G = 40.0, 0.002, 8e-5
    y = np.arange(1, 401, dtype=float) * 0.1 - h  # first sample 0.1 node off the wall
    ux, uz = rotating_channel_analytic(y, h, G, Om, NU)
    ang = ekman_spiral_angle(y, ux, uz, h, exact=(ux, uz))
    # residual 1.1 deg = the measurement distance 0.1 node over delta = 5 nodes
    assert ang["angle_deg"] == pytest.approx(45.0, abs=2.0)
    assert ang["angle_abs_err_deg"] < 1e-9  # identical operator on both sides
    # the angle must rotate as the measurement point moves into the layer
    far = 1.0 / math.sqrt(NU / Om)
    y2 = y + 4.0 * far  # deep inside the layer the angle has rotated clearly
    ux2, uz2 = rotating_channel_analytic(y2, h, G, Om, NU)
    ang2 = ekman_spiral_angle(y2, ux2, uz2, h)
    assert abs(ang2["angle_deg"] - ang["angle_deg"]) > 5.0
    # invariance control: rotating the whole (u_par, u_perp) field rigidly must
    # leave the angle unchanged (catches axis swaps / projection bugs)
    phi = math.radians(30.0)
    p_rot = ux * math.cos(phi) - uz * math.sin(phi)
    q_rot = ux * math.sin(phi) + uz * math.cos(phi)
    ang3 = ekman_spiral_angle(y, p_rot, q_rot, h)
    assert ang3["angle_deg"] == pytest.approx(ang["angle_deg"], abs=1e-9)


# --------------------------------------------------------------------------- #
# stability sweep                                                             #
# --------------------------------------------------------------------------- #
def test_rotation_rate_sweep_detects_the_blowup():
    """A large centrifugal rate must be reported unstable, a small one stable."""

    def make(om):
        ri = 6.0
        u_wall = 11.52 * NU / ri
        sim = CouetteAnnulusSubgrid((28, 28), TAU, ri, 2 * ri, omega_i=u_wall / ri,
                                    omega_o=0.0, omega_frame=float(om))
        r = np.maximum(sim.rad, 1e-12)
        c = circular_couette(r, ri, 2 * ri, u_wall / ri, 0.0)
        sim.initialize_from_velocity(-c["u_theta"] * sim.dy / r, c["u_theta"] * sim.dx / r)
        return sim

    rows = rotation_rate_sweep(make, (0.002, 0.5), max_steps=600, tol=1e-9)
    assert rows[0]["stable"] is True
    assert rows[1]["stable"] is False
    assert rows[1]["centrifugal_over_cs2"] > 1.0