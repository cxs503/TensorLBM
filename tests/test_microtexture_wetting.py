"""Tests for the micro-textured wetting surface (plan-380 n04).

Coverage
--------
* closed-form geometry vs the voxelised mask (``f_s``, ``r``),
* Cassie-Baxter / Wenzel closures, their energy-based crossover, break-through
  pressure scaling,
* the *physical* contact-angle convention of :func:`apparent_angle` on synthetic
  spherical caps, and the volume conservation of :func:`cap_radius`,
* short LBM smoke runs on a textured wall (static Cassie drop, impact
  observables, hydrostatic impalement monotonicity),
* the scaling-fit helpers.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.contact_line.engine import Channel, tanh_profile
from tensorlbm.contact_line.microtexture import (
    PillarTexture,
    TexturedChannel,
    apparent_angle,
    breakthrough_pressure,
    cap_radius,
    cassie_wenzel_crossover_deg,
    cos_theta_cassie_baxter,
    cos_theta_wenzel,
    measure_voxel_geometry,
    predicted_wetting_state,
    run_static_angle,
)
from tensorlbm.contact_line.microtexture_impact import ImpactRunner, scaling_fits


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pitch,width,height",
    [(16, 8, 20), (16, 4, 20), (16, 12, 20), (12, 6, 0)],
)
def test_voxel_geometry_matches_closed_form(pitch: int, width: int, height: int) -> None:
    tex = PillarTexture(pitch=pitch, pillar_width=width, height=height, theta_y_deg=120.0)
    ch = TexturedChannel(nx=4 * pitch, ny=64, wall_rows=2, texture=tex)
    m = measure_voxel_geometry(ch)

    # a flat wall (H = 0) is fully solid at its top row; pillars cover D/P
    assert m["f_s_voxel"] == pytest.approx(1.0 if height == 0 else width / pitch, abs=1e-12)
    # the Wenzel state wets pillar tops, side walls *and* the groove floor
    assert m["r_closed"] == pytest.approx(1.0 + 2.0 * height / pitch)
    if height == 0:  # flat wall: f_s = r = 1 -> Cassie and Wenzel coincide
        assert m["f_s_voxel"] == pytest.approx(1.0)
    assert m["r_voxel"] == pytest.approx(m["r_closed"], rel=1e-12)
    assert m["r_rel_error"] < 1e-12


def test_roughness_formula_limits() -> None:
    flat = PillarTexture(pitch=16, pillar_width=8, height=0, theta_y_deg=100.0)
    assert flat.roughness == pytest.approx(1.0)
    assert flat.gap == 8
    rough = PillarTexture(pitch=16, pillar_width=8, height=40, theta_y_deg=100.0)
    assert rough.roughness == pytest.approx(6.0)


def test_textured_channel_mask_and_planes() -> None:
    tex = PillarTexture(pitch=16, pillar_width=8, height=20, theta_y_deg=130.0)
    ch = TexturedChannel(nx=64, ny=64, wall_rows=2, texture=tex)
    solid = ch.solid_mask()
    assert ch.y_wall() == pytest.approx(21.5)
    assert ch.y_base() == pytest.approx(1.5)
    # base slab
    assert bool(solid[:2, :].all())
    # pillar columns (off = 4) run from row 2 to row 21
    assert bool(solid[2:22, 4:12].all())
    assert not bool(solid[2:22, 12:16].any())
    groove = ch.groove_mask()
    assert groove.sum() == 4 * 8 * 20  # 4 pitches, 8 free columns, 20 rows


def test_channel_requires_periodic_pitch() -> None:
    tex = PillarTexture(pitch=16, pillar_width=8, height=10, theta_y_deg=120.0)
    with pytest.raises(ValueError):
        TexturedChannel(nx=50, ny=64, wall_rows=2, texture=tex)


# ---------------------------------------------------------------------------
# closures
# ---------------------------------------------------------------------------


def test_cassie_baxter_and_wenzel_limits() -> None:
    f_s, r = 0.5, 3.5
    # fully wetting intrinsic angle -> complete wetting in both states
    assert cos_theta_cassie_baxter(1.0, f_s) == pytest.approx(0.0)
    assert cos_theta_cassie_baxter(-1.0, f_s) == pytest.approx(-1.0)
    assert cos_theta_wenzel(1.0, r) == pytest.approx(r)  # caller saturates at 1


def test_crossover_matches_energy_criterion() -> None:
    """Cassie is the energy minimum iff ``cos theta_Y < (f_s-1)/(r-f_s)``."""
    for f_s, r in ((0.5, 3.5), (0.25, 3.5), (0.75, 3.5)):
        # P = 16 and H = 20 give r = 1 + 2H/P = 3.5; width = f_s P
        tex_kw = dict(pitch=16, pillar_width=int(round(f_s * 16)), height=20)
        tex = PillarTexture(theta_y_deg=120.0, **tex_kw)
        assert tex.solid_fraction == pytest.approx(f_s)
        assert tex.roughness == pytest.approx(r)
        theta_c = cassie_wenzel_crossover_deg(f_s, r)
        # E_C - E_W = (1-f_s) sigma + (r-f_s) sigma cos(theta_Y): sign flip at theta_c
        for theta in (theta_c - 5.0, theta_c + 5.0):
            e = (1.0 - f_s) + (r - f_s) * math.cos(math.radians(theta))
            state = predicted_wetting_state(PillarTexture(theta_y_deg=theta, **tex_kw))
            assert (state["state_predicted"] == "cassie") == (e < 0.0)
        assert (
            predicted_wetting_state(
                PillarTexture(theta_y_deg=theta_c + 1.0, **tex_kw)
            )["state_predicted"]
            == "cassie"
        )
        assert (
            predicted_wetting_state(
                PillarTexture(theta_y_deg=theta_c - 1.0, **tex_kw)
            )["state_predicted"]
            == "wenzel"
        )


def test_breakthrough_pressure_scaling() -> None:
    sigma = 0.05611
    textures = [
        PillarTexture(pitch=16, pillar_width=d, height=20, theta_y_deg=120.0)
        for d in (4, 8, 12)
    ]
    p = [breakthrough_pressure(t, sigma)["p_c_abs_2d"] for t in textures]
    assert p == sorted(p)  # monotone increasing with the solid fraction (gap shrinks)
    for t, val in zip(textures, p):
        assert val == pytest.approx(2.0 * sigma * abs(math.cos(math.radians(120.0))) / t.gap)
    # the 3-D force-balance form is exactly twice the 2-D gap value here
    for t in textures:
        b = breakthrough_pressure(t, sigma)
        assert b["p_c_abs_3d"] == pytest.approx(2.0 * b["p_c_abs_2d"])
    # a hydrophilic texture has no barrier (negative p_c)
    hydro = PillarTexture(pitch=16, pillar_width=8, height=20, theta_y_deg=60.0)
    assert breakthrough_pressure(hydro, sigma)["p_c_laplace_2d"] < 0.0


# ---------------------------------------------------------------------------
# angle convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [30.0, 60.0, 90.0, 120.0, 140.0, 155.0])
def test_apparent_angle_convention(theta: float) -> None:
    """A synthetic cap of contact angle theta must be reported as theta."""
    ch = Channel(nx=180, ny=130, wall_rows=2)
    xx, yy = ch.grid()
    y_wall = ch.y_wall()
    rs = 30.0
    yc = y_wall - rs * math.cos(math.radians(theta))
    dist = torch.sqrt((xx - ch.nx / 2.0) ** 2 + (yy - yc) ** 2) - rs
    rho = tanh_profile(ch.shape, dist, 1.957, 0.1596, 1.5)
    rho = torch.where(yy > y_wall, rho, torch.full_like(rho, 0.1596))
    out = apparent_angle(rho, ch.solid_mask(), y_wall, bootstrap=8)
    assert out["theta_deg"] == pytest.approx(theta, abs=0.2)


def test_cap_radius_conserves_volume() -> None:
    for theta in (40.0, 90.0, 130.0, 170.0):
        rs = cap_radius(10.0, theta)
        c = math.cos(math.radians(theta))
        h = rs * (1.0 - c)
        vol_cap = math.pi * h**2 / 3.0 * (3.0 * rs - h)
        vol_sphere = 4.0 / 3.0 * math.pi * 10.0**3
        assert vol_cap == pytest.approx(vol_sphere, rel=1e-12)


# ---------------------------------------------------------------------------
# short LBM smoke runs
# ---------------------------------------------------------------------------


def test_static_textured_drop_is_cassie() -> None:
    # pitch 8 keeps R/P = 2.5 so the composite interface is resolved by several
    # pillars across the drop footprint
    tex = PillarTexture(pitch=8, pillar_width=4, height=16, theta_y_deg=130.0)
    ch = TexturedChannel(nx=144, ny=88, wall_rows=2, texture=tex)
    res = run_static_angle(
        ch, drop_radius=18.0, relax_steps=150, init_angle_deg=150.0, every=10**9
    )
    assert math.isfinite(res["theta_app_deg"])
    assert res["groove_fill"] < 0.25  # grooves stay dry -> Cassie
    assert res["state_observed"] == "cassie"
    assert res["state_predicted"] == "cassie"
    assert res["interface_width_cells"] >= 2.0


def test_impact_observables_are_finite() -> None:
    tex = PillarTexture(pitch=8, pillar_width=4, height=10, theta_y_deg=130.0)
    ch = TexturedChannel(nx=192, ny=88, wall_rows=2, texture=tex)
    runner = ImpactRunner(ch, radius=16.0, we=20.0, tau=1.5)
    res = runner.run(steps=80, every=20, clusters_every=40)
    assert math.isfinite(res["beta_max"]) and res["beta_max"] > 0.5
    assert res["beta_max"] < 6.0
    assert res["Oh"] > 0.0 and res["Re"] > 0.0
    assert res["regime"] in ("rebound", "adhesion", "breakup")
    assert math.isfinite(res["energy"]["e0"]) and res["energy"]["e0"] > 0.0
    # the Guo-scheme SCMP run keeps total mass to a few percent over a short
    # impact (the residual drift is a documented model property, not a bug)
    assert abs(res["mass_total_drift"]) < 0.10


def test_impalement_fill_increases_with_load() -> None:
    from tensorlbm.contact_line.microtexture import run_impalement

    tex = PillarTexture(pitch=16, pillar_width=8, height=16, theta_y_deg=120.0)
    ch = TexturedChannel(nx=32, ny=56, wall_rows=2, texture=tex)
    res = run_impalement(
        ch, column_height=18.0, g_values=(1e-5, 6e-4), steps_per_g=250, refine=0
    )
    fills = [h["groove_fill"] for h in sorted(res["history"], key=lambda h: h["g"])]
    assert math.isfinite(fills[0]) and math.isfinite(fills[1])
    assert fills[1] > fills[0]


# ---------------------------------------------------------------------------
# scaling fits
# ---------------------------------------------------------------------------


def test_scaling_fits_recover_known_exponents() -> None:
    rho, sigma = 1.957, 0.05611
    pts = []
    for r in (20.0, 30.0, 40.0):
        for we in (10.0, 40.0, 160.0):
            tau = 2.2 * math.sqrt(rho * r**3 / sigma)  # Richard/Clanet/Quere
            pts.append(
                {"R": r, "We": we, "beta_max": 0.9 * we**0.25, "tau_contact": tau}
            )
    fits = scaling_fits(pts)
    # beta_max = 0.9 We^0.25 is recovered exactly
    assert fits["beta"]["b"] == pytest.approx(0.25, abs=1e-9)
    assert fits["beta"]["a"] == pytest.approx(0.9, rel=1e-9)
    # tau = 2.2 sqrt(rho R^3/sigma) -> exponent 0.5 against X = rho R^3/sigma
    assert fits["tau"]["exponent"] == pytest.approx(0.5, abs=1e-6)
    assert fits["tau"]["equiv_R_exponent"] == pytest.approx(1.5, abs=1e-6)
    assert fits["tau"]["in_window"] is True
    # and sqrt(We/3) is 15% off the 0.9 We^{1/4} family at these We values
    assert fits["beta"]["max_dev_vs_inviscid"] > 0.0