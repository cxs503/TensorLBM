"""Regression tests for the Flettner rotor-sail (spinning-cylinder) toolbox.

Fast unit tests cover the engineering layer (coefficients, wake profiles,
Weibull statistics, voyage energy bookkeeping).  A short D2Q9 run checks that
the moving-wall bounce-back produces the correct *sign* and a physically
plausible *magnitude* of Magnus lift, and the recorded validation artifacts (if
present) are re-checked against their acceptance thresholds so the regression
is machine-verifiable rather than prose.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.rotor_sail import (
    REFERENCE_TABLE,
    RotorChannelConfig,
    RotorGeometrySpec,
    VoyageInputs,
    apparent_wind,
    build_geometry,
    combined_wake_profile,
    deck_boundary_layer_profile,
    interpolate_coefficients,
    propeller_race_profile,
    rotor_sail_thrust,
    run_rotor_channel,
    run_voyage_assessment,
    strip_thrust,
    uniform_inflow_thrust,
    weibull_bins,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "flettner_rotor"


# --------------------------------------------------------------------------- #
# Geometry and lattice consistency
# --------------------------------------------------------------------------- #
def test_geometry_matches_analytic_disc_area():
    geom = build_geometry(nx=200, ny=160, cx=80.0, cy=79.5, radius=20.0)
    frac = float(geom.solid.float().mean())
    ideal = math.pi * 20.0**2 / (200 * 160)
    assert abs(frac - ideal) / ideal < 0.03
    # every wall link must carry a physical wall fraction in (0, 1]
    d = geom.delta[geom.link]
    assert float(d.min()) > 0.0 and float(d.max()) <= 1.0


def test_wall_fraction_of_axis_aligned_link_is_exact():
    geom = build_geometry(nx=120, ny=120, cx=60.0, cy=59.5, radius=10.0)
    # first fluid node above the cylinder crown is half a cell from the wall
    j = int(math.ceil(59.5 + 10.0))
    q_south = 4  # direction (0, -1) points at the cylinder
    assert bool(geom.link[q_south - 1, j, 60])
    assert abs(float(geom.delta[q_south - 1, j, 60]) - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# Coefficient handling
# --------------------------------------------------------------------------- #
def test_coefficient_interpolation_is_monotone_and_anchored():
    alpha = np.linspace(0.0, 4.0, 41)
    cl, cd = interpolate_coefficients(alpha)
    assert np.all(np.diff(cl) > 0.0)
    cl2, _ = interpolate_coefficients(np.array([2.0]))
    assert abs(cl2[0] - REFERENCE_TABLE["re200_mittal_kumar"]["cl"][2]) < 1e-12


def test_lift_turns_negative_drag_at_high_spin():
    _, cd = interpolate_coefficients(np.array([2.0, 4.0]))
    assert cd[0] <= 0.0 and cd[1] < cd[0]


# --------------------------------------------------------------------------- #
# Inflow / wake models
# --------------------------------------------------------------------------- #
def test_deck_boundary_layer_profile_shape():
    prof = np.asarray(deck_boundary_layer_profile(200, delta=50.0, z0=0.5))
    assert prof.min() >= 0.0 and abs(prof.max() - 1.0) < 1e-12
    assert np.all(np.diff(prof) >= -1e-12)
    assert prof[-1] == pytest.approx(1.0)


def test_propeller_race_adds_speed_only_near_axis():
    race = propeller_race_profile(200, jet_ratio=1.8, race_radius_rows=8.0, centre_row=0.0)
    assert race[0] > race[8] > 1.0
    assert race[100] == pytest.approx(1.0)
    assert race.max() < 1.8


def test_combined_wake_is_below_free_stream_near_deck():
    prof = combined_wake_profile(200, delta_rows=80.0, jet_ratio=1.0)
    assert prof[2] < 0.65 and prof[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Spanwise thrust vs the mean-velocity engineering formula
# --------------------------------------------------------------------------- #
def test_strip_and_mean_velocity_agree_for_uniform_inflow():
    geom = RotorGeometrySpec(n_rotors=2, omega=12.0)
    prof = np.ones(80)
    out = strip_thrust(prof, v_ship=7.2, geom=geom)
    assert abs(out["relative_difference"]) < 1e-9
    ref = uniform_inflow_thrust(7.2, geom)
    assert out["thrust_strip_N"] == pytest.approx(ref["thrust_N"], rel=1e-9)


def test_sheared_inflow_decays_thrust_and_disagrees_with_mean_formula():
    geom = RotorGeometrySpec(n_rotors=1, omega=12.0)
    prof = combined_wake_profile(120, delta_rows=60.0)
    out = strip_thrust(prof, v_ship=7.2, geom=geom)
    uniform = uniform_inflow_thrust(7.2, geom)
    assert out["thrust_strip_N"] < uniform["thrust_N"]
    # the mean-velocity formula cannot reproduce a strongly sheared inflow; the
    # documented acceptance band is |difference| < 20 % for deck-BL shear
    assert abs(out["relative_difference"]) < 0.20


# --------------------------------------------------------------------------- #
# Apparent wind and voyage energy bookkeeping
# --------------------------------------------------------------------------- #
def test_apparent_wind_head_and_beam_cases():
    head_speed, head_angle = apparent_wind(v_ship=7.0, v_true_wind=5.0, wind_angle_rel_ship_deg=180.0)
    assert head_speed == pytest.approx(12.0)  # true wind ahead, 180 deg off the bow
    assert head_angle == pytest.approx(0.0, abs=1e-9)
    beam_speed, beam_angle = apparent_wind(v_ship=7.0, v_true_wind=5.0, wind_angle_rel_ship_deg=90.0)
    assert beam_speed == pytest.approx(math.hypot(7.0, 5.0))
    assert beam_angle == pytest.approx(math.degrees(math.atan2(5.0, 7.0)))


def test_rotor_feathers_against_head_wind():
    geom = RotorGeometrySpec(omega=12.0)
    out = rotor_sail_thrust(v_app=12.0, beta_deg=0.0, geom=geom)
    assert out["feathered"] is True and out["thrust_N"] == 0.0


def test_weibull_bins_normalised_and_mean_close_to_analytic():
    a, k = 8.0, 2.0
    speeds, probs = weibull_bins(a, k, v_max=30.0, n=60)
    assert probs.sum() == pytest.approx(1.0)
    mean = float((speeds * probs).sum())
    assert mean == pytest.approx(a * math.gamma(1.0 + 1.0 / k), rel=0.02)


def test_voyage_assessment_energy_balance():
    rep = run_voyage_assessment({}, VoyageInputs())
    s = rep["summary"]
    assert s["sea_hours_per_year"] > 0.0
    assert s["gross_co2_reduction_t_per_year"] >= s["co2_saved_t_per_year"] > 0.0
    # net = gross - rotor drive power, all through the same SFOC / CO2 factor
    inp = rep["inputs"]
    expected = (
        s["net_power_saving_kW"] * s["sea_hours_per_year"] * inp["sfoc_g_per_kwh"] / 1e6 * inp["co2_factor"]
    )
    assert s["co2_saved_t_per_year"] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# LBM rotating boundary condition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_lbm_magnus_lift_sign_and_magnitude(alpha):
    cfg = RotorChannelConfig(
        nx=240,
        ny=160,
        radius=8.0,
        u_in=0.10,
        re=200.0,
        spin=(alpha,),
        cx=48.0,
        cy=79.5,
        n_steps=700,
        warmup=450,
        sample_every=25,
    )
    res = run_rotor_channel(cfg)
    cl = res["cl"][0]
    if alpha == 0.0:
        assert abs(cl) < 0.35  # symmetric case: no mean lift
    else:
        assert 0.8 < cl < 4.0  # positive Magnus lift, below the potential-flow 2*pi*alpha
        # the control-volume momentum balance must reproduce the wall force
        assert abs(res["cl_cv"][0] - cl) < max(0.6, 0.35 * abs(cl))


# --------------------------------------------------------------------------- #
# Recorded validation artifacts (machine-checkable acceptance)
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict | None:
    p = ARTIFACTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_artifact_laminar_alpha2_vs_literature():
    data = _load("lam_D20_a2.json")
    if data is None:
        pytest.skip("validation artifact not present in this checkout")
    m = data["metrics"]
    ref = 2.29  # Mittal & Kumar (2003), Re=200, alpha=2
    err = abs(m["cl"][0] - ref) / ref
    assert err < 0.05, f"CL={m['cl'][0]:.3f} vs {ref}: {err*100:.1f}%"


def test_artifact_control_volume_crosscheck():
    data = _load("lam_D20_a2.json")
    if data is None:
        pytest.skip("validation artifact not present in this checkout")
    m = data["metrics"]
    rel = abs(m["cl_cv"][0] - m["cl"][0]) / max(abs(m["cl"][0]), 1e-6)
    assert rel < 0.10, f"wall vs control-volume CL differ by {rel*100:.1f}%"