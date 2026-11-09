"""Tests for :mod:`tensorlbm.pancake_ice` (campaign node IN69-N07).

Covers the four acceptance criteria of the node that can be checked as units
(coalescence threshold, geometry window, mechanism split, negative control)
plus the model algebra: collision kernels, sticking, buoyancy, added mass,
power-law fitting and the geometry generators.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import pancake_ice as pi

WAVE = pi.WaveField(wavelength=60.0, amplitude=3.0)  # H/lambda = 0.1


# ---------------------------------------------------------------- wave field
def test_wave_field_deep_water_dispersion() -> None:
    wave = pi.WaveField(wavelength=60.0, amplitude=3.0)
    assert wave.wavenumber == pytest.approx(2.0 * math.pi / 60.0)
    assert wave.omega**2 == pytest.approx(pi.G * wave.wavenumber, rel=1e-12)
    assert wave.period == pytest.approx(2.0 * math.pi / math.sqrt(9.81 * 2.0 * math.pi / 60.0))
    assert wave.steepness == pytest.approx(0.1)


def test_wave_field_orbital_decay() -> None:
    wave = pi.WaveField(wavelength=60.0, amplitude=3.0)
    assert wave.orbital_excursion(0.0) == pytest.approx(3.0)
    # exp(-2pi) decay over one wavelength of depth
    assert wave.orbital_excursion(-60.0) == pytest.approx(3.0 * math.exp(-2.0 * math.pi), rel=1e-12)
    assert wave.orbital_velocity(0.0) == pytest.approx(3.0 * wave.omega)
    assert wave.strain_rate(0.0) == pytest.approx(3.0 * wave.omega * wave.wavenumber)
    with pytest.raises(pi.PancakeIceError):
        wave.orbital_excursion(0.5)


def test_wave_field_rejects_invalid_parameters() -> None:
    with pytest.raises(pi.PancakeIceError):
        pi.WaveField(wavelength=0.0, amplitude=1.0)
    with pytest.raises(pi.PancakeIceError):
        pi.WaveField(wavelength=10.0, amplitude=0.0)
    with pytest.raises(pi.PancakeIceError):
        pi.WaveField(wavelength=10.0, amplitude=6.0)


# -------------------------------------------------------- coalescence (a)
def test_retention_threshold_is_lambda_over_twenty() -> None:
    """At the storm steepness H/lambda = 0.1 the threshold is exactly lambda/20."""
    assert pi.orbital_retention_threshold(WAVE) == pytest.approx(WAVE.wavelength / 20.0)
    assert pi.retention_probability(WAVE.wavelength / 20.0, WAVE) == pytest.approx(0.5)


def test_coalescence_probability_above_threshold() -> None:
    for frac in (1 / 19.0, 1 / 18.0, 1 / 15.0, 1 / 12.0, 1 / 8.0):
        p = pi.retention_probability(frac * WAVE.wavelength, WAVE)
        assert p > 0.5, f"D/lambda=1/{1/frac:.0f} gave p={p:.4f}"


def test_coalescence_probability_below_threshold() -> None:
    for frac in (1 / 22.0, 1 / 25.0, 1 / 30.0, 1 / 50.0, 1 / 100.0):
        p = pi.retention_probability(frac * WAVE.wavelength, WAVE)
        assert p < 0.2, f"D/lambda=1/{1/frac:.0f} gave p={p:.4f}"


def test_retention_probability_is_monotone_in_size() -> None:
    diameters = np.geomspace(0.01, 30.0, 80)
    probs = np.array([pi.retention_probability(float(d), WAVE) for d in diameters])
    assert np.all(np.diff(probs) >= 0.0)
    assert probs[0] < 1e-6 and probs[-1] > 1.0 - 1e-6


def test_retention_probability_rejects_bad_input() -> None:
    with pytest.raises(pi.PancakeIceError):
        pi.retention_probability(0.0, WAVE)
    with pytest.raises(pi.PancakeIceError):
        pi.retention_probability(1.0, WAVE, sharpness=0.0)


# ------------------------------------------------------- collision kernels
def test_settling_velocity_stokes_scaling() -> None:
    w1 = pi.stokes_settling_velocity(2e-3)
    w2 = pi.stokes_settling_velocity(4e-3)
    assert w2 / w1 == pytest.approx(4.0)  # w ~ d^2
    assert w1 > 0.0  # ice rises
    with pytest.raises(pi.PancakeIceError):
        pi.stokes_settling_velocity(-1.0)


def test_frazil_number_density_matches_volume_fraction() -> None:
    d = 2e-3
    n = pi.frazil_number_density(d, 0.05)
    assert n * math.pi * d**3 / 6.0 == pytest.approx(0.05)
    assert pi.frazil_number_density(d, 0.0) == 0.0


def test_collision_kernel_wave_term_matches_smoluchowski() -> None:
    d = 2e-3
    kernel = pi.frazil_collision_kernel(d, d, wave=WAVE, settling_spread=0.0)
    # identical crystals, no settling spread -> settling term must vanish
    assert kernel["beta_settling"] == pytest.approx(0.0)
    expected = (4.0 / 3.0) * WAVE.strain_rate(0.0) * d**3
    assert kernel["beta_wave"] == pytest.approx(expected)


def test_collision_kernel_turbulent_term() -> None:
    d = 2e-3
    eps = 1e-3
    kernel = pi.frazil_collision_kernel(d, d, dissipation=eps)
    expected = 1.294 * math.sqrt(eps / pi.NU_WATER) * d**3
    assert kernel["beta_turbulent"] == pytest.approx(expected, rel=1e-12)
    assert kernel["beta_wave"] == 0.0


def test_collision_kernel_relative_velocity_increases_with_size() -> None:
    small = pi.frazil_collision_kernel(2e-3, wave=WAVE, settling_spread=0.2)
    large = pi.frazil_collision_kernel(2e-1, wave=WAVE, settling_spread=0.2)
    assert large["relative_velocity"] > small["relative_velocity"]


def test_sticking_efficiency_window() -> None:
    assert pi.sticking_efficiency(0.0) == pytest.approx(1.0)
    assert pi.sticking_efficiency(0.02) == pytest.approx(math.exp(-1.0))
    assert pi.sticking_efficiency(1.0) < 1e-10
    with pytest.raises(pi.PancakeIceError):
        pi.sticking_efficiency(-1.0)


# ----------------------------------------------------------- geometry (b)
def test_geometry_aspect_ratio_window() -> None:
    """Every (D, h) pair of the campaign design must sit inside the measured window."""
    for diameter in (0.3, 0.5, 1.0, 2.0, 3.0):
        for ar in (5.0, 8.0, 10.0, 15.0, 20.0):
            h = min(max(diameter / ar, pi.H_PANCAKE_BAND[0]), pi.H_PANCAKE_BAND[1])
            geo = pi.PancakeGeometry(diameter=diameter, thickness=h)
            assert geo.validate()["ok"] is True
            assert 5.0 <= geo.aspect_ratio <= 20.0


def test_geometry_rejects_out_of_band() -> None:
    with pytest.raises(pi.PancakeIceError):
        pi.PancakeGeometry(diameter=1.0, thickness=0.01).validate()  # D/h = 100
    with pytest.raises(pi.PancakeIceError):
        pi.PancakeGeometry(diameter=5.0, thickness=0.15).validate()  # D outside band
    geo = pi.PancakeGeometry(diameter=5.0, thickness=0.15)
    report = geo.validate(strict=False)
    assert report["ok"] is False
    assert any("diameter" in issue for issue in report["issues"])


def test_buoyancy_draft_and_freeboard() -> None:
    geo = pi.PancakeGeometry(diameter=1.0, thickness=0.15, rim_height=0.03)
    # rim mass smeared over the plan area: draft = (rho_i/rho_w)(h + rim volume/area)
    expected = (pi.RHO_ICE / pi.RHO_WATER) * geo.equivalent_thickness
    assert geo.draft == pytest.approx(expected)
    assert geo.draft < geo.thickness  # floats
    assert geo.freeboard_disc > 0.0
    assert geo.freeboard_rim > geo.freeboard_disc
    out = pi.pancake_buoyancy(1.0, 0.15, 0.03)
    assert out["buoyancy_force_N"] == pytest.approx(out["weight_N"], rel=1e-12)


def test_added_mass_of_thin_disc() -> None:
    geo = pi.PancakeGeometry(diameter=1.0, thickness=0.1)
    assert geo.added_mass_face_on == pytest.approx(pi.RHO_WATER / 3.0)
    # thin disc at D/h = 10 is added-mass dominated (rim mass included in the ratio)
    expected = pi.added_mass_coefficient_disc(10.0) * geo.volume_disc / geo.volume_total
    assert geo.added_mass_ratio == pytest.approx(expected, rel=1e-12)
    assert 4.0 < geo.added_mass_ratio < 5.5


def test_bending_stress_scaling_and_breakup_size() -> None:
    wave = pi.WaveField(wavelength=60.0, amplitude=3.0)
    s1 = pi.pancake_bending_stress(1.0, 0.1, wave)
    s2 = pi.pancake_bending_stress(2.0, 0.1, wave)
    assert s2 / s1 == pytest.approx(4.0)  # sigma ~ D^2/h
    sigma_flex = 0.3e6
    d_break = math.sqrt(sigma_flex * 0.1 / (0.75 * pi.RHO_ICE * wave.amplitude * wave.omega**2))
    assert 0.5 * wave.wavelength / 10.0 < d_break < 2.0 * wave.wavelength / 10.0


# ------------------------------------------------------------- growth model
def test_growth_rate_stalls_below_retention_threshold() -> None:
    wave = pi.WaveField(wavelength=20.0, amplitude=1.0)  # A_orb = 1 m
    below = pi.aggregate_growth_rate(0.2, wave=wave)  # D < A_orb
    above = pi.aggregate_growth_rate(1.5, wave=wave)  # D > A_orb
    assert below["retention"] < 1e-6
    assert above["retention"] > 0.5
    # the rim-filtration channel only opens above the threshold
    assert above["dD_dt"] > below["dD_dt"]
    assert above["slip_velocity"] > 0.0


def test_simulate_aggregate_growth_timescales() -> None:
    wave = pi.WaveField(wavelength=20.0, amplitude=1.0)
    result = pi.simulate_aggregate_growth(wave=wave, thickness=0.1, t_max_hours=6.0, dt=1.0)
    # consolidation is fast (minutes), the formation time is thermodynamic (hours)
    assert math.isfinite(result["t_consolidation_h"])
    assert result["t_consolidation_h"] < 1.0
    assert 12.0 < result["t_thermo_h"] < 120.0
    assert 12.0 < result["t_form_h"] < 120.0
    assert result["d_saturation_m"] > result["target_diameter_m"]


def test_thermodynamic_formation_time_scaling() -> None:
    t = pi.thermodynamic_formation_time(0.1, heat_flux=200.0)
    assert t == pytest.approx(pi.RHO_ICE * 0.1 * pi.LATENT_HEAT_FUSION / 200.0)
    assert 40.0 < t / 3600.0 < 45.0  # ~42 h for a 10 cm pancake, consistent with field data
    with pytest.raises(pi.PancakeIceError):
        pi.thermodynamic_formation_time(0.0)
    with pytest.raises(pi.PancakeIceError):
        pi.thermodynamic_formation_time(0.1, heat_flux=0.0)


def test_formation_probability_gates() -> None:
    big = pi.pancake_formation_probability(30.0, WAVE, thickness=0.15)
    assert big["breakup"] == 0.0 and big["probability"] == 0.0
    ok = pi.pancake_formation_probability(3.0, WAVE, thickness=0.3)
    assert ok["buoyancy"] == 1.0 and ok["retention"] == pytest.approx(0.5)


# ------------------------------------------------ resistance model (c,d,e)
def test_negative_control_zero_concentration() -> None:
    for diameter in (0.3, 1.0, 3.0):
        for speed in (1.0, 3.0, 5.0):
            out = pi.pancake_resistance(0.0, diameter, diameter / 10.0, speed)
            assert out["resistance_N"] == 0.0
            assert out["slip_N"] == 0.0 and out["bulldozing_N"] == 0.0


def test_jamming_split_limits() -> None:
    f_slip0, f_bull0 = pi.jamming_split(0.0)
    assert f_slip0 == pytest.approx(1.0) and f_bull0 == pytest.approx(0.0)
    f_slip1, f_bull1 = pi.jamming_split(1.0, c_jam=0.65, p_jam=3.0)
    assert f_bull1 > 0.7
    shares = [pi.jamming_split(c)[0] for c in np.linspace(0.0, 1.0, 20)]
    assert np.all(np.diff(shares) <= 0.0)
    with pytest.raises(pi.PancakeIceError):
        pi.jamming_split(1.5)


def test_raft_thickness_amplification() -> None:
    assert pi.raft_thickness(0.0, 0.1) == pytest.approx(0.1)
    assert pi.raft_thickness(0.65, 0.1) == pytest.approx(0.15)  # 1 + k_raft at C_jam
    assert pi.raft_thickness(0.9, 0.1) > pi.raft_thickness(0.65, 0.1)


def test_mechanism_shares_low_and_high_concentration() -> None:
    low = pi.pancake_resistance(0.1, 1.0, 0.1, 3.0)
    high = pi.pancake_resistance(0.8, 1.0, 0.1, 3.0)
    assert low["slip_share"] > 0.60
    assert high["bulldozing_share"] > 0.50


def test_resistance_monotone_in_concentration_and_speed() -> None:
    r_c = [pi.pancake_resistance(c, 1.0, 0.1, 3.0)["resistance_N"] for c in np.linspace(0.05, 0.95, 12)]
    assert np.all(np.diff(r_c) > 0.0)
    r_v = [pi.pancake_resistance(0.5, 1.0, 0.1, v)["resistance_N"] for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert np.all(np.diff(r_v) > 0.0)


def test_resistance_rejects_invalid_state() -> None:
    with pytest.raises(pi.PancakeIceError):
        pi.pancake_resistance(1.2, 1.0, 0.1, 3.0)
    with pytest.raises(pi.PancakeIceError):
        pi.pancake_resistance(0.5, 0.0, 0.1, 3.0)


def test_power_law_fit_recovers_synthetic_exponents() -> None:
    c, v = np.meshgrid(np.linspace(0.1, 0.8, 8), np.linspace(1.0, 5.0, 6), indexing="ij")
    r = 2.5 * c**1.4 * v**1.9
    fit = pi.power_law_fit(c.ravel(), v.ravel(), r.ravel())
    assert fit["alpha"] == pytest.approx(1.4, rel=1e-6)
    assert fit["beta"] == pytest.approx(1.9, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(pi.PancakeIceError):
        pi.power_law_fit([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0])


def test_sweep_resistance_grid_shape_and_scaling() -> None:
    out = pi.sweep_resistance([0.1, 0.8], [1.0], [0.1], [1.0, 5.0])
    assert out["shape"].tolist() == [2, 1, 1, 2]
    assert out["resistance_N"].shape == (4,)
    fit = pi.power_law_fit(out["C"], out["V"], out["resistance_N"])
    assert 1.0 <= fit["alpha"] <= 2.0
    assert 1.5 <= fit["beta"] <= 2.5


# ------------------------------------------------------- geometry generators
def test_pancake_field_mask_coverage() -> None:
    for c in (0.2, 0.5, 0.8):
        mask = pi.pancake_field_mask(160, 160, concentration=c, diameter=1.0, cell_size=0.05, seed=3)
        assert mask.dtype == bool
        assert 0.85 * c <= mask.mean() <= 1.05 * c + 0.01


def test_pancake_field_mask_rejects_bad_input() -> None:
    with pytest.raises(pi.PancakeIceError):
        pi.pancake_field_mask(64, 64, concentration=0.0, diameter=1.0, cell_size=0.05)
    with pytest.raises(pi.PancakeIceError):
        pi.pancake_field_mask(1, 64, concentration=0.5, diameter=1.0, cell_size=0.05)


def test_pancake_section_mask_geometry() -> None:
    mask = pi.pancake_section_mask(
        120, 80, diameter_cells=40, thickness_cells=4, rim_height_cells=2, rim_width_cells=1
    )
    assert mask.dtype == bool
    core = int(mask.sum())
    assert core > 40 * 4  # rim cells add on top of the plate
    ys, xs = np.nonzero(mask)
    assert ys.max() - ys.min() + 1 == 40 + 2 * 2  # diameter plus rim overhang
    with pytest.raises(pi.PancakeIceError):
        pi.pancake_section_mask(10, 10, diameter_cells=1, thickness_cells=1)


# --------------------------------------------------------------- LBM smoke
def test_lbm_smoke_section_drag_is_positive_and_finite() -> None:
    """Minimal D2Q9 bounce-back run on a pancake section (kernel smoke test)."""
    torch = pytest.importorskip("torch")
    from tensorlbm.boundaries import (
        apply_simple_channel_boundaries,
        bounce_back_cells,
        compute_obstacle_forces,
        make_channel_wall_mask,
    )
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.solver import collide_bgk, stream

    torch.set_num_threads(1)
    nx, ny, size, h_cells = 90, 56, 20, 2
    mask_np = pi.pancake_section_mask(nx, ny, diameter_cells=size, thickness_cells=h_cells)
    mask = torch.as_tensor(mask_np)
    wall = make_channel_wall_mask(ny, nx, mask, device="cpu")
    rho = torch.ones((ny, nx))
    u_in = 0.06
    f = equilibrium(rho, u_in * rho, torch.zeros_like(rho), device="cpu")
    zeros = torch.zeros_like(mask)
    nu = u_in * size / 100.0
    forces = []
    for step in range(120):
        f = collide_bgk(f, tau=3.0 * nu + 0.5)
        f = stream(f)
        fx, _ = compute_obstacle_forces(f, mask)
        f = apply_simple_channel_boundaries(f, u_in=u_in, wall_mask=wall, obstacle_mask=zeros)
        f = bounce_back_cells(f, mask)
        if step > 60:
            forces.append(float(fx.item()))
    fx_mean = float(np.mean(forces))
    assert math.isfinite(fx_mean)
    assert fx_mean > 0.0  # drag in the flow direction