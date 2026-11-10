"""pytest suite for the actuator line model (ALM) propeller module.

Coverage map (criterion ids from campaign node p299w4-n03):

* C1 — blade-element momentum closure vs Wageningen B-series polynomials
  (the calibration family B4-70: every station within 5 % on ``K_T``).
* C2 — Gaussian smearing width enters only through the flow sampling, and the
  deposition itself is eps-independent (conservation tests below).
* C4 — Prandtl tip/root loss: bounds, monotone decay towards the tip, blade
  count scaling, and the induced tip-load reduction.
* C5 — force/mass bookkeeping of the lattice coupling.
* C6 — the suite itself.

Plus geometry/polar unit tests (chord normalisation, lift slope, stall,
cascade and rotational-augmentation switches) and lattice-sampling tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.actuator_line import (
    BladeGeometry,
    GaussianSmearer,
    SectionPolar,
    apply_actuator_line_forces,
    blade_element_forces,
    prandtl_loss_factor,
    prandtl_tip_loss,
    section_cascaded_polar,
    solve_bemt,
    trilinear_sample,
)
from tensorlbm.b_series_open_water import BSeriesPropeller, b_series_open_water

# ---------------------------------------------------------------------------
# reference data (B-series)
# ---------------------------------------------------------------------------


def test_b_series_reference_values_b4_70():
    """Regression lock: the restored B-series coefficients keep their values."""
    prop = BSeriesPropeller(p_d=1.0, ae_a0=0.70, z_blades=4)
    assert prop.kt(0.0) == pytest.approx(0.4547, abs=2e-4)
    assert prop.kt(0.5) == pytest.approx(0.2710, abs=2e-4)
    assert prop.kq(0.5) == pytest.approx(0.04343, abs=2e-4)
    assert prop.j_at_zero_thrust() == pytest.approx(1.062, abs=5e-3)


def test_b_series_efficiency_and_envelope():
    prop = BSeriesPropeller(p_d=0.8, ae_a0=0.55, z_blades=5)
    eta = b_series_open_water(0.6, 0.8, 0.55, 5)["eta_0"]
    assert 0.4 < eta < 0.8
    with pytest.raises(ValueError):
        BSeriesPropeller(p_d=2.0, ae_a0=0.7, z_blades=4)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_chord_distribution_reproduces_expanded_area_ratio():
    geom = BladeGeometry(diameter=32.0, hub_ratio=0.18, ae_a0=0.70, z_blades=4)
    xs = np.linspace(geom.hub_ratio, 1.0, 4001)
    area = geom.z_blades * np.trapezoid(geom.chord(xs), xs * geom.radius)
    ratio = area / (math.pi * geom.diameter**2 / 4.0)
    assert ratio == pytest.approx(0.70, rel=2e-3)


def test_chord_maximum_near_seventy_percent_radius():
    geom = BladeGeometry()
    xs = np.linspace(geom.hub_ratio, 1.0, 401)
    x_peak = xs[int(np.argmax(geom.chord_ratio(xs)))]
    assert 0.66 <= x_peak <= 0.78
    assert 0.30 < float(np.max(geom.chord_ratio(xs))) < 0.45


def test_pitch_angle_at_seven_tenths_matches_pitch_ratio():
    geom = BladeGeometry(p_d=1.0)
    theta = math.degrees(float(geom.pitch_angle(np.array([0.7]))[0]))
    expected = math.degrees(math.atan(1.0 / (2.0 * math.pi * 0.7 * geom.radius / geom.diameter)))
    assert theta == pytest.approx(expected, abs=1e-6)
    assert 20.0 < theta < 30.0


# ---------------------------------------------------------------------------
# section polar
# ---------------------------------------------------------------------------


def test_polar_lift_slope_and_zero_lift_angle():
    pol = SectionPolar()
    alpha = np.radians(np.array([0.0, 2.0, 4.0]))
    cl, _ = pol.cl_cd(alpha)
    slope = (cl[2] - cl[0]) / (alpha[2] - alpha[0])
    assert slope == pytest.approx(pol.cl_alpha, rel=1e-3)
    cl0, _ = pol.cl_cd(np.array([0.0]))
    assert cl0[0] == pytest.approx(-pol.cl_alpha * math.radians(pol.alpha0_deg), rel=1e-3)


def test_polar_post_stall_behaviour():
    pol = SectionPolar()
    cl_lin, cd_lin = pol.cl_cd(np.radians(np.array([6.0])))
    cl_post, cd_post = pol.cl_cd(np.radians(np.array([30.0])))
    assert cd_post[0] > 5.0 * cd_lin[0]  # separated drag
    assert cl_post[0] < cl_lin[0] * 1.5  # lift saturates


def test_cascade_correction_reduces_lift():
    geom = BladeGeometry(cascade_k=0.0)
    g2 = BladeGeometry(cascade_k=0.62)
    alpha = np.radians(np.array([4.0]))
    chord = np.array([geom.chord(0.7)])
    r = np.array([0.7 * geom.radius])
    cl_iso, _ = section_cascaded_polar(geom, alpha, chord, r)
    cl_cas, _ = section_cascaded_polar(g2, alpha, chord, r)
    assert cl_cas[0] < 0.8 * cl_iso[0]


def test_rotational_augmentation_only_past_stall():
    geom = BladeGeometry(use_rot_amp=True)
    chord = np.array([geom.chord(0.7)])
    r = np.array([0.7 * geom.radius])
    alpha_att = np.radians(np.array([4.0]))
    alpha_post = np.radians(np.array([16.0]))
    off = BladeGeometry(use_rot_amp=False, cascade_k=geom.cascade_k)
    cl_on_att, _ = section_cascaded_polar(geom, alpha_att, chord, r)
    cl_off_att, _ = section_cascaded_polar(off, alpha_att, chord, r)
    cl_on_post, _ = section_cascaded_polar(geom, alpha_post, chord, r)
    cl_off_post, _ = section_cascaded_polar(off, alpha_post, chord, r)
    assert cl_on_att[0] == pytest.approx(cl_off_att[0], rel=1e-12)
    assert cl_on_post[0] > cl_off_post[0]


# ---------------------------------------------------------------------------
# Prandtl losses (C4)
# ---------------------------------------------------------------------------


def test_prandtl_tip_loss_bounds_and_decay():
    x = np.linspace(0.2, 1.0, 50)
    phi = np.full_like(x, math.radians(8.0))
    f = prandtl_tip_loss(x, phi, 4)
    assert np.all(f[:-1] > 0.0) and np.all(f <= 1.0)
    assert f[0] > 0.98  # far inboard: no loss
    assert f[-1] < 0.05  # at the tip: full loss
    assert np.all(np.diff(f) <= 1e-12)  # monotone


def test_prandtl_tip_loss_scales_with_blade_count():
    """Standard Prandtl form: more blades -> the loss region narrows, F rises."""
    x = np.array([0.9])
    phi = np.array([math.radians(10.0)])
    f3 = float(prandtl_tip_loss(x, phi, 3)[0])
    f6 = float(prandtl_tip_loss(x, phi, 6)[0])
    assert 0.0 < f3 < f6 < 1.0


def test_prandtl_root_loss_lowers_factor_inboard():
    x = np.linspace(0.2, 1.0, 40)
    phi = np.full_like(x, math.radians(8.0))
    f_only_tip = prandtl_loss_factor(x, phi, 4, hub_ratio=0.0)
    f_with_root = prandtl_loss_factor(x, phi, 4, hub_ratio=0.18)
    assert f_with_root[0] < f_only_tip[0]
    assert f_with_root[-1] == pytest.approx(f_only_tip[-1], rel=1e-12)


# ---------------------------------------------------------------------------
# BEMT closure (C1)
# ---------------------------------------------------------------------------


def test_bemt_matches_b_series_at_design_point():
    geom = BladeGeometry(p_d=1.0, ae_a0=0.70, z_blades=4)
    prop = BSeriesPropeller(p_d=1.0, ae_a0=0.70, z_blades=4)
    sol = solve_bemt(geom, 0.6)
    assert sol["converged"]
    assert abs(sol["KT"] - prop.kt(0.6)) / prop.kt(0.6) < 0.03


@pytest.mark.parametrize("j_val", [0.4, 0.5, 0.6, 0.7, 0.8])
def test_bemt_within_five_percent_across_j(j_val):
    geom = BladeGeometry(p_d=1.0, ae_a0=0.70, z_blades=4)
    prop = BSeriesPropeller(p_d=1.0, ae_a0=0.70, z_blades=4)
    sol = solve_bemt(geom, j_val)
    assert abs(sol["KT"] - prop.kt(j_val)) / prop.kt(j_val) < 0.05


def test_bemt_thrust_decreases_with_advance_ratio():
    geom = BladeGeometry()
    kt = [solve_bemt(geom, j)["KT"] for j in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(a > b for a, b in zip(kt, kt[1:]))


def test_bemt_zero_thrust_advance_ratio():
    geom = BladeGeometry()
    prop = BSeriesPropeller(p_d=1.0, ae_a0=0.70, z_blades=4)
    lo, hi = 0.3, 1.6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if solve_bemt(geom, mid)["KT"] > 0 else (lo, mid)
    assert abs(0.5 * (lo + hi) - prop.j_at_zero_thrust()) < 0.1


def test_bemt_tip_loss_reduces_thrust_and_tip_loading():
    geom = BladeGeometry()
    with_tip = solve_bemt(geom, 0.6, n_sections=60)
    no_tip = solve_bemt(geom, 0.6, n_sections=60, apply_prandtl=False)
    assert with_tip["KT"] < no_tip["KT"]
    x = with_tip["sections"]["x"]
    tip = x >= 0.9
    ratio = np.mean(with_tip["sections"]["dT_dr"][tip]) / np.mean(no_tip["sections"]["dT_dr"][tip])
    assert ratio < 0.9  # the tip carries noticeably less load with Prandtl on


def test_bemt_requires_positive_advance_ratio():
    with pytest.raises(ValueError):
        solve_bemt(BladeGeometry(), 0.0)


# ---------------------------------------------------------------------------
# Gaussian smearing / conservation (C5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1.0, 2.0, 4.0, 8.0])
def test_gaussian_smearing_conserves_total_force(eps):
    shape = (96, 96, 96)
    center = (48.0, 48.0, 48.0)
    extent = (8.0, 8.0, 4.0)
    rng = np.random.default_rng(3)
    points = np.stack(
        [
            center[0] + rng.uniform(-8.0, 8.0, 24),
            center[1] + rng.uniform(-8.0, 8.0, 24),
            center[2] + rng.uniform(-4.0, 4.0, 24),
        ],
        axis=1,
    )
    forces = rng.normal(size=(24, 3))
    smearer = GaussianSmearer(shape, eps=eps, center=center, extent=extent)
    fx, fy, fz = smearer.apply(points, forces)
    assert fx.sum() == pytest.approx(forces[:, 0].sum(), rel=1e-12)
    assert fy.sum() == pytest.approx(forces[:, 1].sum(), rel=1e-12)
    assert fz.sum() == pytest.approx(forces[:, 2].sum(), rel=1e-12)


@pytest.mark.parametrize("eps", [1.0, 2.0, 4.0])
def test_gaussian_smearing_width_grows_with_eps(eps):
    shape = (128, 128, 128)
    center = (64.0, 64.0, 64.0)
    extent = (2.0, 2.0, 2.0)
    points = np.array([[64.0, 64.0, 64.0]])
    forces = np.array([[1.0, 0.0, 0.0]])
    smearer = GaussianSmearer(shape, eps=eps, center=center, extent=extent)
    fx, _, _ = smearer.apply(points, forces)
    idx = np.arange(shape[2], dtype=float)
    weight = fx.sum(axis=(0, 1))
    variance = float(np.sum((idx - center[2]) ** 2 * weight) / np.sum(weight))
    assert variance == pytest.approx(eps**2, rel=0.25)


def test_apply_actuator_line_forces_wrapper_matches_smearer():
    shape = (64, 64, 64)
    smearer = GaussianSmearer(shape, eps=2.0, center=(32.0, 32.0, 32.0), extent=(6.0, 6.0, 6.0))
    loads = {
        "points": np.array([[32.0, 36.0, 32.0], [34.0, 30.0, 33.0]]),
        "force_fluid": np.array([[0.0, 0.0, -1e-3], [0.0, 1e-4, -2e-3]]),
    }
    fx, fy, fz = apply_actuator_line_forces(smearer, loads)
    # the returned grids are ordered (z, y, x) like the LBM field layout
    assert fx.sum() == pytest.approx(loads["force_fluid"][:, 0].sum(), rel=1e-12)
    assert fz.sum() == pytest.approx(loads["force_fluid"][:, 2].sum(), rel=1e-12)


# ---------------------------------------------------------------------------
# lattice sampling
# ---------------------------------------------------------------------------


def test_trilinear_sample_reproduces_linear_field():
    nz, ny, nx = 8, 9, 10
    zz, yy, xx = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    field = 2.0 * xx + 3.0 * yy - 1.5 * zz + 0.5
    zq = np.array([2.3, 4.7])
    yq = np.array([3.1, 5.6])
    xq = np.array([4.4, 7.2])
    got = trilinear_sample(field, zq, yq, xq, periodic=(False, False, False))
    want = 2.0 * xq + 3.0 * yq - 1.5 * zq + 0.5
    assert np.allclose(got, want, atol=1e-10)


def test_trilinear_sample_periodic_wrap():
    field = np.arange(64, dtype=float).reshape(4, 4, 4)
    val = trilinear_sample(
        field, np.array([-0.5]), np.array([0.0]), np.array([0.0]), periodic=(True, True, True)
    )
    assert val[0] == pytest.approx(np.mean([field[3, 0, 0], field[0, 0, 0]]))


# ---------------------------------------------------------------------------
# blade element loads from a flow field
# ---------------------------------------------------------------------------


def _uniform_field(nz, ny, nx, ux_value):
    ux = np.full((nz, ny, nx), ux_value, dtype=float)
    zero = np.zeros_like(ux)
    return ux, zero, zero


def test_blade_element_forces_thrust_and_torque_signs():
    geom = BladeGeometry(diameter=32.0)
    omega = 2.0 * math.pi * 0.0015
    ux, uy, uz = _uniform_field(80, 80, 96, 0.024)
    loads = blade_element_forces(ux, uy, uz, geom, omega=omega, psi=0.3, center=(40, 40, 32))
    assert loads["thrust_on_blade"] > 0.0
    assert loads["torque_on_blade"] > 0.0
    # reaction on the fluid is +x (the stream is accelerated) and spins in +e_t
    assert np.all(loads["force_fluid"][:, 2] >= -1e-15)


def test_blade_element_angles_are_physical():
    geom = BladeGeometry(diameter=32.0)
    omega = 2.0 * math.pi * 0.0015
    ux, uy, uz = _uniform_field(80, 80, 96, 0.024)
    loads = blade_element_forces(ux, uy, uz, geom, omega=omega, psi=0.0, center=(40, 40, 32))
    alpha_deg = np.degrees(loads["alpha"])
    assert np.all(alpha_deg > 0.0)
    assert alpha_deg.max() < 30.0  # inlet sections are not fully stalled here
    assert loads["W"].min() > 0.0


def test_blade_element_forces_scale_with_tip_speed():
    geom = BladeGeometry(diameter=32.0)
    ux, uy, uz = _uniform_field(80, 80, 96, 0.024)
    slow = blade_element_forces(ux, uy, uz, geom, omega=2 * math.pi * 0.0010, psi=0.0, center=(40, 40, 32))
    fast = blade_element_forces(ux, uy, uz, geom, omega=2 * math.pi * 0.0020, psi=0.0, center=(40, 40, 32))
    assert fast["thrust_on_blade"] > slow["thrust_on_blade"]


def test_blade_element_marker_count_and_positions():
    geom = BladeGeometry(diameter=32.0, z_blades=4)
    ux, uy, uz = _uniform_field(80, 80, 96, 0.024)
    loads = blade_element_forces(
        ux, uy, uz, geom, omega=2 * math.pi * 0.0015, psi=0.0, center=(40, 40, 32), n_sections=12
    )
    assert loads["points"].shape == (4 * 12, 3)
    radial = np.hypot(loads["points"][:, 0] - 40.0, loads["points"][:, 1] - 40.0)
    assert radial.max() <= 0.5 * geom.diameter + 1e-9
    assert radial.min() >= geom.hub_radius

# ---------------------------------------------------------------------------
# lattice coupling: closed-box momentum budget and a smoke run of the runner
# ---------------------------------------------------------------------------


def test_closed_box_momentum_budget_under_alm_forcing():
    """Guo-type ALM forcing must inject exactly the applied momentum (<1 %)."""
    import torch

    from tensorlbm.actuator_line import AlmOpenWaterConfig, GaussianSmearer
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

    nz, ny, nx = 48, 48, 64
    rho = torch.ones((nz, ny, nx), dtype=torch.float64)
    u_in = 0.02
    ux = torch.full_like(rho, u_in)
    z = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, z, z)

    geom = BladeGeometry(diameter=16.0)
    smearer = GaussianSmearer(
        shape=(nz, ny, nx),
        eps=2.0,
        center=(24.0, 24.0, 24.0),
        extent=(geom.radius + 1.0, geom.radius + 1.0, 4.0),
    )
    loads = blade_element_forces(
        ux.numpy(),
        z.numpy(),
        z.numpy(),
        geom,
        omega=2.0 * math.pi * 0.003,
        psi=0.0,
        center=(24.0, 24.0, 24.0),
        n_sections=10,
    )
    box = smearer.apply_box(loads["points"], loads["force_fluid"])
    force = torch.from_numpy(box)
    sl = smearer.box_slices()

    _, ux0, _, _ = macroscopic3d(f)
    mom0 = float(ux0.sum())
    f[:, sl[0], sl[1], sl[2]] += equilibrium3d(
        rho[sl], ux0[sl] + force[2], z[sl] + force[1], z[sl] + force[0]
    ) - equilibrium3d(rho[sl], ux0[sl], z[sl], z[sl])
    _, ux1, _, _ = macroscopic3d(f)
    injected = float(force[2].sum())
    momentum_gain = float(ux1.sum()) - mom0
    assert injected > 0.0  # propeller adds +x momentum to the stream
    assert abs(momentum_gain - injected) / abs(injected) < 0.01
    del AlmOpenWaterConfig


def test_alm_open_water_runner_smoke():
    """End-to-end ALM lattice run: positive thrust, conserved mass."""
    from tensorlbm.actuator_line import AlmOpenWaterConfig, run_alm_open_water

    cfg = AlmOpenWaterConfig(
        nx=48,
        ny=40,
        nz=40,
        diameter=16.0,
        j_target=0.5,
        rpm_lu=0.003,
        eps_over_dx=2.0,
        n_sections=8,
        n_steps=10,
        warmup_steps=4,
        sample_every=2,
        ramp_steps=2,
    )
    out = run_alm_open_water(cfg)
    assert out["KT_force"] > 0.0
    assert out["KQ_force"] > 0.0
    assert out["mass_drift_rel"] < 1e-3
    assert len(out["samples"]) >= 2
    assert math.isfinite(out["KT_flux"])
