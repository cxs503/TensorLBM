"""Regression tests for the KCS self-propulsion chain (``tensorlbm.selfpropulsion``).

The tests cover, in order of the physics chain:

1. KP505 geometry consistency (expanded area ratio) -- geometry must be validated
   before any prediction is trusted.
2. Blade-element/momentum open-water solution vs the measured KP505 curve,
   including the section-polar uncertainty band and the zero-thrust advance
   ratio.
3. Body-force projection: exact momentum accounting of the Guo source and of the
   actuator-disk field (thrust / torque totals).
4. Field diagnostics: pressure jump, swirl, propeller-plane wake statistics,
   hull pressure-drag integration.
5. Self-propulsion point (``T (1 - t) = R_T``), power chain
   ``P_E -> P_T -> P_D`` and the ``eta_H / eta_R / eta_0`` split.
6. Reynolds scaling of the wake fraction and the acceptance criteria against the
   SIMMAN 2010 benchmark values (w = 0.28, t = 0.22).
7. A short LBM smoke run of the actuator-disk tunnel case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.selfpropulsion import (
    KCS_MODEL,
    KP505Geometry,
    SectionPolar,
    apply_body_force_guo,
    blade_element_momentum,
    build_actuator_disk_field,
    disk_pressure_jump,
    disk_swirl,
    empirical_thrust_deduction,
    empirical_wake_fraction,
    fit_wake_reynolds_exponent,
    holtrop_relative_rotative_efficiency,
    hull_pressure_resistance,
    ittc78_resistance,
    kp505_reference_table,
    lattice_force_scale,
    open_water_series,
    plane_wake_stats,
    power_chain,
    relative_rotative_efficiency,
    rescale_wake_fraction,
    run_actuator_disk_momentum_case,
    solve_self_propulsion,
    solve_self_propulsion_curve,
    thrust_deduction_from_pressure,
)

GEOM = KP505Geometry()
POLAR = SectionPolar()


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------


def test_kp505_geometry_area_ratio():
    """The chord table must reproduce the published EAR = 0.80 after scaling."""
    assert GEOM.raw_area_ratio() == pytest.approx(0.711, abs=0.02)
    assert GEOM.area_ratio() == pytest.approx(0.80, abs=0.005)
    assert GEOM.ear_scale > 1.0


def test_kp505_pitch_and_chord_monotone_shapes():
    rr = np.linspace(GEOM.hub_ratio, 1.0, 40)
    chord = GEOM.chord(rr)
    assert chord.min() > 0.0
    assert chord.max() < 0.5 * GEOM.diameter
    beta = np.rad2deg(GEOM.pitch_angle(rr))
    assert np.all(np.diff(beta) < 0.0)      # pitch angle decreases outward
    assert 30.0 < beta[0] < 60.0
    assert 5.0 < beta[-1] < 20.0


# ---------------------------------------------------------------------------
# 2. open water: BEM vs measurement
# ---------------------------------------------------------------------------


def test_open_water_design_point_within_uncertainty_band():
    """At J = 0.7 the measurement must fall inside the section-polar band."""
    ref = kp505_reference_table()
    idx = ref["J"].index(0.7)
    sol = blade_element_momentum(GEOM, POLAR, 0.7)
    kts, kqs = [], []
    # documented section-polar uncertainty: lift slope, zero-lift angle (camber)
    # and profile drag level of the NACA-66-mod family used by KP505
    for cl_alpha in (5.0, 5.7, 6.2):
        for alpha0 in (-4.5, -3.5, -2.5):
            for cd0 in (0.0060, 0.0085, 0.0120):
                pol = SectionPolar(cl_alpha=cl_alpha, alpha_0_deg=alpha0, cd0=cd0)
                s = blade_element_momentum(GEOM, pol, 0.7)
                kts.append(s["kt"])
                kqs.append(10.0 * s["kq"])
    assert min(kts) <= ref["KT"][idx] <= max(kts)
    assert min(kqs) <= 10.0 * ref["KQ"][idx] <= max(kqs)
    # nominal polar (no calibration on KT/KQ): within 15 % of the measurement
    assert abs(sol["kt"] - ref["KT"][idx]) / ref["KT"][idx] < 0.15
    assert abs(10.0 * sol["kq"] - 10.0 * ref["KQ"][idx]) / (10.0 * ref["KQ"][idx]) < 0.15


def test_open_water_curve_physical_bounds():
    rows = open_water_series(GEOM, POLAR)
    kt = np.array([r["kt"] for r in rows])
    eta = np.array([r["eta0"] for r in rows])
    js = np.array([r["j"] for r in rows])
    assert kt[0] > kt[-1]                              # thrust decreases with J
    assert np.all(np.isfinite(eta[1:-1]))
    assert 0.55 < float(np.nanmax(eta)) < 0.80
    # zero thrust between the last positive and the first negative KT
    j_zero = float(np.interp(0.0, kt[::-1], js[::-1]))
    assert 0.95 < j_zero < 1.20


def test_bem_section_loading_is_positive_and_bounded():
    sol = blade_element_momentum(GEOM, POLAR, 0.7)
    dT = np.array(sol["dT_dr"])
    rr = np.array(sol["r_over_R"])
    assert np.all(dT[rr < 0.98] > 0.0)      # tip section may unload (pancake)
    assert np.max(np.array(sol["a"])) < 0.95
    assert np.all(np.array(sol["cl"]) <= SectionPolar().cl_max + 1e-9)


# ---------------------------------------------------------------------------
# 3. body-force projection and momentum accounting
# ---------------------------------------------------------------------------


def test_guo_forcing_injects_exact_momentum():
    nz, ny, nx = 6, 7, 8
    f = equilibrium3d(torch.ones(nz, ny, nx), torch.full((nz, ny, nx), 0.04),
                      torch.zeros(nz, ny, nx), torch.zeros(nz, ny, nx))
    fx = torch.zeros(nz, ny, nx)
    fy = torch.zeros(nz, ny, nx)
    fz = torch.zeros(nz, ny, nx)
    fx[2, :, :] = 1.0e-4
    fy[3, :, :] = -2.0e-4
    fz[4, :, :] = 3.0e-4
    g = apply_body_force_guo(f, fx, fy, fz)
    for comp, field in enumerate((fx, fy, fz)):
        before = float((f * C[:, comp][:, None, None, None]).sum())
        after = float((g * C[:, comp][:, None, None, None]).sum())
        assert (after - before) == pytest.approx(float(field.sum()), rel=1e-3)


def test_actuator_disk_field_totals():
    rr = np.linspace(0.2, 1.0, 50)
    load = np.clip(1.0 - (rr - 0.75) ** 2 / 0.6, 0.05, None)
    thrust, torque = 3.0e-3, 2.0e-4
    disk = build_actuator_disk_field(
        24, 40, 48, center=(20.0, 20.0, 12.0), radius_cells=8.0,
        hub_radius_cells=2.0, r_over_R=rr, dT_dr=load, dQ_dr=load,
        thrust_lattice=thrust, torque_lattice=torque, thickness_cells=3)
    assert disk["n_cells"] > 0
    assert disk["sum_fx"] == pytest.approx(thrust, rel=1e-6)
    assert disk["sum_angular_momentum"] == pytest.approx(torque, rel=1e-5)
    # force acts downstream (+x) on the fluid; the ship reaction is -x
    assert disk["sum_fx"] > 0.0
    assert float(disk["fx"][disk["mask"]].min()) > 0.0
    assert float(disk["fx"][~disk["mask"]].abs().max()) == 0.0
    assert disk["mask"].dtype == torch.bool


def test_actuator_disk_field_rejects_empty_mask():
    with pytest.raises(ValueError):
        build_actuator_disk_field(8, 8, 8, center=(4.0, 4.0, 4.0),
                                  radius_cells=10.0, hub_radius_cells=9.0,
                                  r_over_R=[0.2, 1.0], dT_dr=[1.0, 1.0],
                                  dQ_dr=[1.0, 1.0], thrust_lattice=1.0,
                                  torque_lattice=1.0)


def test_lattice_force_scale_dimensional_consistency():
    # F_phys = F_lat * rho * dx^4 / dt^2 ; doubling dx raises the force unit 16x
    s1 = lattice_force_scale(999.5, 0.01, 1e-3)
    s2 = lattice_force_scale(999.5, 0.02, 1e-3)
    assert s2 / s1 == pytest.approx(16.0)
    with pytest.raises(ValueError):
        lattice_force_scale(1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. field diagnostics
# ---------------------------------------------------------------------------


def test_disk_pressure_jump_reads_step():
    nz, ny, nx = 16, 24, 40
    field = torch.full((nz, ny, nx), 3.0)          # p = 1.0 everywhere
    field[:, :, 21:] = 3.0 * 1.01                  # p = 1.01 downstream
    out = disk_pressure_jump(field, center=(20.0, 12.0, 8.0), radius_cells=5.0,
                             thickness_cells=3, offset_cells=2)
    assert out["delta_p"] == pytest.approx(0.01, rel=5e-2)
    assert out["x_upstream"] < 20 < out["x_downstream"]


def test_disk_swirl_recovers_rotation():
    nz, ny, nx = 16, 24, 24
    cy, cz = 12.0, 8.0
    yc = torch.arange(ny, dtype=torch.float32)[None, :]
    zc = torch.arange(nz, dtype=torch.float32)[:, None]
    dy = yc - cy
    dz = zc - cz
    r = torch.sqrt(dy**2 + dz**2)
    omega = 0.02
    uy = (-omega * dz).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    uz = (omega * dy).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    out = disk_swirl(uy, uz, center=(0.0, cy, cz), radius_cells=6.0, x_index=5)
    assert out["u_theta_mean"] == pytest.approx(omega * float(r[r <= 6.0].mean()), rel=5e-2)
    assert out["n_cells"] > 0


def test_plane_wake_stats_uniform_and_deficit():
    nz, ny = 20, 20
    cy, cz = 10.0, 10.0
    u = torch.full((nz, ny), 1.0)
    zc = torch.arange(nz, dtype=torch.float32)[:, None]
    yc = torch.arange(ny, dtype=torch.float32)[None, :]
    r = torch.sqrt((yc - cy) ** 2 + (zc - cz) ** 2)
    u[r <= 4.0] = 0.5
    stats = plane_wake_stats(u, center=(0.0, cy, cz), radius_cells=4.0,
                             hub_radius_cells=1.0, freestream=1.0, n_radial=3,
                             n_sectors=4)
    assert stats["w"] == pytest.approx(0.5, abs=0.02)
    assert stats["u_mean_over_U"] == pytest.approx(0.5, abs=0.02)
    assert len(stats["radial_profile"]) == 3
    assert len(stats["sector_profile"]) == 4
    with pytest.raises(ValueError):
        plane_wake_stats(u, center=(0.0, cy, cz), radius_cells=4.0,
                         hub_radius_cells=1.0, freestream=0.0)


def test_hull_pressure_resistance_sign_and_magnitude():
    nz, ny, nx = 6, 8, 20
    mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    mask[:, 3:5, 8:12] = True                    # a bar
    rho = torch.ones(nz, ny, nx)
    rho[:, 3:5, 7] = 3.0 * 1.02                  # high pressure upstream face
    rho[:, 3:5, 12] = 3.0 * 0.98                 # low pressure downstream face
    out = hull_pressure_resistance(mask, rho)
    assert out["n_faces_plus"] > 0 and out["n_faces_minus"] > 0
    # drag = -sum(p n_x dA): upstream face (n_x = -1) contributes +p, the
    # downstream face contributes -p
    # 6 z-planes x 2 y-cells on each x face
    expected = 6 * 2 * (1.02 - 0.98)
    assert out["pressure_resistance"] == pytest.approx(expected, rel=1e-5)
    assert out["n_faces_plus"] == 12 and out["n_faces_minus"] == 12


def test_thrust_deduction_from_pressure_sign():
    out = thrust_deduction_from_pressure(110.0, 100.0, 100.0)
    assert out["delta_R"] == pytest.approx(10.0)
    assert out["t"] == pytest.approx(0.10)
    with pytest.raises(ValueError):
        thrust_deduction_from_pressure(1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 5. self-propulsion point and power chain
# ---------------------------------------------------------------------------


def test_self_propulsion_point_satisfies_balance():
    w = empirical_wake_fraction(KCS_MODEL.cb)
    t = empirical_thrust_deduction(w)
    sp = solve_self_propulsion(resistance=82.0, speed=KCS_MODEL.speed,
                               geometry=GEOM, polar=POLAR, wake_fraction=w,
                               thrust_deduction=t, rho=KCS_MODEL.rho)
    assert sp["thrust"] * (1.0 - t) == pytest.approx(82.0, rel=1e-8)
    assert 7.0 < sp["n_rps"] < 12.0
    assert sp["kq"] > 0.0 and sp["eta0"] > 0.0
    assert sp["iterations"] < 120


def test_self_propulsion_curve_solver_matches_bem_energy():
    ref = kp505_reference_table()
    w = empirical_wake_fraction(KCS_MODEL.cb)
    t = empirical_thrust_deduction(w)
    sp = solve_self_propulsion_curve(
        resistance=82.0, speed=KCS_MODEL.speed, diameter=GEOM.diameter,
        rho=KCS_MODEL.rho, wake_fraction=w, thrust_deduction=t,
        j_table=ref["J"], kt_table=ref["KT"], kq_table=ref["KQ"])
    assert sp["thrust"] * (1.0 - t) == pytest.approx(82.0, rel=1e-9)
    assert 8.5 < sp["n_rps"] < 10.5           # published KCS self-propulsion n ~ 9.4
    assert 0.60 < sp["eta0"] < 0.75
    chain = power_chain(resistance=82.0, speed=KCS_MODEL.speed, thrust=sp["thrust"],
                        torque=sp["torque"], n_rps=sp["n_rps"],
                        wake_fraction=w, thrust_deduction=t, eta0=sp["eta0"],
                        eta_R=holtrop_relative_rotative_efficiency(
                            prismatic_coefficient=KCS_MODEL.cb / KCS_MODEL.cm,
                            lcb_percent_l=-1.5, pitch_ratio=GEOM.design_pitch_ratio))
    assert 0.70 < chain["eta_D"] < 0.78
    # eta_H > 1 for a full hull form: P_T < P_E < P_D
    assert chain["P_T"] < chain["P_E"] < chain["P_D"]


def test_self_propulsion_requires_bracket():
    with pytest.raises(ValueError):
        solve_self_propulsion(resistance=1.0e6, speed=KCS_MODEL.speed, geometry=GEOM,
                              polar=POLAR, wake_fraction=0.28,
                              thrust_deduction=0.21, n_bounds=(1.0, 2.0))


def test_power_chain_identity():
    w, t = 0.2755, 0.2089
    resistance = 82.1
    thrust = resistance / (1.0 - t)          # self-propulsion balance
    chain = power_chain(resistance=resistance, speed=KCS_MODEL.speed, thrust=thrust,
                        torque=4.15, n_rps=9.45, wake_fraction=w,
                        thrust_deduction=t)
    assert chain["eta_H"] == pytest.approx(chain["eta_H_check"], rel=1e-9)
    assert chain["eta_H"] == pytest.approx((1 - t) / (1 - w), rel=1e-12)
    assert chain["P_E"] == pytest.approx(82.1 * KCS_MODEL.speed, rel=1e-12)
    assert chain["P_T"] == pytest.approx(thrust * KCS_MODEL.speed * (1 - w), rel=1e-12)
    assert chain["P_D"] == pytest.approx(2 * math.pi * 9.45 * 4.15, rel=1e-12)
    assert chain["eta_D"] == pytest.approx(chain["eta_H"] * chain["eta_R"] * chain["eta_0"],
                                           rel=1e-9)


def test_relative_rotative_efficiency_near_unity_for_near_uniform_wake():
    wake_field = {
        "w": 0.28,
        "radial_profile": [{"r_over_R": r, "w": 0.28} for r in (0.3, 0.5, 0.7, 0.9)],
        "sector_profile": [{"theta_mid": th, "w": 0.28} for th in
                           np.linspace(-math.pi, math.pi, 8, endpoint=False)],
    }
    out = relative_rotative_efficiency(GEOM, POLAR, wake_field=wake_field,
                                       n_rps=9.45, speed=KCS_MODEL.speed,
                                       rho=KCS_MODEL.rho, n_sections=24)
    assert 0.90 < out["eta_R"] < 1.10


# ---------------------------------------------------------------------------
# 6. Reynolds scaling and benchmark acceptance
# ---------------------------------------------------------------------------


def test_rescale_wake_fraction_and_exponent_fit():
    re_list = [1.0e3, 2.0e3, 4.0e3]
    w_pot = 0.05
    w_list = [w_pot + 0.4 * (r / 1.0e3) ** -0.2 for r in re_list]
    fit = fit_wake_reynolds_exponent(re_list, w_list, w_potential=w_pot)
    assert fit["exponent"] == pytest.approx(0.2, abs=1e-6)
    scaled = rescale_wake_fraction(w_total=0.45, w_potential=0.05,
                                   re_measured=1.0e4, re_target=1.4e7,
                                   exponent=fit["exponent"])
    assert scaled["w"] < 0.45
    assert scaled["w"] == pytest.approx(0.05 + 0.40 * (1.0e4 / 1.4e7) ** 0.2, rel=1e-9)


def test_kcs_benchmark_criteria_pass():
    """Acceptance regression: w, t and eta_H vs SIMMAN 2010."""
    w = empirical_wake_fraction(KCS_MODEL.cb)
    t = empirical_thrust_deduction(w)
    assert abs(w - KCS_MODEL.reference_wake_fraction) / KCS_MODEL.reference_wake_fraction < 0.10
    assert abs(t - KCS_MODEL.reference_thrust_deduction) / KCS_MODEL.reference_thrust_deduction < 0.15
    eta_h = (1 - t) / (1 - w)
    eta_h_ref = (1 - KCS_MODEL.reference_thrust_deduction) / \
        (1 - KCS_MODEL.reference_wake_fraction)
    assert abs(eta_h - eta_h_ref) / eta_h_ref < 0.05


def test_ittc78_resistance_matches_model_test_scale():
    res = ittc78_resistance(length=KCS_MODEL.length, beam=KCS_MODEL.beam,
                            draft=KCS_MODEL.draft, cb=KCS_MODEL.cb,
                            speed=KCS_MODEL.speed, rho=KCS_MODEL.rho,
                            nu=KCS_MODEL.nu, wetted_area=KCS_MODEL.wetted_area,
                            residual_coefficient=0.58e-3)
    assert 1.3e7 < res["reynolds"] < 1.5e7
    # model-scale KCS total resistance at Fr = 0.26 is 82 N (SIMMAN 2010)
    assert res["R_T"] == pytest.approx(KCS_MODEL.reference_resistance, rel=0.05)


# ---------------------------------------------------------------------------
# 7. LBM smoke run
# ---------------------------------------------------------------------------


def test_actuator_disk_tunnel_smoke():
    out = run_actuator_disk_momentum_case(
        nz=12, ny=18, nx=48, radius_cells=5.0, hub_radius_cells=1.5,
        thrust_lattice=2.0e-3, torque_lattice=5.0e-5, n_steps=220,
        warmup_steps=140, sample_steps=40, tau=0.505, smagorinsky_cs=0.1)
    assert out["checks"]["thrust_accounting_rel_error"] < 1e-6
    assert math.isfinite(out["measured"]["u_disk"])
    assert out["measured"]["u_disk"] > out["config"]["u_in"]       # jet is accelerated
    assert len(out["axial_profile"]) == out["config"]["nx"]