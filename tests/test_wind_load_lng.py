"""Tests for the LNG superstructure wind-load module (P16-N7).

Fast unit/integration tests only (no long simulation): geometry and unit
consistency, lattice invariants, the wall-model inversion, force integration
against a brute-force reference, the statistical helpers and the
ship-platform coefficient-table contract.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tensorlbm.wind_load_lng import (
    B_LOG,
    C_LATTICE,
    KAPPA,
    OPP,
    W_LATTICE,
    Box,
    LatticeUnits,
    VoxelGeometry,
    WindLoadSolver,
    cube_case,
    gci_study,
    lng_moss_superstructure,
    solve_friction_velocity,
    strouhal_from_series,
    voxelise,
)
from tensorlbm.wind_load_table import (
    SCHEMA,
    _mean,
    acceptance_criteria,
    build_table,
    interpolate,
    to_platform_payload,
    validate_table,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_lattice_weights_and_opposites():
    assert len(C_LATTICE) == 19
    assert abs(sum(W_LATTICE) - 1.0) < 1e-12
    for i, c in enumerate(C_LATTICE):
        assert C_LATTICE[OPP[i]] == tuple(-v for v in c)
    # isotropy: sum w c_a c_b = cs^2 delta_ab
    w = np.array(W_LATTICE)
    c = np.array(C_LATTICE, dtype=float)
    for a in range(3):
        for b in range(3):
            val = float((w * c[:, a] * c[:, b]).sum())
            assert abs(val - (1.0 / 3.0 if a == b else 0.0)) < 1e-12


def _projected_area_2d(boxes, axis_norm: int, res: float = 0.05) -> float:
    """Exact-ish projected area of a box union (2-D fine raster)."""
    ext = {0: (0.0, 0.0), 1: (0.0, 0.0)}
    lo = [min(getattr(b, n) for b in boxes) for n in ("x0", "y0", "z0")]
    hi = [max(getattr(b, n) for b in boxes) for n in ("x1", "y1", "z1")]
    k1, k2 = [k for k in range(3) if k != axis_norm]
    n1 = int(np.ceil((hi[k1] - lo[k1]) / res))
    n2 = int(np.ceil((hi[k2] - lo[k2]) / res))
    c1 = lo[k1] + (np.arange(n1) + 0.5) * res
    c2 = lo[k2] + (np.arange(n2) + 0.5) * res
    A = np.zeros((n2, n1), dtype=bool)
    A1, A2 = np.meshgrid(c1, c2)
    for b in boxes:
        l1, u1 = getattr(b, f"xyz"[k1] + "0"), getattr(b, f"xyz"[k1] + "1")
        l2, u2 = getattr(b, f"xyz"[k2] + "0"), getattr(b, f"xyz"[k2] + "1")
        A |= (A1 >= l1) & (A1 < u1) & (A2 >= l2) & (A2 < u2)
    return float(A.sum()) * res * res


def test_voxel_projection_areas_match_analytic_projection():
    """Stepped voxel projected areas agree with a fine 2-D projection."""
    boxes = lng_moss_superstructure()
    g = voxelise(boxes, 1.0, 0.0)
    a_frontal = _projected_area_2d(boxes, axis_norm=0, res=0.05)
    a_lateral = _projected_area_2d(boxes, axis_norm=1, res=0.1)
    assert abs(g.area_frontal - a_frontal) / a_frontal < 0.05
    assert abs(g.area_lateral - a_lateral) / a_lateral < 0.05
    # physics check: both projections are of the order of the enclosing
    # rectangle (64 m x 50 m) times the solid fraction, and the plan area is
    # the largest of the three for this spread-out superstructure
    assert 400.0 < g.area_lateral < 1100.0
    assert 400.0 < g.area_frontal < 1100.0
    assert g.area_plan > g.area_lateral


def test_geometry_rotation_swaps_projected_areas():
    g0 = voxelise(lng_moss_superstructure(), 1.0, 0.0)
    g90 = voxelise(lng_moss_superstructure(), 1.0, 90.0)
    assert abs(g90.area_frontal - g0.area_lateral) / g0.area_lateral < 0.05
    assert abs(g90.area_lateral - g0.area_frontal) / g0.area_frontal < 0.05
    # a rotated stepped body keeps its volume
    assert abs(g90.solid_volume - g0.solid_volume) / g0.solid_volume < 0.05


def test_geometry_extends_enclose_the_body():
    g = voxelise(lng_moss_superstructure(), 1.0, 30.0)
    L, B, H = g.length, g.breadth, g.height
    assert 24.0 <= H <= 25.5                 # 25 m above deck, stepped
    assert L >= 50.0 and B >= 30.0           # rotated body still fits
    assert g.solid.sum() > 0
    assert g.solid_volume > 5000.0           # m^3 of superstructure


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_lattice_units_are_self_consistent():
    u = LatticeUnits.from_velocity(dx=0.5, u_phys=15.0, u_lat=0.1)
    assert abs(u.dt - 0.5 * 0.1 / 15.0) < 1e-15
    assert abs(u.nu_lat - u.nu_phys * u.dt / u.dx ** 2) < 1e-15
    assert u.tau0 > 0.5
    # lattice and physical Reynolds numbers must agree
    assert abs(u.re_lattice(50.0) - u.re_phys(25.0)) / u.re_lattice(50.0) < 1e-9
    # force conversion: C_X is invariant between unit systems
    f_lat = 0.417
    a_f = 300.0
    c_from_lat = f_lat / (0.5 * u.u_lat ** 2 * (a_f / u.dx ** 2))
    f_phys = float(u.force_to_physical(f_lat))
    c_from_phys = f_phys / (0.5 * u.rho_phys * u.u_phys ** 2 * a_f)
    assert abs(c_from_lat - c_from_phys) / abs(c_from_lat) < 1e-9


def test_lattice_units_from_reynolds_reproduces_target():
    u = LatticeUnits.from_reynolds(dx=0.125, re_target=3000.0, length_m=1.0,
                                   u_lat=0.08)
    assert abs(u.re_lattice(1.0 / 0.125) - 3000.0) / 3000.0 < 1e-9


# ---------------------------------------------------------------------------
# solver invariants
# ---------------------------------------------------------------------------

def test_equilibrium_recovers_density_and_momentum():
    g = cube_case(side=1.0, cells_per_side=4)
    u = LatticeUnits.from_velocity(dx=g.dx, u_phys=5.0)
    s = WindLoadSolver(g, u)
    rho = 1.0 + 0.01 * torch.rand(s.shape)
    vel = torch.zeros((3,) + s.shape)
    vel[0] = 0.03
    vel[1] = -0.01
    f = s.equilibrium(rho, vel)
    rho2, vel2 = s.macroscopic(f)
    assert torch.allclose(rho, rho2, atol=1e-5)
    assert torch.allclose(vel, vel2, atol=1e-6)


def test_uniform_flow_is_preserved_and_mass_balanced():
    g = cube_case(side=1.0, cells_per_side=4)
    u = LatticeUnits.from_velocity(dx=g.dx, u_phys=5.0)
    s = WindLoadSolver(g, u)
    for _ in range(15):
        s.step()
    rho, vel = s.macroscopic(s.f)
    fluid = s.fluid
    assert abs(float(vel[0][fluid].mean()) - s.u_in) < 5e-3
    # the density-pinned outlet keeps a small O(1e-3) instantaneous flux
    # imbalance (dM/dt of the domain); the time-averaged balance is what the
    # campaign reports through mass_budget()
    assert s.mass_flux_error() < 1e-2
    assert s.mass_budget(0)["n_samples"] == s.step_count
    # no body force expected in a body-free streamwise direction
    assert abs(s.coefficient_summary(0)["C_Y"]) < 5e-3


def test_smagorinsky_viscosity_vanishes_at_equilibrium():
    g = cube_case(side=1.0, cells_per_side=4)
    u = LatticeUnits.from_velocity(dx=g.dx, u_phys=5.0)
    s = WindLoadSolver(g, u)
    rho = torch.ones(s.shape)
    vel = torch.zeros((3,) + s.shape)
    vel[0] = 0.08
    feq = s.equilibrium(rho, vel)
    nu_t = s.smagorinsky_viscosity(feq, feq, rho)
    assert float(nu_t.max()) < 1e-10


def test_momentum_exchange_matches_bruteforce():
    """Gather-based Ladd force == brute-force masked sum."""
    g = cube_case(side=1.0, cells_per_side=4)
    u = LatticeUnits.from_velocity(dx=g.dx, u_phys=5.0)
    s = WindLoadSolver(g, u)
    torch.manual_seed(0)
    f_col = torch.rand_like(s.f) * 0.01 + 0.001
    F, M = s.momentum_exchange(f_col)

    F_ref = np.zeros(3)
    M_ref = np.zeros(3)
    X = s.X.expand(s.shape)
    Y = s.Y.expand(s.shape)
    Z = s.Z.expand(s.shape)
    for i in range(19):
        ci = np.array(C_LATTICE[i], dtype=float)
        mask = s.mask_enter_bruteforce = None
        # brute force: fluid cell x with solid neighbour at x + c_i
        # nb[x] = solid[x + c_i]  (roll by -c_i)
        nb = torch.roll(s.solid, shifts=(-int(ci[2]), -int(ci[1]), -int(ci[0])),
                        dims=(0, 1, 2))
        sel = s.fluid & nb
        p = f_col[i] * sel
        F_ref += 2.0 * ci * float(p.sum())
        sx = 2.0 * float((X * p).sum())
        sy = 2.0 * float((Y * p).sum())
        sz = 2.0 * float((Z * p).sum())
        M_ref += np.cross([sx, sy, sz], ci)
    assert np.allclose(F, F_ref, rtol=1e-5, atol=1e-8)
    assert np.allclose(M, M_ref, rtol=1e-5, atol=1e-6)


def test_wall_model_friction_velocity_satisfies_log_law():
    ut = np.array([0.01, 0.05, 0.2])
    y = 0.5
    nu = 1e-4
    u_tau = solve_friction_velocity(ut, y, nu, iters=8)
    resid = u_tau * (np.log(y * u_tau / nu) / KAPPA + B_LOG) - ut
    assert np.max(np.abs(resid) / ut) < 2e-3
    # larger tangential velocity -> larger friction velocity (monotone)
    assert u_tau[0] < u_tau[1] < u_tau[2]


def test_wall_model_returns_force_opposing_the_flow():
    g = cube_case(side=1.0, cells_per_side=4)
    u = LatticeUnits.from_velocity(dx=g.dx, u_phys=5.0)
    s = WindLoadSolver(g, u)
    for _ in range(10):
        s.step()
    rho, vel = s.macroscopic(s.f)
    nu = torch.full(s.shape, s.nu0)
    F, _ = s.wall_model(vel, rho, nu)
    assert F[0] > 0.0                     # drag on the body opposes +x flow
    assert s.n_wall_cells > 0


def test_probe_reading_matches_grid_point():
    g = cube_case(side=1.0, cells_per_side=4)
    u = LatticeUnits.from_velocity(dx=g.dx, u_phys=5.0)
    s = WindLoadSolver(g, u)
    for _ in range(5):
        s.step()
    v = s.probe(-1.0, 0.0, 1.0)
    assert v.shape == (3,)
    assert abs(v[0] - s.u_in) < 0.02


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_gci_recovers_second_order_and_extrapolates():
    f_exact = 0.80
    c = 0.05
    f_fine = f_exact + c * 0.5 ** 2
    f_med = f_exact + c * 1.0 ** 2
    f_coarse = f_exact + c * 2.0 ** 2
    res = gci_study(f_coarse, f_med, f_fine, r=2.0)
    assert abs(res["p_observed"] - 2.0) < 1e-6
    assert abs(res["f_extrapolated"] - f_exact) < 0.01
    assert 0.0 < res["U_GCI"] < 0.1


def test_strouhal_recovers_synthetic_frequency():
    fs, dt = 5.0, 0.01
    t = np.arange(2048) * dt
    sig = np.sin(2 * np.pi * fs * t) + 0.05 * np.random.default_rng(0).normal(
        size=t.size)
    res = strouhal_from_series(sig, dt, length_m=25.0, u_phys=15.0)
    assert abs(res["f_peak_hz"] - fs) / fs < 0.02
    assert abs(res["St"] - fs * 25.0 / 15.0) / res["St"] < 0.02
    assert res["peak_energy_fraction"] > 0.5


# ---------------------------------------------------------------------------
# coefficient table / platform contract
# ---------------------------------------------------------------------------

def _fake_record(angle, dx, cx, cy, cn, ck):
    return {
        "angle_deg": angle,
        "geometry": {
            "dx_m": dx, "area_frontal_m2": 300.0, "area_lateral_m2": 620.0,
            "extent_m": [50.0, 30.0, 25.0], "solid_volume_m3": 12000.0,
            "grid_xyz": [100, 60, 50],
        },
        "units": {"rho_phys_kgm3": 1.225, "u_phys_ms": 15.0, "dx_m": dx},
        "coefficients": {"C_X": cx, "C_Y": cy, "C_N": cn, "C_K": ck,
                         "C_X_std": 0.01, "C_Y_std": 0.01},
    }


def test_table_build_and_interpolation():
    recs = [
        _fake_record(0.0, 1.0, 0.70, 0.00, 0.00, 0.00),
        _fake_record(90.0, 1.0, 0.20, 0.85, 0.10, 0.05),
        _fake_record(180.0, 1.0, -0.55, 0.00, 0.00, 0.00),
        _fake_record(270.0, 1.0, 0.20, -0.85, -0.10, -0.05),
    ]
    table = build_table(recs, geometry_name="LNG-Moss-superstructure")
    assert table["schema"] == SCHEMA
    validate_table(table)
    mid = interpolate(table, 45.0)
    assert abs(mid["C_X"] - 0.45) < 1e-9
    assert abs(mid["C_Y"] - 0.425) < 1e-9
    # periodic wrap-around between 270 and 360/0
    wrapped = interpolate(table, 315.0)
    assert abs(wrapped["C_X"] - 0.45) < 1e-9
    # finest grid wins when several resolutions are supplied for one angle
    recs2 = recs + [_fake_record(0.0, 0.5, 0.68, 0.0, 0.0, 0.0)]
    table2 = build_table(recs2, geometry_name="x")
    assert abs(table2["C_X"][0] - 0.68) < 1e-9


def test_platform_payload_is_physically_consistent():
    recs = [_fake_record(a, 1.0, cx, cy, cn, ck) for a, cx, cy, cn, ck in
            [(0.0, 0.70, 0.0, 0.0, 0.0), (90.0, 0.20, 0.85, 0.10, 0.05)]]
    table = build_table(recs, geometry_name="x")
    payload = to_platform_payload(
        table, angle_deg=0.0, wind_speed=15.0,
        ship={"length": 290.0, "beam": 46.0, "draft": 11.5,
              "displacement": 120000.0})
    wind = payload["wind"]
    q = 0.5 * 1.225 * 15.0 ** 2
    assert abs(wind["forces"]["F_x_N"] - 0.70 * q * 300.0) < 1e-6
    assert abs(wind["coefficients"]["C_X_AL"] - 0.70 * 300.0 / 620.0) < 1e-9
    json.dumps(payload)          # must be serialisable for the HTTP request


def test_table_rejects_bad_schema():
    with pytest.raises(ValueError):
        validate_table({"schema": "nope"})
    recs = [_fake_record(0.0, 1.0, 0.7, 0.0, 0.0, 0.0),
            _fake_record(90.0, 1.0, 0.2, 0.8, 0.1, 0.05)]
    table = build_table(recs, geometry_name="x")
    table["C_Y"] = table["C_Y"][:-1]
    with pytest.raises(ValueError):
        validate_table(table)


def test_acceptance_criteria_flags_failures_honestly():
    ok = acceptance_criteria(
        C_X_0=0.70, C_X_90=0.21, ref_C_X_0=0.72, ref_C_X_90=0.20,
        U_GCI=0.03, St=0.11, St_ref=0.10, mass_error=1e-5, pytest_passed=12,
        table_path="artifacts/p16n7/wind_load_table.json",
        reference_class="published correlation",
    )
    assert ok["all_pass"] is True
    bad = acceptance_criteria(
        C_X_0=1.50, C_X_90=0.21, ref_C_X_0=0.72, ref_C_X_90=0.20,
        U_GCI=0.12, St=0.30, St_ref=0.10, mass_error=1e-2, pytest_passed=3,
        table_path="", reference_class="none",
    )
    assert bad["all_pass"] is False
    assert bad["a_C_X_0_within_10pct"]["pass"] is False
    assert bad["c_GCI_under_5pct"]["pass"] is False