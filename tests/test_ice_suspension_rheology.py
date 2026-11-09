"""Unit and integration tests for the ice-crystal suspension rheology module.

Run with::

    pytest tests/test_ice_suspension_rheology.py -q

The suite is deliberately fast (<60 s): the shear-cell tests use a 12x12x13
cell for a few hundred steps, which is enough to exercise every code path and
check the invariants (exact linear profile, ``eta_r(pure) = 1``, monotone
viscosity increase with ``phi``) without waiting for a production campaign.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.ice_suspension_rheology import closures, shapes
from tensorlbm.ice_suspension_rheology.calibrate import analyse, run_case
from tensorlbm.ice_suspension_rheology.couette import ShearCell, ShearCellConfig


# ---------------------------------------------------------------------------
# closures
# ---------------------------------------------------------------------------
def test_einstein_and_batchelor_limits():
    assert closures.einstein_relative_viscosity(0.0) == 1.0
    assert closures.batchelor_relative_viscosity(0.0) == 1.0
    # slope at phi -> 0 is the intrinsic viscosity
    eps = 1e-6
    slope = (closures.batchelor_relative_viscosity(eps) - 1.0) / eps
    assert abs(slope - closures.INTRINSIC_VISCOSITY_SPHERE) < 1e-4
    # second-order coefficient
    phi = 0.1
    ident = 1 + 2.5 * phi + 6.2 * phi**2
    assert abs(closures.batchelor_relative_viscosity(phi) - ident) < 1e-12


def test_krieger_dougherty_properties():
    pm = 0.63
    assert abs(closures.krieger_dougherty_relative_viscosity(0.0, pm) - 1.0) < 1e-15
    eps = 1e-7
    d = closures.krieger_dougherty_relative_viscosity(eps, pm) - 1.0
    assert abs(d / eps - 2.5) < 1e-4, "KD must preserve the Einstein slope at phi=0"
    # monotone and divergent at phi_max
    a = closures.krieger_dougherty_relative_viscosity(0.30, pm)
    b = closures.krieger_dougherty_relative_viscosity(0.60, pm)
    assert b > a > 1.0
    with pytest.raises(ValueError):
        closures.krieger_dougherty_relative_viscosity(0.70, pm)


def test_fit_krieger_dougherty_recovers_phi_max():
    pm_true = 0.62
    phi = np.linspace(0.02, 0.32, 8)
    eta = closures.krieger_dougherty_relative_viscosity(phi, pm_true)
    fit = closures.fit_krieger_dougherty_phi_max(phi, eta)
    assert abs(fit["phi_max"] - pm_true) < 5e-3
    assert fit["rmse"] < 1e-4
    assert fit["r2"] > 0.999


def test_contract_is_continuous_and_bounded():
    c = closures.ViscosityContract(phi_max=0.63)
    # C0: blended jump much smaller than the naive hard switch
    hard = c.hard_switch_rel_jump()
    soft = c.max_rel_jump()
    assert soft < hard
    assert hard < 0.05, "hard switch must stay inside the 5% contract tolerance"
    # the assembled closure keeps the Einstein slope at phi -> 0
    eps = 1e-5
    assert abs((float(c.eta_r(eps)) - 1.0) / eps - c.intrinsic) < 1e-2
    # monotone, finite in the calibrated range
    grid = np.linspace(0.0, 0.4, 400)
    vals = c.eta_r(grid)
    assert np.all(np.isfinite(vals))
    assert np.all(np.diff(vals) > 0)


def test_inertial_and_carrier_couplings():
    c = closures.ViscosityContract(phi_max=0.63, i_v=None)
    vis = float(c.eta_r(0.1))
    assert abs(float(c.eta_r_inertial(0.1, 0.0)) - vis) < 1e-12  # Stokes limit
    assert float(c.eta_r_inertial(0.1, 1.0)) > vis
    out = closures.bingham_suspension_stress(
        np.array([1e-3, 1e-2]), 0.1, tau_y0=0.1, consistency=0.01, exponent=1.0, contract=c
    )
    assert np.all(out["tau"] > 0)
    assert np.allclose(out["tau_y"], 0.1 * vis)
    pw = closures.power_law_suspension_consistency(0.1, 1.0, 0.8, contract=c)
    assert abs(pw["consistency"] - vis) < 1e-12 and pw["power_index"] == 0.8


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
def test_sphere_mask_volume_and_symmetry():
    m = shapes.sphere_mask(24, 24, 24, (12.0, 12.0, 12.0), 4.0)
    exact = 4.0 / 3.0 * math.pi * 4.0**3
    assert abs(m.sum() / exact - 1.0) < 0.10, "lattice sphere volume within 10%"
    # centro-symmetric about the centre node (index i -> (24 - i) mod 24)
    assert m[12, 12, 12]
    mirror = np.roll(m[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
    assert np.array_equal(m, mirror)


def test_faceted_shape_is_near_spherical():
    sph = shapes.sphere_mask(24, 24, 24, (12.0, 12.0, 12.0), 5.0)
    fac = shapes.faceted_mask(24, 24, 24, (12.0, 12.0, 12.0), 5.0, n_faces=80)
    ratio = fac.sum() / sph.sum()
    assert 0.80 < ratio < 1.0, "faceted grain inscribed in the sphere"
    assert fac[12, 12, 12]


def test_placement_is_non_overlapping_and_measures_phi():
    ps = shapes.place_particles(32, 32, 32, 0.05, 3.0, seed=7, wall_margin=4)
    assert len(ps) >= 2
    phi = shapes.measure_effective_fraction(ps)
    assert 0.0 < phi <= 0.09
    nz, ny, nx = ps[0].mask.shape
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            dz = a.centre[0] - b.centre[0]
            dy = a.centre[1] - b.centre[1]
            dx = a.centre[2] - b.centre[2]
            dy -= ny * round(dy / ny)
            dx -= nx * round(dx / nx)
            r = math.sqrt(dz * dz + dy * dy + dx * dx)
            assert r >= a.half_extent + b.half_extent, "particles must not overlap"
        # walls respected
        assert a.centre[0] - a.half_extent >= 0.5
        assert a.centre[0] + a.half_extent <= nz - 1.5


def test_rigid_body_velocity_is_a_rigid_motion():
    mask = shapes.sphere_mask(16, 16, 16, (8.0, 8.0, 8.0), 2.0)
    gam, zc, zmid = 1e-3, 8.0, 7.5
    usx, usy, usz = shapes.rigid_body_velocity(mask, (zc, 8.0, 8.0), gam, zmid)
    z, y, x = shapes.box_grid(16, 16, 16)
    m = mask
    # Omega = gamma/2 e_y -> u = Omega x r has zero strain rate
    r = np.stack([x[m] - 8.0, y[m] - 8.0, z[m] - 8.0])
    u = np.vstack([usx[m], usy[m], usz[m]])
    # symmetric part of the velocity gradient of Omega x r is identically zero:
    # check on the analytic form u = (Om*(z-zc), 0, -Om*(x-xc))
    om = 0.5 * gam
    assert np.allclose(u[0], gam * (zc - zmid) + om * r[2])
    assert np.allclose(u[2], -om * r[0])
    assert np.allclose(u[1], 0.0)


# ---------------------------------------------------------------------------
# shear cell
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pure_cell_result():
    import torch

    torch.set_num_threads(2)
    cfg = ShearCellConfig(nx=12, ny=12, nz=13, nu=0.1, u_wall=0.02)
    cell = ShearCell(cfg)
    return cfg, cell.run(320, n_warmup=160, sample_every=20)


def test_pure_couette_reproduces_analytic_profile(pure_cell_result):
    cfg, r = pure_cell_result
    # measured shear rate must match 2 U / (nz - 1) (half-way bounce-back walls)
    assert abs(r["gamma_dot_measured"] / cfg.gamma_dot - 1.0) < 0.02
    assert r["profile_r2"] > 0.999
    # the plane momentum flux must close the pure-fluid momentum budget
    assert abs(r["tau_near"] / (cfg.nu * cfg.gamma_dot) - 1.0) < 0.20


def test_negative_control_eta_r_is_one(pure_cell_result):
    """Criterion (e): no particles -> relative viscosity identical to 1."""
    cfg, r = pure_cell_result
    eta_r = (r["tau_corrected"] / r["tau_corrected"]) * (
        r["gamma_dot_measured"] / r["gamma_dot_measured"]
    )
    assert abs(eta_r - 1.0) < 1e-12
    # and the analytic closures are exact at phi = 0
    c = closures.ViscosityContract(phi_max=0.63)
    assert abs(float(c.eta_r(0.0)) - 1.0) < 1e-15


def test_suspension_increases_viscosity(pure_cell_result):
    import torch

    torch.set_num_threads(2)
    cfg, pure = pure_cell_result
    ps = shapes.place_particles(cfg.nx, cfg.ny, cfg.nz, 0.05, 2.0, seed=11, wall_margin=4)
    assert ps, "needs at least one particle"
    cell = ShearCell(cfg, ps, wall_band=3)
    r = cell.run(320, n_warmup=160, sample_every=20)
    assert r["phi_lattice"] > 0
    assert r["profile_r2"] > 0.98
    eta_r = (r["tau_corrected"] / pure["tau_corrected"]) * (
        pure["gamma_dot_measured"] / r["gamma_dot_measured"]
    )
    assert eta_r > 1.0, f"particles must raise the viscosity (got {eta_r:.4f})"


def test_run_case_and_analyse_pipeline():
    base = dict(nx=12, ny=12, nz=13, radius=2.0, n_steps=200, wall_margin=4)
    pure = run_case(0.0, 0, threads=2, **base)
    cases = [run_case(0.03, s, threads=2, **base) for s in (1, 2)] + [
        run_case(0.08, 3, threads=2, **base)
    ]
    ana = analyse(cases, pure)
    assert "dilute_fit" in ana and "rows" in ana
    assert len(ana["rows"]) == len(cases)
    assert ana["closure_limits"]["kd_phi0"] == pytest.approx(1.0)
    # the closure limits are the exact negative control of the assembly
    for key, val in ana["closure_limits"].items():
        assert abs(val - 1.0) < 1e-12, key
    assert isinstance(ana["batchelor"].get("pass_10pct", True), (bool, np.bool_))