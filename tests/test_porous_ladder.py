"""Tests for :mod:`tensorlbm.porous_ladder` (campaign node p339w2-n06).

Two families:

* **analytic identities / estimator round-trips** — exact by construction, they
  protect the closure estimators independently of the (expensive) lattice runs;
* **small lattice certificates** — the D3Q19 driver is certified on plane
  Poiseuille (``K = h^2/12``) and plane Couette (linear profile) before any
  porous-media number is trusted.

The production ladder lives in ``benchmarks/pending/porous_ladder/run.py``;
these tests are the fast regression guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import tensorlbm.porous_ladder as pl

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


# ---------------------------------------------------------------- analytics
def test_kozeny_carman_reference_value():
    """phi=0.4, d_p=1 -> K = 0.4^3/(180*0.6^2) = 9.8765e-4."""
    K = pl.kozeny_carman_permeability(0.4, 1.0)
    assert K == pytest.approx(0.4**3 / (180 * 0.6**2), rel=1e-14)
    assert K == pytest.approx(9.876543209876544e-04, rel=1e-12)
    # scaling: K ~ d_p^2
    assert pl.kozeny_carman_permeability(0.4, 2.0) == pytest.approx(4 * K, rel=1e-14)
    # the 150-constant variant is the Ergun-viscous one
    assert pl.kozeny_carman_permeability(0.4, 1.0, constant=150.0) == pytest.approx(
        K * 180 / 150, rel=1e-14
    )


def test_kozeny_constant_roundtrip():
    for phi in (0.36, 0.45, 0.52):
        K = pl.kozeny_carman_permeability(phi, 16.0)
        assert pl.kozeny_constant(phi, 16.0, K) == pytest.approx(180.0, rel=1e-12)
    with pytest.raises(ValueError):
        pl.kozeny_carman_permeability(1.0, 1.0)


def test_ergun_matches_darcy_viscous_and_scaling():
    phi, d, mu, rho = 0.45, 16.0, 1.0 / 6.0, 1.0
    u = 1e-6  # Stokes limit
    g = pl.ergun_pressure_gradient(phi, d, u, rho=rho, mu=mu)
    K150 = phi**3 * d**2 / (150 * (1 - phi) ** 2)
    assert g["viscous"] == pytest.approx(mu * u / K150, rel=1e-12)
    assert g["inertial"] / g["total"] < 1e-4
    # inertial/viscous = (1.75/150) Re_p/(1-phi)
    for uu in (0.01, 0.05, 0.2):
        gg = pl.ergun_pressure_gradient(phi, d, uu, rho=rho, mu=mu)
        re_p = uu * d / mu
        assert gg["inertial"] / gg["viscous"] == pytest.approx(
            1.75 / 150 * re_p / (1 - phi), rel=1e-12
        )
        assert gg["inertial"] == pytest.approx(
            pl.ergun_pressure_gradient(phi, d, 2 * uu, mu=mu)["inertial"] / 4, rel=1e-12
        )


def test_brinkman_screening_length_definition():
    assert pl.brinkman_screening_length(4.0, 1.0) == pytest.approx(2.0)
    assert pl.brinkman_screening_length(4.0, 0.25) == pytest.approx(4.0)
    assert pl.pore_reynolds(0.1, 10.0, 0.5) == pytest.approx(2.0)


# ------------------------------------------------------------------ geometry
def test_generate_pack_no_overlap_and_porosity():
    pack = pl.generate_sphere_pack((32, 32, 32), 10.0, 0.45, seed=11)
    stats = pl.pack_statistics(pack.as_mask(), 10.0)
    assert pack.meta["max_residual_overlap"] < 0.02 * pack.radius
    # The library bounce-back puts the no-slip wall at the solid-node centres, so
    # the raster radius is r + RASTER_OFFSET: the *voxel* solid fraction is
    # therefore larger than the nominal one by the half-cell shell.  Correcting
    # the shell recovers the nominal solid fraction (this is the digitiser audit).
    assert abs(stats["phi_mask"] - pack.phi_nominal) < 0.02
    # the flow-effective solid fraction subtracts the half-cell wall shell
    eff = pl.effective_geometry(stats["phi_mask"], 10.0, stats["d_sauter_mask"],
                                phi_nominal=pack.phi_nominal)
    assert eff["phi_eff"] > pack.phi_nominal  # wall sits half a cell inside
    assert eff["d_eff"] == pytest.approx(10.0 - 1.0)
    # Digitised (staircase) surfaces are *inflated*: the counted face area
    # exceeds the analytic 6(1-phi)/d_p, so the mask-derived Sauter diameter is
    # systematically LOW (0.6-1.0 d_p at d_p~10).  This is exactly why the
    # ladder uses the *nominal* geometric d_p and reports the mask value only as
    # an audit number.
    assert 0.60 < stats["d_sauter_over_dp"] < 1.02
    # min-image pair distance never below the contact distance
    d = pl._min_image(pack.centers, pack.shape, pack.periodic)
    dist = np.sqrt((d**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 2 * pack.radius * 0.97


def test_bcc_bed_interface_is_flat_and_periodic_in_plane():
    bed = pl.bcc_bed(4, 30.0, 7.845, free_height=12)
    nz, ny, nx = bed.shape
    y_top = bed.meta["y_bed_top"]
    ys = bed.centers[:, 1]
    assert ys.max() + bed.radius == pytest.approx(y_top, rel=1e-12)
    assert nz == nx == 30
    mask = bed.as_mask()
    assert not mask[:, int(math.ceil(y_top)) + 1 :, :].any()  # nothing above
    phi_bed = 1 - mask[:, : int(y_top), :].mean()
    assert phi_bed == pytest.approx(bed.phi_nominal, abs=0.03)


# ----------------------------------------------------- volume-averaged model
def _composite(alpha=0.5, K=40.0, phi=0.85, y_int=60.0, free_top=90.0, accel=2e-5):
    return pl.composite_brinkman_profile(
        K=K, phi=phi, alpha=alpha, y_int=y_int, free_top=free_top, nu=1.0 / 6.0,
        accel_free=accel, accel_bed=accel,
    )


def test_composite_profile_satisfies_all_boundary_conditions():
    K, phi, alpha, y_int, free_top = 40.0, 0.85, 0.5, 60.0, 90.0
    r = _composite(alpha, K, phi, y_int, free_top)
    y, u = r["y"], r["u_super"]
    assert u[0] == pytest.approx(0.0, abs=1e-15)                      # floor no-slip
    u_int_bed = np.interp(y_int, y, u) / phi                          # continuity
    u_int_free = r["u_int"]
    assert u_int_bed == pytest.approx(u_int_free, rel=1e-12)
    # B-J condition
    bj = alpha / math.sqrt(K) * (r["u_int"] - r["u_D"])
    assert r["slope"] == pytest.approx(bj, rel=1e-12)
    # top wall no-slip (wall_speed = 0)
    assert r["u_free"][-1] == pytest.approx(0.0, abs=1e-12)
    # bed solution is the analytic Darcy-Brinkman one
    lam = math.sqrt(K)  # mu_eff = mu
    assert r["lam"] == pytest.approx(lam, rel=1e-12)
    assert r["u_D"] == pytest.approx(K * phi * 2e-5 / (1 / 6), rel=1e-12)


def test_beavers_joseph_alpha_estimator_roundtrip():
    for alpha in (0.1, 0.5, 1.0, 3.0):
        r = _composite(alpha=alpha)
        sel = r["y"] > r["y"][np.argmin(np.abs(r["y"] - 60.0))]
        yf, uf = r["y"][sel], r["u_super"][sel]
        fit = pl.free_region_quadratic_fit(yf[1:-1], uf[1:-1], 60.0, drop=1)
        a_rec = pl.beavers_joseph_alpha(r["K"], fit["slope"], fit["u_int"], r["u_D"])
        assert a_rec == pytest.approx(alpha, rel=2e-3)


def test_decay_length_fit_roundtrip():
    lam = 7.0
    y = np.arange(0.0, 60.0)
    u = np.exp(y / lam)
    fit = pl.fit_decay_length(y, u, y_lo=10.0, y_hi=40.0)
    assert fit["lam"] == pytest.approx(lam, rel=1e-6)
    assert fit["decay_r2"] > 0.999
    assert math.isnan(pl.fit_decay_length(y[:3], u[:3], y_lo=0, y_hi=2)["lam"])


# ------------------------------------------------------- lattice certificates
def _channel_mask(ny: int, nz: int, nx: int) -> np.ndarray:
    solid = np.zeros((nz, ny, nx), bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    return solid


def test_lattice_plane_poiseuille_returns_h2_over_12():
    ny, nz, nx = 24, 8, 8
    sim = pl.PorousLBM3D(_channel_mask(ny, nz, nx), tau=1.0)
    accel = 1e-4
    sim.run(8000, accel=accel, check_every=400, rtol=1e-7, min_steps=2000)
    _, ux, _, _ = sim.macroscopic()
    u = float((ux * sim.fluid).sum() / sim.fluid.sum())
    K = pl.darcy_permeability(sim.nu, u, accel)
    h = ny - 1  # library bounce-back puts the no-slip wall at the solid-node centres
    # 5 %: the residual is the O(dx) wall-position error of the node-centre
    # bounce-back (measured wall span 30.95 vs 31 on ny=32)
    assert K / (h * h / 12.0) == pytest.approx(1.0, abs=0.05)


def test_lattice_plane_couette_is_linear():
    ny, nz, nx = 24, 8, 8
    sim = pl.PorousLBM3D(_channel_mask(ny, nz, nx), tau=1.0)
    u_w = 0.05
    sim.run(6000, top_wall_speed=u_w, check_every=400, rtol=1e-7, min_steps=2000)
    _, ux, _, _ = sim.macroscopic()
    prof = pl.plane_average(ux, sim.fluid)
    h = ny - 1  # library bounce-back puts the no-slip wall at the solid-node centres
    inner = prof[3 : ny - 3]
    slope = np.polyfit(np.arange(3, ny - 3), inner, 1)[0]
    assert slope == pytest.approx(u_w / h, rel=0.03)


def test_lattice_mass_conservation_and_mach_number():
    ny, nz, nx = 16, 6, 6
    sim = pl.PorousLBM3D(_channel_mask(ny, nz, nx), tau=1.0)
    mass0 = float(sim.f.sum())
    sim.run(600, accel=1e-4, check_every=200, rtol=1e-9, min_steps=100)
    mass1 = float(sim.f.sum())
    rho, ux, _, _ = sim.velocity()
    # mass conservation is a property of the f-sum; a driven LBM flow is
    # legitimately slightly compressible, so rho != 1 by ~1e-5 is expected
    assert abs(mass1 - mass0) / abs(mass0) < 1e-4  # float32 forcing round-off
    assert float(rho.mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(ux.abs().max()) * math.sqrt(3.0) < 0.1