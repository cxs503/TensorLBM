"""Tests for the rotating-reference-frame LBM module.

The suite is deliberately layered: the analytic helpers and the pseudo-force
algebra are checked in closed form, the *solver* is checked against exact
invariants of a rotating flow (geostrophic fixed point, no-stress / no-slip
walls, mass conservation of the volumetric source) rather than against
self-consistency, and one integration test reproduces a (coarse) Ekman spiral.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.rotating_frame import (
    RotatingFrameLBM,
    axial_uniformity,
    centrifugal_accel,
    coriolis_accel,
    ekman_delta,
    ekman_profile,
    fit_ekman_thickness,
    fit_power_law,
    guo_forcing_delta,
    lattice_scaling,
    spiral_phase_depth,
)

torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# 1. Pseudo-forces                                                            #
# --------------------------------------------------------------------------- #
def test_coriolis_accel_is_minus_two_omega_cross_u():
    ux = torch.tensor([[2.0]])
    uy = torch.tensor([[-3.0]])
    uz = torch.tensor([[0.5]])
    w = 0.25
    ax, ay, az = coriolis_accel(ux, uy, uz, (0.0, 0.0, w))
    # Omega x u = (0,0,w) x (2,-3,0.5) = (-w*u_y, w*u_x, 0) = (+0.75, +0.5, 0)
    assert float(ax) == pytest.approx(-2.0 * (-w * -3.0))
    assert float(ay) == pytest.approx(-2.0 * (w * 2.0))
    assert float(az) == pytest.approx(0.0)


def test_coriolis_accel_is_perpendicular_to_velocity_and_axis():
    rng = torch.Generator().manual_seed(0)
    u = torch.randn(3, 5, 5, generator=rng)
    omega = torch.tensor([0.1, -0.2, 0.3])
    a = torch.stack(coriolis_accel(u[0], u[1], u[2], omega.tolist()))
    assert float((a * u).sum()) == pytest.approx(0.0, abs=1e-6)  # Coriolis does no work
    oz = omega / omega.norm()
    along = (a * oz.view(3, 1, 1)).sum(dim=0)
    assert float(along.abs().max()) == pytest.approx(0.0, abs=1e-6)


def test_centrifugal_accel_matches_omega_squared_r():
    x = torch.tensor([[[1.0, 2.0]]])
    y = torch.tensor([[[0.0, -3.0]]])
    ax, ay, az = centrifugal_accel(x, y, None, (0.0, 0.0, 2.0))
    assert torch.allclose(ax, 4.0 * x)
    assert torch.allclose(ay, 4.0 * y)
    assert torch.allclose(az, torch.zeros_like(x))


def test_centrifugal_field_is_curl_free():
    """grad x (Omega^2 r/2) = 0 -> the term can be absorbed into the pressure."""
    omega = (0.3, -0.2, 0.5)
    n = 6
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    zz = torch.zeros_like(xx)
    ax, ay, az = centrifugal_accel(xx, yy, zz, omega)
    # 2-D curl (d a_y/dx - d a_x/dy) must vanish; the z component of the 3-D curl
    curl_z = torch.gradient(ay, dim=1)[0] - torch.gradient(ax, dim=0)[0]
    assert float(curl_z.abs().max()) < 1e-12


def test_guo_forcing_matches_library_implementation():
    from tensorlbm.free_surface_common import guo_force_delta_3d

    rng = torch.Generator().manual_seed(1)
    shape = (3, 4, 5)
    Fx, Fy, Fz = (torch.randn(shape, generator=rng) for _ in range(3))
    ux, uy, uz = (0.01 * torch.randn(shape, generator=rng) for _ in range(3))
    tau = 0.8
    mine = guo_forcing_delta(Fx, Fy, Fz, ux, uy, uz, tau)
    ref = guo_force_delta_3d(Fx, Fy, Fz, ux, uy, uz, tau, torch.device("cpu"))
    assert torch.allclose(mine, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Analytic reference solution                                              #
# --------------------------------------------------------------------------- #
def test_ekman_delta_and_phase_depth():
    nu, f = 0.1, 0.02
    assert ekman_delta(nu, f) == pytest.approx(math.sqrt(2 * nu / f))
    # written with the frame rotation, delta = sqrt(nu/Omega)
    assert ekman_delta(nu, 2 * 0.01) == pytest.approx(math.sqrt(nu / 0.01))
    assert spiral_phase_depth(nu, f) == pytest.approx(math.pi * ekman_delta(nu, f))


def test_ekman_profile_satisfies_the_linear_ode():
    """Ekman spiral solves the steady linear rotating balance nu w'' = i f w."""
    nu, f, U_g = 0.1, 0.02, 0.03
    delta = ekman_delta(nu, f)
    z = torch.linspace(0.0, 3 * delta, 2001, dtype=torch.float64)
    up, uq = ekman_profile(z / delta, U_g)
    h = float(z[1] - z[0])
    # differentiate the real and imaginary parts separately and verify
    #   nu u'' = -f v ,   nu v'' = +f u      (i.e. nu w'' = i f w)
    u2 = torch.gradient(torch.gradient(up, spacing=h)[0], spacing=h)[0]
    v2 = torch.gradient(torch.gradient(uq, spacing=h)[0], spacing=h)[0]
    scale = float(f * U_g / nu)
    assert float((nu * u2 + f * uq).abs()[10:-10].max()) / scale < 2e-3
    assert float((nu * v2 - f * (up - U_g)).abs()[10:-10].max()) / scale < 2e-3
    # no-slip wall at zeta = 0 and geostrophic recovery aloft
    assert float(abs(up[0])) < 1e-12 and float(abs(uq[0])) < 1e-12
    assert float(abs(uq[-1])) < 1e-2 * U_g
    assert float(abs(up[-1] - U_g)) < 0.1 * U_g


def test_ekman_thickness_fit_recovers_analytic_delta():
    U_g, delta = 0.02, ekman_delta(0.1, 0.02)
    z = torch.arange(1, 120, dtype=torch.float64)
    up, uq = ekman_profile(z / delta, U_g)
    gen = torch.Generator().manual_seed(7)
    up = up + 0.002 * U_g * torch.randn(len(z), generator=gen, dtype=torch.float64)
    uq = uq + 0.002 * U_g * torch.randn(len(z), generator=gen, dtype=torch.float64)
    fit = fit_ekman_thickness(z, up, uq)
    assert abs(fit["delta"] - delta) / delta < 0.01
    assert abs(fit["U_g"] - U_g) / U_g < 0.01


def test_power_law_fit():
    r = torch.exp(torch.linspace(math.log(4.0), math.log(20.0), 12, dtype=torch.float64))
    fit = fit_power_law(r, 3.5 * r**-1.0)
    assert fit["exponent"] == pytest.approx(-1.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_axial_uniformity():
    shape = (8, 4, 4)
    rng = torch.Generator().manual_seed(3)
    uni = torch.ones(shape) + 0.01 * torch.randn(shape, generator=rng)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[1:-1] = True
    assert axial_uniformity(uni, mask)["max_rel_dev"] < 0.03
    sheared = torch.ones(shape) * torch.linspace(0.5, 1.5, shape[0]).view(-1, 1, 1)
    assert axial_uniformity(sheared, mask)["max_rel_dev"] > 0.3


def test_lattice_scaling_round_trip():
    s = lattice_scaling(nu_phys=0.01, f_phys=0.01, delta_nodes=20.0, nu_lat=0.1)
    assert s["delta_phys"] == pytest.approx(math.sqrt(2 * 0.01 / 0.01))
    assert s["f_lat_check"] == pytest.approx(s["f_lat"], rel=1e-12)
    assert s["dx"] * 20.0 == pytest.approx(s["delta_phys"])


# --------------------------------------------------------------------------- #
# 3. Solver invariants                                                        #
# --------------------------------------------------------------------------- #
def _box(shape=(6, 6, 6), **kw):
    return RotatingFrameLBM(shape, **kw)


def test_geostrophic_state_is_a_fixed_point():
    """-2 Omega x u + F = 0 is exactly maintained by the Guo-forced stepper."""
    nu, omega = 0.05, 0.02
    f = 2 * omega
    U_g = 0.02
    sim = _box(tau=3 * nu + 0.5, omega=(0, 0, omega), body_force=(0.0, f * U_g, 0.0))
    ones = torch.ones((6, 6, 6))
    sim.initialize_from_velocity(U_g * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    sim.step(400)
    _, ux, uy, uz = sim.fields()
    assert abs(float(ux.mean()) - U_g) / U_g < 5e-3
    assert float(uy.abs().max()) < 1e-4
    assert float(uz.abs().max()) < 1e-6


def test_no_slip_walls_dissipate_uniform_flow():
    """A uniform flow in a closed no-slip box must decay towards rest."""
    nu = 0.05
    shape = (6, 6, 6)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[0] = solid[-1] = True
    sim = RotatingFrameLBM(shape, tau=3 * nu + 0.5, solid=solid)
    ones = torch.ones(shape)
    sim.initialize_from_velocity(0.02 * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    _, ux0, _, _ = sim.fields()
    sim.step(300)
    _, ux1, _, _ = sim.fields()
    fluid = ~solid
    assert float(ux1[fluid].abs().max()) < 0.5 * float(ux0[fluid].abs().max())


def test_free_slip_walls_do_not_brake_uniform_flow():
    """Specular walls carry no shear: a uniform tangential flow is preserved."""
    nu = 0.05
    shape = (6, 6, 6)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[0] = solid[-1] = True
    slip = torch.zeros_like(solid)
    slip[0] = slip[-1] = True
    sim = RotatingFrameLBM(shape, tau=3 * nu + 0.5, solid=solid, slip_solid=slip)
    ones = torch.ones(shape)
    sim.initialize_from_velocity(0.02 * ones, torch.zeros_like(ones), torch.zeros_like(ones))
    _, ux0, _, _ = sim.fields()
    sim.step(300)
    _, ux1, uy1, uz1 = sim.fields()
    fluid = ~solid
    assert float(ux1[fluid].mean()) == pytest.approx(float(ux0[fluid].mean()), rel=2e-3)
    assert float(uz1[fluid].abs().max()) < 1e-5


def test_mass_source_injects_at_the_prescribed_rate():
    """Continuity with a volumetric source: dM/dt = sum(S); no momentum added."""
    shape = (4, 4, 4)
    S = torch.zeros(shape)
    S[1, 1, 1] = 1e-4
    # float64: in float32 the *measurement* of dM loses ~2 % because the mass
    # injected per step (1e-4) sits 6 orders below the total mass (64)
    sim = RotatingFrameLBM(shape, tau=0.8, mass_source=S, dtype=torch.float64)
    m0 = float(sim.fields()[0].sum())
    steps = 200
    sim.step(steps)
    m1 = float(sim.fields()[0].sum())
    expect = 1e-4 * steps
    assert (m1 - m0) == pytest.approx(expect, rel=1e-6)
    _, ux, uy, uz = sim.fields()
    assert float(max(ux.abs().max(), uy.abs().max(), uz.abs().max())) < 1e-2


def test_no_dynamics_inside_solids():
    shape = (5, 5, 5)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[2, 2, 2] = True
    sim = RotatingFrameLBM(shape, tau=0.8, solid=solid)
    sim.step(50)
    rho, ux, uy, uz = sim.fields()
    assert float(rho[2, 2, 2]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 4. Integration: coarse Ekman spiral                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.physics
def test_ekman_spiral_coarse():
    """Coarse rotating-frame Ekman layer reproduces the analytic spiral."""
    nu = 0.1
    tau = 3 * nu + 0.5
    delta_nodes = 8.0
    f = 2 * nu / delta_nodes**2
    delta = ekman_delta(nu, f)
    nz = int(round(6 * delta_nodes))
    ny = nx = 3
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[0] = True
    solid[-1] = True
    slip = torch.zeros_like(solid)
    slip[-1] = True
    U_g = 0.02
    sim = RotatingFrameLBM(
        (nz, ny, nx),
        tau=tau,
        omega=(0, 0, 0.5 * f),
        body_force=(0.0, f * U_g, 0.0),
        solid=solid,
        slip_solid=slip,
    )
    period = int(round(2 * math.pi / (0.5 * f)))
    conv = sim.run_until_steady(tol=2e-4, max_steps=40000, period=period)
    assert conv["converged"]

    _, ux, uy, _ = sim.fields()
    fluid = ~solid
    cnt = fluid.sum(dim=(1, 2)).clamp(min=1)
    up = (ux * fluid).sum(dim=(1, 2)) / cnt
    uq = (uy * fluid).sum(dim=(1, 2)) / cnt
    z = torch.arange(nz, dtype=torch.float64)
    fit = fit_ekman_thickness(z[1:-1], up[1:-1], uq[1:-1])
    assert abs(fit["delta"] - delta) / delta < 0.10
    a_par, a_perp = ekman_profile(torch.tensor([1.0], dtype=torch.float64), U_g)
    zi = int(round(delta))
    assert abs(float(up[zi]) - float(a_par[0])) / U_g < 0.05
    assert abs(float(uq[zi]) - float(a_perp[0])) / U_g < 0.05
