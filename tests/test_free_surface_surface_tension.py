"""Acceptance + unit tests for the free-surface CSF surface-tension module.

Physics anchors (all with their analytic reference):

* ``kappa = -div(n_hat)`` equals ``1/R`` (2-D) / ``2/R`` (3-D);
* ``Delta p = sigma kappa`` (Young-Laplace);
* ``T = 2 pi sqrt(rho R^3 / (8 sigma))`` (Rayleigh n = 2);
* Marangoni force is purely tangential with magnitude ``|grad_s sigma| delta_s``;
* the interface transport conserves ``sum(phi)``.

The long-running acceptance sweeps (3 radii, 4 grids, 5 Ma) live in
``scripts/run_free_surface_tension_verification.py``; the tests below use the
smallest grid that still exercises the physics.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.free_surface_tension import (
    CS2,
    CapillarySolver,
    SolverConfig,
    csf_force,
    curvature,
    get_lattice,
    gradient,
    interface_delta,
    interface_normal,
    laplace_pressure,
    marangoni_force,
    rayleigh_period,
    tanh_droplet_phi,
    tanh_ellipse_phi,
    tanh_flat_phi,
    tanh_sphere_phi,
)
from tensorlbm.free_surface_tension import verification as V

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# lattice primitives                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_equilibrium_moments_exact():
    lat = get_lattice(2)
    rho = torch.rand(8, 8, dtype=torch.float64) + 0.5
    u = torch.randn(2, 8, 8, dtype=torch.float64) * 0.05
    f = lat.equilibrium(rho, u)
    r, mom = lat.moments(f)
    assert torch.allclose(r, rho, atol=1e-13)
    assert torch.allclose(mom, rho.unsqueeze(0) * u, atol=1e-13)


@pytest.mark.unit
def test_guo_source_has_zero_mass_and_correct_momentum():
    lat = get_lattice(3)
    u = torch.zeros(3, 6, 6, 6, dtype=torch.float64)
    force = torch.zeros(3, 6, 6, 6, dtype=torch.float64)
    force[0] = 1e-3
    tau = torch.full((6, 6, 6), 0.8, dtype=torch.float64)
    src = lat.guo_source(u, force, tau)
    assert abs(float(src.sum())) < 1e-16
    mom = torch.stack(
        [(src * lat.cf[:, a].view(-1, 1, 1, 1)).sum(0) for a in range(3)], dim=0
    )
    assert torch.allclose(mom[0], force[0] * (1.0 - 0.5 / tau), atol=1e-15)


@pytest.mark.unit
def test_streaming_conserves_mass_and_opposite_pairs():
    lat = get_lattice(2)
    f = torch.rand(lat.Q, 10, 12, dtype=torch.float64)
    out = lat.stream(f)
    assert abs(float(out.sum() - f.sum())) < 1e-12
    opp = lat.opposite()
    assert bool((lat.c[opp] == -lat.c).all())


# --------------------------------------------------------------------------- #
# interface geometry                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_delta_s_integrates_to_unity_across_interface():
    # 1-D tanh profile: the surface delta must carry unit "mass" across it
    y = torch.arange(0, 40, dtype=torch.float64) + 0.5
    phi2 = 0.5 * (1.0 - torch.tanh(2.0 * (y - 20.0) / 3.0)).unsqueeze(1).repeat(1, 3)
    delta = interface_delta(phi2)[2:-2]  # drop the periodic-wrap rows
    assert float(delta.sum()) == pytest.approx(3.0, rel=0.03)  # 3 columns -> 3x unit


@pytest.mark.unit
@pytest.mark.physics
def test_curvature_of_2d_circle():
    L, R = 64, 16
    phi = tanh_droplet_phi((L, L), (L / 2, L / 2), R, width=3.0)
    k = curvature(phi)
    d = interface_delta(phi)
    band = d > 0.25 * float(d.max())
    k_band = float((k[band] * d[band]).sum() / d[band].sum())
    assert k_band == pytest.approx(1.0 / R, rel=0.03)


@pytest.mark.unit
@pytest.mark.physics
def test_curvature_of_3d_sphere():
    L, R = 24, 8
    phi = tanh_sphere_phi((L, L, L), (L / 2, L / 2, L / 2), R, width=3.0)
    k = curvature(phi)
    d = interface_delta(phi)
    band = d > 0.25 * float(d.max())
    k_band = float((k[band] * d[band]).sum() / d[band].sum())
    assert k_band == pytest.approx(2.0 / R, rel=0.05)


@pytest.mark.unit
@pytest.mark.physics
def test_curvature_sign_droplet_positive_bubble_negative():
    L = 48
    drop = tanh_droplet_phi((L, L), (L / 2, L / 2), 12, width=3.0)
    bubble = 1.0 - drop
    band = interface_delta(drop) > 0.25 * float(interface_delta(drop).max())
    assert float(curvature(drop)[band].mean()) > 0.0
    assert float(curvature(bubble)[band].mean()) < 0.0


# --------------------------------------------------------------------------- #
# CSF force                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_csf_force_points_toward_the_liquid_inside():
    L, R = 48, 12
    phi = tanh_droplet_phi((L, L), (L / 2, L / 2), R, width=3.0)
    f = csf_force(phi, sigma=0.05)
    band = interface_delta(phi) > 0.25 * float(interface_delta(phi).max())
    yy, xx = torch.meshgrid(
        torch.arange(L, dtype=torch.float64) + 0.5,
        torch.arange(L, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    rx, ry = xx - L / 2, yy - L / 2
    radial = f[0] * rx + f[1] * ry  # must be negative (inward) on the +x/+y side
    assert float(radial[band].mean()) < 0.0


@pytest.mark.unit
def test_csf_force_scales_linearly_and_vanishes_without_sigma():
    L, R = 40, 10
    phi = tanh_droplet_phi((L, L), (L / 2, L / 2), R, width=3.0)
    assert float(csf_force(phi, 0.0).abs().max()) == 0.0
    f1 = csf_force(phi, 0.02)
    f2 = csf_force(phi, 0.04)
    assert torch.allclose(2.0 * f1, f2, atol=1e-14)


@pytest.mark.unit
def test_csf_force_matches_sigma_kappa_grad_phi():
    L, R = 40, 10
    phi = tanh_droplet_phi((L, L), (L / 2, L / 2), R, width=3.0)
    sigma = 0.03
    f = csf_force(phi, sigma)
    k = curvature(phi)
    gx, gy = gradient(phi)
    assert torch.allclose(f[0], sigma * k * gx, atol=1e-14)
    assert torch.allclose(f[1], sigma * k * gy, atol=1e-14)


# --------------------------------------------------------------------------- #
# Marangoni                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_marangoni_force_is_purely_tangential():
    L = 32
    phi = tanh_flat_phi((L, L), L / 2, width=3.0)
    f = marangoni_force(phi, (1e-5, 0.0))
    n = interface_normal(phi)
    # f . n_hat must vanish everywhere: no normal component is added
    dot = f[0] * n[0] + f[1] * n[1]
    assert float(dot.abs().max()) < 1e-15


@pytest.mark.unit
def test_marangoni_force_magnitude_and_vanishing_gradient():
    L = 32
    phi = tanh_flat_phi((L, L), L / 2, width=3.0)
    g = 1e-5
    f = marangoni_force(phi, (g, 0.0))
    delta = interface_delta(phi)
    band = delta > 0.25 * float(delta.max())
    band[:2] = False
    band[-2:] = False  # the periodic wrap in y is not part of the flat interface
    # on a flat interface n_hat = +-y, so |grad_s sigma| = |d sigma/dx| and
    # f_x = g * delta_s.  Integrating the force across the interface must give
    # back the imposed tangential stress g.
    assert torch.allclose(f[0][band] / delta[band], torch.full_like(f[0][band], g), atol=1e-12)
    # integrate across the interface (rows 2..-2 exclude the periodic wrap, i.e.
    # the second crossing that a periodic flat interface inevitably has)
    assert float(f[0][2:-2].sum(dim=0).mean()) == pytest.approx(g, rel=1e-6)
    assert abs(float(f[1].sum())) < 1e-12
    assert float(marangoni_force(phi, (0.0, 0.0)).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# analytic anchors                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_laplace_and_rayleigh_formulas():
    assert laplace_pressure(0.05, 10.0, ndim=2) == pytest.approx(0.005)
    assert laplace_pressure(0.05, 10.0, ndim=3) == pytest.approx(0.01)
    T = rayleigh_period(0.02, 12.0, rho=1.0, mode=2)
    assert T == pytest.approx(2 * math.pi * math.sqrt(12.0**3 / (8 * 0.02)))
    # higher modes oscillate faster: T(n) ~ 1/sqrt(n(n-1)(n+2))
    assert rayleigh_period(0.02, 12.0, mode=3) < T


@pytest.mark.unit
def test_zero_crossings_and_period_on_synthetic_signal():
    t = [float(i) for i in range(0, 600)]
    y = [math.cos(2 * math.pi * ti / 200.0) for ti in t]
    cr = V.zero_crossings(t, y)
    T, std, n_half = V.period_from_crossings(cr)
    assert T == pytest.approx(200.0, rel=0.01)
    assert std < 1.0
    assert n_half == len(cr) - 1


@pytest.mark.unit
def test_linear_fit_and_power_law_on_synthetic_data():
    x = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    y = [3.0 * xi + 0.1 for xi in x]
    a, b, r2 = V.linear_fit(x, y)
    assert a == pytest.approx(3.0) and b == pytest.approx(0.1) and r2 == pytest.approx(1.0)
    h = [1 / 8, 1 / 12, 1 / 18, 1 / 26]
    u = [3e-4 * hi**2 for hi in h]
    slope, _, r2p = V.power_law_slope(h, u)
    assert slope == pytest.approx(2.0, rel=1e-6) and r2p == pytest.approx(1.0)


@pytest.mark.unit
def test_parabola_at_is_exact_for_a_quadratic():
    prof = [(j + 0.5) ** 2 for j in range(10)]
    for y0 in (3.5, 4.0, 6.25):
        assert V.parabola_at(prof, y0) == pytest.approx(y0**2, rel=1e-12)


# --------------------------------------------------------------------------- #
# operator consistency with the existing free-surface module                  #
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_curvature_matches_free_surface_common():
    from tensorlbm.free_surface_common import mean_curvature_3d

    phi = tanh_sphere_phi((20, 20, 20), (10, 10, 10), 6.0, width=3.0)
    ours = curvature(phi)
    theirs = mean_curvature_3d(phi)
    d = interface_delta(phi)
    band = d > 0.25 * float(d.max())
    assert torch.allclose(ours[band], theirs[band], atol=1e-10)


# --------------------------------------------------------------------------- #
# solver behaviour                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_conservative_clamp_preserves_mass():
    phi = torch.rand(20, 20, dtype=torch.float64)
    phi[3:8, 3:8] += 0.4  # overshoot above 1
    phi[12:16, 12:16] -= 0.3  # undershoot below 0
    out = CapillarySolver.conservative_clamp(phi)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert abs(float(out.sum() - phi.sum())) < 1e-9


@pytest.mark.unit
def test_sharpen_restores_a_thin_interface_and_keeps_the_contour():
    phi = tanh_ellipse_phi((48, 48), (24, 24), (10.6, 9.4), width=3.0)
    smeared = phi.clone()
    for _ in range(20):
        smeared = 0.25 * (
            torch.roll(smeared, 1, 0)
            + torch.roll(smeared, -1, 0)
            + torch.roll(smeared, 1, 1)
            + torch.roll(smeared, -1, 1)
        )
    solver = CapillarySolver((48, 48), SolverConfig(sigma=0.05, sharpen=True))
    sharp = smeared
    for _ in range(4):  # one re-init per LBM step in production; 4 here for a 20x smeared field
        sharp = solver.sharpen_phi(sharp, mass_target=float(phi.sum()))
    d_sharp = interface_delta(sharp)
    d_phi = interface_delta(phi)
    # peak gradient (inverse interface thickness) is restored
    assert float(d_sharp.max()) == pytest.approx(float(d_phi.max()), rel=0.15)
    assert abs(float(sharp.sum() - phi.sum())) < 1e-3 * float(phi.sum())


@pytest.mark.physics
def test_static_droplet_reaches_laplace_pressure():
    """Short, coarse Young-Laplace check (the 3-radius sweep is the acceptance run)."""
    R, L, sigma = 12, 48, 0.05
    solver = CapillarySolver(
        (L, L), SolverConfig(sigma=sigma, tau_liquid=0.9, tau_gas=0.9, init_laplace=True)
    )
    solver.set_phi(tanh_droplet_phi((L, L), (L / 2, L / 2), R, width=3.0))
    solver.step(600)
    dp = solver.pressure_jump(R, shell=3.0)
    assert dp == pytest.approx(laplace_pressure(sigma, R, ndim=2), rel=0.06)
    assert solver.max_speed() < 0.02


@pytest.mark.physics
def test_pressure_jump_grows_with_surface_tension():
    R, L = 10, 40
    out = []
    for sigma in (0.02, 0.04):
        solver = CapillarySolver((L, L), SolverConfig(sigma=sigma, init_laplace=True))
        solver.set_phi(tanh_droplet_phi((L, L), (L / 2, L / 2), R, width=3.0))
        solver.step(300, advect=False)
        out.append(solver.pressure_jump(R, shell=3.0))
    assert out[1] / out[0] == pytest.approx(2.0, rel=0.05)


@pytest.mark.physics
def test_mass_conservation_of_moving_interface():
    # conservative flux form + conservative_clamp: sum(phi) to round-off
    solver = CapillarySolver((48, 48), SolverConfig(sigma=0.05, sharpen=False))
    solver.set_phi(tanh_ellipse_phi((48, 48), (24, 24), (10.6, 9.4), width=3.0))
    m0 = solver.phi_mass()
    for _ in range(200):
        solver.step(1, advect=True)
    assert abs(solver.phi_mass() - m0) / m0 < 1e-9


@pytest.mark.unit
def test_3d_solver_stays_finite():
    L, R = 16, 4
    solver = CapillarySolver(
        (L, L, L), SolverConfig(sigma=0.05, dtype=torch.float32, init_laplace=True)
    )
    solver.set_phi(tanh_sphere_phi((L, L, L), (L / 2, L / 2, L / 2), R, width=3.0))
    solver.step(30, advect=True)
    assert bool(torch.isfinite(solver.f).all())
    assert bool(torch.isfinite(solver.phi).all())
    assert CS2 == pytest.approx(1.0 / 3.0)


@pytest.mark.physics
def test_marangoni_drives_flow_up_the_surface_tension_gradient():
    """Flow goes toward increasing sigma: grad sigma = +x -> u_x > 0."""
    solver = CapillarySolver(
        (32, 4), SolverConfig(sigma=0.0, tau_liquid=0.9, tau_gas=0.9, init_laplace=False)
    )
    solver.set_phi(tanh_flat_phi((32, 4), 16.0, width=3.0))
    solver.add_wall(axis=1, side="both")
    solver.step(800, advect=False, grad_sigma=(1e-5, 0.0))
    prof = solver.u[0].mean(dim=1)
    assert float(prof[15]) > 0.0
    assert float(prof[0]) == pytest.approx(0.0, abs=1e-12)  # no-slip wall
    # analytic two-layer Couette value
    mu = (0.9 - 0.5) / 3.0
    u_ref = 1e-5 * 15.5 * 15.5 / (31.0 * mu)
    assert float(prof[15]) == pytest.approx(u_ref, rel=0.35)


@pytest.mark.unit
def test_verification_thresholds_are_consistent_with_report():
    """Guard: the acceptance numbers used in the report must be these values."""
    assert laplace_pressure(0.05, 16.0, ndim=2) == pytest.approx(3.125e-3)
    assert laplace_pressure(0.05, 16.0, ndim=3) == pytest.approx(6.25e-3)