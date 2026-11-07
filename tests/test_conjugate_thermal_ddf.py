"""Tests for the DDF thermal / conjugate-heat-transfer validation package.

Covers the lattice invariants, the numerical closures (analytic pure-conduction,
diffusion and viscosity limits), the conjugate interface coupling, the Nusselt
diagnostics and the two literature correlations used as references.

All tests are sized to run in a few seconds each (no long validation runs).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.conjugate_thermal import (
    C5,
    C9,
    OPP5,
    OPP9,
    W5,
    W9,
    Conduction2D,
    DDF2D,
    bilinear_sample,
    cavity_natural_convection,
    channel_nusselt_fully_developed,
    churchill_bernstein,
    conjugate_channel,
    cylinder_forced_convection,
    de_vahl_davis_nu,
    nu_cylinder_polar,
    rayleigh_growth_rate,
)
from tensorlbm.conjugate_thermal.ddf2d import (
    lagrange_derivative_weights,
    shifted,
)

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# Lattice invariants
# ---------------------------------------------------------------------------
def test_d2q9_weights_normalised():
    assert float(W9.sum()) == pytest.approx(1.0, abs=1e-15)
    assert float(W5.sum()) == pytest.approx(1.0, abs=1e-15)


def test_d2q5_isotropy():
    """Second moment of the D2Q5 weights must be cs^2 * I with cs^2 = 1/3."""
    c = C5.to(torch.float64)
    m = (W5.reshape(-1, 1) * c).t() @ c
    assert float(m[0, 0]) == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert float(m[1, 1]) == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert float(m[0, 1]) == pytest.approx(0.0, abs=1e-15)


def test_opposite_index_is_involution():
    for opp, q in ((OPP9, 9), (OPP5, 5)):
        for i in range(q):
            assert int(opp[int(opp[i])]) == i


def test_lattice_directions_are_opposite():
    for c, opp in ((C9, OPP9), (C5, OPP5)):
        for i, dir_i in enumerate(opp):
            assert torch.equal(c[i] + c[int(dir_i)], torch.zeros(2, dtype=c.dtype))


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def test_lagrange_derivative_weights_wall_stencil():
    w = lagrange_derivative_weights([1.0, 2.0, 3.0], 0.5)
    assert w == pytest.approx([-2.0, 3.0, -1.0], abs=1e-14)
    w2 = lagrange_derivative_weights([0.5, 1.5, 2.5], 0.5)
    assert w2 == pytest.approx([-1.5, 2.0, -0.5], abs=1e-14)


def test_lagrange_weights_are_exact_on_quadratic():
    # p(x) = 3x^2 - 2x + 1 -> p'(0.5) = 1
    xs = [1.0, 2.0, 3.0]
    ys = [3 * x * x - 2 * x + 1 for x in xs]
    w = lagrange_derivative_weights(xs, 0.5)
    assert sum(wi * yi for wi, yi in zip(w, ys)) == pytest.approx(1.0, abs=1e-12)


def test_bilinear_sample_exact_on_linear_field():
    ny, nx = 9, 7
    y = torch.arange(ny, dtype=torch.float64).reshape(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).reshape(1, -1)
    field = 2.0 * x + 3.0 * y + 1.0
    xs = torch.tensor([0.0, 1.5, 3.25], dtype=torch.float64)
    ys = torch.tensor([0.5, 4.0, 7.75], dtype=torch.float64)
    got = bilinear_sample(field, xs, ys)
    assert torch.allclose(got, 2.0 * xs + 3.0 * ys + 1.0, atol=1e-12)


def test_shifted_matches_indexing():
    a = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    out = shifted(a, 1, -1, fill=-99.0)
    # out[y, x] = a[y + cy, x + cx] with cx = -1, cy = 1
    assert out[0, 1] == pytest.approx(float(a[1, 0]))
    assert out[1, 2] == pytest.approx(float(a[2, 1]))
    assert out[0, 0] == pytest.approx(-99.0)   # x - 1 out of range
    assert out[2, 3] == pytest.approx(-99.0)   # y + 1 out of range


# ---------------------------------------------------------------------------
# Equilibrium / moment consistency
# ---------------------------------------------------------------------------
def test_flow_equilibrium_moments():
    d = DDF2D(6, 8, nu=0.05, alpha=0.05)
    rho = torch.full((6, 8), 1.3, dtype=torch.float64)
    ux = torch.full((6, 8), 0.04, dtype=torch.float64)
    uy = torch.full((6, 8), -0.02, dtype=torch.float64)
    f = d.equilibrium_flow(rho, ux, uy)
    assert torch.allclose(f.sum(0), rho, atol=1e-14)
    assert torch.allclose((f * d._cx).sum(0), rho * ux, atol=1e-14)
    assert torch.allclose((f * d._cy).sum(0), rho * uy, atol=1e-14)


def test_thermal_equilibrium_moment():
    d = DDF2D(6, 8, nu=0.05, alpha=0.05)
    T = torch.full((6, 8), 0.7, dtype=torch.float64)
    ux = torch.full((6, 8), 0.03, dtype=torch.float64)
    uy = torch.zeros((6, 8), dtype=torch.float64)
    g = d.equilibrium_thermal(T, ux, uy)
    assert torch.allclose(g.sum(0), T, atol=1e-14)


def test_macroscopic_recovers_uniform_state():
    d = DDF2D(5, 7, nu=0.04, alpha=0.04)
    d.init_uniform(ux=0.02, uy=-0.01, T=0.25)
    rho, ux, uy, T = d.macroscopic()
    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-13)
    assert torch.allclose(ux, torch.full_like(ux, 0.02), atol=1e-13)
    assert torch.allclose(uy, torch.full_like(uy, -0.01), atol=1e-13)
    assert torch.allclose(T, torch.full_like(T, 0.25), atol=1e-13)


def test_invalid_relaxation_raises():
    with pytest.raises(ValueError):
        DDF2D(4, 4, nu=0.7, alpha=0.05)  # tau = 2.6 > 2.5
    with pytest.raises(ValueError):
        DDF2D(4, 4, nu=0.05, alpha=1.0)  # tau_T = 3.5 > 3


# ---------------------------------------------------------------------------
# Analytic limits: conduction, diffusion, viscosity
# ---------------------------------------------------------------------------
def _box_conduction(ny=6, nx=14, alpha=0.2, nsteps=4000):
    d = DDF2D(ny, nx, nu=0.1, alpha=alpha)
    d.set_box_walls(1)
    left = torch.zeros(ny, nx, dtype=torch.bool)
    right = torch.zeros(ny, nx, dtype=torch.bool)
    left[:, 0] = True
    right[:, -1] = True
    d.set_temperature_bc(left, DDF2D.DIRICHLET, 1.0)
    d.set_temperature_bc(right, DDF2D.DIRICHLET, 0.0)
    d.build_thermal_links()
    d.init_uniform(T=0.5)
    d.step(nsteps)
    d.macroscopic()
    return d


def test_pure_conduction_nusselt_is_unity():
    d = _box_conduction()
    H = d.nx - 2
    q = d.wall_flux_normal("x", 0)
    nu = q[1:-1] * H / d.alpha
    assert float(nu.min()) == pytest.approx(1.0, abs=1e-8)
    assert float(nu.max()) == pytest.approx(1.0, abs=1e-8)


def test_pure_conduction_profile_is_linear():
    d = _box_conduction()
    H = d.nx - 2
    ana = 1.0 - (torch.arange(d.nx, dtype=torch.float64) - 0.5) / H
    interior = d.T[1:-1, 1:-1]
    ref = ana[1:-1].reshape(1, -1).expand_as(interior)
    assert float((interior - ref).abs().max()) < 1e-8


def test_adiabatic_walls_do_not_leak_heat():
    """With only side walls isothermal, the field must be y-invariant."""
    d = _box_conduction(ny=8, nx=12, alpha=0.2, nsteps=3000)
    spread = float(d.T[1:-1, 1:-1].std(dim=0).max())
    assert spread < 1e-10


def test_thermal_diffusivity_matches_analytic():
    nx = 32
    alpha = 0.03
    d = DDF2D(8, nx, nu=0.02, alpha=alpha, dtype=torch.float64)
    x = torch.arange(nx, dtype=torch.float64)
    T = 0.5 + 0.3 * torch.sin(2 * math.pi * x / nx).reshape(1, -1).expand(8, nx).clone()
    d.init_fields(T=T)
    k = 2 * math.pi / nx

    def amp():
        _, _, _, Tf = d.macroscopic()
        return float(((Tf[3] - Tf[3].mean()) * torch.sin(k * x)).sum())

    a0 = amp()
    d.step(2000)
    a1 = amp()
    alpha_eff = math.log(a0 / a1) / 2000 / k ** 2
    assert alpha_eff == pytest.approx(alpha, rel=0.01)


def test_kinematic_viscosity_matches_analytic():
    nx = 32
    nu = 0.02
    d = DDF2D(nx, nx, nu=nu, alpha=0.03, dtype=torch.float64)
    x = torch.arange(nx, dtype=torch.float64)
    X, Y = x.reshape(1, -1), x.reshape(-1, 1)
    k = 2 * math.pi / nx
    ux = 0.01 * torch.sin(k * X) * torch.cos(k * Y)
    uy = -0.01 * torch.cos(k * X) * torch.sin(k * Y)
    rho = torch.ones(nx, nx, dtype=torch.float64)
    d.f = d.equilibrium_flow(rho, ux.expand(nx, nx).clone(), uy.expand(nx, nx).clone())

    def amp():
        _, u1, _, _ = d.macroscopic()
        return float((u1 * torch.sin(k * X) * torch.cos(k * Y)).sum())

    a0 = amp()
    d.step(1500)
    a1 = amp()
    nu_eff = math.log(a0 / a1) / 1500 / (2 * k ** 2)
    assert nu_eff == pytest.approx(nu, rel=0.02)


def test_periodic_domain_conserves_energy():
    """Toroidal streaming is a permutation, so sum(g) is conserved exactly."""
    d = DDF2D(10, 10, nu=0.05, alpha=0.05, dtype=torch.float64)
    T = torch.rand(10, 10, dtype=torch.float64)
    d.init_fields(T=T)
    e0 = d.total_energy()
    d.step(500)
    assert d.total_energy() == pytest.approx(e0, rel=1e-12)
    _, _, _, Tf = d.macroscopic()
    assert float(Tf.min()) > 0.0
    assert float((Tf - Tf.mean()).abs().max()) < float((T - T.mean()).abs().max())


def test_solid_slab_conduction_linear():
    s = Conduction2D(11, 8, alpha=0.05)
    s.add_link(3, row=10, T=1.0)   # interface half a cell above row 10
    s.add_link(4, row=1, T=0.0)    # cold wall half a cell below row 1
    s.init_uniform(0.5)
    s.step(8000)
    T = s.macroscopic()[:, 4]
    for i in range(1, 11):
        assert float(T[i]) == pytest.approx((i - 0.5) / 10.0, abs=2e-3)


def test_conduction_adiabatic_link_zero_flux():
    s = Conduction2D(8, 6, alpha=0.05)
    s.add_link(3, row=0, T=0.5)     # adiabatic top (reflection)
    s.add_link(4, row=7, T=0.5)
    s.init_uniform(0.5)
    s.step(500)
    T = s.macroscopic()
    assert float((T - 0.5).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# Buoyancy coupling
# ---------------------------------------------------------------------------
def test_buoyancy_force_direction():
    """A warm patch must accelerate upward (gravity along -y)."""
    d = DDF2D(8, 8, nu=0.05, alpha=0.05, gravity=1e-3, beta=1.0, T_ref=0.0)
    T = torch.full((8, 8), 0.0, dtype=torch.float64)
    T[2:6, 2:6] = 1.0
    d.init_fields(T=T)
    d.step(20)
    _, _, uy, _ = d.macroscopic()
    assert float(uy[1:7, 1:7].mean()) > 0.0


# ---------------------------------------------------------------------------
# Nusselt diagnostics and correlations
# ---------------------------------------------------------------------------
def test_cylinder_polar_nusselt_on_synthetic_field():
    """Nu(theta) from a prescribed 1/r-free linear radial profile is constant."""
    ny = nx = 81
    y = torch.arange(ny, dtype=torch.float64).reshape(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).reshape(1, -1)
    cx = cy = 40.0
    r = ((x - cx) ** 2 + (y - cy) ** 2).sqrt().clamp_min(1e-12)
    d_eff = 21.0
    r_wall = 10.5
    # T = 1 - (r - r_wall)/d_eff  -> dT/dr = -1/d_eff -> Nu = 1
    T = 1.0 - (r - r_wall) / d_eff
    res = nu_cylinder_polar(T, cx, cy, r_wall, d_eff, ntheta=180)
    assert res["nu_mean"] == pytest.approx(1.0, rel=2e-2)


def test_churchill_bernstein_reference_values():
    # Re = 100, Pr = 0.7 is the case-(a) reference point
    assert churchill_bernstein(100.0, 0.7) == pytest.approx(5.156, abs=5e-3)
    nu = [churchill_bernstein(re, 0.7) for re in (40.0, 100.0, 200.0)]
    assert nu[0] < nu[1] < nu[2]
    # scaling with Re^0.5 dominates at moderate Re
    assert nu[2] / nu[1] == pytest.approx(math.sqrt(2.0), rel=0.12)


def test_de_vahl_davis_lookup():
    assert de_vahl_davis_nu(1e4) == pytest.approx(2.243)
    assert de_vahl_davis_nu(1e5) == pytest.approx(4.519)
    with pytest.raises(KeyError):
        de_vahl_davis_nu(1234.0)


def test_fully_developed_channel_nusselt():
    assert channel_nusselt_fully_developed() == pytest.approx(7.541)


# ---------------------------------------------------------------------------
# Case runners (smoke, small settings)
# ---------------------------------------------------------------------------
def test_cavity_smoke_finite_and_balanced():
    r = cavity_natural_convection(ra=1e4, n=20, nsteps=3000, check_every=500,
                                  dtype=torch.float32)
    assert math.isfinite(r["nu_hot"]) and r["nu_hot"] > 1.0
    assert r["energy_balance_rel"] < 1e-2
    assert r["fields"]["T"].shape == (20, 20)


def test_cylinder_smoke_positive_nusselt():
    r = cylinder_forced_convection(re=40.0, n_d=8, nsteps=1200, check_every=200,
                                   tol=0.0, dtype=torch.float32)
    assert 0.5 < r["nu_mean"] < 20.0
    assert r["nu_front"] > r["nu_rear"]  # stagnation point transfers most heat
    assert r["n_wall_links"] > 0


def test_conjugate_interface_flux_conservation_machine_precision():
    r = conjugate_channel(hf=10, ts=5, length=48, nsteps=1500, check_every=250,
                          dtype=torch.float64)
    assert r["resid_interface"] < 1e-12
    assert r["dev_center_max"] < 0.5  # not yet steady, but physically bounded


def test_conjugate_steady_center_temperature_within_network():
    r = conjugate_channel(hf=12, ts=6, length=96, nsteps=6000, check_every=500,
                          dtype=torch.float32)
    assert r["dev_center_tail"] < 0.05
    assert r["T_center"][-2] < r["T_bulk"][-2] < 1.0


def test_rayleigh_growth_rate_smoke():
    r = rayleigh_growth_rate(ra=2500.0, n=20, nsteps=2000, sample_every=100,
                             dtype=torch.float32)
    assert math.isfinite(r["sigma"])
    assert r["e_max"] > 0.0


def test_rayleigh_growth_rate_amplitude_is_resolved():
    """Strongly subcritical decays; strongly supercritical saturates at a finite
    amplitude.  The *amplitude* is the robust discriminator - the rate itself is
    only resolved for large distances from the threshold."""
    sub = rayleigh_growth_rate(ra=200.0, n=16, nsteps=900, sample_every=100,
                               dtype=torch.float32)
    sup = rayleigh_growth_rate(ra=6000.0, n=16, nsteps=900, sample_every=100,
                               dtype=torch.float32)
    assert sup["e_max"] > 10.0 * sub["e_max"]