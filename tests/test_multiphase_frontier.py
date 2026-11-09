"""Tests for the campaign p40w2-n02 multiphase frontier surface.

Fast suite (a few seconds): every solver is checked against the *closed form* or
an exact invariant of the same physics, on grids small enough for CI.

* rheology  : quadrature vs power-law closed form, FV channel/pipe vs
              quadrature, Newtonian limit, second-order convergence
* cavitation: Rayleigh collapse time, Minnaert frequency, volume-fraction
              closure, Schnerr-Sauer geometric identity
* VIV       : structure integrator vs the analytic damped oscillator, spectral
              estimators, staircase body area, frame-transformation invariants
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.multiphase_frontier import (
    CarreauYasuda,
    Cross,
    PowerLaw,
    RayleighPlesset,
    channel_finite_volume,
    dominant_frequency,
    elastic_mount_response,
    free_decay_verlet,
    make_model,
    minnaert_frequency,
    pipe_finite_volume,
    poiseuille_channel_profile,
    poiseuille_pipe_profile,
    profile_error,
    rayleigh_collapse_time,
    schnerr_sauer_source,
)
from tensorlbm.multiphase_frontier.cavitation_dynamics import (
    alpha_from_radius,
    radius_from_alpha,
)
from tensorlbm.multiphase_frontier.viv import (
    VIVFrameSolver,
    period_from_peaks,
    stationary_strouhal,
)

MODELS = ["power_law", "power_law_shear_thickening", "carreau_yasuda", "cross"]


# ---------------------------------------------------------------------------
# rheology
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", MODELS)
def test_apparent_viscosity_is_positive_and_monotone(key):
    """Shear thinning must give d(mu)/d(gamma) < 0 and vice versa."""
    model = make_model(key)
    g = np.logspace(-6, 2, 200)
    mu = model.mu(g)
    assert np.all(mu > 0.0)
    sign = np.sign(np.diff(mu))
    thinning = not (key.endswith("shear_thickening"))
    if thinning:
        assert np.all(sign <= 1e-12), f"{key}: mu must not increase with gamma"
    else:
        assert np.all(sign >= -1e-12), f"{key}: mu must not decrease"


def test_power_law_closed_form_matches_quadrature():
    """The quadrature solver reproduces the analytic power-law profile."""
    pl = PowerLaw(K=0.02, n=0.6)
    y = np.linspace(-16, 16, 65)
    exact = pl.channel_profile_exact(y, 16.0, 3.73e-4)
    quad = poiseuille_channel_profile(pl, y, 16.0, 3.73e-4, force_quadrature=True)
    err = profile_error(quad, exact)
    assert err["rel_linf"] < 1e-6, err


@pytest.mark.parametrize("key", MODELS)
def test_channel_finite_volume_matches_quadrature(key):
    """Plane-Poiseuille FV solution vs the quadrature reference (<0.5 %)."""
    model = make_model(key)
    res = channel_finite_volume(model, n_y=64, h=1.0, grad_p=5e-3)
    ref = poiseuille_channel_profile(model, res["y"], 1.0, 5e-3)
    err = profile_error(res["u"], ref)
    assert err["rel_l2"] < 5e-3, err
    assert abs(res["u_max"] - ref.max()) / ref.max() < 5e-3


@pytest.mark.parametrize("key", MODELS)
def test_pipe_finite_volume_matches_quadrature(key):
    """Pipe-Poiseuille FV solution vs the quadrature reference (<0.5 %)."""
    model = make_model(key)
    res = pipe_finite_volume(model, n_r=128, R=1.0, grad_p=0.3)
    ref = poiseuille_pipe_profile(model, res["r"], 1.0, 0.3)
    err = profile_error(res["u"], ref)
    assert err["rel_l2"] < 5e-3, err


def test_newtonian_limit_is_exact():
    """n = 1 must reproduce the analytic parabola to discretisation accuracy."""
    mu = 0.05
    model = PowerLaw(K=mu, n=1.0)
    res = pipe_finite_volume(model, n_r=200, R=1.0, grad_p=1.0)
    exact = 1.0 * 1.0 ** 2 / (4.0 * mu)
    assert abs(res["u_max"] - exact) / exact < 1e-4
    res2 = channel_finite_volume(model, n_y=200, h=1.0, grad_p=1.0)
    exact2 = 1.0 * 1.0 ** 2 / (2.0 * mu)
    assert abs(res2["u_max"] - exact2) / exact2 < 1e-4


def test_finite_volume_is_second_order_accurate():
    """Refining by 4x must shrink the profile error by roughly 16x."""
    model = make_model("carreau_yasuda")
    err = []
    for n in (32, 128):
        res = channel_finite_volume(model, n_y=n, h=1.0, grad_p=1e-2)
        ref = poiseuille_channel_profile(model, res["y"], 1.0, 1e-2)
        err.append(profile_error(res["u"], ref)["rel_l2"])
    assert err[1] < err[0] / 8.0, err


def test_models_have_distinct_physics():
    """The three constitutive laws must not collapse onto each other."""
    g = np.array([1e-3, 1e-1, 1e1])
    mu = {k: make_model(k).mu(g) for k in MODELS}
    assert not np.allclose(mu["carreau_yasuda"], mu["cross"])
    assert not np.allclose(mu["power_law"], mu["carreau_yasuda"])


# ---------------------------------------------------------------------------
# cavitation dynamics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dp", [1.0e5, 1.0e6])
def test_rayleigh_collapse_time(dp):
    """Integrated Rayleigh-Plesset collapse vs 0.9147 R0 sqrt(rho/dp)."""
    rho, p_v = 1.0e3, 2.3e3
    rp = RayleighPlesset(R0=1.0e-4, p_inf=p_v + dp, p_v=p_v, rho=rho,
                         sigma=0.0, mu=0.0)
    out = rp.collapse_time(dt=rayleigh_collapse_time(1.0e-4, dp, rho) / 50000.0)
    assert abs(out["T_c"] - out["T_rayleigh"]) / out["T_rayleigh"] < 0.01


def test_minnaert_frequency_against_linear_response():
    """Natural frequency of a small gas bubble vs (1/2pi R0) sqrt(3 kappa p/rho)."""
    R0, p0, rho = 1.0e-4, 1.0e5, 1.0e3
    rp = RayleighPlesset(R0=R0, p_inf=p0, p_v=0.0, rho=rho, sigma=0.0, mu=0.0,
                         p_g0=p0, kappa=1.4, R_init_frac=1.0e-3)
    T_m = 1.0 / minnaert_frequency(R0, p0, rho, 1.4)
    res = rp.integrate(t_end=4.0 * T_m, dt=T_m / 2000.0)
    t, R = res["t"], res["R"]
    spec = np.abs(np.fft.rfft((R - R.mean()) * np.hanning(len(R))))
    fr = np.fft.rfftfreq(len(R), d=float(t[1] - t[0]))
    spec[0] = 0.0
    f_num = fr[int(np.argmax(spec))]
    assert abs(f_num - 1.0 / T_m) / (1.0 / T_m) < 0.05, (f_num, 1.0 / T_m)


def test_alpha_radius_closure_roundtrip():
    n_bub = 1.0e8
    a = np.array([1e-4, 0.01, 0.3, 0.7])
    R = radius_from_alpha(a, n_bub)
    a_back = alpha_from_radius(R, n_bub)
    assert np.allclose(a, a_back, rtol=1e-10)


def test_schnerr_sauer_geometric_identity():
    """d alpha/dt = 3 alpha (1-alpha) Rdot / R holds identically along RP."""
    rho, n_bub = 1.0e3, 1.0e8
    rp = RayleighPlesset(R0=2.0e-4, p_inf=5.0e4, p_v=2.3e3, rho=rho,
                         sigma=0.0728, mu=1.0e-3)
    T = rayleigh_collapse_time(2.0e-4, 5.0e4, rho)
    res = rp.integrate(t_end=0.5 * T, dt=T / 8000.0)
    R, Rd, t = res["R"], res["Rdot"], res["t"]
    a = alpha_from_radius(R, n_bub)
    mask = (a > 1e-3) & (a < 0.9)
    S_num = np.gradient(a, t)[mask]
    S_geo = (3.0 * a * (1.0 - a) / R * Rd)[mask]
    rel = np.abs(S_num - S_geo) / np.maximum(np.abs(S_geo), 1e-12)
    assert np.median(rel) < 0.02, np.median(rel)


def test_schnerr_sauer_sign_follows_pressure():
    """Production when p < p_v, destruction otherwise."""
    a = np.array([0.1])
    p_low, p_high = 1.0e3, 1.0e5          # below and above p_v = 2.3e3
    src_low = schnerr_sauer_source(a, np.array([p_low]), 2.3e3, 1.0e3, 1e-4)
    src_high = schnerr_sauer_source(a, np.array([p_high]), 2.3e3, 1.0e3, 1e-4)
    assert src_low[0] > 0.0 > src_high[0]
    assert abs(src_low[0] / src_high[0]) == pytest.approx(
        np.sqrt(2.3e3 - p_low) / np.sqrt(p_high - 2.3e3), rel=1e-12)


# ---------------------------------------------------------------------------
# VIV
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("zeta", [0.0, 0.05])
def test_structure_integrator_matches_analytic_decay(zeta):
    """Velocity-Verlet vs the exact damped-oscillator solution."""
    w = 2.0 * np.pi / 50.0
    n, dt = 2000, 0.25
    num = free_decay_verlet(w, zeta, 1.0, n, dt=dt)
    ana = elastic_mount_response(w, zeta, 1.0, n, dt=dt)
    assert np.max(np.abs(num["x"] - ana["x"])) < 5e-3


def test_dominant_frequency_recovers_sinusoid():
    t = np.arange(4000, dtype=float)
    f0 = 1.0 / 973.0
    y = np.sin(2 * np.pi * f0 * t)
    assert abs(dominant_frequency(t, y)["f"] - f0) / f0 < 0.02
    pk = period_from_peaks(t, y)
    assert abs(pk["T"] * f0 - 1.0) < 0.01
    assert pk["n_peaks"] >= 3


def test_stationary_strouhal_trend():
    """St(Re) of a fixed cylinder increases monotonically with Re."""
    Re = np.array([100.0, 200.0, 500.0, 1000.0])
    st = np.array([stationary_strouhal(r) for r in Re])
    assert np.all(np.diff(st) > 0.0)
    assert abs(stationary_strouhal(100.0) - 0.164) < 0.01


def test_staircase_body_area_close_to_circle():
    s = VIVFrameSolver(D=16, nx=256, ny=160, Re=100.0, U=0.1, fixed=True)
    exact = np.pi * (0.5 * 16) ** 2
    assert abs(s.solid_area - exact) / exact < 0.03
    assert s.links > 100          # enough boundary links for the load integral


def test_solver_rejects_unstable_relaxation_time():
    with pytest.raises(ValueError):
        VIVFrameSolver(D=16, nx=128, ny=96, Re=1000.0, U=0.1, fixed=True)


def test_frame_transform_preserves_symmetry_and_mass():
    """A fixed cylinder in an exactly symmetric setup keeps Cl = 0 and rho ~ 1.

    This is the invariant that the *accelerating-frame* formulation must
    respect: with no structural motion the pseudo-acceleration vanishes and the
    solver is a plain fixed-cylinder LBM.
    """
    s = VIVFrameSolver(D=12, nx=192, ny=160, Re=100.0, U=0.1, fixed=True)
    for _ in range(400):
        out = s.step()
    assert abs(out["cl"]) < 1e-3
    assert 0.9 < s.rho.min() and s.rho.max() < 1.1
    assert out["cd"] > 0.5                # a cylinder at Re=100 pushes ~1.3


def test_elastic_mount_responds_to_lift():
    """With the structure free, a lift imbalance must produce motion."""
    s = VIVFrameSolver(D=12, nx=192, ny=160, Re=100.0, U=0.1, m_star=10.0,
                       zeta=0.0, U_r=5.0, x0_frac=0.05)
    assert s.x_s == pytest.approx(0.05 * 12)
    x0 = s.x_s
    for _ in range(300):
        s.step()
    assert np.isfinite(s.x_s) and s.x_s != x0
    assert abs(s.a_s) >= 0.0


def test_grid_and_geometry_guards():
    s = VIVFrameSolver(D=10, nx=200, ny=140, Re=100.0, U=0.1, U_r=4.0,
                       m_star=10.0)
    assert s.U_r == 4.0
    assert s.f_n == pytest.approx(0.1 / (4.0 * 10.0))
    assert s.omega_n == pytest.approx(2 * np.pi * s.f_n)
    assert s.k == pytest.approx(s.m * s.omega_n ** 2)