"""P171W1-N02 regression guards for the Lighthill volume quadrature.

These tests encode the two findings of the maturity pass:

1. the Kirchhoff surface integral needs the normal derivative
   ``dG/dn_y = -G'(r) (rhat . n)`` -- a missing minus sign produced a
   constant 0.583 dB (0.9351 amplitude) bias against the volume integral;
2. negative controls: a global sign flip of ``T_ij`` is a pure pi phase
   shift of the linear acoustic field, while a component flip or a pure
   monopole destroys the four-lobe directivity / frequency law.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.lighthill_volume import (
    co_rotating_vortex_pair,
    kirchhoff_surface_2d,
    rotating_quadrupole_analysis,
    volume_quadrupole_frequency_domain,
    gaussian_quadrupole_farfield_exact,
)

C0 = 1.0 / math.sqrt(3.0)


def _gaussian_quadrupole(omega=0.3, sigma_over_lambda=0.05, dx_over_sigma=0.25):
    k = omega / C0
    lam = 2 * math.pi / k
    sigma = sigma_over_lambda * lam
    half = 4.0 * sigma
    dx = dx_over_sigma * sigma
    n = int(round(2 * half / dx)) + 1
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")
    g = np.exp(-(xx ** 2 + yy ** 2) / sigma ** 2)
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    T = (g[..., None, None] * A).reshape(-1, 2, 2)
    src = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return omega, lam, sigma, T, src, np.full(xx.size, dx * dx)


def _pair_spectrum(M=0.05, n_t=48, n_obs=64, r_over_lambda=10.0):
    half_sep, core, mask_radius, dx = 6.0, 2.5, 16.0, 0.5
    gamma = M * C0 * 4.0 * math.pi * half_sep
    omega_rot = gamma / (4.0 * math.pi * half_sep ** 2)
    omega2 = 2.0 * omega_rot
    k = omega2 / C0
    lam = 2.0 * math.pi / k
    r = r_over_lambda * lam
    L = mask_radius + 0.5
    x = np.arange(-L, L + 1e-12, dx)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mask = (xx ** 2 + yy ** 2) <= mask_radius ** 2
    xs, ys = xx[mask], yy[mask]
    src = np.stack([xs, ys], axis=-1)
    w = np.full(xs.size, dx * dx)
    th = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    period = math.pi / omega_rot
    dt = period / n_t
    T_hat = np.zeros((xs.size, 2, 2), dtype=complex)
    for it in range(n_t):
        t = it * dt
        ux, uy = co_rotating_vortex_pair(xs, ys, gamma, half_sep, core, t)
        T = np.empty((xs.size, 2, 2))
        T[:, 0, 0] = ux * ux
        T[:, 0, 1] = T[:, 1, 0] = ux * uy
        T[:, 1, 1] = uy * uy
        T_hat += T * np.exp(-1j * omega2 * t) * dt
    T_hat *= 2.0 / period
    return dict(src=src, w=w, obs=obs, th=th, T_hat=T_hat, omega2=omega2,
                omega_rot=omega_rot)


def _pressure(s, T_hat, omega):
    return volume_quadrupole_frequency_domain(
        T_hat, s["src"], s["w"], s["obs"], omega, C0, ndim=2, batch_size=16)


def test_kirchhoff_normal_derivative_sign():
    """Regression: surface and volume chains must agree to << 3 dB (actual 1e-3 dB)."""
    omega, lam, sigma, T, src, w = _gaussian_quadrupole()
    r_surface = lam
    n_surf = 128
    ths = np.linspace(0, 2 * np.pi, n_surf, endpoint=False)
    S = np.stack([r_surface * np.cos(ths), r_surface * np.sin(ths)], axis=-1)
    nrm = S / r_surface
    dr = lam / 400.0

    def vol(obs):
        return volume_quadrupole_frequency_domain(
            T, src, w, np.atleast_2d(obs), omega, C0, ndim=2, batch_size=16)

    p_s = vol(S)
    dp_dn = (vol(S + dr * nrm) - vol(S - dr * nrm)) / (2 * dr)
    ds = 2 * np.pi * r_surface / n_surf
    obs = np.stack([2 * lam * np.ones(4),
                    2 * lam * np.array([0.0, 0.3, 1.0, 1.3])], axis=-1)
    obs = np.stack([2 * lam * np.cos(np.array([0.0, 0.3, 1.0, 1.3])),
                    2 * lam * np.sin(np.array([0.0, 0.3, 1.0, 1.3]))], axis=-1)
    p_vol = vol(obs)
    p_kir = kirchhoff_surface_2d(
        p_s, dp_dn, S, nrm, np.full(n_surf, ds), obs, omega, C0)
    ratio = np.abs(p_kir) / np.abs(p_vol)
    # The inherited (wrong-sign) implementation gives 0.9351, i.e. 0.58 dB.
    assert np.max(np.abs(ratio - 1.0)) < 1e-3
    assert np.max(np.abs(20 * np.log10(ratio))) < 0.01


def test_global_sign_flip_is_a_pure_phase_shift():
    """T_ij -> -T_ij must leave frequency, |amplitude| and |pattern| invariant."""
    s = _pair_spectrum()
    p = _pressure(s, s["T_hat"], s["omega2"])
    p_neg = _pressure(s, -s["T_hat"], s["omega2"])
    a = rotating_quadrupole_analysis(p, s["th"], s["omega2"])
    b = rotating_quadrupole_analysis(-p, s["th"], s["omega2"])
    assert np.allclose(p_neg, -p, rtol=1e-10, atol=1e-18)
    assert abs(b["amplitude"] - a["amplitude"]) / a["amplitude"] < 1e-10
    assert b["pattern_corr_cos2"] == pytest.approx(a["pattern_corr_cos2"], abs=1e-10)
    # phase-sensitive operator check: the flip is visible as a complex sign change
    assert np.mean(np.abs(p_neg - p) / np.abs(p)) == pytest.approx(2.0, rel=1e-10)


def test_component_sign_flip_breaks_four_lobe_directivity():
    s = _pair_spectrum()
    T = s["T_hat"].copy()
    T[:, 0, 0] *= -1.0
    a = rotating_quadrupole_analysis(_pressure(s, T, s["omega2"]), s["th"], s["omega2"])
    assert a["pattern_corr_cos2"] < 0.9
    assert abs(a["winding"]) < 1.0


def test_pure_monopole_radiates_at_rotation_frequency_and_is_isotropic():
    s = _pair_spectrum()
    sigma, L, dx = 2.5, 16.0, 0.5
    x = np.arange(-L, L + 1e-12, dx)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mask = (xx ** 2 + yy ** 2) <= 16.0 ** 2
    xs, ys = xx[mask], yy[mask]
    src = np.stack([xs, ys], axis=-1)
    w = np.full(xs.size, dx * dx)
    g = np.exp(-(xs ** 2 + ys ** 2) / sigma ** 2)
    T_mono = np.eye(2)[None, :, :] * (0.05 ** 2) * g[:, None, None]
    p1 = volume_quadrupole_frequency_domain(
        T_mono, src, w, s["obs"], s["omega_rot"], C0, ndim=2, batch_size=16)
    p2 = volume_quadrupole_frequency_domain(
        np.zeros_like(T_mono), src, w, s["obs"], s["omega2"], C0, ndim=2, batch_size=16)
    a = rotating_quadrupole_analysis(p1, s["th"], s["omega_rot"])
    assert np.abs(p1).mean() > np.abs(p2).mean()   # energy at f_rot, not 2 f_rot
    assert a["isotropy"] < 1e-6                     # no four-lobe structure
    assert a["pattern_corr_cos2"] < 0.5


def test_operator_anchor_against_exact_closed_form():
    """C1 regression: 2-D volume operator <= 2 % against the exact Hankel solution."""
    omega, lam, sigma, T, src, w = _gaussian_quadrupole(sigma_over_lambda=0.1)
    k = omega / C0
    r = 30 * lam
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    p_num = volume_quadrupole_frequency_domain(
        T, src, w, obs, omega, C0, ndim=2, batch_size=12)
    p_ana = gaussian_quadrupole_farfield_exact(obs, omega, C0, 1.0, sigma)
    ref = np.abs(p_ana)
    lobe = ref > 0.2 * ref.max()
    err = np.abs(np.abs(p_num[lobe]) - ref[lobe]) / ref[lobe]
    assert err.max() < 0.02
