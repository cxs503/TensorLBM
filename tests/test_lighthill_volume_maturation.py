"""Maturation tests for the Lighthill volume-quadrupole chain (P170W2B-N03).

These tests encode the acceptance criteria of the maturation node on top of the
inherited P19-N06 / P171W1-N04 work:

A. operator anchor    : 2-D volume-quadrupole integral vs the exact
                        (non-compact) far-field closed form      -> <= 2 %
B. retarded time      : delayed-time 3-D FW-H volume term vs the exact
                        frequency-domain operator               -> <= 2 %
C. vortex-pair physics: peak at 2 f_rot, phase-locked |cos 2theta| four lobes,
                        winding number 2, Mach-normalised amplitude ~ M^3
D. permeable surface  : Kirchhoff surface integral vs volume integral at
                        r/lambda >= 2                              -> <= 3 dB
E. compact limit      : exact finite-size vs point-quadrupole closed form, and
                        closed-form acoustic power vs the numerical far-field
                        intensity integral                     -> <= 2 %

The heavy vortex-pair analysis is cached per module (``_CACHE``) so the suite
runs in well under two minutes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.lighthill_volume import (
    analytical_gaussian_quadrupole_farfield,
    co_rotating_vortex_pair,
    compact_quadrupole_power,
    compute_lighthill_tensor,
    gaussian_quadrupole_farfield_3d,
    gaussian_quadrupole_farfield_exact,
    kirchhoff_surface_2d,
    rotating_quadrupole_analysis,
    volume_quadrupole_frequency_domain,
    volume_quadrupole_time_domain,
)

C0 = 1.0 / math.sqrt(3.0)
_CACHE: dict = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def gaussian_source_2d(omega, sigma, extent_sigma=5.0, dx_over_sigma=0.25):
    half = extent_sigma * sigma
    dx = dx_over_sigma * sigma
    n = int(round(2 * half / dx)) + 1
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")
    g = np.exp(-(xx ** 2 + yy ** 2) / sigma ** 2)
    T = g[..., None, None] * np.array([[1.0, 0.0], [0.0, -1.0]])
    return (T.reshape(-1, 2, 2), np.stack([xx.ravel(), yy.ravel()], axis=-1),
            np.full(xx.size, dx * dx))


def _farfield_error_2d(omega, sigma, r_over_lambda, n_obs=24, dx_over_sigma=0.25):
    k = omega / C0
    lam = 2 * math.pi / k
    T, src, w = gaussian_source_2d(omega, sigma, dx_over_sigma=dx_over_sigma)
    th = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    r = r_over_lambda * lam
    obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
    p_num = volume_quadrupole_frequency_domain(T, src, w, obs, omega, C0, ndim=2,
                                               batch_size=12)
    p_ana = gaussian_quadrupole_farfield_exact(obs, omega, C0, 1.0, sigma)
    ref = np.abs(p_ana)
    lobe = ref > 0.2 * ref.max()          # the 45-degree nulls are ill-conditioned
    pointwise = np.abs(np.abs(p_num[lobe]) - ref[lobe]) / ref[lobe]
    l2 = np.sqrt(np.mean((np.abs(p_num) - ref) ** 2)) / ref.max()
    return float(pointwise.max()), float(l2)


def _pair_field(M, half_sep=6.0, core=2.5):
    gamma = M * C0 * 4.0 * math.pi * half_sep
    return gamma, gamma / (4.0 * math.pi * half_sep ** 2)


def vortex_pair_analysis(M, mask_radius=16.0, dx=0.5, n_obs=64, n_t=96,
                         r_over_lambda=10.0, harmonics=(1, 2, 3)):
    """Far-field complex pressure of the rotating pair (cached per M)."""
    key = ("vp", M, mask_radius, dx, n_obs, n_t, r_over_lambda, harmonics)
    if key in _CACHE:
        return _CACHE[key]

    half_sep, core = 6.0, 2.5
    gamma, omega_rot = _pair_field(M, half_sep, core)
    omega2 = 2.0 * omega_rot
    lam = 2.0 * math.pi * C0 / omega2
    r = r_over_lambda * lam

    L = mask_radius + 0.5
    x = np.arange(-L, L + 1e-12, dx)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    mask = (xx ** 2 + yy ** 2) <= mask_radius ** 2
    xs, ys = xx[mask], yy[mask]
    src = np.stack([xs, ys], axis=-1)
    w = np.full(xs.size, dx * dx)
    th = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)

    def T_at(t):
        ux, uy = co_rotating_vortex_pair(xs, ys, gamma, half_sep, core, t)
        T = np.empty((xs.size, 2, 2))
        T[:, 0, 0] = ux * ux
        T[:, 0, 1] = T[:, 1, 0] = ux * uy
        T[:, 1, 1] = uy * uy
        return T

    period = math.pi / omega_rot
    dt = period / n_t
    T_hat = np.zeros((xs.size, 2, 2), dtype=complex)
    for it in range(n_t):
        t = it * dt
        T_hat += T_at(t) * np.exp(-1j * omega2 * t) * dt
    T_hat *= 2.0 / period
    p_hat = volume_quadrupole_frequency_domain(T_hat, src, w, obs, omega2, C0,
                                               ndim=2, batch_size=16)
    analysis = rotating_quadrupole_analysis(p_hat, th, omega2)

    harm = {}
    per_full = 2 * math.pi / omega_rot
    nts = 4 * n_t
    dts = per_full / nts
    cache = [T_at(it * dts) for it in range(nts)]
    for n in harmonics:
        w_n = n * omega_rot
        T_hat_n = np.zeros((xs.size, 2, 2), dtype=complex)
        for it in range(nts):
            T_hat_n += cache[it] * np.exp(-1j * w_n * it * dts) * dts
        T_hat_n *= 2.0 / per_full
        p_n = volume_quadrupole_frequency_domain(T_hat_n, src, w, obs, w_n, C0,
                                                 ndim=2, batch_size=16)
        harm[n] = float(np.abs(p_n).mean())

    out = dict(p_hat=p_hat, th=th, analysis=analysis, harmonic=harm,
               omega_rot=omega_rot, omega2=omega2, lam=lam, r=r)
    _CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# A. Operator anchor
# ---------------------------------------------------------------------------

class TestAOperatorAnchor:
    """2-D volume-quadrupole operator vs exact far-field closed form."""

    @pytest.mark.parametrize("s_lam", [0.02, 0.10, 0.20])
    def test_non_compact_source_below_2pct(self, s_lam):
        omega = 0.15
        lam = 2 * math.pi * C0 / omega
        max_err, l2 = _farfield_error_2d(omega, s_lam * lam, r_over_lambda=30.0)
        assert max_err < 0.02, f"sigma/lambda={s_lam}: lobe error {max_err:.2e}"
        assert l2 < 0.02, f"sigma/lambda={s_lam}: L2 error {l2:.2e}"

    def test_grid_convergence(self):
        omega = 0.15
        lam = 2 * math.pi * C0 / omega
        sigma = 0.10 * lam
        errs = [_farfield_error_2d(omega, sigma, 30.0, dx_over_sigma=d)[0]
                for d in (1.0, 0.5, 0.25)]
        assert errs[-1] < 0.02
        assert errs[-1] <= errs[0], f"no convergence: {errs}"

    def test_far_field_truncation_scales_like_one_over_kr(self):
        """Residual of the closed form is the O(1/(kr)) stationary-phase term."""
        omega = 0.15
        lam = 2 * math.pi * C0 / omega
        sigma = 0.10 * lam
        _, l2_small = _farfield_error_2d(omega, sigma, 10.0)
        _, l2_large = _farfield_error_2d(omega, sigma, 80.0)
        assert l2_large < l2_small
        assert l2_large < 0.005


# ---------------------------------------------------------------------------
# B. Retarded-time (delayed-time) volume term
# ---------------------------------------------------------------------------

def _retarded_time_case(spp=64, include_near_field=True):
    key = ("td", spp, include_near_field)
    if key in _CACHE:
        return _CACHE[key]
    omega, sigma = 0.8, 0.25
    lam = 2 * math.pi * C0 / omega
    per = 2 * math.pi / omega
    dx = sigma / 2.5
    n = 11
    x = (np.arange(n) - n // 2) * dx
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    g = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
    TT = g[..., None, None] * np.diag([1.0, -1.0, 0.0])
    org = (-(n // 2) * dx,) * 3
    r = 25 * lam
    th = np.array([0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3])
    obs = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=-1)
    dt = per / spp
    n_t = int((25 + 2.5) * spp)
    T = np.empty((n_t, n, n, n, 3, 3), dtype=np.float32)
    for it in range(n_t):
        T[it] = (TT * math.cos(omega * it * dt)).astype(np.float32)

    p_td, t_td = volume_quadrupole_time_domain(T, dx, obs, C0, dt, grid_origin=org,
                                               include_near_field=include_near_field)
    p_fx = volume_quadrupole_frequency_domain(
        TT.reshape(-1, 3, 3),
        np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1),
        np.full(n ** 3, dx ** 3), obs, omega, C0, ndim=3, batch_size=6)
    p_ana = gaussian_quadrupole_farfield_3d(obs, omega, C0, 1.0, sigma)

    start = int(np.argmax(t_td > (r + 2.0) / C0))
    sel = slice(start, start + spp)
    tt = t_td[sel]
    errs, phases = [], []
    for i in range(obs.shape[0]):
        A = np.column_stack([np.cos(omega * tt), np.sin(omega * tt)])
        coef, *_ = np.linalg.lstsq(A, p_td[i, sel], rcond=None)
        amp = float(np.hypot(*coef))
        ph = float(math.atan2(-coef[1], coef[0]))
        errs.append(abs(amp - abs(p_fx[i])) / abs(p_fx[i]))
        phases.append(abs(np.angle(np.exp(1j * (ph - np.angle(p_fx[i]))))))
    out = dict(err=float(max(errs)), phase=float(max(phases)),
               err_vs_analytic=float(max(abs(np.hypot(*np.linalg.lstsq(
                   np.column_stack([np.cos(omega * tt), np.sin(omega * tt)]),
                   p_td[i, sel], rcond=None)[0]) - abs(p_ana[i])) / abs(p_ana[i])
                   for i in range(obs.shape[0]))),
               p_td=p_td, t_td=t_td, sel=sel, omega=omega, amp_ana=float(abs(p_ana[0])))
    _CACHE[key] = out
    return out


class TestBRetardedTime:
    """Delayed-time 3-D FW-H volume term."""

    def test_matches_exact_operator_within_2pct(self):
        case = _retarded_time_case(spp=64, include_near_field=True)
        assert case["err"] < 0.02, f"amplitude error {case['err']:.2e}"
        assert case["phase"] < 0.05, f"phase error {case['phase']:.2e} rad"

    def test_matches_analytic_farfield_within_2pct(self):
        case = _retarded_time_case(spp=64, include_near_field=True)
        assert case["err_vs_analytic"] < 0.02, f"vs analytic: {case['err_vs_analytic']:.2e}"

    def test_near_field_terms_fix_the_phase(self):
        """The 1/r^2 and 1/r^3 corrections remove the far-field phase bias."""
        far_only = _retarded_time_case(spp=64, include_near_field=False)
        three_term = _retarded_time_case(spp=64, include_near_field=True)
        assert far_only["phase"] > 10 * three_term["phase"]
        assert three_term["phase"] < 1e-3

    def test_time_resolution_convergence(self):
        coarse = _retarded_time_case(spp=48, include_near_field=True)
        fine = _retarded_time_case(spp=96, include_near_field=True)
        assert fine["err"] <= coarse["err"]
        assert fine["err"] < 0.01

    def test_signal_reproduces_harmonic_amplitude(self):
        case = _retarded_time_case(spp=64, include_near_field=True)
        A = np.column_stack([np.cos(case["omega"] * case["t_td"][case["sel"]]),
                             np.sin(case["omega"] * case["t_td"][case["sel"]])])
        coef, *_ = np.linalg.lstsq(A, case["p_td"][0, case["sel"]], rcond=None)
        amp = float(np.hypot(*coef))
        assert abs(amp - case["amp_ana"]) / case["amp_ana"] < 0.02


# ---------------------------------------------------------------------------
# C. Physics anchor: co-rotating vortex pair
# ---------------------------------------------------------------------------

class TestCVortexPair:
    """Frequency, directivity and amplitude laws of the rotating quadrupole."""

    @pytest.mark.parametrize("M", [0.05, 0.10])
    def test_peak_frequency_is_twice_rotation_frequency(self, M):
        res = vortex_pair_analysis(M)
        harm = res["harmonic"]
        peak = max(harm, key=lambda n: harm[n])
        assert peak == 2, f"M={M}: dominant harmonic {peak}*Omega (expected 2*Omega)"
        assert harm[2] / harm[1] > 10.0, f"fundamental not suppressed: {harm}"
        assert harm[2] / harm[3] > 10.0, f"3*Omega not suppressed: {harm}"

    @pytest.mark.parametrize("M", [0.05, 0.10])
    def test_four_lobe_directivity(self, M):
        res = vortex_pair_analysis(M)
        an = res["analysis"]
        corr = max(an["pattern_corr_cos2"], an["pattern_corr_sin2"])
        assert corr >= 0.95, f"M={M}: phase-locked pattern corr {corr:.4f}"
        assert abs(an["winding"]) == pytest.approx(2.0, abs=0.05)

    def test_rms_amplitude_is_isotropic_while_pattern_spins(self):
        """A rotating quadrupole has |p_hat| independent of theta; the four-lobe
        pattern rotates at Omega = omega/2, so a time-averaged RMS directivity
        is (spuriously) isotropic."""
        an = vortex_pair_analysis(0.05)["analysis"]
        assert an["isotropy"] < 0.02, f"|p_hat| anisotropy {an['isotropy']:.4f}"

    def test_amplitude_scales_as_mach_cubed(self):
        machs = [0.05, 0.075, 0.10]
        amps = np.array([vortex_pair_analysis(M)["analysis"]["amplitude"] for M in machs])
        norm = amps / (C0 ** 2 * np.array(machs))
        slope_raw = np.polyfit(np.log(machs), np.log(amps), 1)[0]
        slope_norm = np.polyfit(np.log(machs), np.log(norm), 1)[0]
        # raw amplitude at fixed r/lambda: quadrupole ~ M^4 (aeroacoustic M^3 law
        # times the geometric factor of the constant-r/lambda measurement point)
        assert abs(slope_raw - 4.0) / 4.0 < 0.10, f"raw slope {slope_raw:.3f}"
        # Mach-normalised amplitude p/(rho0 c0^2 M) ~ M^3: the classical law
        assert abs(slope_norm - 3.0) / 3.0 < 0.10, f"normalised slope {slope_norm:.3f}"

    def test_compressible_part_of_lighthill_tensor(self):
        """Isentropic closure cancels delta_ij(p - c0^2 rho) identically; a
        non-isentropic (entropy) spot keeps it, and it changes the far field."""
        x = np.arange(-20, 20 + 1e-12, 0.5)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        gamma, _ = _pair_field(0.05)
        ux, uy = co_rotating_vortex_pair(xx, yy, gamma, 6.0, 2.5, 0.0)
        u = np.stack([ux, uy], axis=0)

        # isentropic: p' = -rho0 |u|^2 / 2, rho = rho0 + p'/c0^2.  Note the
        # baseline: the trace term delta_ij (p - c0^2 rho) only cancels when the
        # *absolute* pressure is passed (p0 = c0^2 rho0) -- a classic trap.
        p_isen = -0.5 * (ux ** 2 + uy ** 2)
        rho_isen = 1.0 + p_isen / C0 ** 2
        T_isen = compute_lighthill_tensor(rho_isen, u, p=p_isen + C0 ** 2, c0=C0)
        T_reyn = compute_lighthill_tensor(rho_isen, u, p=None, c0=C0)
        trace = np.abs(T_isen - T_reyn).max()
        scale = np.abs(T_reyn).max()
        assert trace / scale < 1e-12, "isentropic trace term should vanish identically"

        # non-isentropic entropy spot: p' = -rho0|u|^2/2 + p_ent, rho unchanged
        p_ent = 0.02 * np.exp(-(xx ** 2 + yy ** 2) / 8.0)
        T_ent = compute_lighthill_tensor(rho_isen, u, p=p_isen + C0 ** 2 + p_ent, c0=C0)
        trace_ent = np.abs(T_ent - T_reyn).max()
        assert trace_ent / scale > 1e-3, "entropy source must activate the trace term"

        # the compressible part changes the radiated field measurably
        omega = 0.3
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        obs = np.stack([40.0 * np.cos(th), 40.0 * np.sin(th)], axis=-1)
        src = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        w = np.full(xx.size, 0.25)
        p_ref = volume_quadrupole_frequency_domain(T_reyn.reshape(-1, 2, 2), src, w,
                                                   obs, omega, C0, ndim=2, batch_size=16)
        p_full = volume_quadrupole_frequency_domain(T_ent.reshape(-1, 2, 2), src, w,
                                                    obs, omega, C0, ndim=2, batch_size=16)
        change = np.abs(p_full - p_ref).max() / np.abs(p_ref).max()
        assert change > 1e-3, f"entropy trace term has no acoustic effect ({change:.2e})"

    def test_prescribed_pair_rotates_rigidly(self):
        """u(y, t) = R(wt) u(R(-wt) y, 0): the source field is a rigid rotation."""
        x = np.arange(-12, 12 + 1e-12, 0.5)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        gamma, omega_rot = _pair_field(0.05)
        quarter = 0.25 * 2 * math.pi / omega_rot
        ux1, uy1 = co_rotating_vortex_pair(xx, yy, gamma, 6.0, 2.5, quarter)

        ang = omega_rot * quarter
        ca, sa = math.cos(ang), math.sin(ang)
        xb = ca * xx + sa * yy
        yb = -sa * xx + ca * yy
        uxb, uyb = co_rotating_vortex_pair(xb, yb, gamma, 6.0, 2.5, 0.0)
        ux_expect = ca * uxb - sa * uyb
        uy_expect = sa * uxb + ca * uyb
        assert np.abs(ux1 - ux_expect).max() < 1e-10
        assert np.abs(uy1 - uy_expect).max() < 1e-10


# ---------------------------------------------------------------------------
# D. Permeable surface (Kirchhoff) vs volume integral
# ---------------------------------------------------------------------------

class TestDSurfaceVsVolume:
    def test_kirchhoff_matches_volume_within_3dB(self):
        omega = 0.3
        k = omega / C0
        lam = 2 * math.pi / k
        sigma = lam / 20.0
        T, src, w = gaussian_source_2d(omega, sigma, extent_sigma=4.0)

        def volume(obs):
            return volume_quadrupole_frequency_domain(T, src, w, np.atleast_2d(obs),
                                                      omega, C0, ndim=2, batch_size=16)

        r_s = 1.0 * lam
        n_surf = 256
        ths = np.linspace(0, 2 * np.pi, n_surf, endpoint=False)
        S = np.stack([r_s * np.cos(ths), r_s * np.sin(ths)], axis=-1)
        nrm = S / r_s
        dr = lam / 400.0
        p_s = volume(S)
        dp_dn = (volume(S + dr * nrm) - volume(S - dr * nrm)) / (2 * dr)
        ds = 2 * np.pi * r_s / n_surf

        angles = np.array([0.0, 0.3, 1.0, 1.3])
        for rol in (2.0, 5.0, 10.0):
            r = rol * lam
            obs = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=-1)
            p_vol = volume(obs)
            p_kir = kirchhoff_surface_2d(p_s, dp_dn, S, nrm, np.full(n_surf, ds), obs,
                                         omega, C0)
            d_spl = np.abs(20 * np.log10(np.abs(p_kir) / np.abs(p_vol))).max()
            assert d_spl <= 3.0, f"r/lambda={rol}: |dSPL|={d_spl:.2f} dB"


# ---------------------------------------------------------------------------
# E. Compact limit and closed-form cross-checks
# ---------------------------------------------------------------------------

class TestECompactLimit:
    def test_exact_vs_point_quadrupole_in_compact_limit(self):
        omega = 0.3
        lam = 2 * math.pi * C0 / omega
        ang = np.deg2rad(np.array([0.0, 15.0, 30.0, 60.0, 75.0, 90.0]))
        obs = 500.0 * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        for s_lam in (0.01, 0.02):
            sigma = s_lam * lam
            p_exact = gaussian_quadrupole_farfield_exact(obs, omega, C0, 1.0, sigma)
            p_point = analytical_gaussian_quadrupole_farfield(obs, omega, C0, 1.0,
                                                              sigma, (0.0, 0.0), "xx-yy")
            dev = np.abs(np.abs(p_exact) - np.abs(p_point)) / np.abs(p_point)
            assert dev.max() < 0.02, f"sigma/lambda={s_lam}: dev {dev.max():.4f}"
            # the deviation is the analytical finite-size factor
            theo = 1.0 - math.exp(-(omega / C0 * sigma) ** 2 / 4.0)
            assert abs(dev.max() - theo) < 5e-3

    def test_compact_power_closed_form_2d(self):
        omega = 0.3
        lam = 2 * math.pi * C0 / omega
        sigma = 0.005 * lam
        Q = np.array([[1.0, 0.0], [0.0, -1.0]]) * math.pi * sigma ** 2
        r = 4000.0
        th = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
        obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)
        p = gaussian_quadrupole_farfield_exact(obs, omega, C0, 1.0, sigma)
        p_num = float(np.sum(np.abs(p) ** 2 / (2 * C0) * r * (2 * np.pi / th.size)))
        p_cf = compact_quadrupole_power(Q, omega, C0, 1.0, 2)
        assert abs(p_num - p_cf) / p_cf < 0.02

    def test_compact_power_closed_form_3d(self):
        omega = 0.8
        lam = 2 * math.pi * C0 / omega
        sigma = 0.005 * lam
        Q = np.diag([1.0, -1.0, 0.0]) * math.pi ** 1.5 * sigma ** 3
        r = 4000.0
        n_theta, n_phi = 128, 64
        th = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / (2 * n_theta)
        ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        TG, PG = np.meshgrid(th, ph, indexing="ij")
        obs = np.stack([r * np.sin(TG) * np.cos(PG), r * np.sin(TG) * np.sin(PG),
                        r * np.cos(TG)], axis=-1).reshape(-1, 3)
        p = gaussian_quadrupole_farfield_3d(obs, omega, C0, 1.0, sigma)
        d_omega = np.sin(TG).ravel() * (np.pi / n_theta) * (2 * np.pi / n_phi)
        p_num = float(np.sum(np.abs(p) ** 2 / (2 * C0) * r ** 2 * d_omega))
        p_cf = compact_quadrupole_power(Q, omega, C0, 1.0, 3)
        assert abs(p_num - p_cf) / p_cf < 0.02

    def test_compact_power_is_trace_free_and_positive(self):
        omega, sigma = 0.3, 0.01
        Q_traceless = np.array([[1.0, 0.0], [0.0, -1.0]]) * math.pi * sigma ** 2
        Q_with_trace = Q_traceless + 3.0 * np.eye(2)
        p1 = compact_quadrupole_power(Q_traceless, omega, C0)
        p2 = compact_quadrupole_power(Q_with_trace, omega, C0)
        assert p1 > 0.0
        assert p1 == pytest.approx(p2, rel=1e-12)  # trace part does not radiate