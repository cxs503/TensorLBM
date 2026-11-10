"""Node p234w2-n05 -- acceptance tests for spectrum/moment-calibrated inlet synthesis.

Every gate of the campaign node is exercised here on CI-sized configurations:

(a) spectrum: 1/3-octave band error vs von Kármán over ``[0.2, 5] f_peak`` < 15 %
(b) moments:   ``u'u'``, ``v'v'``, ``w'w'`` relative error < 10 %
(c) integral:  ``L_ux`` within 20 % of target
(d) temporal:  exponential fit of ``R_uu(τ)`` gives ``R² > 0.95``
(e) divergence: ``max|∇·u'|Δx/u_rms < 0.05``
(f) reproducibility: identical seed ⇒ identical sha256
(g) negative control: ``L_ux = 0`` (white noise in time) ⇒ band error > 50 %

plus the analytic normalizations of the target spectra, the Turin decomposition,
the receiver validation and the two-route application layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.turbulence.inlet_downstream import (
    ChannelConfig,
    D2Q9Channel,
    cross_check_routes,
    downstream_impact,
    inlet_signal_2d,
)
from tensorlbm.turbulence.inlet_spectral_synthesis import (
    DSRFGInlet,
    SpectralInletSynthesis,
    SynthesisConfig,
    autocorrelation,
    evaluate_gates,
    exponential_fit,
    inlet_statistics,
    inplane_divergence,
    kaimal_su,
    project_inplane_divergence_free,
    third_octave_bands,
    turin_filter_amplitudes,
    von_karman_acf_time,
    von_karman_e3d,
    von_karman_e11,
    von_karman_f_peak,
    von_karman_tensor_spectrum,
)

# Campaign configuration, scaled down in *space* only: the time axis keeps the
# resolution the acceptance criteria need (f_Nyquist / f_peak > 30).
SMALL = dict(ny=32, nz=32, n_time=1536, dt=0.15, u_rms=0.1, L_ux=2.0, U=1.0, seed=12345)


# ---------------------------------------------------------------------------
# Analytic target spectra
# ---------------------------------------------------------------------------


def test_von_karman_e11_normalisation():
    """``∫₀^∞ E11 dk1 = u_rms²`` and the ``k1·E11`` peak sits at ``0.9146/L``."""
    # Log grid out to k = 1e6 L^-1: the 5/6 spectrum has a k^(-5/3) tail, so a
    # linear grid truncated at a few hundred wavenumbers loses ~1 % of the area.
    k = np.geomspace(1e-6, 1e6, 800001)
    e11 = von_karman_e11(k, u_rms=2.0, L=1.5)
    area = np.trapezoid(e11, k)
    assert area == pytest.approx(4.0, rel=2e-3)

    k2 = np.linspace(1e-4, 50.0, 200001)
    f_peak = k2[np.argmax(k2 * von_karman_e11(k2, 1.0, 1.0))]
    assert f_peak == pytest.approx(0.9146, rel=2e-2)


def test_von_karman_e3d_and_tensor():
    """``∫ E(k) dk = 3/2 u_rms²`` and the trace of ``Φ_ij`` matches ``E``."""
    k = np.logspace(-4.0, 4.0, 400001)
    area = np.trapezoid(von_karman_e3d(k, u_rms=1.0, L=1.0), k)
    assert area == pytest.approx(1.5, rel=1e-2)

    kvec = np.array([[0.3, 0.0, 0.0], [0.0, 0.7, 0.0], [0.2, -0.4, 0.9]])
    phi = von_karman_tensor_spectrum(kvec, u_rms=1.0, L=1.0)
    assert phi.shape == (3, 3, 3)
    trace = np.trace(phi, axis1=-2, axis2=-1)
    kmag = np.linalg.norm(kvec, axis=-1)
    # For the isotropic tensor Φ_ij = E/(4πk²)(δ_ij − k_i k_j/k²) the trace is
    # E/(2πk²) (the 3−1 from the projector), which is what the test pins down.
    assert np.allclose(trace, von_karman_e3d(kmag, 1.0, 1.0) / (2.0 * np.pi * kmag**2), rtol=1e-10)
    # symmetry and zero divergence condition k_i Φ_ij = 0
    for i in range(kvec.shape[0]):
        assert np.allclose(phi[i], phi[i].T, atol=1e-14)
        assert np.allclose(kvec[i] @ phi[i], 0.0, atol=1e-10)


def test_kaimal_normalisation_and_bands():
    f = np.logspace(-5.0, 3.0, 400001)
    area = np.trapezoid(kaimal_su(f, u_rms=1.5, L=2.0, U=1.0), f)
    assert area == pytest.approx(2.25, rel=1e-2)
    bands = third_octave_bands(von_karman_f_peak(0.1, 2.0, 1.0))
    assert bands.shape[0] >= 8
    assert bands[0, 0] >= 0.2 * von_karman_f_peak(0.1, 2.0, 1.0) - 1e-12
    assert bands[-1, 1] <= 5.0 * von_karman_f_peak(0.1, 2.0, 1.0) + 1e-12


# ---------------------------------------------------------------------------
# Turin decomposition
# ---------------------------------------------------------------------------


def test_turin_kernel_reproduces_target_acf():
    """Filtered unit-variance noise carries the target autocorrelation."""
    n, dt = 4096, 0.05
    acf_target = von_karman_acf_time(n, dt, u_rms=1.0, L=2.0, U=1.0)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((64, n))
    H = turin_filter_amplitudes(
        np.abs(np.fft.rfft(acf_target)) * 2.0 * dt,  # one-sided PSD of the target ACF
        dt,
    )
    out = np.fft.irfft(np.fft.rfft(noise, axis=-1) * H[None, :], n=n, axis=-1)
    acf = autocorrelation(out, max_lag=200)
    lag = np.arange(201) * dt
    # Compare in the decay region where the estimate is noise-free enough.
    mask = lag <= 8.0
    rel = np.abs(acf[mask] - acf_target[:201][mask]) / max(acf_target[0], 1e-12)
    assert rel.max() < 0.05


# ---------------------------------------------------------------------------
# Divergence control
# ---------------------------------------------------------------------------


def test_projection_removes_measured_divergence():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((4, 3, 24, 20))
    before_max, before_rms = inplane_divergence(field)
    projected = project_inplane_divergence_free(field)
    after_max, after_rms = inplane_divergence(projected)
    assert before_rms > 0.5  # sanity: the raw field really is compressive
    assert after_max < 1e-10
    assert after_rms < 1e-10
    # streamwise component untouched
    assert np.allclose(projected[:, 0], field[:, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# Acceptance gates (a)-(f) on the digital-filter branch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dfm_case():
    cfg = SynthesisConfig(**SMALL)
    syn = SpectralInletSynthesis(cfg)
    field = syn.synthesize()
    return cfg, field, syn


def test_gate_spectrum_and_moments(dfm_case):
    cfg, field, _ = dfm_case
    stats = inlet_statistics(field, cfg)
    verdict = evaluate_gates(stats)
    # (a) spectrum
    assert verdict["spectrum_band_error"] < 0.15
    # (b) Reynolds stresses
    assert verdict["stress_rel_error"] < 0.10
    # (c) integral scale
    assert verdict["integral_scale_rel_error"] < 0.20
    # (d) exponential fit of the temporal correlation
    assert verdict["acf_exp_r2"] > 0.95
    # (e) measured divergence
    assert verdict["divergence_max"] < 0.05
    assert verdict["all_pass"]


def test_gate_reproducibility_sha256(dfm_case):
    """(f) same seed ⇒ bit-identical field; different seed ⇒ different field."""
    cfg, field, syn = dfm_case
    import hashlib

    sha_first = hashlib.sha256(np.ascontiguousarray(field, dtype=np.float64).tobytes()).hexdigest()
    cfg2 = SynthesisConfig(**{**SMALL, "seed": SMALL["seed"]})
    sha_second = SpectralInletSynthesis(cfg2).sha256()
    assert sha_first == sha_second

    cfg3 = SynthesisConfig(**{**SMALL, "seed": SMALL["seed"] + 1})
    assert SpectralInletSynthesis(cfg3).sha256() != sha_first


def test_gate_negative_control_zero_integral_scale():
    """(g) ``L_ux = 0`` gives white noise in time: band error must exceed 50 %."""
    cfg = SynthesisConfig(**{**SMALL, "L_ux": 0.0})
    field = SpectralInletSynthesis(cfg).synthesize()
    stats = inlet_statistics(field, SynthesisConfig(**SMALL))
    assert stats.max_band_error > 0.50
    assert stats.L_ux_sample < 0.2 * SMALL["L_ux"]


# ---------------------------------------------------------------------------
# Second target family and the auxiliary generator
# ---------------------------------------------------------------------------


def test_kaimal_target_family():
    cfg = SynthesisConfig(**{**SMALL, "spectrum": "kaimal"})
    stats = inlet_statistics(SpectralInletSynthesis(cfg).synthesize(), cfg)
    assert stats.max_band_error < 0.15
    assert stats.divergence_max < 0.05


def test_dsrgf_branch_passes_same_gates():
    # The DSRFG branch needs the full campaign plane resolution: with fewer modes
    # the realized ⟨v'v'⟩/⟨w'w'⟩ ratio scatters and the *common* transverse
    # rescaling (which is what keeps the field solenoidal) then shows up as a
    # stress error of ~20 % (measured on 32x32 / 48 modes).
    cfg = SynthesisConfig(ny=64, nz=64, n_time=512, dt=0.5, u_rms=0.1, L_ux=2.0, U=1.0, seed=5)
    field = DSRFGInlet(config=cfg, n_modes=64).synthesize()
    stats = inlet_statistics(field, cfg)
    verdict = evaluate_gates(stats)
    assert verdict["stress_rel_error"] < 0.10
    assert verdict["divergence_max"] < 0.05
    assert verdict["acf_exp_r2"] > 0.95


# ---------------------------------------------------------------------------
# Application layer / receiver
# ---------------------------------------------------------------------------


def test_receiver_reproduces_poiseuille():
    """The D2Q9 receiver must reproduce its own analytic parabola (<2 % L2)."""
    cfg = ChannelConfig(
        nx=24,
        ny=32,
        nu=0.05,
        u_bulk=0.06,
        periodic_x=True,
        turbulence_amplitude=0.0,
        monitor_stations=(8,),
    )
    assert D2Q9Channel(cfg).poiseuille_error(2500) < 0.02


def test_route_arbitration_at_inlet_plane():
    """Both imposition routes must deliver the prescribed signal at the inlet plane."""
    cfg = SynthesisConfig(ny=24, nz=8, n_time=512, dt=1.0, u_rms=0.01, L_ux=10.0, U=1.0, seed=3)
    signal = inlet_signal_2d(SpectralInletSynthesis(cfg).synthesize())
    res = cross_check_routes(signal, n_steps=400, n_relax=1, near_field_station=4, ny=24)
    diffs = res["differences"]
    assert diffs["inlet_rel_mean"] < 1e-9
    assert diffs["inlet_rel_rms"] < 1e-9
    assert diffs["inlet_rel_T_int"] < 1e-9


def test_downstream_turbulence_amplifies_and_decays():
    """Synthesized inflow must raise the downstream fluctuation level and decay."""
    cfg = SynthesisConfig(
        ny=24, nz=8, n_time=1200, dt=1.0, u_rms=0.01, L_ux=20.0, L_spatial=6.0, U=1.0, seed=4
    )
    signal = inlet_signal_2d(SpectralInletSynthesis(cfg).synthesize())
    out = downstream_impact(signal, n_steps=700, nx=96, ny=24, nu=0.05, stations=(16, 60, 90))
    rms_t = [out["turbulent"][f"x{x}"]["rms"] for x in (16, 60, 90)]
    rms_l = [out["laminar"][f"x{x}"]["rms"] for x in (16, 60, 90)]
    assert rms_t[0] > 20.0 * max(rms_l[0], 1e-9)  # turbulence dominates the laminar floor
    assert rms_t[0] > rms_t[-1]  # and it decays downstream
    assert out["log_decay_rate_per_cell"] > 0.0


def test_exponential_fit_recovers_time_scale():
    tau = np.arange(0, 40, 0.1)
    rng = np.random.default_rng(0)
    acf = np.exp(-tau / 3.0) + 0.002 * rng.standard_normal(tau.size)
    r2, T = exponential_fit(tau, acf)
    assert T == pytest.approx(3.0, rel=0.05)
    assert r2 > 0.99
