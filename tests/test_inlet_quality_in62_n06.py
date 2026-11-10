"""IN62-N06 — tests for divergence-free synthetic inflow and the development section.

Covered capabilities (see ``docs/IN62_N06_INLET_QUALITY.md``):

* ``tensorlbm.turbulence.dfsem_inflow``
    - Reynolds-stress algebra (``stress_tensor`` / ``stress_cholesky``),
    - divergence diagnostics (``divergence_rms``, ``divergence_ratio``),
    - a-posteriori divergence projection (``project_divergence_free``),
    - DFSEM inlet (curl-of-stream-function) with target ``R_ij`` and projection,
    - classical SEM inlet as the documented negative control,
    - digital-filter (Klein) inlet with a Lund transformation layer.
* ``tensorlbm.turbulence.inlet_development``
    - D3Q19 uniform-stream development section, ``R_ij(x)`` / pressure statistics,
    - ``L_dev`` (95 % self-preservation length) definition on synthetic profiles,
    - the Comte-Bellot & Corrsin decay-exponent fit on a synthetic power law.

The tests are sized for CI (a few seconds); the full production run lives in
``experiments/in62_n06_inlet_quality/run_inlet_quality.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.turbulence.dfsem_inflow import (
    DFSEMInlet,
    DigitalFilterInlet,
    divergence_ratio,
    divergence_rms,
    lund_transform,
    project_divergence_free,
    sample_stress,
    stress_cholesky,
    stress_tensor,
)
from tensorlbm.turbulence.inlet_development import (
    DevelopmentSectionConfig,
    check_kernels_against_library,
    decay_exponent,
    development_length,
    run_development_section,
)

TARGET = dict(uu=4.0e-4, vv=2.0e-4, ww=2.0e-4)
NY = NZ = 32


# ---------------------------------------------------------------------------
# Reynolds-stress algebra
# ---------------------------------------------------------------------------


def test_stress_tensor_is_symmetric() -> None:
    R = stress_tensor(1.0, 2.0, 3.0, 0.1, -0.2, 0.3)
    assert R.shape == (3, 3)
    assert torch.allclose(R, R.transpose(0, 1))


def test_stress_cholesky_reconstructs_tensor() -> None:
    R = stress_tensor(4.0e-4, 2.0e-4, 2.0e-4, 0.5e-4)
    L = stress_cholesky(R)
    assert torch.allclose(L @ L.transpose(0, 1), R, atol=1e-16)


def test_stress_cholesky_rejects_indefinite_tensor() -> None:
    # uv too large for the given normal stresses -> negative eigenvalue
    R = stress_tensor(1.0e-4, 1.0e-4, 1.0e-4, 5.0e-4)
    with pytest.raises(ValueError):
        stress_cholesky(R)


def test_sample_stress_recovers_known_covariance() -> None:
    gen = torch.Generator().manual_seed(0)
    A = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.2, 0.1, 0.6]], dtype=torch.float64)
    eps = torch.randn((3, 400_000), generator=gen, dtype=torch.float64)
    u = (A @ eps).reshape(3, 400, 1000)
    S = sample_stress(u)
    assert torch.allclose(S, A @ A.transpose(0, 1), rtol=0.03, atol=3e-4)


# ---------------------------------------------------------------------------
# Divergence diagnostics and projection
# ---------------------------------------------------------------------------


def test_divergence_rms_vanishes_for_analytic_solenoidal_field() -> None:
    ny = nz = 32
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    z = torch.arange(nz, dtype=torch.float64).view(1, -1)
    ky, kz = 2.0 * math.pi * 2.0 / ny, 2.0 * math.pi * 3.0 / nz
    # u' = (0, dψ/dz, -dψ/dy) with ψ = sin(ky y) sin(kz z) is exactly solenoidal
    psi = torch.sin(ky * y) * torch.sin(kz * z)
    v = kz * torch.cos(kz * z) * torch.sin(ky * y)
    w = -ky * torch.cos(ky * y) * torch.sin(kz * z)
    u = torch.stack([torch.zeros_like(psi), v, w])
    assert float(divergence_rms(u)) < 1e-12
    assert float(divergence_ratio(u)) < 1e-12


def test_projection_reduces_divergence_by_two_orders() -> None:
    gen = torch.Generator().manual_seed(3)
    u = torch.randn((3, NY, NZ), generator=gen)
    before = float(divergence_ratio(u))
    after = float(divergence_ratio(project_divergence_free(u)))
    assert after < 1e-6
    assert before / max(after, 1e-12) > 1e3


# ---------------------------------------------------------------------------
# DFSEM / SEM inlets
# ---------------------------------------------------------------------------


def test_dfsem_projected_inlet_is_divergence_free_and_matches_stress() -> None:
    R = stress_tensor(**TARGET)
    gen = DFSEMInlet(ny=NY, nz=NZ, stress=R, length_scale=6.0, seed=1, dtype=torch.float64)
    seq = gen.sequence(24)
    assert torch.isfinite(seq).all()

    # ensemble over snapshots == what the LBM development section will see
    S = sample_stress(seq.movedim(0, 1).reshape(3, -1))
    assert float(((S - R) / R[0, 0]).abs().amax()) < 0.05

    for snap in seq[:4]:
        assert float(divergence_ratio(snap.float())) < 1.0e-3


def test_sem_negative_control_has_measurable_divergence() -> None:
    """Unprojected classical SEM must carry a compressive part (negative control)."""
    R = stress_tensor(**TARGET)
    dfsem = DFSEMInlet(ny=NY, nz=NZ, stress=R, length_scale=6.0, seed=1,
                       divergence_free=True, projection=False, dtype=torch.float64)
    sem = DFSEMInlet(ny=NY, nz=NZ, stress=R, length_scale=6.0, seed=1,
                     divergence_free=False, projection=False, dtype=torch.float64)
    d_dfsem = float(divergence_ratio(dfsem.sequence(1)[0].float()))
    d_sem = float(divergence_ratio(sem.sequence(1)[0].float()))
    assert d_sem > 0.05  # O(10^-1): a compressive inlet
    # curl-of-stream-function kernels carry ~1 order of magnitude less
    # divergence; the projection (tested above) removes the remaining part
    assert d_sem / max(d_dfsem, 1e-12) > 5.0


def test_dfsem_rejects_degenerate_configuration() -> None:
    R = stress_tensor(**TARGET)
    with pytest.raises(ValueError):
        DFSEMInlet(ny=2, nz=4, stress=R)
    with pytest.raises(ValueError):
        DFSEMInlet(ny=8, nz=8, stress=R, stress_correction="nope")


# ---------------------------------------------------------------------------
# Digital filter + Lund transformation
# ---------------------------------------------------------------------------


def test_lund_transform_imposes_target_covariance_exactly() -> None:
    R_target = stress_tensor(4.0e-4, 1.0e-4, 2.0e-4, 0.3e-4)
    gen = torch.Generator().manual_seed(7)
    raw = torch.randn((3, 64, 8, 8), generator=gen, dtype=torch.float64)
    out = lund_transform(raw, R_target)
    assert torch.allclose(sample_stress(out), R_target, rtol=1e-12, atol=1e-18)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-15)


def test_digital_filter_with_lund_is_exact_and_raw_filter_is_not() -> None:
    R = stress_tensor(**TARGET)
    kw = dict(ny=24, nz=24, stress=R, length_scale=4.0, time_scale=6.0, n_steps=96, seed=2)
    lund = DigitalFilterInlet(**kw, apply_lund=True, dtype=torch.float64).sequence()
    raw = DigitalFilterInlet(**kw, apply_lund=False, dtype=torch.float64).sequence()

    def rel_error(seq: torch.Tensor) -> float:
        S = sample_stress(seq.movedim(0, 1).reshape(3, -1))
        return float(((S - R) / R[0, 0]).abs().amax())

    err_lund, err_raw = rel_error(lund), rel_error(raw)
    assert err_lund < 1e-6
    assert err_raw > 10.0 * err_lund  # the Lund layer is what buys the match


def test_digital_filter_integral_scale_scales_with_filter_width() -> None:
    """Two-point correlation length grows with the Gaussian filter width."""
    def corr_length(length_scale: float) -> float:
        seq = DigitalFilterInlet(ny=64, nz=8, stress=stress_tensor(**TARGET),
                                 length_scale=length_scale, time_scale=4.0,
                                 n_steps=64, apply_lund=False,
                                 seed=5, dtype=torch.float64).sequence()
        u = seq[:, 0]  # streamwise component, (T, ny, nz)
        a = u - u.mean()
        ac = torch.fft.irfft(torch.abs(torch.fft.rfft(a, dim=1)) ** 2, n=64, dim=1).mean(dim=(0, 2))
        ac = ac / ac[0]
        below = torch.nonzero(ac < 0.5, as_tuple=False)
        return float(below[0]) if below.numel() else float(len(ac))

    short, long = corr_length(2.0), corr_length(8.0)
    assert long > short


# ---------------------------------------------------------------------------
# Development section (D3Q19)
# ---------------------------------------------------------------------------


def test_kernels_match_library_equilibrium() -> None:
    audit = check_kernels_against_library()
    assert audit["equilibrium_max_abs_diff"] < 1e-6
    assert audit["macroscopic_max_abs_diff"] < 1e-6
    assert audit["collision_max_abs_diff"] < 1e-6
    assert audit["field_magnitude"] > 0.0


def test_development_section_returns_finite_statistics() -> None:
    R = stress_tensor(**TARGET)
    gen = DFSEMInlet(ny=12, nz=12, stress=R, length_scale=4.0, seed=4, dtype=torch.float32)
    cfg = DevelopmentSectionConfig(nx=32, ny=12, nz=12, tau=0.505, n_steps=80,
                                   n_warmup=40, sample_every=5, ramp_cells=2)
    res = run_development_section(gen.sequence(80), cfg)
    assert res["R"].shape == (32, 3, 3)
    assert torch.isfinite(res["R"]).all()
    assert torch.isfinite(res["pressure_rms"]).all()
    assert res["samples"] > 0
    assert res["config"]["u_rms_inlet"] > 0.0


def test_development_length_of_a_self_preserving_profile_is_small() -> None:
    """A profile whose shape and decay rate are stationary from x=2 has L_dev~2."""
    nx = 40
    x = torch.arange(nx, dtype=torch.float64) + 1.0
    tke = 4.0e-4 * (x / 2.0) ** (-0.5)  # pure power law: constant local exponent
    R = torch.zeros((nx, 3, 3), dtype=torch.float64)
    R[:, 0, 0] = tke / 0.66  # fixed anisotropy shape R_vv = R_ww = 0.17 R_uu
    R[:, 1, 1] = 0.17 * R[:, 0, 0]
    R[:, 2, 2] = 0.17 * R[:, 0, 0]
    out = development_length(R, tol=0.05, x_start=2, window=4)
    assert math.isfinite(out["L_dev"])
    assert out["L_dev"] < 8.0


def test_development_length_detects_a_non_stationary_shape() -> None:
    """An inlet relaxing from isotropic to anisotropic never satisfies the test."""
    nx = 40
    x = torch.arange(nx, dtype=torch.float64) + 1.0
    decay = (x / x[0]) ** (-0.5)
    frac = 1.0 - 0.5 * torch.exp(-(x - 1.0) / 20.0)  # R_vv/R_uu drifts 0.5 -> 1
    R = torch.zeros((nx, 3, 3), dtype=torch.float64)
    R[:, 0, 0] = 4.0e-4 * decay
    R[:, 1, 1] = frac * R[:, 0, 0]
    R[:, 2, 2] = frac * R[:, 0, 0]
    out = development_length(R, tol=0.05, x_start=1, window=3)
    # Either never reached, or reached only after the shape has relaxed (>10 cells)
    assert not math.isfinite(out["L_dev"]) or out["L_dev"] > 10.0


def test_development_length_shape_only_ignores_the_decay_rate() -> None:
    """A stationary *shape* with a non-power-law decay: only the shape criterion passes."""
    nx = 60
    x = torch.arange(nx, dtype=torch.float64) + 1.0
    tke = 4.0e-4 * torch.exp(-x / 12.0)  # exponential: d ln k/d ln x varies strongly
    R = torch.zeros((nx, 3, 3), dtype=torch.float64)
    R[:, 0, 0] = tke / 1.34
    R[:, 1, 1] = 0.17 * R[:, 0, 0]
    R[:, 2, 2] = 0.17 * R[:, 0, 0]
    shape = development_length(R, tol=0.05, x_start=4, window=5, require_rate=False)
    strict = development_length(R, tol=0.05, x_start=4, window=5, require_rate=True)
    assert math.isfinite(shape["L_dev"])
    assert shape["L_dev"] <= 6.0
    assert not math.isfinite(strict["L_dev"]) or strict["L_dev"] > shape["L_dev"] + 10.0


# ---------------------------------------------------------------------------
# Decay-exponent fit (Comte-Bellot & Corrsin comparison machinery)
# ---------------------------------------------------------------------------


def test_decay_exponent_recovers_a_synthetic_power_law() -> None:
    t = torch.arange(0, 2000, 10, dtype=torch.float64)
    tke = 1.2e-3 * (t + 40.0) ** (-1.30)
    fit = decay_exponent(t, tke)
    assert fit["n"] == pytest.approx(1.30, rel=1e-3)
    assert fit["rms_residual"] < 1e-6


def test_decay_exponent_is_insensitive_to_the_amplitude() -> None:
    t = torch.arange(0, 2000, 10, dtype=torch.float64)
    a = decay_exponent(t, 1.0e-3 * (t + 10.0) ** (-1.2))
    b = decay_exponent(t, 7.0e-3 * (t + 10.0) ** (-1.2))
    assert a["n"] == pytest.approx(b["n"], rel=1e-6)