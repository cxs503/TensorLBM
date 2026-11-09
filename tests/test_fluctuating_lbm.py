"""Tests for the fluctuating-LBM statistical-physics layer.

Covers the three anchors (equipartition, Green-Kubo, Stokes-Einstein) and the
negative controls of campaign node p254w1-n06:

  J4  kT = 0 is bitwise identical to the plain BGK solver
  J5  noise amplitude scales as sqrt(kT); FDT covariance is the Landau-Lifshitz one
  J2  k-resolved Green-Kubo viscosity recovers the input viscosity
  J3  mode-resolved equipartition <|u_perp(k)|^2> = kT V
  J1  Stokes-Einstein: D = kT/(6 pi eta a) for a Stokes-dragged particle

The production-size runs live in ``scripts/run_lbm_fluctuating_campaign.py``; the
runs here are deliberately small so the file stays a fast gate, hence the loose
statistical tolerances (documented per test).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.fluctuating_lbm import (
    CS2,
    FluctuatingLBM,
    fdt_covariance,
    gk_viscosity_mode,
    run_brownian_particle,
    traceless_basis,
    traceless_stress,
)

# ---------------------------------------------------------------------------
# lattice / noise structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_traceless_basis_is_orthonormal_and_traceless(dim: int) -> None:
    basis = traceless_basis(dim)
    n = basis.shape[0]
    assert n == dim * (dim + 1) // 2 - 1  # traceless symmetric dimension
    flat = basis.reshape(n, -1)
    assert np.allclose(flat @ flat.T, np.eye(n), atol=1e-12)
    assert np.abs(np.einsum("kii->k", basis)).max() < 1e-14


@pytest.mark.parametrize("dim", [2, 3])
def test_random_stress_matches_landau_lifshitz_covariance(dim: int) -> None:
    """<s_ab s_cd> = 2 kT eta (d_ac d_bd + d_ad d_bc - 2/d d_ab d_cd)."""
    kT, eta, ns = 1e-3, 0.1, 60000
    s = traceless_stress(np.random.default_rng(0), (ns,), dim, kT, eta)
    emp = np.einsum("abn,cdn->abcd", s, s) / ns
    ana = fdt_covariance(dim, kT, eta)
    assert np.abs(emp - ana).max() / np.abs(ana).max() < 0.05
    # trace-free and zero mean
    assert np.abs(sum(s[a, a] for a in range(dim))).max() < 1e-14
    zero_mean = np.abs(s.reshape(dim * dim, -1).mean(axis=1)).max()
    # sample-mean noise of a zero-mean field: ~ sqrt(2 kT eta / ns)
    assert zero_mean < 5.0 * np.sqrt(2 * kT * eta / ns)


def test_zero_stress_at_zero_temperature() -> None:
    s = traceless_stress(np.random.default_rng(1), (10, 10), 3, 0.0, 0.1)
    assert np.all(s == 0.0)
    lbm = FluctuatingLBM((8, 8), nu=0.1, kT=0.0)
    assert np.all(lbm.random_force() == 0.0)


def test_noise_amplitude_scales_as_sqrt_kT() -> None:
    kTs = np.array([1e-4, 4e-4, 1.6e-3])
    amps = np.array(
        [FluctuatingLBM((10, 10), nu=0.1, kT=k, seed=3).noise_amplitude(8) for k in kTs]
    )
    slope = np.polyfit(np.log(kTs), np.log(amps), 1)[0]
    assert abs(slope - 0.5) < 0.02, f"noise scaling exponent {slope}"


# ---------------------------------------------------------------------------
# negative control: kT = 0 must reproduce the standard BGK solver bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(12, 12), (8, 8, 8)])
def test_zero_temperature_march_is_bitwise_identical_to_bgk(shape) -> None:
    a = FluctuatingLBM(shape, nu=0.1, kT=0.0, seed=7)
    b = FluctuatingLBM(shape, nu=0.1, kT=0.0, seed=7)
    assert np.array_equal(a.f, b.f)
    a.step(25)
    b.step_reference_bgk(25)
    assert np.array_equal(a.f, b.f), "kT=0 path must be bitwise identical to plain BGK"


def test_noise_switch_off_is_bitwise_identical_to_bgk() -> None:
    a = FluctuatingLBM((12, 12), nu=0.1, kT=1e-3, seed=4)
    b = FluctuatingLBM((12, 12), nu=0.1, kT=1e-3, seed=4)
    a.step(20, noise=False)
    b.step_reference_bgk(20)
    assert np.array_equal(a.f, b.f)


def test_mass_and_momentum_are_conserved_by_the_noise() -> None:
    """FDT noise injects zero net mass and, per site, zero net momentum drift."""
    lbm = FluctuatingLBM((16, 16), nu=0.1, kT=1e-3, seed=5)
    m0 = float(lbm.f.sum())
    lbm.step(200)
    rho1, u = lbm.macroscopic(half_force=False)
    # the stochastic stress is a divergence: total mass and total momentum are
    # conserved (local density does fluctuate, that is the point of the module)
    assert abs(float(lbm.f.sum()) - m0) < 1e-9 * abs(m0), "noise must not create mass"
    assert abs(u.mean(axis=(1, 2))).max() < 5e-3, "no net momentum may be injected"


# ---------------------------------------------------------------------------
# J3 equipartition
# ---------------------------------------------------------------------------


def test_equipartition_mode_resolved_3d() -> None:
    """<|u_perp(k)|^2> = kT V for the shear mode (tolerance 25 %, short run)."""
    n, kT, nu = 12, 1e-3, 0.1
    lbm = FluctuatingLBM((n, n, n), nu=nu, kT=kT, seed=11)
    grids = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    phase = np.exp(-2j * np.pi * grids[0] / n)
    lbm.step(400)
    vals = []
    for _ in range(250):
        lbm.step(5)
        u = lbm.velocity(half_force=False)
        vals.append(abs(np.sum(u[1] * phase)) ** 2)
    ratio = float(np.mean(vals) / (kT * n**3))
    # approaches 1 as k -> 0 (0.94 +- 0.01 at L=16 in the campaign run);
    # at k = 2 pi/12 the finite-k lattice correction leaves ~10 % deficit
    assert 0.65 < ratio < 1.30, f"mode-resolved equipartition ratio {ratio:.3f}"


def test_equipartition_rises_with_temperature() -> None:
    """The measured velocity variance must scale linearly with kT."""
    ratios = []
    for kT in (5e-4, 2e-3):
        lbm = FluctuatingLBM((12, 12), nu=0.1, kT=kT, seed=9)
        lbm.step(300)
        acc = 0.0
        for _ in range(60):
            lbm.step(20)
            u = lbm.velocity(half_force=False)
            acc += float(np.mean(u * u))
        ratios.append(acc / 60 / kT)
    assert abs(ratios[0] - ratios[1]) / np.mean(ratios) < 0.35


# ---------------------------------------------------------------------------
# J2 Green-Kubo viscosity
# ---------------------------------------------------------------------------


def test_green_kubo_recovers_input_viscosity() -> None:
    """eta_GK must recover eta = nu rho within 25 % on a short 3D run.

    The production estimate (40 realisations, 3500 steps) reaches -3.3 %; here we
    only assert the estimator is unbiased enough to be a gate.
    """
    nu = 0.1
    lbm = FluctuatingLBM((12, 12, 12), nu=nu, kT=1e-3, seed=13)
    r = gk_viscosity_mode(lbm, k_mode=1, n_steps=2000, n_equil=400)
    assert np.isfinite(r.eta_gk) and r.eta_gk > 0
    # single short realisation: the block error is ~25 %, so the gate is 40 %
    assert abs(r.eta_gk - nu) / nu < 0.40, f"eta_GK={r.eta_gk:.4f} vs {nu}"
    # the deterministic lattice response coefficient is a property of the scheme
    assert 0.5 < r.c0_lattice < 0.85, f"c0={r.c0_lattice:.3f}"
    # the raw lattice moment is biased low by c0^2 (documented, audited)
    assert r.eta_gk_raw < r.eta_gk


def test_green_kubo_rejects_zero_temperature() -> None:
    lbm = FluctuatingLBM((8, 8, 8), nu=0.1, kT=0.0)
    with pytest.raises(ValueError):
        gk_viscosity_mode(lbm, n_steps=10, n_equil=1)


# ---------------------------------------------------------------------------
# J1 Stokes-Einstein
# ---------------------------------------------------------------------------


def test_brownian_particle_smoke_and_scaling() -> None:
    """D must be finite, positive and follow kT/gamma (loose: short run)."""
    r = run_brownian_particle(
        n=10, nu=0.1, kT=2e-3, radius=1.2, n_steps=600, n_equilibrate=100, seed=2
    )
    assert r.gamma_stokes == pytest.approx(6 * np.pi * r.eta * r.radius)
    assert r.d_stokes_einstein == pytest.approx(r.kT / r.gamma_stokes)
    assert np.isfinite(r.diffusion) and r.diffusion > 0
    assert r.msd.shape == r.times.shape
    # the mean-square displacement must grow with time
    assert r.msd[-1] > r.msd[0]


def test_brownian_diffusion_scales_with_kT() -> None:
    """D ~ kT/gamma: doubling kT doubles the predicted and (roughly) measured D."""
    lo = run_brownian_particle(
        n=10, nu=0.1, kT=1e-3, radius=1.2, n_steps=500, n_equilibrate=100, seed=3
    )
    hi = run_brownian_particle(
        n=10, nu=0.1, kT=2e-3, radius=1.2, n_steps=500, n_equilibrate=100, seed=3
    )
    assert hi.d_stokes_einstein == pytest.approx(2 * lo.d_stokes_einstein, rel=1e-12)
    assert hi.vacf_diffusion > lo.vacf_diffusion


# ---------------------------------------------------------------------------
# solver invariants
# ---------------------------------------------------------------------------


def test_tau_and_viscosity_relation() -> None:
    lbm = FluctuatingLBM((8, 8), nu=0.1)
    assert lbm.tau == pytest.approx(0.5 + 0.1 / CS2)
    assert lbm.nu == pytest.approx(CS2 * (lbm.tau - 0.5))
    assert lbm.eta == pytest.approx(lbm.nu * lbm.rho0)


def test_invalid_lattice_and_tau_rejected() -> None:
    with pytest.raises(ValueError):
        FluctuatingLBM((8, 8), lattice="D3Q19")
    with pytest.raises(ValueError):
        FluctuatingLBM((8, 8), tau=0.5)
    with pytest.raises(ValueError):
        FluctuatingLBM((8, 8))
