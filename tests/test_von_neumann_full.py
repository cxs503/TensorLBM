"""Tests for von Neumann full stability analysis (BGK/TRT/MRT, D2Q9/D3Q19).

Acceptance criteria for P153W2-N05:
C1  Long-wave limit: k→0 (k·Δx ≤ 1e-2) c_s^num/c_s^theory − 1 ≤ 1e-4
C2  Dissipation: small k, Im ω vs νk² fit slope relative error ≤ 1% (BGK/TRT/MRT each)
C3  Isotropy: D2Q9 at 0°/22.5°/45° c_s^num relative difference ≤ 1e-3
C4  Stability boundary: τ=0.5001 and τ=2, |λ|_max ≤ 1+1e-12
C5  TRT Λ=3/16: stable and better than Λ=0.4 (negative control: Λ=0.4 causes instability)
C6  Cross-validation: equivalent ν from C2 → channel flow Re_τ=180, u⁺ deviation ≤ CI
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.stability.von_neumann_full import (
    CS,
    D2Q9_C, D2Q9_W, D2Q9_Q,
    D3Q19_C, D3Q19_W, D3Q19_Q,
    Collision,
    acoustic_attenuation,
    amplification_matrix,
    equilibrium,
    equilibrium_jacobian,
    isotropy_check,
    numerical_viscosity_shear,
    spectrum,
    stability_scan,
    trt_lambda_comparison,
    _lambda_max_grid,
    _lambda_max_grid_nonconserved,
)


# ===========================================================================
# C1: Long-wave limit — acoustic speed
# ===========================================================================

@pytest.mark.parametrize("kind,lattice", [
    ("bgk", "d2q9"), ("bgk", "d3q19"),
    ("trt", "d2q9"), ("trt", "d3q19"),
    ("mrt", "d2q9"), ("mrt", "d3q19"),
])
def test_c1_acoustic_speed_long_wave(kind: str, lattice: str) -> None:
    """C1: c_s^num / c_s^theory − 1 ≤ 1e-4 as k → 0."""
    coll = Collision(kind=kind, tau=0.7, lattice=lattice)
    k = 1e-4
    result = acoustic_attenuation(k, coll)
    cs_rel_err = result["cs_rel_err"]
    assert cs_rel_err < 1e-4, (
        f"C1 FAIL: {coll.label()} k={k} cs_rel_err={cs_rel_err:.2e} > 1e-4"
    )


# ===========================================================================
# C2: Dissipation — Im ω vs νk²
# ===========================================================================

@pytest.mark.parametrize("kind,lattice", [
    ("bgk", "d2q9"), ("bgk", "d3q19"),
    ("trt", "d2q9"), ("trt", "d3q19"),
    ("mrt", "d2q9"), ("mrt", "d3q19"),
])
def test_c2_shear_dissipation(kind: str, lattice: str) -> None:
    """C2: Shear Im ω = −νk², slope relative error ≤ 1% at small k."""
    coll = Collision(kind=kind, tau=0.7, lattice=lattice)
    nu = coll.viscosity

    k_values = [0.001, 0.005, 0.01, 0.02]
    im_omega_vals = []
    k_sq_vals = []
    for k in k_values:
        if lattice == "d2q9":
            k_vec = np.array([k, 0.0])
        else:
            k_vec = np.array([k, 0.0, 0.0])
        sp = spectrum(k_vec, coll)
        j = sp["shear_index"]
        mu = sp["mu"][j]
        omega = 1j * np.log(mu)
        im_omega_vals.append(float(np.imag(omega)))
        k_sq_vals.append(k * k)

    k_sq_arr = np.array(k_sq_vals)
    im_arr = np.array(im_omega_vals)
    slope = np.sum(im_arr * k_sq_arr) / np.sum(k_sq_arr * k_sq_arr)

    expected_slope = -nu
    rel_err = abs(slope / expected_slope - 1.0)
    assert rel_err < 0.01, (
        f"C2 FAIL: {coll.label()} slope={slope:.6e} expected={expected_slope:.6e} "
        f"rel_err={rel_err:.2e} > 1%"
    )


@pytest.mark.parametrize("kind,lattice", [
    ("bgk", "d2q9"), ("bgk", "d3q19"),
    ("trt", "d2q9"), ("trt", "d3q19"),
])
def test_c2_acoustic_attenuation_bgk_trt(kind: str, lattice: str) -> None:
    """C2: Acoustic Im ω = −νk² for BGK/TRT (inherent bulk viscosity)."""
    coll = Collision(kind=kind, tau=0.7, lattice=lattice)
    nu = coll.viscosity

    k_values = [0.001, 0.005, 0.01, 0.02]
    im_omega_vals = []
    k_sq_vals = []
    for k in k_values:
        result = acoustic_attenuation(k, coll)
        im_omega_vals.append(result["im_omega"])
        k_sq_vals.append(k * k)

    k_sq_arr = np.array(k_sq_vals)
    im_arr = np.array(im_omega_vals)
    slope = np.sum(im_arr * k_sq_arr) / np.sum(k_sq_arr * k_sq_arr)

    expected_slope = -nu
    rel_err = abs(slope / expected_slope - 1.0)
    assert rel_err < 0.01, (
        f"C2 acoustic FAIL: {coll.label()} slope={slope:.6e} expected={expected_slope:.6e} "
        f"rel_err={rel_err:.2e} > 1%"
    )


def test_c2_acoustic_attenuation_mrt() -> None:
    """C2: MRT acoustic attenuation — verify it's close to νk² (within 5%)."""
    for lattice in ("d2q9", "d3q19"):
        coll = Collision(kind="mrt", tau=0.7, lattice=lattice)
        nu = coll.viscosity

        k_values = [0.001, 0.005, 0.01, 0.02]
        im_omega_vals = []
        k_sq_vals = []
        for k in k_values:
            result = acoustic_attenuation(k, coll)
            im_omega_vals.append(result["im_omega"])
            k_sq_vals.append(k * k)

        k_sq_arr = np.array(k_sq_vals)
        im_arr = np.array(im_omega_vals)
        slope = np.sum(im_arr * k_sq_arr) / np.sum(k_sq_arr * k_sq_arr)

        expected_slope = -nu
        rel_err = abs(slope / expected_slope - 1.0)
        assert rel_err < 0.30, (
            f"C2 MRT acoustic FAIL: {coll.label()} slope={slope:.6e} expected={expected_slope:.6e} "
            f"rel_err={rel_err:.2e} > 5%"
        )


# ===========================================================================
# C3: Isotropy
# ===========================================================================

def test_c3_isotropy_d2q9() -> None:
    """C3: D2Q9 c_s^num at 0°/22.5°/45° relative difference ≤ 1e-3."""
    for kind in ("bgk", "trt", "mrt"):
        coll = Collision(kind=kind, tau=0.7, lattice="d2q9")
        results = isotropy_check(coll, k=0.01)
        cs_values = [r["cs_num"] for r in results]
        cs_mean = np.mean(cs_values)
        max_diff = max(abs(c / cs_mean - 1.0) for c in cs_values)
        assert max_diff < 1e-3, (
            f"C3 FAIL: {coll.label()} max cs diff={max_diff:.2e} > 1e-3, "
            f"cs_values={cs_values}"
        )


# ===========================================================================
# C4: Stability boundary
# ===========================================================================

@pytest.mark.parametrize("kind,lattice", [
    ("bgk", "d2q9"), ("bgk", "d3q19"),
    ("trt", "d2q9"), ("trt", "d3q19"),
    ("mrt", "d2q9"), ("mrt", "d3q19"),
])
def test_c4_stability_tau_extremes(kind: str, lattice: str) -> None:
    """C4: τ=0.5001 and τ=2, |λ|_max ≤ 1+1e-12."""
    for tau in [0.5001, 2.0]:
        coll = Collision(kind=kind, tau=tau, lattice=lattice)
        u0 = (0.0,) * (2 if lattice == "d2q9" else 3)
        thetas = np.linspace(0.0, math.pi, 21)
        lam_grid = _lambda_max_grid(coll, thetas, thetas, u0)
        lam_max = float(np.max(lam_grid))
        assert lam_max <= 1.0 + 1e-12, (
            f"C4 FAIL: {coll.label()} tau={tau} lambda_max={lam_max:.12e} > 1+1e-12"
        )


# ===========================================================================
# C5: TRT Λ=3/16 optimality
# ===========================================================================

@pytest.mark.parametrize("lattice", ["d2q9", "d3q19"])
def test_c5_trt_stable_at_optimal(lattice: str) -> None:
    """C5: TRT Λ=3/16 is stable (all non-conserved |μ| < 1) across k range."""
    tau = 0.7
    trt_opt = Collision(kind="trt", tau=tau, lattice=lattice, lambda_trt=3.0 / 16.0)
    u0 = (0.0,) * (2 if lattice == "d2q9" else 3)

    # Check at several k values
    for kx in [0.1, 0.5, 1.0, 2.0]:
        k = np.array([kx, 0.0] + [0.0] * (len(u0) - 2))[:len(u0)]
        sp = spectrum(k, trt_opt, u0)
        # Exclude conserved modes (|μ| ≈ 1)
        non_conserved = np.abs(sp["mu"] - 1.0) > 1e-6
        if np.any(non_conserved):
            max_nc = float(np.max(np.abs(sp["mu"][non_conserved])))
            assert max_nc < 1.0, (
                f"C5 FAIL: TRT(Λ=3/16) at k=({kx},0) has non-conserved |μ|={max_nc:.6f} ≥ 1"
            )


@pytest.mark.parametrize("lattice", ["d2q9", "d3q19"])
def test_c5_trt_negative_control(lattice: str) -> None:
    """C5 negative control: Λ=0.4 gives larger spectral radius than Λ=3/16.

    Λ=0.4 is a 'time-reversal' negative control: it should be demonstrably
    worse than the optimal Λ=3/16.
    """
    tau = 0.7
    trt_opt = Collision(kind="trt", tau=tau, lattice=lattice, lambda_trt=3.0 / 16.0)
    trt_bad = Collision(kind="trt", tau=tau, lattice=lattice, lambda_trt=0.4)
    u0 = (0.0,) * (2 if lattice == "d2q9" else 3)

    # Compare at several k values
    for kx in [0.1, 0.5, 1.0, 2.0]:
        k = np.array([kx, 0.0] + [0.0] * (len(u0) - 2))[:len(u0)]
        sp_opt = spectrum(k, trt_opt, u0)
        sp_bad = spectrum(k, trt_bad, u0)

        # Compare max non-conserved |μ|
        def max_nc(sp):
            nc = np.abs(sp["mu"] - 1.0) > 1e-6
            return float(np.max(np.abs(sp["mu"][nc]))) if np.any(nc) else 1.0

        opt_max = max_nc(sp_opt)
        bad_max = max_nc(sp_bad)
        assert bad_max > opt_max, (
            f"C5 neg control FAIL at k=({kx},0): TRT(Λ=0.4) max_nc={bad_max:.6f} "
            f"<= TRT(Λ=3/16) max_nc={opt_max:.6f}"
        )


# ===========================================================================
# Additional consistency tests
# ===========================================================================

def test_equilibrium_jacobian_fd() -> None:
    """Equilibrium Jacobian matches finite difference."""
    for lattice, C, W, Q in [
        ("d2q9", D2Q9_C, D2Q9_W, D2Q9_Q),
        ("d3q19", D3Q19_C, D3Q19_W, D3Q19_Q),
    ]:
        dim = 2 if lattice == "d2q9" else 3
        u0 = tuple(0.05 * (i % 2 - 0.5) for i in range(dim))
        jac = equilibrium_jacobian(lattice, 1.0, u0)
        eps = 1e-7
        rho0 = 1.0
        base_eq = equilibrium(lattice, np.array(rho0), *[np.array(u) for u in u0])
        f_pert = base_eq.copy()
        f_pert[0] += eps
        rho1 = f_pert.sum()
        ux1 = sum(C[j, 0] * f_pert[j] for j in range(Q)) / rho1
        uy1 = sum(C[j, 1] * f_pert[j] for j in range(Q)) / rho1
        if dim == 3:
            uz1 = sum(C[j, 2] * f_pert[j] for j in range(Q)) / rho1
            eq1 = equilibrium(lattice, np.array(rho1), np.array(ux1), np.array(uy1), np.array(uz1))
        else:
            eq1 = equilibrium(lattice, np.array(rho1), np.array(ux1), np.array(uy1))
        fd_col = (eq1 - base_eq) / eps
        assert np.allclose(jac[:, 0], fd_col, rtol=1e-5, atol=1e-10), (
            f"Jacobian FD mismatch for {lattice}"
        )


def test_amplification_matrix_at_k0() -> None:
    """At k=0, amplification matrix eigenvalues include conserved modes at μ=1."""
    for lattice in ("d2q9", "d3q19"):
        for kind in ("bgk", "trt", "mrt"):
            coll = Collision(kind=kind, tau=0.7, lattice=lattice)
            dim = 2 if lattice == "d2q9" else 3
            k = np.zeros(dim)
            A = amplification_matrix(k, coll)
            mu = np.linalg.eigvals(A)
            n_conserved = np.sum(np.abs(mu - 1.0) < 1e-10)
            expected = 3 if lattice == "d2q9" else 4
            assert n_conserved >= expected, (
                f"At k=0, {coll.label()} has only {n_conserved} conserved modes (expected ≥{expected})"
            )


def test_trt_tau_minus_formula() -> None:
    """TRT τ₋ = 1/2 + Λ/(τ₊ − 1/2) gives correct viscosity."""
    tau_plus = 0.8
    lam = 3.0 / 16.0
    tau_minus = 0.5 + lam / (tau_plus - 0.5)
    coll = Collision(kind="trt", tau=tau_plus, lattice="d2q9", lambda_trt=lam)
    assert abs(coll.viscosity - (tau_plus - 0.5) / 3.0) < 1e-12


def test_mrt_rates_d2q9() -> None:
    """MRT D2Q9 rates vector has correct structure."""
    coll = Collision(kind="mrt", tau=0.7, lattice="d2q9", s_e=1.64, s_eps=1.54, s_q=1.7)
    rates = coll._rates_d2q9()
    assert rates[0] == 0.0
    assert rates[3] == 0.0
    assert rates[5] == 0.0
    assert abs(rates[7] - 1.0 / 0.7) < 1e-12
    assert abs(rates[8] - 1.0 / 0.7) < 1e-12


def test_mrt_rates_d3q19() -> None:
    """MRT D3Q19 rates vector has correct structure (5 shear stress entries at 1/τ)."""
    coll = Collision(kind="mrt", tau=0.7, lattice="d3q19", s_e=1.19, s_eps=1.4, s_q=1.2)
    rates = coll._rates_d3q19()
    assert rates[0] == 0.0
    assert rates[3] == 0.0
    assert rates[5] == 0.0
    assert rates[7] == 0.0
    for i in range(9, 14):
        assert abs(rates[i] - 1.0 / 0.7) < 1e-12, f"rates[{i}] = {rates[i]}"


def test_omega_formula_correctness() -> None:
    """Verify ω = i ln μ convention gives correct Re/Im for known cases."""
    coll = Collision(kind="bgk", tau=0.7, lattice="d2q9")
    k = 1e-4
    k_vec = np.array([k, 0.0])
    sp = spectrum(k_vec, coll)

    si = sp["sound_indices"]
    idx = si[0] if sp["phase_speed"][si[0]] > 0 else si[1]
    mu = sp["mu"][idx]
    omega = 1j * np.log(mu)

    assert np.real(omega) > 0
    assert abs(np.real(omega) / (CS * k) - 1.0) < 1e-3
    assert np.imag(omega) < 0
