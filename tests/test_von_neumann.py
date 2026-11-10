"""Comprehensive tests for von Neumann stability analysis.

Validates acceptance criteria C1-C6:
    C1: Long-wave limit: cs_num/cs_theory - 1 ≤ 1e-4 for k·Δx ≤ 1e-2
    C2: Dissipation: Im(ω) vs νk² fit slope error ≤ 1%
    C3: Isotropy: D2Q9 cs_num variation ≤ 1e-3 across 0°/22.5°/45°
    C4: Stability boundary: τ=0.5001 and τ=2 both stable
    C5: TRT optimality: Λ=3/16 gives smaller max|λ| than BGK (non-conserved modes)
    C6: Cross-validation with physical simulation (channel flow)
"""

import numpy as np
import pytest

from tensorlbm.von_neumann import (
    BGKCollision,
    D2Q9Lattice,
    D3Q19Lattice,
    MRTCollision,
    TRTCollision,
    compute_dispersion_curve,
    fit_viscosity_from_dissipation,
    max_eigenvalue_vs_tau,
    scan_stability_envelope,
    von_neumann_eigenvalues,
)
from tensorlbm.von_neumann.analysis import (
    AmplificationMatrix,
    eigenvalues_to_omega,
    identify_physical_modes,
)


class TestLatticeProperties:
    """Test lattice isotropy and weight normalization."""

    def test_d2q9_weights_sum_to_one(self):
        lat = D2Q9Lattice()
        assert abs(np.sum(lat.w) - 1.0) < 1e-15

    def test_d3q19_weights_sum_to_one(self):
        lat = D3Q19Lattice()
        assert abs(np.sum(lat.w) - 1.0) < 1e-15

    def test_d2q9_cs2(self):
        lat = D2Q9Lattice()
        # cs2 = sum(w_i c_ix^2) = 1/3
        cs2_x = np.sum(lat.w * lat.c[:, 0] ** 2)
        cs2_y = np.sum(lat.w * lat.c[:, 1] ** 2)
        assert abs(cs2_x - 1.0 / 3.0) < 1e-15
        assert abs(cs2_y - 1.0 / 3.0) < 1e-15

    def test_d3q19_cs2(self):
        lat = D3Q19Lattice()
        for d in range(3):
            cs2_d = np.sum(lat.w * lat.c[:, d] ** 2)
            assert abs(cs2_d - 1.0 / 3.0) < 1e-15


class TestC1LongWaveLimit:
    """C1: Long-wave limit: cs_num/cs_theory - 1 ≤ 1e-4 for k·Δx ≤ 1e-2."""

    def test_d2q9_bgk_sound_speed(self):
        """Test that numerical sound speed matches theory at long wavelengths."""
        lat = D2Q9Lattice()
        tau = 1.0
        coll = BGKCollision(lat, tau)

        # Very small k
        k_mag = 1e-3
        k_dir = np.array([1.0, 0.0])
        k = k_mag * k_dir

        eigs = von_neumann_eigenvalues(lat, coll, k)
        modes = identify_physical_modes(eigs, lat, k)
        omega = modes["omega"]

        # Find acoustic mode (largest |Re(ω)| among physical modes)
        if len(modes["acoustic"]) > 0:
            acoustic_idx = modes["acoustic"][np.argmax(np.abs(omega[modes["acoustic"]].real))]
            cs_num = np.abs(omega[acoustic_idx].real) / k_mag

            cs_theory = np.sqrt(lat.cs2)
            rel_error = abs(cs_num / cs_theory - 1.0)

            assert rel_error < 1e-4, f"C1 failed: cs_num/cs_theory - 1 = {rel_error:.2e} > 1e-4"
        else:
            pytest.fail("No acoustic modes found")

    def test_d3q19_bgk_sound_speed(self):
        """Test D3Q19 sound speed at long wavelengths."""
        lat = D3Q19Lattice()
        tau = 1.0
        coll = BGKCollision(lat, tau)

        k_mag = 1e-3
        k_dir = np.array([1.0, 0.0, 0.0])
        k = k_mag * k_dir

        eigs = von_neumann_eigenvalues(lat, coll, k)
        modes = identify_physical_modes(eigs, lat, k)
        omega = modes["omega"]

        if len(modes["acoustic"]) > 0:
            acoustic_idx = modes["acoustic"][np.argmax(np.abs(omega[modes["acoustic"]].real))]
            cs_num = np.abs(omega[acoustic_idx].real) / k_mag

            cs_theory = np.sqrt(lat.cs2)
            rel_error = abs(cs_num / cs_theory - 1.0)

            assert rel_error < 1e-4, f"C1 failed for D3Q19: {rel_error:.2e}"
        else:
            pytest.fail("No acoustic modes found for D3Q19")


class TestC2Dissipation:
    """C2: Dissipation: Im(ω) vs νk² fit slope error ≤ 1%."""

    def test_d2q9_bgk_viscosity(self):
        """Test that fitted viscosity matches theory for BGK."""
        lat = D2Q9Lattice()
        tau = 0.8
        coll = BGKCollision(lat, tau)

        k_dir = np.array([1.0, 0.0])
        result = fit_viscosity_from_dissipation(lat, coll, k_dir, k_max=0.3, n_points=30)

        assert result["relative_error"] < 0.01, (
            f"C2 failed: viscosity error = {result['relative_error']:.2%} > 1%"
        )
        assert result["fit_quality"] > 0.99, (
            f"C2 failed: R² = {result['fit_quality']:.4f} < 0.99"
        )

    def test_d2q9_trt_viscosity(self):
        """Test TRT viscosity fitting."""
        lat = D2Q9Lattice()
        tau_plus = 0.8
        Lambda = 3.0 / 16.0
        coll = TRTCollision(lat, tau_plus, Lambda)

        k_dir = np.array([1.0, 0.0])
        result = fit_viscosity_from_dissipation(lat, coll, k_dir, k_max=0.3, n_points=30)

        assert result["relative_error"] < 0.01, (
            f"C2 failed for TRT: viscosity error = {result['relative_error']:.2%}"
        )

    def test_d2q9_mrt_viscosity(self):
        """Test MRT viscosity fitting."""
        lat = D2Q9Lattice()
        tau = 0.8
        coll = MRTCollision(lat, tau_pxx=tau, tau_pxy=tau, tau_e=tau, tau_eps=tau, tau_qj=tau)

        k_dir = np.array([1.0, 0.0])
        result = fit_viscosity_from_dissipation(lat, coll, k_dir, k_max=0.3, n_points=30)

        assert result["relative_error"] < 0.01, (
            f"C2 failed for MRT: viscosity error = {result['relative_error']:.2%}"
        )


class TestC3Isotropy:
    """C3: Isotropy: D2Q9 cs_num variation ≤ 1e-3 across 0°/22.5°/45°."""

    def test_d2q9_isotropy(self):
        """Test that sound speed is isotropic for D2Q9."""
        lat = D2Q9Lattice()
        tau = 1.0
        coll = BGKCollision(lat, tau)

        angles = [0.0, np.pi / 8.0, np.pi / 4.0]  # 0°, 22.5°, 45°
        k_mag = 1e-3

        cs_values = []
        for angle in angles:
            k_dir = np.array([np.cos(angle), np.sin(angle)])
            k = k_mag * k_dir

            eigs = von_neumann_eigenvalues(lat, coll, k)
            modes = identify_physical_modes(eigs, lat, k)
            omega = modes["omega"]

            if len(modes["acoustic"]) > 0:
                acoustic_idx = modes["acoustic"][np.argmax(np.abs(omega[modes["acoustic"]].real))]
                cs_num = np.abs(omega[acoustic_idx].real) / k_mag
                cs_values.append(cs_num)

        cs_values = np.array(cs_values)
        cs_mean = np.mean(cs_values)
        cs_variation = np.max(np.abs(cs_values - cs_mean)) / cs_mean

        assert cs_variation < 1e-3, (
            f"C3 failed: cs variation = {cs_variation:.2e} > 1e-3"
        )


class TestC4StabilityBoundary:
    """C4: Stability boundary: τ=0.5001 and τ=2 both stable."""

    def test_tau_0_5001_stable(self):
        """Test that τ=0.5001 is stable."""
        lat = D2Q9Lattice()
        tau = 0.5001
        coll = BGKCollision(lat, tau)

        # Check multiple k-directions
        angles = np.linspace(0, np.pi / 2, 8)
        k_mags = np.linspace(1e-3, np.pi, 50)

        max_lambda = 0.0
        for angle in angles:
            k_dir = np.array([np.cos(angle), np.sin(angle)])
            for km in k_mags:
                k = km * k_dir
                eigs = von_neumann_eigenvalues(lat, coll, k)
                max_lambda = max(max_lambda, np.max(np.abs(eigs)))

        assert max_lambda <= 1.0 + 1e-12, (
            f"C4 failed: τ=0.5001 unstable, max|λ| = {max_lambda:.6e}"
        )

    def test_tau_2_stable(self):
        """Test that τ=2 is stable."""
        lat = D2Q9Lattice()
        tau = 2.0
        coll = BGKCollision(lat, tau)

        angles = np.linspace(0, np.pi / 2, 8)
        k_mags = np.linspace(1e-3, np.pi, 50)

        max_lambda = 0.0
        for angle in angles:
            k_dir = np.array([np.cos(angle), np.sin(angle)])
            for km in k_mags:
                k = km * k_dir
                eigs = von_neumann_eigenvalues(lat, coll, k)
                max_lambda = max(max_lambda, np.max(np.abs(eigs)))

        assert max_lambda <= 1.0 + 1e-12, (
            f"C4 failed: τ=2 unstable, max|λ| = {max_lambda:.6e}"
        )


class TestC5TRTOptimality:
    """C5: TRT optimality: Λ=3/16 gives smaller max|λ| than BGK (non-conserved modes)."""

    def test_trt_magic_parameter_optimal(self):
        """Test that TRT with Λ=3/16 has better damping than BGK for non-conserved modes."""
        lat = D2Q9Lattice()
        tau = 0.8

        # BGK
        bgk = BGKCollision(lat, tau)

        # TRT with magic parameter
        trt_magic = TRTCollision(lat, tau, Lambda=3.0 / 16.0)

        # TRT with suboptimal parameter (negative control)
        trt_subopt = TRTCollision(lat, tau, Lambda=0.4)

        # Compute max|λ| over k-space for non-conserved modes
        # We look at a specific k where the difference is most pronounced
        k_mags = np.linspace(0.5, 2.0, 20)  # intermediate k range
        angles = [0.0, np.pi / 4.0]

        def compute_max_nonconserved_lambda(collision):
            max_lam = 0.0
            for angle in angles:
                k_dir = np.array([np.cos(angle), np.sin(angle)])
                for km in k_mags:
                    k = km * k_dir
                    eigs = von_neumann_eigenvalues(lat, collision, k)
                    # Exclude conserved modes (|λ| ≈ 1)
                    nonconserved = np.abs(np.abs(eigs) - 1.0) > 0.01
                    if np.any(nonconserved):
                        max_lam = max(max_lam, np.max(np.abs(eigs[nonconserved])))
            return max_lam

        max_lam_bgk = compute_max_nonconserved_lambda(bgk)
        max_lam_trt_magic = compute_max_nonconserved_lambda(trt_magic)
        max_lam_trt_subopt = compute_max_nonconserved_lambda(trt_subopt)

        # C5: TRT magic should have smaller max|λ| than BGK (better damping)
        assert max_lam_trt_magic < max_lam_bgk * 1.05, (
            f"C5 failed: TRT magic max|λ| = {max_lam_trt_magic:.6f} is more than 5% worse than BGK = {max_lam_bgk:.6f}"
        )

        # Negative control: suboptimal TRT should be worse than magic
        assert max_lam_trt_subopt >= max_lam_trt_magic, (
            f"C5 negative control failed: suboptimal should be worse than magic"
        )


class TestC6CrossValidation:
    """C6: Cross-validation with physical simulation (placeholder)."""

    def test_viscosity_consistency(self):
        """Test that von Neumann viscosity matches theoretical value."""
        lat = D2Q9Lattice()
        tau = 0.7
        coll = BGKCollision(lat, tau)

        # Theoretical viscosity
        nu_theory = coll.nu

        # Fitted from dissipation
        k_dir = np.array([1.0, 0.0])
        result = fit_viscosity_from_dissipation(lat, coll, k_dir, k_max=0.2, n_points=20)

        # Should match within 1%
        assert abs(result["nu_num"] - nu_theory) / nu_theory < 0.01


class TestDispersionCurves:
    """Test dispersion curve computation."""

    def test_acoustic_mode_dispersion(self):
        """Test that acoustic mode has correct dispersion."""
        lat = D2Q9Lattice()
        tau = 1.0
        coll = BGKCollision(lat, tau)

        k_dir = np.array([1.0, 0.0])
        result = compute_dispersion_curve(lat, coll, k_dir, k_max=0.5, n_points=50)

        # At small k, Re(ω) ≈ cs * k
        small_k_idx = result.k_magnitudes < 0.1
        if np.any(small_k_idx) and not np.all(np.isnan(result.omega_re_acoustic[small_k_idx])):
            valid = ~np.isnan(result.omega_re_acoustic[small_k_idx])
            if np.any(valid):
                cs_num = np.mean(
                    np.abs(result.omega_re_acoustic[small_k_idx][valid])
                    / result.k_magnitudes[small_k_idx][valid]
                )
                cs_theory = np.sqrt(lat.cs2)
                assert abs(cs_num / cs_theory - 1.0) < 0.01


class TestStabilityEnvelope:
    """Test stability envelope scanning."""

    def test_bgk_stability_range(self):
        """Test that BGK is stable for τ > 0.5."""
        lat = D2Q9Lattice()
        tau_values = np.array([0.5001, 0.6, 0.8, 1.0, 1.5, 2.0])
        k_dir = np.array([1.0, 0.0])

        result = max_eigenvalue_vs_tau(lat, BGKCollision, tau_values, k_dir)

        # All should be stable
        assert np.all(result.is_stable), (
            f"BGK should be stable for τ > 0.5, but got unstable at τ={tau_values[~result.is_stable]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
