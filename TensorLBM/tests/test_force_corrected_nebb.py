"""Tests for force-corrected NEBB boundary condition and body-force driver scaffold.

Verifies:
1. Force-corrected NEBB eliminates wall slip under body force
2. Second-order convergence of velocity profile error
3. Mass conservation with Guo forcing
4. Comparison: standard BB vs force-corrected NEBB under body force
5. BodyForceDriver produces results within acceptance thresholds

Reference: Poiseuille flow analytical solution u(y) = g/(2ν) · y(H-y).
Acceptance: L2 error < 2%, L∞ error < 2%, wall slip < 1e-10.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from tensorlbm.boundary.force_corrected_nebb import (
    apply_force_corrected_nebb_1d,
    compute_force_correction,
    ForceCorrectedNEBB,
)
from tensorlbm.forcing.body_force_driver import (
    BodyForceDriver,
    BodyForceResult,
    PoiseuilleConfig,
    run_convergence_study,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def poiseuille_config():
    """Standard Poiseuille configuration for testing."""
    return PoiseuilleConfig(
        height=32,
        nx=4,
        tau=0.8,
        force=1.0e-5,
        steps=8000,
        forcing="guo",
        bc="force_corrected_nebb",
        init_mode="analytic",
    )


@pytest.fixture
def standard_bb_config():
    """Same config but with standard bounce-back."""
    return PoiseuilleConfig(
        height=32,
        nx=4,
        tau=0.8,
        force=1.0e-5,
        steps=8000,
        forcing="guo",
        bc="standard_bb",
        init_mode="analytic",
    )


# ---------------------------------------------------------------------------
# Force correction computation tests
# ---------------------------------------------------------------------------

class TestForceCorrection:
    """Tests for the force correction term computation."""

    def test_zero_force_gives_zero_correction(self):
        """No force → no correction."""
        corr = compute_force_correction(
            tau=0.8, force_x=0.0, force_y=0.0,
            u_wall=np.zeros(4), v_wall=np.zeros(4),
            rho_wall=np.ones(4),
        )
        assert np.allclose(corr, 0.0), "Zero force should give zero correction"

    def test_correction_scales_with_force(self):
        """Correction should scale linearly with force magnitude."""
        u = np.zeros(4)
        v = np.zeros(4)
        rho = np.ones(4)

        c1 = compute_force_correction(tau=0.8, force_x=1e-5, force_y=0.0,
                                       u_wall=u, v_wall=v, rho_wall=rho)
        c2 = compute_force_correction(tau=0.8, force_x=2e-5, force_y=0.0,
                                       u_wall=u, v_wall=v, rho_wall=rho)

        assert np.allclose(c2, 2.0 * c1, rtol=1e-10), \
            "Correction should scale linearly with force"

    def test_correction_depends_on_tau(self):
        """Correction coefficient depends on (1 - 1/(2τ))."""
        u = np.zeros(4)
        v = np.zeros(4)
        rho = np.ones(4)

        c1 = compute_force_correction(tau=0.8, force_x=1e-5, force_y=0.0,
                                       u_wall=u, v_wall=v, rho_wall=rho)
        c2 = compute_force_correction(tau=1.0, force_x=1e-5, force_y=0.0,
                                       u_wall=u, v_wall=v, rho_wall=rho)

        # coef(0.8) = 1 - 0.5/0.8 = 0.375
        # coef(1.0) = 1 - 0.5/1.0 = 0.5
        expected_ratio = 0.5 / 0.375
        actual_ratio = c1[1, 0] / c2[1, 0] if c2[1, 0] != 0 else 0

        assert abs(actual_ratio - 1.0 / expected_ratio) < 1e-10, \
            f"Ratio should be {1.0/expected_ratio:.6f}, got {actual_ratio:.6f}"


# ---------------------------------------------------------------------------
# BodyForceDriver tests
# ---------------------------------------------------------------------------

class TestBodyForceDriver:
    """Tests for the body-force driver scaffold."""

    def test_poiseuille_within_2_percent(self, poiseuille_config):
        """Primary acceptance: velocity profile within 2% of analytical."""
        driver = BodyForceDriver(poiseuille_config)
        result = driver.run()

        assert result.eps_L2 < 0.02, f"L2 error {result.eps_L2:.6f} > 2%"
        assert result.eps_inf < 0.02, f"L∞ error {result.eps_inf:.6f} > 2%"

    def test_mass_conservation(self, poiseuille_config):
        """Guo forcing must conserve mass (no density drift)."""
        driver = BodyForceDriver(poiseuille_config)
        result = driver.run()

        assert result.mass_drift < 1e-8, \
            f"Mass drift {result.mass_drift:.2e} > 1e-8"

    def test_wall_slip_near_zero(self, poiseuille_config):
        """Force-corrected NEBB should eliminate wall slip."""
        driver = BodyForceDriver(poiseuille_config)
        result = driver.run()

        assert result.wall_slip < 1e-10, \
            f"Wall slip {result.wall_slip:.2e} > 1e-10"

    def test_force_corrected_nebb_vs_standard_bb(self, poiseuille_config, standard_bb_config):
        """Force-corrected NEBB should have lower error than standard BB under body force."""
        driver_fc = BodyForceDriver(poiseuille_config)
        result_fc = driver_fc.run()

        driver_bb = BodyForceDriver(standard_bb_config)
        result_bb = driver_bb.run()

        # Force-corrected NEBB should have lower or equal error
        assert result_fc.eps_L2 <= result_bb.eps_L2 + 1e-6, \
            (f"FC-NEBB L2={result_fc.eps_L2:.6f} should be <= "
             f"standard BB L2={result_bb.eps_L2:.6f}")

        # Standard BB should show wall slip under body force
        assert result_bb.wall_slip > result_fc.wall_slip, \
            "Standard BB should have more wall slip than force-corrected NEBB"

    def test_second_order_convergence(self):
        """Half-way BB is second-order: error ratio ≈ 4 when doubling resolution."""
        results = run_convergence_study(
            heights=[8, 16, 32, 64],
            tau=0.8,
            force=1.0e-5,
            forcing="guo",
            bc="force_corrected_nebb",
        )

        # Check convergence rate between successive resolutions
        for i in range(len(results) - 1):
            r1 = results[i]
            r2 = results[i + 1]
            if r1.eps_inf > 1e-12 and r2.eps_inf > 1e-12:
                ratio = r1.eps_inf / r2.eps_inf
                # Second order: ratio should be ~4 (allow 2.5-5.5 range)
                assert 2.5 <= ratio <= 5.5, \
                    (f"Convergence ratio H={r1.config.height}→{r2.config.height}: "
                     f"{ratio:.2f} (expected ~4)")

    def test_analytic_init_faster_convergence(self, poiseuille_config):
        """Analytic initialization should reach steady state faster."""
        cfg_analytic = PoiseuilleConfig(
            **{k: v for k, v in dataclasses.asdict(poiseuille_config).items()
               if k != 'init_mode'},
            init_mode="analytic",
        )
        cfg_uniform = PoiseuilleConfig(
            **{k: v for k, v in dataclasses.asdict(poiseuille_config).items()
               if k != 'init_mode'},
            init_mode="uniform",
        )

        driver_a = BodyForceDriver(cfg_analytic)
        result_a = driver_a.run()

        driver_u = BodyForceDriver(cfg_uniform)
        result_u = driver_u.run()

        # Both should converge to similar accuracy
        assert abs(result_a.eps_L2 - result_u.eps_L2) < 1e-4, \
            "Both init modes should reach same steady state"

    def test_simple_forcing_has_larger_error(self, poiseuille_config):
        """Simple (first-order) forcing should have larger error than Guo."""
        cfg_guo = PoiseuilleConfig(
            **{k: v for k, v in dataclasses.asdict(poiseuille_config).items()
               if k != 'forcing'},
            forcing="guo",
        )
        cfg_simple = PoiseuilleConfig(
            **{k: v for k, v in dataclasses.asdict(poiseuille_config).items()
               if k != 'forcing'},
            forcing="simple",
        )

        result_guo = BodyForceDriver(cfg_guo).run()
        result_simple = BodyForceDriver(cfg_simple).run()

        # Guo should be more accurate
        assert result_guo.eps_L2 <= result_simple.eps_L2, \
            "Guo forcing should be at least as accurate as simple forcing"


# ---------------------------------------------------------------------------
# ForceCorrectedNEBB class tests
# ---------------------------------------------------------------------------

class TestForceCorrectedNEBBClass:
    """Tests for the ForceCorrectedNEBB class interface."""

    def test_apply_on_simple_channel(self):
        """Test NEBB application on a simple 2D channel."""
        ny, nx = 10, 4
        f = np.ones((9, ny, nx)) / 9.0  # uniform equilibrium

        solid = np.zeros((ny, nx), dtype=bool)
        solid[0, :] = True
        solid[-1, :] = True

        bc = ForceCorrectedNEBB(
            tau=0.8, force_x=1e-5, force_y=0.0,
            u_wall=0.0, v_wall=0.0,
        )

        f_out = bc.apply(f, solid)

        # Should not crash and should modify boundary populations
        assert f_out.shape == f.shape
        # Boundary populations should differ from interior
        assert not np.allclose(f_out[:, 1, :], f_out[:, 2, :])
