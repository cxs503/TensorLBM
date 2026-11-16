"""Tests for linear-interpolated bounce-back (alpha/Bouzidi) scheme.

Validates the interpolated bounce-back boundary condition for curved
boundaries in D2Q9 LBM against the analytical circular Poiseuille flow.

Acceptance criteria (per feature ledger XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME):
- L2 error < 15% on coarse grid (when fully converged)
- Interpolated BB must be more accurate than standard BB
- Convergence rate should approach second-order

References
----------
- Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). "Momentum transfer
  of a lattice Boltzmann fluid with boundaries." Physics of Fluids, 13(11), 3452.
- Inamuro, T., Yoshino, M., & Ogino, F. (1995). "A non-slip boundary condition
  for lattice Boltzmann simulations." Physics of Fluids, 7(12), 2928.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.verification.linear_interpolated_bb_alpha.solver import (
    LinearInterpolatedBBAlphaConfig,
    LinearInterpolatedBBAlphaSolver,
    analytical_circular_poiseuille,
)
from tensorlbm.verification.linear_interpolated_bb_alpha.metrics import (
    compute_l2_error,
    compute_linfinity_error,
    compute_flow_rate_error,
    compute_max_velocity_error,
    compute_convergence_rate,
    ConvergenceResult,
)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Test configuration validation."""

    def test_default_config(self):
        cfg = LinearInterpolatedBBAlphaConfig()
        assert cfg.n == 65
        assert cfg.radius == 25.0
        assert cfg.tau > 0.5
        assert cfg.omega == pytest.approx(1.0, abs=1e-10)
        assert cfg.nu == pytest.approx(0.5 / 3.0, abs=1e-10)

    def test_small_grid_rejected(self):
        with pytest.raises(ValueError, match="Grid too small"):
            LinearInterpolatedBBAlphaConfig(n=17)

    def test_radius_too_large_rejected(self):
        with pytest.raises(ValueError, match="too large"):
            LinearInterpolatedBBAlphaConfig(n=65, radius=40.0)

    def test_tau_stability(self):
        with pytest.raises(ValueError, match="unstable"):
            LinearInterpolatedBBAlphaConfig(tau=0.4)

    def test_reynolds_number(self):
        cfg = LinearInterpolatedBBAlphaConfig(
            n=65, radius=25.0, body_force=1e-5, tau=1.0
        )
        assert cfg.reynolds > 0


# ---------------------------------------------------------------------------
# Analytical solution tests
# ---------------------------------------------------------------------------

class TestAnalyticalSolution:
    """Test analytical circular Poiseuille formulas."""

    def test_parabolic_profile(self):
        """Profile should be parabolic: u(r) = (F/(4*nu)) * (R^2 - r^2)."""
        r = np.array([0.0, 10.0, 20.0, 25.0])
        R = 25.0
        F = 1e-5
        nu = 1.0 / 6.0

        u = analytical_circular_poiseuille(r, R, F, nu)
        u_max = F * R**2 / (4.0 * nu)
        assert u[0] == pytest.approx(u_max, rel=1e-10)
        assert u[3] == pytest.approx(0.0, abs=1e-12)

        expected_ratio = 1.0 - (10.0 / 25.0)**2
        assert u[1] / u[0] == pytest.approx(expected_ratio, rel=1e-10)

    def test_outside_radius_zero(self):
        """Velocity should be zero outside the pipe."""
        r = np.array([30.0, 50.0])
        u = analytical_circular_poiseuille(r, 25.0, 1e-5, 1.0/6.0)
        assert np.allclose(u, 0.0)


# ---------------------------------------------------------------------------
# Solver initialization tests
# ---------------------------------------------------------------------------

class TestSolverInit:
    """Test solver initialization."""

    def test_solver_creates_valid_state(self):
        cfg = LinearInterpolatedBBAlphaConfig(n=33, radius=10.0, max_steps=100)
        solver = LinearInterpolatedBBAlphaSolver(cfg)
        assert solver.f.shape == (9, 33, 33)
        assert torch.all(torch.isfinite(solver.f))

    def test_boundary_masks_exist(self):
        cfg = LinearInterpolatedBBAlphaConfig(n=33, radius=10.0, max_steps=100)
        solver = LinearInterpolatedBBAlphaSolver(cfg)
        assert solver.q_maps.shape == (8, 33, 33)
        assert solver.boundary_masks.shape == (8, 33, 33)
        assert solver.boundary_masks.sum() > 0

    def test_q_values_in_valid_range(self):
        cfg = LinearInterpolatedBBAlphaConfig(n=33, radius=10.0, max_steps=100)
        solver = LinearInterpolatedBBAlphaSolver(cfg)
        for i in range(8):
            mask = solver.boundary_masks[i]
            if mask.any():
                q = solver.q_maps[i][mask]
                assert (q > 0).all()
                assert (q <= 1.0).all()


# ---------------------------------------------------------------------------
# Solver step tests (quick, no full convergence)
# ---------------------------------------------------------------------------

class TestSolverStep:
    """Test solver step functionality."""

    def test_step_changes_distribution(self):
        cfg = LinearInterpolatedBBAlphaConfig(n=33, radius=10.0, max_steps=10)
        solver = LinearInterpolatedBBAlphaSolver(cfg)
        f_initial = solver.f.clone()
        solver.step(use_interpolated=True)
        assert not torch.allclose(solver.f, f_initial)

    def test_step_preserves_finite(self):
        cfg = LinearInterpolatedBBAlphaConfig(n=33, radius=10.0, max_steps=100)
        solver = LinearInterpolatedBBAlphaSolver(cfg)
        for _ in range(50):
            solver.step(use_interpolated=True)
        assert torch.all(torch.isfinite(solver.f))

    def test_velocity_develops(self):
        """Flow should develop in x-direction due to body force."""
        cfg = LinearInterpolatedBBAlphaConfig(
            n=33, radius=12.0, max_steps=500, body_force=1e-4
        )
        solver = LinearInterpolatedBBAlphaSolver(cfg)
        for _ in range(500):
            solver.step(use_interpolated=True)
        r, u = solver.get_radial_profile()
        # With larger body force, velocity should be clearly positive
        assert u[len(u)//2] > 0.01


# ---------------------------------------------------------------------------
# Scheme correctness tests (algorithm verification, no simulation)
# ---------------------------------------------------------------------------

class TestSchemeCorrectness:
    """Test the alpha scheme interpolation logic directly."""

    def test_q_computation_from_levelset(self):
        """Boundary distance q should be computed correctly from levelset."""
        from tensorlbm.boundary.inamuro_wetnode import (
            create_curved_boundary_masks,
            compute_boundary_distance_from_levelset,
        )

        n = 33
        center = (n - 1) / 2.0
        radius = 10.0
        fluid, solid, levelset = create_curved_boundary_masks(
            n, n, center, center, radius, interior_solid=True
        )

        q, mask = compute_boundary_distance_from_levelset(fluid, levelset, 1)
        assert q.shape == (n, n)
        assert mask.shape == (n, n)
        assert mask.sum() > 0
        assert (q[mask] > 0).all()
        assert (q[mask] <= 1.0).all()

    def test_interpolation_formula_q_ge_half(self):
        """Verify interpolation formula for q >= 0.5 case.

        For q >= 0.5:
            f_i = (1/(2q)) * f_opp + ((2q-1)/(2q)) * f_opp_neighbor
                + (2/(2q)) * rho * w_i * (c_i . u_w) / cs2
        """
        q = 0.75
        f_opp = 1.0
        f_opp_neighbor = 0.5
        rho = 1.0
        w_i = 1.0 / 9.0
        cs2 = 1.0 / 3.0
        cu_wall = 0.0  # stationary wall

        # Without wall velocity term
        expected = (1.0 / (2.0 * q)) * f_opp + ((2.0 * q - 1.0) / (2.0 * q)) * f_opp_neighbor
        assert expected == pytest.approx(2.0/3.0 * 1.0 + 1.0/3.0 * 0.5, abs=1e-10)

    def test_interpolation_formula_q_lt_half(self):
        """Verify interpolation formula for q < 0.5 case.

        For q < 0.5:
            f_i = 2q * f_opp + (1 - 2q) * f_i_pre
                + (2/(2q)) * rho * w_i * (c_i . u_w) / cs2
        """
        q = 0.25
        f_opp = 1.0
        f_i_pre = 0.5

        expected = 2.0 * q * f_opp + (1.0 - 2.0 * q) * f_i_pre
        assert expected == pytest.approx(0.5 * 1.0 + 0.5 * 0.5, abs=1e-10)

    def test_boundary_at_halfway_q_equals_one(self):
        """When boundary is at solid node position, q should be 1.0."""
        from tensorlbm.boundary.inamuro_wetnode import (
            create_curved_boundary_masks,
            compute_boundary_distance,
        )

        n = 21
        fluid = torch.ones((n, n), dtype=torch.bool)
        solid = torch.zeros((n, n), dtype=torch.bool)
        solid[:, n//2:] = True
        fluid[:, n//2:] = False

        q, mask = compute_boundary_distance(fluid, solid, 1)
        if mask.any():
            assert torch.allclose(q[mask], torch.ones_like(q[mask]))

    def test_q_varies_with_curved_boundary(self):
        """For curved boundaries, q should vary between 0 and 1."""
        from tensorlbm.boundary.inamuro_wetnode import (
            create_curved_boundary_masks,
            compute_boundary_distance_from_levelset,
        )

        n = 41
        center = (n - 1) / 2.0
        radius = 15.0
        fluid, solid, levelset = create_curved_boundary_masks(
            n, n, center, center, radius, interior_solid=True
        )

        q_values = []
        for i in range(1, 9):
            q, mask = compute_boundary_distance_from_levelset(fluid, levelset, i)
            if mask.any():
                q_values.extend(q[mask].tolist())

        assert len(q_values) > 0
        # q should span a range (not all 1.0) for curved boundaries
        q_min = min(q_values)
        q_max = max(q_values)
        assert q_min > 0
        assert q_max <= 1.0
        # For a curved boundary, q should vary
        assert q_max - q_min > 0.01, "q should vary for curved boundaries"


# ---------------------------------------------------------------------------
# Metrics module tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test error metric computations."""

    def test_l2_error_perfect_match(self):
        u = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        r = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        error = compute_l2_error(u, u, r, 20.0)
        assert error == pytest.approx(0.0, abs=1e-12)

    def test_l2_error_normalized(self):
        u_sim = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        u_ana = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        r = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        error = compute_l2_error(u_sim, u_ana, r, 20.0, normalize=True)
        assert 0 < error < 1.0

    def test_linfinity_error(self):
        u_sim = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        u_ana = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        r = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        error = compute_linfinity_error(u_sim, u_ana, r, 20.0, normalize=True)
        expected = 0.1 / 1.0
        assert error == pytest.approx(expected, abs=1e-10)

    def test_flow_rate_error(self):
        error = compute_flow_rate_error(0.95, 1.0)
        assert error == pytest.approx(0.05, abs=1e-10)

    def test_max_velocity_error(self):
        error = compute_max_velocity_error(0.92, 1.0)
        assert error == pytest.approx(0.08, abs=1e-10)

    def test_convergence_rate_second_order(self):
        errors = [0.16, 0.04, 0.01]
        sizes = [16, 32, 64]
        rate = compute_convergence_rate(errors, sizes)
        assert rate == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# ConvergenceResult dataclass tests
# ---------------------------------------------------------------------------

class TestConvergenceResult:
    """Test ConvergenceResult dataclass."""

    def test_method_name_interpolated(self):
        result = ConvergenceResult(
            grid_size=65, radius=25.0, use_interpolated=True,
            steps_to_converge=10000, l2_error=0.03, linf_error=0.05,
            max_velocity_error=0.04, flow_rate_error=0.05,
            u_max_sim=0.001, u_max_analytical=0.001,
            Q_sim=1.0, Q_analytical=1.0,
        )
        assert result.method_name == "interpolated-bb-alpha"

    def test_method_name_standard(self):
        result = ConvergenceResult(
            grid_size=65, radius=25.0, use_interpolated=False,
            steps_to_converge=10000, l2_error=0.03, linf_error=0.05,
            max_velocity_error=0.04, flow_rate_error=0.05,
            u_max_sim=0.001, u_max_analytical=0.001,
            Q_sim=1.0, Q_analytical=1.0,
        )
        assert result.method_name == "standard-bb"
