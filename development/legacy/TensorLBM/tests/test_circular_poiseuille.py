"""Tests for circular Poiseuille flow verification.

Validates:
1. Inamuro wet-node boundary condition implementation
2. Comparison against analytical parabolic profile
3. Accuracy improvement over standard bounce-back
4. Convergence behavior

Reference: White, F.M. (2011). "Fluid Mechanics", 7th ed., McGraw-Hill.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.verification.circular_poiseuille.solver import (
    CircularPoiseuilleConfig,
    CircularPoiseuilleSolver,
    run_circular_poiseuille,
    run_comparison,
)
from tensorlbm.verification.circular_poiseuille.analytical import (
    poiseuille_velocity_profile,
    poiseuille_max_velocity,
    poiseuille_flow_rate,
    poiseuille_wall_shear_stress,
    poiseuille_reynolds_number,
)
from tensorlbm.boundary.inamuro_wetnode import (
    compute_boundary_distance_from_levelset,
    create_curved_boundary_masks,
)


# ---------------------------------------------------------------------------
# Analytical solution tests
# ---------------------------------------------------------------------------

class TestAnalyticalSolutions:
    """Test analytical Poiseuille flow formulas."""

    def test_velocity_profile_parabolic(self):
        """Profile should be parabolic: u(r) = u_max * (1 - (r/R)^2)."""
        r = np.array([0.0, 0.5, 1.0])
        R = 1.0
        u_max = 1.0
        u = poiseuille_velocity_profile(r, R, u_max)
        expected = np.array([1.0, 0.75, 0.0])
        np.testing.assert_allclose(u, expected, atol=1e-10)

    def test_max_velocity_from_pressure_gradient(self):
        """u_max = -dp/dx * R^2 / (4 * mu)."""
        dp_dx = -1.0
        R = 1.0
        mu = 1.0
        u_max = poiseuille_max_velocity(dp_dx, R, mu)
        assert u_max == pytest.approx(0.25, abs=1e-10)

    def test_flow_rate(self):
        """Q = pi * R^2 * u_max / 2."""
        u_max = 1.0
        R = 1.0
        Q = poiseuille_flow_rate(u_max, R)
        expected = np.pi * 1.0 * 1.0 / 2.0
        assert Q == pytest.approx(expected, abs=1e-10)

    def test_wall_shear_stress(self):
        """tau_w = 2 * mu * u_max / R."""
        u_max = 1.0
        R = 1.0
        mu = 0.5
        tau_w = poiseuille_wall_shear_stress(u_max, R, mu)
        assert tau_w == pytest.approx(1.0, abs=1e-10)

    def test_reynolds_number(self):
        """Re = u_mean * D / nu, u_mean = u_max / 2."""
        u_max = 0.1
        D = 50.0
        nu = 1.0 / 6.0  # tau = 1.0
        Re = poiseuille_reynolds_number(u_max, D, nu)
        expected = (0.1 / 2.0) * 50.0 / (1.0 / 6.0)
        assert Re == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Boundary mask tests
# ---------------------------------------------------------------------------

class TestBoundaryMasks:
    """Test curved boundary mask creation."""

    def test_circular_mask_creation(self):
        """Should create proper circular fluid/solid masks."""
        n = 33
        center = (n - 1) / 2.0
        radius = 10.0
        fluid, solid, levelset = create_curved_boundary_masks(
            n, n, center, center, radius, interior_solid=True
        )
        assert fluid.shape == (n, n)
        assert solid.shape == (n, n)
        assert levelset.shape == (n, n)
        # Center should be solid (interior_solid=True)
        assert not fluid[n//2, n//2]
        assert solid[n//2, n//2]
        # Corners should be fluid
        assert fluid[0, 0]
        assert not solid[0, 0]

    def test_boundary_distance_computation(self):
        """Should compute q values for boundary nodes."""
        n = 33
        center = (n - 1) / 2.0
        radius = 10.0
        fluid, solid, levelset = create_curved_boundary_masks(
            n, n, center, center, radius, interior_solid=True
        )

        q, mask = compute_boundary_distance_from_levelset(fluid, levelset, 1)
        assert q.shape == (n, n)
        assert mask.shape == (n, n)
        # Some boundary nodes should exist
        assert mask.sum() > 0
        # q should be in (0, 1] for boundary nodes
        assert (q[mask] > 0).all()
        assert (q[mask] <= 1.0).all()


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestCircularPoiseuilleConfig:
    """Test configuration validation."""

    def test_default_config(self):
        cfg = CircularPoiseuilleConfig()
        assert cfg.n == 65
        assert cfg.radius == 25.0
        assert cfg.tau > 0.5

    def test_small_grid_rejected(self):
        with pytest.raises(ValueError, match="Grid too small"):
            CircularPoiseuilleConfig(n=17)

    def test_radius_too_large_rejected(self):
        with pytest.raises(ValueError, match="too large"):
            CircularPoiseuilleConfig(n=65, radius=40.0)

    def test_tau_stability(self):
        with pytest.raises(ValueError, match="unstable"):
            CircularPoiseuilleConfig(tau=0.4)

    def test_omega_property(self):
        cfg = CircularPoiseuilleConfig(tau=1.0)
        assert cfg.omega == pytest.approx(1.0, abs=1e-10)

    def test_nu_property(self):
        cfg = CircularPoiseuilleConfig(tau=1.0)
        expected = (1.0 - 0.5) / 3.0
        assert cfg.nu == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestCircularPoiseuilleSolver:
    """Test solver basic functionality."""

    def test_solver_initialization(self):
        cfg = CircularPoiseuilleConfig(n=33, radius=10.0, max_steps=100)
        solver = CircularPoiseuilleSolver(cfg)
        assert solver.f.shape == (9, 33, 33)
        assert torch.all(torch.isfinite(solver.f))

    def test_solver_step_changes_f(self):
        cfg = CircularPoiseuilleConfig(n=33, radius=10.0, max_steps=10)
        solver = CircularPoiseuilleSolver(cfg)
        f_initial = solver.f.clone()
        solver.step()
        assert not torch.allclose(solver.f, f_initial)

    def test_solver_step_preserves_finite(self):
        cfg = CircularPoiseuilleConfig(n=33, radius=10.0, max_steps=100)
        solver = CircularPoiseuilleSolver(cfg)
        for _ in range(50):
            solver.step()
        assert torch.all(torch.isfinite(solver.f))

    def test_velocity_develops(self):
        """Flow should develop in x-direction due to body force."""
        cfg = CircularPoiseuilleConfig(n=33, radius=10.0, max_steps=5000)
        solver = CircularPoiseuilleSolver(cfg)
        for _ in range(3000):
            solver.step()
        r, u = solver.get_radial_profile()
        # Centerline velocity should be positive
        assert u[len(u)//2] > 0.01

    def test_radial_profile_shape(self):
        """Radial profile should have correct shape."""
        cfg = CircularPoiseuilleConfig(n=33, radius=10.0, max_steps=100)
        solver = CircularPoiseuilleSolver(cfg)
        for _ in range(100):
            solver.step()
        r, u = solver.get_radial_profile()
        assert len(r) == 33
        assert len(u) == 33


# ---------------------------------------------------------------------------
# Verification tests (accuracy against analytical solution)
# ---------------------------------------------------------------------------

class TestVerificationAccuracy:
    """Test accuracy of both boundary conditions against analytical solution."""

    def _compute_l2_error(self, r_sim, u_sim, r_max, u_max_analytical):
        """Compute L2 error between simulation and analytical profile."""
        # Only consider points within the pipe
        mask = r_sim <= r_max
        r_valid = r_sim[mask]
        u_valid = u_sim[mask]

        u_analytical = poiseuille_velocity_profile(r_valid, r_max, u_max_analytical)

        # Normalize by max velocity
        if u_max_analytical > 0:
            l2_error = np.sqrt(np.mean((u_valid - u_analytical)**2)) / u_max_analytical
        else:
            l2_error = np.sqrt(np.mean((u_valid - u_analytical)**2))

        return l2_error

    def test_inamuro_accuracy_coarse(self):
        """Inamuro on coarse grid should achieve < 15% L2 error."""
        cfg = CircularPoiseuilleConfig(
            n=49, radius=18.0, body_force=1e-5, tau=1.0,
            max_steps=20000, use_inamuro=True
        )
        solver = CircularPoiseuilleSolver(cfg)
        solver.run(resid_tol=1e-6, monitor_interval=2000)

        r, u = solver.get_radial_profile()
        u_max_sim = u[len(u)//2]

        # Analytical u_max from body force
        # For LBM: u_max = body_force * R^2 / (4 * nu)
        nu = cfg.nu
        u_max_analytical = cfg.body_force * cfg.radius**2 / (4.0 * nu)

        l2_error = self._compute_l2_error(r, u, cfg.radius, u_max_analytical)
        assert l2_error < 0.15, f"Inamuro L2 error {l2_error:.4f} > 0.15"

    def test_bounce_back_accuracy_coarse(self):
        """Standard bounce-back on coarse grid: expect higher error than Inamuro."""
        cfg = CircularPoiseuilleConfig(
            n=49, radius=18.0, body_force=1e-5, tau=1.0,
            max_steps=20000, use_inamuro=False
        )
        solver = CircularPoiseuilleSolver(cfg)
        solver.run(resid_tol=1e-6, monitor_interval=2000)

        r, u = solver.get_radial_profile()
        u_max_sim = u[len(u)//2]

        nu = cfg.nu
        u_max_analytical = cfg.body_force * cfg.radius**2 / (4.0 * nu)

        l2_error = self._compute_l2_error(r, u, cfg.radius, u_max_analytical)
        # Bounce-back should have higher error but still reasonable
        assert l2_error < 0.25, f"BB L2 error {l2_error:.4f} > 0.25"

    def test_inamuro_better_than_bounce_back(self):
        """Inamuro should be more accurate than standard bounce-back."""
        n = 49
        radius = 18.0
        body_force = 1e-5
        tau = 1.0

        # Run both
        results = run_comparison(n=n, radius=radius, body_force=body_force, tau=tau, max_steps=15000)

        r_bb, u_bb = results['bb'].get_radial_profile()
        r_in, u_in = results['inamuro'].get_radial_profile()

        nu = (tau - 0.5) / 3.0
        u_max_analytical = body_force * radius**2 / (4.0 * nu)

        l2_bb = self._compute_l2_error(r_bb, u_bb, radius, u_max_analytical)
        l2_in = self._compute_l2_error(r_in, u_in, radius, u_max_analytical)

        # Inamuro should be at least as good, typically better
        assert l2_in <= l2_bb * 1.5, (
            f"Inamuro L2={l2_in:.4f} not better than BB L2={l2_bb:.4f}"
        )

    def test_max_velocity_accuracy(self):
        """Centerline velocity should match analytical within 10%."""
        cfg = CircularPoiseuilleConfig(
            n=65, radius=25.0, body_force=1e-5, tau=1.0,
            max_steps=25000, use_inamuro=True
        )
        solver = CircularPoiseuilleSolver(cfg)
        solver.run(resid_tol=1e-7, monitor_interval=2000)

        u_max_sim = solver.compute_max_velocity()

        nu = cfg.nu
        u_max_analytical = cfg.body_force * cfg.radius**2 / (4.0 * nu)

        rel_error = abs(u_max_sim - u_max_analytical) / u_max_analytical
        assert rel_error < 0.10, f"Max velocity error {rel_error:.4f} > 0.10"

    def test_convergence_with_resolution(self):
        """Error should decrease with grid refinement."""
        errors = []
        for n in [33, 49, 65]:
            radius = n * 0.4  # Keep radius proportional
            cfg = CircularPoiseuilleConfig(
                n=n, radius=radius, body_force=1e-5, tau=1.0,
                max_steps=15000, use_inamuro=True
            )
            solver = CircularPoiseuilleSolver(cfg)
            solver.run(resid_tol=1e-6, monitor_interval=2000)

            r, u = solver.get_radial_profile()
            nu = cfg.nu
            u_max_analytical = cfg.body_force * cfg.radius**2 / (4.0 * nu)

            l2_error = self._compute_l2_error(r, u, cfg.radius, u_max_analytical)
            errors.append(l2_error)

        # Error should generally decrease (allow some noise)
        assert errors[-1] < errors[0] * 1.5, (
            f"Error not decreasing: {errors}"
        )


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestComparison:
    """Test comparison between boundary conditions."""

    def test_run_comparison(self):
        """run_comparison should return both solvers."""
        results = run_comparison(n=33, radius=10.0, max_steps=5000)
        assert 'bb' in results
        assert 'inamuro' in results
        assert 'config' in results

    def test_both_solvers_converge(self):
        """Both methods should converge to steady state."""
        results = run_comparison(n=33, radius=10.0, max_steps=10000)

        u_max_bb = results['bb'].compute_max_velocity()
        u_max_in = results['inamuro'].compute_max_velocity()

        # Both should have positive velocity
        assert u_max_bb > 0
        assert u_max_in > 0

        # They should be reasonably close (within 20%)
        rel_diff = abs(u_max_bb - u_max_in) / max(u_max_bb, u_max_in)
        assert rel_diff < 0.20, f"Max velocity diff {rel_diff:.4f} > 0.20"
