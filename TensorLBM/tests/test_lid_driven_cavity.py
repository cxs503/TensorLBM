"""Tests for lid-driven cavity verification module.

Validates D2Q9 LBM lid-driven cavity against Ghia et al. (1982) reference data.

Test cases:
- Re=100: Low Re, well-resolved, should match Ghia within 5%
- Re=400: Moderate Re, should match within 8%
- Re=1000: Higher Re, boundary layer thinner, may need finer grid
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.verification.lid_driven_cavity.solver import (
    CavityConfig,
    CavitySolver,
)
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    GhiaReference,
    get_ghia_reference,
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    interpolate_ghia,
)
from tensorlbm.verification.lid_driven_cavity.verification import (
    VerificationResult,
    compute_errors,
    run_single_case,
)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestCavityConfig:
    """Test CavityConfig validation and derived properties."""

    def test_default_config(self):
        cfg = CavityConfig()
        assert cfg.n == 129
        assert cfg.re == 100.0
        assert cfg.u_lid == 0.1
        assert cfg.tau > 0.5

    def test_tau_computed_from_re(self):
        """tau should be computed from Re and u_lid."""
        cfg = CavityConfig(n=129, re=100, u_lid=0.1)
        L = cfg.n - 1
        nu = cfg.u_lid * L / cfg.re
        expected_tau = nu / (1.0/3.0) + 0.5
        assert abs(cfg.tau - expected_tau) < 1e-10

    def test_small_grid_rejected(self):
        with pytest.raises(ValueError, match="Grid too small"):
            CavityConfig(n=9)

    def test_high_velocity_rejected(self):
        with pytest.raises(ValueError, match="too high"):
            CavityConfig(u_lid=0.5)

    def test_omega_property(self):
        cfg = CavityConfig()
        assert abs(cfg.omega - 1.0/cfg.tau) < 1e-10


# ---------------------------------------------------------------------------
# Reference data tests
# ---------------------------------------------------------------------------

class TestReferenceData:
    """Test Ghia reference data integrity."""

    def test_ghia_re100_data(self):
        assert GHIA_RE100.re == 100.0
        assert len(GHIA_RE100.y) == len(GHIA_RE100.u)
        assert len(GHIA_RE100.x) == len(GHIA_RE100.v)
        # u should be negative in lower half (primary vortex)
        assert np.all(GHIA_RE100.u[:7] < 0)
        # v should be negative in left half
        assert np.all(GHIA_RE100.v[:7] < 0)

    def test_ghia_re400_data(self):
        assert GHIA_RE400.re == 400.0
        assert len(GHIA_RE400.y) == len(GHIA_RE400.u)

    def test_ghia_re1000_data(self):
        assert GHIA_RE1000.re == 1000.0
        assert len(GHIA_RE1000.y) == len(GHIA_RE1000.u)

    def test_get_ghia_reference_rejects_untabulated_re(self):
        """Re=5000 lives in Ghia Table III/IV, not here: it must raise, not fall back."""
        import pytest

        with pytest.raises(KeyError):
            get_ghia_reference(5000)
        with pytest.raises(KeyError):
            get_ghia_reference(3200)

    def test_get_ghia_reference(self):
        ref = get_ghia_reference(100)
        assert ref.re == 100.0
        ref = get_ghia_reference(400)
        assert ref.re == 400.0

    def test_interpolate_ghia(self):
        """Interpolation should return values at requested points."""
        y_test = np.array([0.0, 0.5, 1.0])
        u_interp = interpolate_ghia(GHIA_RE100, y_test, axis='u')
        assert len(u_interp) == 3
        # At y=0 and y=1, u should be 0 (no-slip walls)
        assert u_interp[0] == pytest.approx(0.0, abs=0.01)
        assert u_interp[-1] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestCavitySolver:
    """Test CavitySolver basic functionality."""

    def test_solver_initialization(self):
        cfg = CavityConfig(n=33, re=100, max_steps=100)
        solver = CavitySolver(cfg)
        assert solver.f.shape == (9, 33, 33)
        assert np.all(np.isfinite(solver.f.numpy()))

    def test_solver_step_changes_f(self):
        cfg = CavityConfig(n=33, re=100, max_steps=10)
        solver = CavitySolver(cfg)
        f_initial = solver.f.clone()
        solver.step()
        assert not torch.allclose(solver.f, f_initial)

    def test_solver_step_preserves_finite(self):
        cfg = CavityConfig(n=33, re=100, max_steps=100)
        solver = CavitySolver(cfg)
        for _ in range(50):
            solver.step()
        assert np.all(np.isfinite(solver.f.numpy()))

    def test_mass_conservation(self):
        """Total mass should be approximately conserved."""
        cfg = CavityConfig(n=33, re=100, max_steps=100)
        solver = CavitySolver(cfg)
        mass_initial = solver.f.sum().item()

        for _ in range(50):
            solver.step()

        mass_final = solver.f.sum().item()
        # Mass change should be small (< 1%)
        rel_change = abs(mass_final - mass_initial) / mass_initial
        assert rel_change < 0.01

    def test_get_profiles(self):
        cfg = CavityConfig(n=33, re=100, max_steps=500)
        solver = CavitySolver(cfg)
        for _ in range(200):
            solver.step()

        y, u, x, v = solver.get_profiles()
        assert len(y) == 33
        assert len(u) == 33
        assert len(x) == 33
        assert len(v) == 33
        # Velocities should be finite
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# Verification tests (lightweight, low Re, coarse grid)
# ---------------------------------------------------------------------------

class TestVerification:
    """Test verification against Ghia reference."""

    def test_compute_errors(self):
        """Error computation should work correctly."""
        y_sim = np.array([0.0, 0.5, 1.0])
        u_sim = np.array([0.0, -0.15, 0.0])
        x_sim = np.array([0.0, 0.5, 1.0])
        v_sim = np.array([0.0, -0.07, 0.0])

        errors = compute_errors(y_sim, u_sim, x_sim, v_sim, GHIA_RE100)
        assert 'l2_u' in errors
        assert 'l2_v' in errors
        assert 'max_u' in errors
        assert 'max_v' in errors
        assert errors['l2_u'] >= 0
        assert errors['l2_v'] >= 0

    def test_re100_coarse(self):
        """Re=100 on coarse grid should show correct trends."""
        cfg = CavityConfig(n=33, re=100, max_steps=5000, u_lid=0.1)
        solver = CavitySolver(cfg)

        # Run for enough steps to develop flow
        for _ in range(3000):
            solver.step()

        y, u, x, v = solver.get_profiles()

        # Primary vortex: u should be negative in lower half
        assert np.min(u) < -0.05
        # v should be negative in left half
        assert np.min(v) < -0.02

    def test_re100_medium_grid(self):
        """Re=100 on medium grid should approach Ghia within 15%."""
        cfg = CavityConfig(n=65, re=100, max_steps=15000, u_lid=0.1)
        solver = CavitySolver(cfg)

        for _ in range(10000):
            solver.step()

        y, u, x, v = solver.get_profiles()
        ref = get_ghia_reference(100)
        errors = compute_errors(y, u, x, v, ref)

        # At n=65, expect L2 error < 15%
        assert errors['l2_u'] < 0.15, f"L2 u error {errors['l2_u']:.4f} > 0.15"
        assert errors['l2_v'] < 0.15, f"L2 v error {errors['l2_v']:.4f} > 0.15"

    def test_u_min_location(self):
        """Minimum u should be near y=0.5 for Re=100."""
        cfg = CavityConfig(n=65, re=100, max_steps=15000, u_lid=0.1)
        solver = CavitySolver(cfg)

        for _ in range(10000):
            solver.step()

        y, u, _, _ = solver.get_profiles()
        u_min_idx = np.argmin(u)
        u_min_y = y[u_min_idx]

        # Ghia: u_min at y ≈ 0.5 for Re=100
        assert 0.35 < u_min_y < 0.65, f"u_min at y={u_min_y:.3f}, expected ~0.5"


# Import torch for the solver test
import torch
