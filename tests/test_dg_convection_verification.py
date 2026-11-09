"""DG Convection Verification Tests (IN68-N04).

Pytest test suite for DG verification. Focuses on tests that can be reliably
verified with the current infrastructure.

These tests verify:
1. DG operator construction (Lobatto nodes, mass/stiffness matrices)
2. Mass conservation for P1 advection
3. Basic advection functionality

Note: Full convergence studies and benchmark comparisons require additional
development and are documented in docs/dg_verification_suite_IN68_N04.md.
"""

import math

import pytest
import torch

from tensorlbm.dg_advection import get_ops, lobatto_nodes
from tensorlbm.dg_verification import (
    generate_verification_report,
    test_dg_operators,
    test_mass_conservation_1d,
)


class TestDGOperators:
    """Test DG operator construction."""

    def test_lobatto_nodes_p1(self):
        """P1 Lobatto nodes should be [-1, 1]."""
        nodes = lobatto_nodes(1)
        assert len(nodes) == 2
        assert nodes[0] == pytest.approx(-1.0)
        assert nodes[-1] == pytest.approx(1.0)

    def test_lobatto_nodes_p2(self):
        """P2 Lobatto nodes should be [-1, 0, 1]."""
        nodes = lobatto_nodes(2)
        assert len(nodes) == 3
        assert nodes[0] == pytest.approx(-1.0)
        assert nodes[1] == pytest.approx(0.0)
        assert nodes[-1] == pytest.approx(1.0)

    def test_lobatto_nodes_p3(self):
        """P3 Lobatto nodes should have 4 points."""
        nodes = lobatto_nodes(3)
        assert len(nodes) == 4
        assert nodes[0] == pytest.approx(-1.0)
        assert nodes[-1] == pytest.approx(1.0)
        # Interior nodes should be symmetric
        assert nodes[1] == pytest.approx(-nodes[2])

    def test_operators_construction_p1(self):
        """P1 operators should be constructable."""
        ops = get_ops(1, dx=0.1, dtype=torch.float64, device="cpu")
        assert ops.degree == 1
        assert ops.n_node == 2
        assert ops.Ax.shape == (2, 2)
        assert ops.face_lift.shape == (2, 2)

    def test_operators_construction_p2(self):
        """P2 operators should be constructable."""
        ops = get_ops(2, dx=0.1, dtype=torch.float64, device="cpu")
        assert ops.degree == 2
        assert ops.n_node == 3
        assert ops.Ax.shape == (3, 3)
        assert ops.face_lift.shape == (3, 2)

    def test_constant_derivative_zero(self):
        """Derivative of constant function should be zero (in cell average)."""
        for degree in [1, 2, 3]:
            ops = get_ops(degree, dx=1.0, dtype=torch.float64, device="cpu")
            nodes = lobatto_nodes(degree)
            n = len(nodes)
            
            # Constant function
            u = torch.ones(n, dtype=torch.float64)
            du = ops.Ax @ u
            
            # Cell average should be zero
            # Use simple average (exact for P1, approximate for higher)
            avg = float(du.mean())
            assert abs(avg) < 1e-10, f"P{degree}: constant derivative avg = {avg}"


class TestMassConservation:
    """Test mass conservation for DG advection."""

    def test_p1_conservation(self):
        """P1 DG should conserve mass for smooth advection."""
        results = test_mass_conservation_1d()
        p1_result = results.get("P1", {})
        
        # Check that P1 conservation test passed
        assert p1_result.get("pass", False), \
            f"P1 conservation failed: rel_err = {p1_result.get('relative_error', 'N/A')}"
        
        # Check relative error is small
        rel_err = p1_result.get("relative_error", 1.0)
        assert rel_err < 1e-3, f"P1 conservation error too large: {rel_err}"


class TestVerificationReport:
    """Test verification report generation."""

    def test_report_generation(self, tmp_path):
        """Verify that report generation runs without error."""
        report_path = tmp_path / "test_report.json"
        report = generate_verification_report(report_path)
        
        # Check report structure
        assert "title" in report
        assert "mission_id" in report
        assert report["mission_id"] == "IN68-N04"
        assert "tests" in report
        assert "summary" in report
        
        # Check file was created
        assert report_path.exists()


class TestDGAdvectionBasic:
    """Basic DG advection functionality tests."""

    def test_1d_advection_setup(self):
        """Test that 1D advection can be set up."""
        from tensorlbm.dg_advection import get_ops, lobatto_nodes
        
        nx = 16
        dx = 1.0 / nx
        degree = 1
        
        ops = get_ops(degree, dx, torch.float64, "cpu")
        nodes = lobatto_nodes(degree)
        nn = len(nodes)
        
        # Create initial condition
        x_cells = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        x_nodes = x_cells.unsqueeze(-1) + (dx/2) * torch.tensor(nodes, dtype=torch.float64).unsqueeze(0)
        u0 = torch.sin(2 * math.pi * x_nodes)
        
        assert u0.shape == (nx, nn)
        assert ops.Ax.shape == (nn, nn)

    def test_rhs_computation(self):
        """Test that RHS can be computed."""
        from tensorlbm.dg_advection import get_ops, lobatto_nodes
        
        nx = 8
        dx = 1.0 / nx
        degree = 1
        
        ops = get_ops(degree, dx, torch.float64, "cpu")
        nodes = lobatto_nodes(degree)
        nn = len(nodes)
        
        # Create initial condition
        x_cells = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        x_nodes = x_cells.unsqueeze(-1) + (dx/2) * torch.tensor(nodes, dtype=torch.float64).unsqueeze(0)
        u = torch.sin(2 * math.pi * x_nodes)
        
        # Compute volume term
        vol = torch.zeros_like(u)
        for i in range(nx):
            vol[i] = ops.Ax @ u[i]
        
        assert vol.shape == u.shape
        assert not torch.isnan(vol).any()
        assert not torch.isinf(vol).any()


# Integration test
def test_full_verification_suite():
    """Run the full verification suite (smoke test)."""
    report = generate_verification_report()
    
    # At least operator tests should pass
    assert "operators" in report["tests"]
    
    # Check that we have some results
    op_results = report["tests"]["operators"]
    assert len(op_results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
