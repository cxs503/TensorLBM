"""Pytest tests for aero-turbulence verification framework.

These tests verify the framework structure and basic functionality.
Full verification runs require longer computation times.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TestReferenceData:
    """Test reference data interpolation."""
    
    def test_abbott_cl_interpolation(self):
        from tensorlbm.verification.aero_turbulence import abbott_cl_interp
        
        # Test key points
        assert abs(abbott_cl_interp(0.0)) < 0.01
        assert abs(abbott_cl_interp(5.0) - 0.543) < 0.01
        assert abs(abbott_cl_interp(10.0) - 0.94) < 0.05
    
    def test_abbott_cd_interpolation(self):
        from tensorlbm.verification.aero_turbulence import abbott_cd_interp
        
        # Cd should be positive and increase with alpha
        cd_0 = abbott_cd_interp(0.0)
        cd_10 = abbott_cd_interp(10.0)
        assert cd_0 > 0
        assert cd_10 > cd_0
    
    def test_thin_airfoil_theory(self):
        from tensorlbm.verification.aero_turbulence import thin_airfoil_cl, thin_airfoil_cl_slope
        
        # Slope should be 2*pi per radian
        assert abs(thin_airfoil_cl_slope() - 2 * np.pi) < 0.01
        
        # Cl at 5 degrees
        cl_5 = thin_airfoil_cl(5.0)
        assert abs(cl_5 - 0.548) < 0.01
    
    def test_mkm1999_profiles(self):
        from tensorlbm.verification.aero_turbulence import (
            mkm1999_mean_velocity_profile,
            mkm1999_reynolds_stress_profile,
            mkm1999_tke_profile,
        )
        
        y_plus = np.array([1.0, 10.0, 30.0, 50.0, 100.0])
        
        # Mean velocity should increase with y+
        u_plus = mkm1999_mean_velocity_profile(y_plus)
        assert all(u_plus[i] < u_plus[i+1] for i in range(len(u_plus)-1))
        
        # Reynolds stress should be positive (magnitude)
        uv_plus = mkm1999_reynolds_stress_profile(y_plus)
        assert all(uv_plus >= 0)
        
        # TKE should be positive
        tke_plus = mkm1999_tke_profile(y_plus)
        assert all(tke_plus >= 0)


class TestAirfoilGeometry:
    """Test NACA0012 geometry generation."""
    
    def test_thickness_distribution(self):
        from tensorlbm.verification.aero_turbulence import naca0012_thickness
        
        xc = np.linspace(0, 1, 100)
        yt = naca0012_thickness(xc)
        
        # Max thickness should be ~0.06 (12% thickness)
        assert abs(yt.max() - 0.06) < 0.001
        
        # Thickness at LE and TE should be small
        assert yt[0] < 0.01
        assert yt[-1] < 0.01
    
    def test_mask_generation(self):
        from tensorlbm.verification.aero_turbulence import build_naca0012_mask
        
        mask = build_naca0012_mask(100, 50, 30.0, 0.0)
        
        assert mask.dtype == torch.bool
        assert mask.shape == (50, 100)
        assert mask.sum() > 0  # Should have solid cells


class TestVanDriestDamping:
    """Test van Driest damping function."""
    
    def test_damping_profile(self):
        from tensorlbm.verification.aero_turbulence import van_driest_damping
        
        y_plus = torch.tensor([0.0, 5.0, 10.0, 25.0, 50.0, 100.0])
        D = van_driest_damping(y_plus, A_plus=25.0)
        
        # D should be 0 at wall
        assert abs(D[0].item()) < 0.01
        
        # D should approach 1 away from wall
        assert D[-1].item() > 0.95
        
        # D should be monotonically increasing
        assert all(D[i] <= D[i+1] for i in range(len(D)-1))


class TestGCI:
    """Test Grid Convergence Index computation."""
    
    def test_convergent_case(self):
        from tensorlbm.verification.aero_turbulence import compute_gci
        
        # Monotonically converging values
        result = compute_gci([1.0, 1.05, 1.07], [100, 200, 400])
        
        assert result.p > 0
        assert result.gci_fine >= 0
        assert result.convergence_status in ["converged", "marginal", "not_converged"]
    
    def test_oscillatory_case(self):
        from tensorlbm.verification.aero_turbulence import compute_gci
        
        # Oscillatory convergence
        result = compute_gci([1.0, 1.1, 1.05], [100, 200, 400])
        
        assert result.convergence_status == "oscillatory"
    
    def test_converged_case(self):
        from tensorlbm.verification.aero_turbulence import compute_gci
        
        # All values the same
        result = compute_gci([1.0, 1.0, 1.0], [100, 200, 400])
        
        assert result.convergence_status == "converged"
        assert result.gci_fine == 0.0


class TestConfigurations:
    """Test configuration dataclasses."""
    
    def test_airfoil_config(self):
        from tensorlbm.verification.aero_turbulence import AirfoilSimConfig
        
        config = AirfoilSimConfig(
            chord=100.0,
            alpha_deg=5.0,
            u_in=0.05,
            re=500.0,
        )
        
        assert config.ma > 0
        assert config.nu > 0
        assert config.tau > 0.5
    
    def test_les_channel_config(self):
        from tensorlbm.verification.aero_turbulence import LESChannelConfig
        
        config = LESChannelConfig(
            nx=256,
            ny=65,
            re_tau=180.0,
            u_tau=0.005,
        )
        
        assert config.H > 0
        assert config.nu > 0
        assert config.tau > 0.5
        assert config.body_force > 0
        
        # Should validate without error
        config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
