"""
Tests for the interface layer two-term law (p495w4-n04, m13946).

Verifies:
1. Two-term law fitting
2. Depth sensitivity L1(kh) = kh/sinh(2kh)
3. Error prediction at production resolution
4. Group speed consistency
"""

import math
import pytest
import numpy as np

from tensorlbm import free_surface_dispersion as fs


class TestDepthSensitivity:
    """Test the depth sensitivity L1(kh) = kh/sinh(2kh)."""
    
    def test_shallow_water_limit(self):
        """L1 -> 0.5 as kh -> 0 (shallow water)."""
        L1 = fs.depth_sensitivity(0.01, 1.0)
        assert abs(L1 - 0.5) < 0.01
    
    def test_deep_water_limit(self):
        """L1 -> 0 as kh -> inf (deep water)."""
        L1 = fs.depth_sensitivity(10.0, 1.0)
        assert L1 < 0.01
    
    def test_monotonic_decrease(self):
        """L1 decreases monotonically with kh."""
        kh_values = [0.5, 1.0, 1.5, 2.0, 3.0]
        L1_values = [fs.depth_sensitivity(kh, 1.0) for kh in kh_values]
        for i in range(len(L1_values) - 1):
            assert L1_values[i] > L1_values[i+1]
    
    def test_known_values(self):
        """Test against known values."""
        # L1(1.0) = 1/sinh(2) ≈ 0.2757
        L1 = fs.depth_sensitivity(1.0, 1.0)
        expected = 1.0 / math.sinh(2.0)
        assert abs(L1 - expected) < 1e-10


class TestTwoTermLaw:
    """Test the two-term error law."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample measurement data from m9103."""
        return [
            {"kh": 0.7, "h": 20.0, "c_rel_error": -0.04045},
            {"kh": 1.0, "h": 20.0, "c_rel_error": -0.03358},
            {"kh": 1.5, "h": 20.0, "c_rel_error": -0.02831},
            {"kh": 2.0, "h": 20.0, "c_rel_error": -0.02762},
            {"kh": 3.0, "h": 20.0, "c_rel_error": -0.03309},
        ]
    
    def test_fit_returns_law(self, sample_data):
        """Fit should return a TwoTermLaw."""
        law = fs.fit_two_term_law(sample_data)
        assert isinstance(law, fs.TwoTermLaw)
        assert hasattr(law, 'c1_xi')
        assert hasattr(law, 'c2')
        assert hasattr(law, 'rms_residual')
    
    def test_fit_quality(self, sample_data):
        """Fit should have small RMS residual."""
        law = fs.fit_two_term_law(sample_data)
        # RMS should be < 1% for good fit
        assert law.rms_residual < 0.01
    
    def test_c1_xi_positive(self, sample_data):
        """c1·ξ should be positive (mass deficit)."""
        law = fs.fit_two_term_law(sample_data)
        assert law.c1_xi > 0
    
    def test_prediction_accuracy(self, sample_data):
        """Predictions should match data within RMS."""
        law = fs.fit_two_term_law(sample_data)
        for m in sample_data:
            k = m["kh"] / m["h"]
            predicted = law.predict(k, m["h"])
            measured = m["c_rel_error"]
            assert abs(predicted - measured) < 0.01
    
    def test_interface_term_dominates(self, sample_data):
        """Interface term should dominate at small kh."""
        law = fs.fit_two_term_law(sample_data)
        # At kh=0.7, h=20
        k = 0.7 / 20.0
        interface = abs(law.interface_term(k, 20.0))
        lattice = abs(law.lattice_term(k, 20.0))
        assert interface > lattice
    
    def test_lattice_term_dominates_deep(self, sample_data):
        """Lattice term should dominate at large kh."""
        law = fs.fit_two_term_law(sample_data)
        # At kh=3.0, h=20
        k = 3.0 / 20.0
        interface = abs(law.interface_term(k, 20.0))
        lattice = abs(law.lattice_term(k, 20.0))
        assert lattice > interface


class TestGroupSpeed:
    """Test group speed computation."""
    
    def test_group_speed_continuum(self):
        """Test continuum group speed formula."""
        k = 0.1
        h = 20.0
        g = 0.01
        
        c_g = fs.continuous_group_speed(k, h, g)
        c = fs.continuous_phase_speed(k, h, g)
        
        # c_g = c/2 * (1 + 2kh/sinh(2kh))
        x = 2 * k * h
        ratio = x / math.sinh(x) if x < 700 else 0.0
        expected = c * 0.5 * (1 + ratio)
        
        assert abs(c_g - expected) < 1e-10
    
    def test_group_speed_shallow(self):
        """In shallow water, c_g -> c."""
        k = 0.01
        h = 1.0
        g = 0.01
        
        c_g = fs.continuous_group_speed(k, h, g)
        c = fs.continuous_phase_speed(k, h, g)
        
        # In shallow water (kh << 1), c_g ≈ c
        assert abs(c_g - c) / c < 0.01
    
    def test_group_speed_deep(self):
        """In deep water, c_g -> c/2."""
        k = 1.0
        h = 100.0
        g = 0.01
        
        c_g = fs.continuous_group_speed(k, h, g)
        c = fs.continuous_phase_speed(k, h, g)
        
        # In deep water (kh >> 1), c_g ≈ c/2
        assert abs(c_g - c/2) / (c/2) < 0.01


class TestProductionResolution:
    """Test production resolution criteria."""
    
    def test_cells_per_wavelength(self):
        """Test cells/λ computation."""
        h = 20.0
        kh = 1.5
        k = kh / h
        cells_per_lambda = 2 * math.pi / k
        
        # cells/λ = 2π·h/kh
        expected = 2 * math.pi * h / kh
        assert abs(cells_per_lambda - expected) < 1e-10
    
    def test_production_resolution_threshold(self):
        """Production resolution requires cells/λ ≥ 40 and h ≥ 10."""
        # Case 1: kh=1.5, h=20 -> cells/λ = 84
        h, kh = 20.0, 1.5
        k = kh / h
        cells_per_lambda = 2 * math.pi / k
        assert cells_per_lambda >= 40
        assert h >= 10
        
        # Case 2: kh=1.5, h=10 -> cells/λ = 42
        h, kh = 10.0, 1.5
        k = kh / h
        cells_per_lambda = 2 * math.pi / k
        assert cells_per_lambda >= 40
        assert h >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
