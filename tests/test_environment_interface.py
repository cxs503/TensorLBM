"""Tests for environment field driving interface.

Validates wind profile, current profile, and temperature field implementations
for ice navigation and marine applications.

Test Coverage
-------------
1. Wind Profile (Log Law):
   - Reference condition satisfaction
   - Monotonicity
   - Error < 5% tolerance

2. Current Profile (Uniform):
   - Uniformity at all depths
   - Zero shear
   - Error < 1% tolerance

3. Temperature Field (Conservative):
   - Energy conservation
   - Positivity
   - Conservative formulation

References
----------
- Garratt, J.R. (1994). The Atmospheric Boundary Layer.
- API RP 2PS (2013). Planning, Designing, and Constructing Fixed Offshore Platforms.
- Incropera, F.P. et al. (2007). Fundamentals of Heat and Mass Transfer.
"""

import numpy as np
import pytest

from tensorlbm.environment import (
    WindProfile,
    LogLawWindProfile,
    PowerLawWindProfile,
    CurrentProfile,
    UniformCurrentProfile,
    PowerLawCurrentProfile,
    TemperatureField,
    ConservativeTemperatureField,
)
from tensorlbm.environment.wind_profile import WindProfileConfig
from tensorlbm.environment.current_profile import CurrentProfileConfig
from tensorlbm.environment.temperature_field import TemperatureFieldConfig


class TestWindProfileLogLaw:
    """Test suite for logarithmic law wind profile."""
    
    def test_reference_condition(self):
        """Test that log law satisfies reference condition at z_ref."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
            roughness_length=0.0002,
        )
        profile = LogLawWindProfile(config)
        
        # Calculate speed at reference height
        U_calc = profile.get_speed(config.reference_height)
        
        # Check error
        error = abs(U_calc - config.reference_speed) / config.reference_speed
        
        # Validation criterion: error < 0.05 (5%)
        assert error < 0.05, f"Log law error {error:.4f} exceeds 5% tolerance"
    
    def test_monotonicity(self):
        """Test that wind speed increases with height."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
            roughness_length=0.0002,
        )
        profile = LogLawWindProfile(config)
        
        # Test at multiple heights
        heights = np.linspace(1.0, 100.0, 20)
        speeds = profile.get_profile(heights)
        
        # Check monotonicity
        assert np.all(np.diff(speeds) >= 0), "Wind speed should increase with height"
    
    def test_zero_at_roughness_length(self):
        """Test that wind speed is zero at roughness length."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
            roughness_length=0.0002,
        )
        profile = LogLawWindProfile(config)
        
        # Speed at roughness length should be zero
        U_z0 = profile.get_speed(config.roughness_length)
        assert U_z0 == 0.0, "Wind speed should be zero at roughness length"
    
    def test_velocity_vector(self):
        """Test velocity vector calculation."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
            roughness_length=0.0002,
            wind_direction=90.0,  # From East
        )
        profile = LogLawWindProfile(config)
        
        u, v, w = profile.get_velocity_vector(10.0)
        
        # Check vertical component is zero
        assert w == 0.0, "Vertical wind component should be zero"
        
        # Check magnitude
        speed = np.sqrt(u**2 + v**2)
        expected_speed = profile.get_speed(10.0)
        assert abs(speed - expected_speed) < 1e-6, "Velocity magnitude mismatch"
    
    def test_validation_method(self):
        """Test validation method returns correct metrics."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
            roughness_length=0.0002,
        )
        profile = LogLawWindProfile(config)
        
        validation = profile.validate(tolerance=0.05)
        
        assert "relative_error" in validation
        assert "is_valid" in validation
        assert validation["is_valid"] == True
        assert validation["relative_error"] < 0.05


class TestWindProfilePowerLaw:
    """Test suite for power law wind profile."""
    
    def test_reference_condition(self):
        """Test that power law satisfies reference condition."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
        )
        profile = PowerLawWindProfile(config, power_law_exponent=7.0)
        
        U_calc = profile.get_speed(config.reference_height)
        error = abs(U_calc - config.reference_speed) / config.reference_speed
        
        assert error < 0.05, f"Power law error {error:.4f} exceeds 5% tolerance"
    
    def test_monotonicity(self):
        """Test that wind speed increases with height."""
        config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=10.0,
        )
        profile = PowerLawWindProfile(config, power_law_exponent=7.0)
        
        heights = np.linspace(1.0, 100.0, 20)
        speeds = profile.get_profile(heights)
        
        assert np.all(np.diff(speeds) >= 0), "Wind speed should increase with height"


class TestCurrentProfileUniform:
    """Test suite for uniform current profile."""
    
    def test_uniformity(self):
        """Test that current speed is uniform at all depths."""
        config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=1.0,
        )
        profile = UniformCurrentProfile(config)
        
        # Test at multiple depths
        depths = np.linspace(0, 100, 11)
        speeds = profile.get_profile(depths)
        
        # All speeds should equal reference
        expected = np.full_like(depths, config.reference_speed)
        max_error = np.max(np.abs(speeds - expected)) / config.reference_speed
        
        # Validation criterion: error < 0.01 (1%)
        assert max_error < 0.01, f"Uniform profile error {max_error:.6f} exceeds 1% tolerance"
    
    def test_zero_shear(self):
        """Test that uniform profile has zero shear."""
        config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=1.0,
        )
        profile = UniformCurrentProfile(config)
        
        depths = np.linspace(0, 100, 11)
        speeds = profile.get_profile(depths)
        
        # Calculate shear
        shear = np.diff(speeds) / np.diff(depths)
        max_shear = np.max(np.abs(shear))
        
        assert max_shear < 1e-10, f"Uniform profile should have zero shear, got {max_shear}"
    
    def test_validation_method(self):
        """Test validation method returns correct metrics."""
        config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=1.0,
        )
        profile = UniformCurrentProfile(config)
        
        validation = profile.validate(tolerance=0.01)
        
        assert "max_relative_error" in validation
        assert "is_valid" in validation
        assert validation["is_valid"] == True
        assert validation["max_relative_error"] < 0.01
        assert validation["is_uniform"] == True
    
    def test_velocity_vector(self):
        """Test velocity vector calculation."""
        config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=1.0,
            current_direction=45.0,
        )
        profile = UniformCurrentProfile(config)
        
        u, v, w = profile.get_velocity_vector(10.0)
        
        # Check vertical component is zero
        assert w == 0.0, "Vertical current component should be zero"
        
        # Check magnitude
        speed = np.sqrt(u**2 + v**2)
        assert abs(speed - config.reference_speed) < 1e-6, "Velocity magnitude mismatch"


class TestCurrentProfilePowerLaw:
    """Test suite for power law current profile."""
    
    def test_reference_condition(self):
        """Test that power law satisfies reference condition."""
        config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=1.0,
        )
        profile = PowerLawCurrentProfile(config, power_law_exponent=7.0)
        
        U_calc = profile.get_speed(config.reference_depth)
        error = abs(U_calc - config.reference_speed) / config.reference_speed
        
        assert error < 0.05, f"Power law error {error:.4f} exceeds 5% tolerance"


class TestTemperatureFieldConservative:
    """Test suite for conservative temperature field."""
    
    def test_energy_conservation(self):
        """Test that conservative formulation conserves energy."""
        config = TemperatureFieldConfig(
            reference_temperature=273.15,
            conservative=True,
        )
        
        # Initial condition: uniform temperature
        def initial_T(x, y, z):
            return 273.15
        
        field = ConservativeTemperatureField(
            config=config,
            initial_temperature=initial_T,
        )
        
        # Create test domain
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        z = np.linspace(0, 10, 11)
        
        # Check conservation
        cons = field.check_conservation(x, y, z, t=0.0)
        
        # Validation criterion: conservative == True
        assert cons["is_conservative"] == True, "Temperature field should be conservative"
    
    def test_positivity(self):
        """Test that temperature is positive (absolute scale)."""
        config = TemperatureFieldConfig(
            reference_temperature=273.15,  # 0°C in Kelvin
            conservative=True,
        )
        
        def initial_T(x, y, z):
            return 273.15
        
        field = ConservativeTemperatureField(
            config=config,
            initial_temperature=initial_T,
        )
        
        # Create test domain
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        z = np.linspace(0, 10, 11)
        
        T_field = field.get_field(x, y, z, t=0.0)
        
        assert np.all(T_field >= 0), "Temperature should be non-negative (Kelvin)"
    
    def test_validation_method(self):
        """Test validation method returns correct metrics."""
        config = TemperatureFieldConfig(
            reference_temperature=273.15,
            conservative=True,
        )
        
        def initial_T(x, y, z):
            return 273.15
        
        field = ConservativeTemperatureField(
            config=config,
            initial_temperature=initial_T,
        )
        
        validation = field.validate()
        
        assert "is_conservative" in validation
        assert "is_positive" in validation
        assert "is_valid" in validation
        assert validation["is_conservative"] == True
        assert validation["is_positive"] == True
        assert validation["is_valid"] == True
        assert validation["conservative_formulation"] == True
    
    def test_heat_flux_calculation(self):
        """Test heat flux calculation."""
        config = TemperatureFieldConfig(
            reference_temperature=273.15,
            thermal_conductivity=0.6,
        )
        
        # Linear temperature gradient
        def initial_T(x, y, z):
            return 273.15 + 0.1 * x  # 0.1 K/m gradient in x
        
        field = ConservativeTemperatureField(
            config=config,
            initial_temperature=initial_T,
        )
        
        qx, qy, qz = field.get_heat_flux(5.0, 5.0, 5.0, t=0.0)
        
        # Expected: qx = -k * dT/dx = -0.6 * 0.1 = -0.06 W/m²
        expected_qx = -0.6 * 0.1
        assert abs(qx - expected_qx) < 1e-3, f"Heat flux mismatch: {qx} vs {expected_qx}"


class TestEnvironmentInterfaceIntegration:
    """Integration tests for complete environment interface."""
    
    def test_combined_environment_setup(self):
        """Test setting up complete environment (wind + current + temperature)."""
        # Wind profile
        wind_config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=15.0,
            roughness_length=0.001,  # Smooth ice
        )
        wind = LogLawWindProfile(wind_config)
        
        # Current profile
        current_config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=0.5,
        )
        current = UniformCurrentProfile(current_config)
        
        # Temperature field
        temp_config = TemperatureFieldConfig(
            reference_temperature=268.15,  # -5°C
            conservative=True,
        )
        
        def initial_T(x, y, z):
            return 268.15
        
        temperature = ConservativeTemperatureField(
            config=temp_config,
            initial_temperature=initial_T,
        )
        
        # Validate all components
        wind_val = wind.validate()
        current_val = current.validate()
        temp_val = temperature.validate()
        
        assert wind_val["is_valid"] == True
        assert current_val["is_valid"] == True
        assert temp_val["is_valid"] == True
    
    def test_ice_navigation_scenario(self):
        """Test environment setup for ice navigation scenario."""
        # Arctic conditions
        wind_config = WindProfileConfig(
            reference_height=10.0,
            reference_speed=20.0,  # Strong wind
            roughness_length=0.01,  # Rough ice
            air_density=1.3,  # Cold air
        )
        wind = LogLawWindProfile(wind_config)
        
        current_config = CurrentProfileConfig(
            reference_depth=10.0,
            reference_speed=0.3,  # Slow current
            water_density=1027.0,  # Cold seawater
        )
        current = UniformCurrentProfile(current_config)
        
        temp_config = TemperatureFieldConfig(
            reference_temperature=263.15,  # -10°C
            conservative=True,
        )
        
        def initial_T(x, y, z):
            return 263.15
        
        temperature = ConservativeTemperatureField(
            config=temp_config,
            initial_temperature=initial_T,
        )
        
        # Check environmental loads
        wind_pressure = wind.get_dynamic_pressure(10.0)
        current_pressure = current.get_dynamic_pressure(10.0)
        
        assert wind_pressure > 0, "Wind dynamic pressure should be positive"
        assert current_pressure > 0, "Current dynamic pressure should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
