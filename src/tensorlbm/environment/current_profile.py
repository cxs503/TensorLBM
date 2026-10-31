"""Current profile models for ocean and ice navigation.

Implements uniform and power law current profiles for marine applications.

Physical Background
-------------------
Ocean currents in ice-infested waters can be modeled as:

1. Uniform Profile:
   u(z) = U_ref (constant with depth)
   
   This is appropriate for:
   - Barotropic flows
   - Well-mixed water columns
   - Simplified engineering analysis

2. Power Law Profile:
   u(z) = U_ref * (z/z_ref)^(1/n)
   
   This represents:
   - Boundary layer effects near seabed
   - Wind-driven surface currents
   - Sheared flow profiles

3. Logarithmic Profile (tidal currents):
   u(z) = (u*/κ) * ln(z/z0)
   
   Similar to atmospheric log law but for water.

Current Characteristics
-----------------------
- Typical ocean currents: 0.1 - 2.0 m/s
- Tidal currents: 0.5 - 3.0 m/s
- Ice drift currents: 0.01 - 0.5 m/s

References
----------
- API RP 2PS (2013). Planning, Designing, and Constructing Fixed Offshore Platforms.
- DNV-OS-C101 (2013). Design of Offshore Steel Structures.
- ISO 19906 (2019). Petroleum and natural gas industries — Arctic offshore structures.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CurrentProfileConfig:
    """Configuration for current profile calculation.
    
    Attributes
    ----------
    reference_depth : float
        Reference depth for current speed (m, positive downward)
    reference_speed : float
        Current speed at reference depth (m/s)
    current_direction : float
        Current direction in degrees (0 = North, 90 = East)
    water_density : float
        Water density (kg/m³)
    profile_type : str
        Type of current profile ('uniform', 'power_law', 'logarithmic')
    """
    reference_depth: float = 10.0
    reference_speed: float = 1.0
    current_direction: float = 0.0
    water_density: float = 1025.0
    profile_type: Literal['uniform', 'power_law', 'logarithmic'] = 'uniform'


class CurrentProfile(ABC):
    """Abstract base class for current profile models."""
    
    def __init__(self, config: Optional[CurrentProfileConfig] = None):
        """Initialize current profile.
        
        Parameters
        ----------
        config : CurrentProfileConfig, optional
            Configuration parameters. Uses defaults if None.
        """
        self.config = config or CurrentProfileConfig()
    
    @abstractmethod
    def get_speed(self, depth: float) -> float:
        """Get current speed at specified depth.
        
        Parameters
        ----------
        depth : float
            Depth below water surface (m, positive downward)
            
        Returns
        -------
        float
            Current speed at depth (m/s)
        """
        pass
    
    @abstractmethod
    def get_profile(self, depths: np.ndarray) -> np.ndarray:
        """Get current speed profile at multiple depths.
        
        Parameters
        ----------
        depths : np.ndarray
            Array of depths (m, positive downward)
            
        Returns
        -------
        np.ndarray
            Array of current speeds (m/s)
        """
        pass
    
    def get_velocity_vector(self, depth: float) -> tuple[float, float, float]:
        """Get current velocity vector at specified depth.
        
        Parameters
        ----------
        depth : float
            Depth below water surface (m, positive downward)
            
        Returns
        -------
        tuple[float, float, float]
            (u, v, w) velocity components (m/s)
        """
        speed = self.get_speed(depth)
        direction_rad = math.radians(self.config.current_direction)
        
        # Oceanographic convention: direction is where current goes TO
        u = speed * math.sin(direction_rad)  # East component
        v = speed * math.cos(direction_rad)  # North component
        w = 0.0  # Vertical component (neglected for horizontal currents)
        
        return (u, v, w)
    
    def get_dynamic_pressure(self, depth: float) -> float:
        """Get dynamic pressure at specified depth.
        
        Parameters
        ----------
        depth : float
            Depth below water surface (m, positive downward)
            
        Returns
        -------
        float
            Dynamic pressure (Pa)
        """
        speed = self.get_speed(depth)
        return 0.5 * self.config.water_density * speed ** 2


class UniformCurrentProfile(CurrentProfile):
    """Uniform current profile (constant with depth).
    
    This is the simplest current model, appropriate for:
    - Barotropic flows
    - Well-mixed water columns
    - Simplified engineering analysis
    - Ice navigation where vertical shear is negligible
    
    Formula
    -------
    u(z) = U_ref (constant)
    
    Validation
    ----------
    The uniform profile should satisfy:
    1. u(z) = U_ref for all z
    2. du/dz = 0 (no shear)
    3. Error metric: max|u(z) - U_ref| / U_ref < 0.01 (1% tolerance)
    """
    
    def __init__(self, config: Optional[CurrentProfileConfig] = None):
        """Initialize uniform current profile.
        
        Parameters
        ----------
        config : CurrentProfileConfig, optional
            Configuration parameters
        """
        super().__init__(config)
    
    def get_speed(self, depth: float) -> float:
        """Get current speed at specified depth (uniform).
        
        Parameters
        ----------
        depth : float
            Depth below water surface (m, positive downward)
            
        Returns
        -------
        float
            Current speed at depth (m/s)
        """
        # Uniform profile: constant speed at all depths
        return self.config.reference_speed
    
    def get_profile(self, depths: np.ndarray) -> np.ndarray:
        """Get current speed profile at multiple depths.
        
        Parameters
        ----------
        depths : np.ndarray
            Array of depths (m, positive downward)
            
        Returns
        -------
        np.ndarray
            Array of current speeds (m/s)
        """
        # Uniform profile: all depths have same speed
        return np.full_like(depths, self.config.reference_speed, dtype=float)
    
    def validate(self, tolerance: float = 0.01) -> dict:
        """Validate uniform profile implementation.
        
        Parameters
        ----------
        tolerance : float
            Relative error tolerance (default 0.01 = 1%)
            
        Returns
        -------
        dict
            Validation results with error metrics
        """
        U_ref = self.config.reference_speed
        
        # Check uniformity at multiple depths
        depths = np.linspace(0, 100, 11)
        speeds = self.get_profile(depths)
        
        # All speeds should equal reference
        max_error = np.max(np.abs(speeds - U_ref)) / U_ref if U_ref > 0 else 0.0
        
        # Check zero shear
        shear = np.diff(speeds) / np.diff(depths)
        max_shear = np.max(np.abs(shear))
        
        return {
            "reference_speed_ms": U_ref,
            "max_relative_error": max_error,
            "max_shear_1s": max_shear,
            "is_valid": max_error < tolerance,
            "is_uniform": max_shear < 1e-10,
            "profile_type": "uniform",
        }


class PowerLawCurrentProfile(CurrentProfile):
    """Power law current profile for sheared flows.
    
    Represents boundary layer effects near seabed or wind-driven
    surface currents.
    
    Formula
    -------
    u(z) = U_ref * (z/z_ref)^(1/n)
    
    where n is the power law exponent:
    - n = 7 for typical ocean boundary layer
    - n = 5 for strong wind-driven currents
    - n = 10 for weak shear conditions
    
    References
    ----------
    - Dyer, A.J. (1970). A review of drag profiles in the ocean.
    - API RP 2PS (2013). Current force coefficients.
    """
    
    def __init__(
        self,
        config: Optional[CurrentProfileConfig] = None,
        power_law_exponent: float = 7.0,
    ):
        """Initialize power law current profile.
        
        Parameters
        ----------
        config : CurrentProfileConfig, optional
            Configuration parameters
        power_law_exponent : float
            Power law exponent n (default 7.0)
        """
        super().__init__(config)
        self.power_law_exponent = power_law_exponent
    
    def get_speed(self, depth: float) -> float:
        """Get current speed at specified depth using power law.
        
        Parameters
        ----------
        depth : float
            Depth below water surface (m, positive downward)
            
        Returns
        -------
        float
            Current speed at depth (m/s)
        """
        z_ref = self.config.reference_depth
        U_ref = self.config.reference_speed
        
        if depth <= 0:
            return 0.0
        
        # u(z) = U_ref * (z/z_ref)^(1/n)
        return U_ref * (depth / z_ref) ** (1.0 / self.power_law_exponent)
    
    def get_profile(self, depths: np.ndarray) -> np.ndarray:
        """Get current speed profile at multiple depths.
        
        Parameters
        ----------
        depths : np.ndarray
            Array of depths (m, positive downward)
            
        Returns
        -------
        np.ndarray
            Array of current speeds (m/s)
        """
        return np.array([self.get_speed(d) for d in depths])
    
    def validate(self, tolerance: float = 0.05) -> dict:
        """Validate power law implementation against reference conditions.
        
        Parameters
        ----------
        tolerance : float
            Relative error tolerance (default 0.05 = 5%)
            
        Returns
        -------
        dict
            Validation results with error metrics
        """
        z_ref = self.config.reference_depth
        U_ref = self.config.reference_speed
        
        # Check reference condition
        U_calc = self.get_speed(z_ref)
        error = abs(U_calc - U_ref) / U_ref if U_ref > 0 else 0.0
        
        # Check monotonicity (speed increases with depth for surface-driven)
        depths = np.linspace(z_ref * 0.1, z_ref * 2, 10)
        speeds = self.get_profile(depths)
        is_monotonic = np.all(np.diff(speeds) >= 0)
        
        return {
            "reference_depth_m": z_ref,
            "reference_speed_ms": U_ref,
            "calculated_speed_ms": U_calc,
            "relative_error": error,
            "is_valid": error < tolerance,
            "is_monotonic": is_monotonic,
            "power_law_exponent": self.power_law_exponent,
        }
