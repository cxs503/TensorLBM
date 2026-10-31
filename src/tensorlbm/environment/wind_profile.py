"""Wind profile models for atmospheric boundary layer.

Implements logarithmic law and power law wind profiles for marine and
ice navigation applications.

Physical Background
-------------------
The atmospheric boundary layer (ABL) wind profile is governed by the
Monin-Obukhov similarity theory. For neutral stability conditions, the
logarithmic law is:

    u(z) = (u*/κ) * ln(z/z0)

where:
- u* is the friction velocity (m/s)
- κ ≈ 0.41 is the von Kármán constant
- z0 is the aerodynamic roughness length (m)
- z is the height above ground (m)

For engineering applications, the power law is often used:

    u(z) = U_ref * (z/z_ref)^(1/n)

where n is the power law exponent (typically 7 for neutral ABL).

Roughness Lengths (z0)
----------------------
- Open water: 0.0002 m
- Smooth ice: 0.001 m
- Rough ice: 0.01 m
- Snow-covered: 0.005 m
- Urban: 1.0 m

References
----------
- Garratt, J.R. (1994). The Atmospheric Boundary Layer.
- WMO Guide to Meteorological Instruments and Methods of Observation.
- DNV-OS-C101 (2013). Design of Offshore Steel Structures.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# Von Kármán constant
VON_KARMAN_CONSTANT = 0.41


@dataclass
class WindProfileConfig:
    """Configuration for wind profile calculation.
    
    Attributes
    ----------
    reference_height : float
        Reference height for wind speed (m)
    reference_speed : float
        Wind speed at reference height (m/s)
    roughness_length : float
        Aerodynamic roughness length z0 (m)
    wind_direction : float
        Wind direction in degrees (0 = North, 90 = East)
    air_density : float
        Air density (kg/m³)
    stability : str
        Atmospheric stability class ('neutral', 'stable', 'unstable')
    """
    reference_height: float = 10.0
    reference_speed: float = 10.0
    roughness_length: float = 0.0002  # Open water default
    wind_direction: float = 0.0
    air_density: float = 1.225
    stability: Literal['neutral', 'stable', 'unstable'] = 'neutral'


class WindProfile(ABC):
    """Abstract base class for wind profile models."""
    
    def __init__(self, config: Optional[WindProfileConfig] = None):
        """Initialize wind profile.
        
        Parameters
        ----------
        config : WindProfileConfig, optional
            Configuration parameters. Uses defaults if None.
        """
        self.config = config or WindProfileConfig()
    
    @abstractmethod
    def get_speed(self, height: float) -> float:
        """Get wind speed at specified height.
        
        Parameters
        ----------
        height : float
            Height above ground/water surface (m)
            
        Returns
        -------
        float
            Wind speed at height (m/s)
        """
        pass
    
    @abstractmethod
    def get_profile(self, heights: np.ndarray) -> np.ndarray:
        """Get wind speed profile at multiple heights.
        
        Parameters
        ----------
        heights : np.ndarray
            Array of heights (m)
            
        Returns
        -------
        np.ndarray
            Array of wind speeds (m/s)
        """
        pass
    
    def get_velocity_vector(self, height: float) -> tuple[float, float, float]:
        """Get wind velocity vector at specified height.
        
        Parameters
        ----------
        height : float
            Height above ground/water surface (m)
            
        Returns
        -------
        tuple[float, float, float]
            (u, v, w) velocity components (m/s)
        """
        speed = self.get_speed(height)
        direction_rad = math.radians(self.config.wind_direction)
        
        # Meteorological convention: direction is where wind comes FROM
        # Convert to where wind goes TO
        u = -speed * math.sin(direction_rad)  # East component
        v = -speed * math.cos(direction_rad)  # North component
        w = 0.0  # Vertical component (neglected for ABL)
        
        return (u, v, w)
    
    def get_dynamic_pressure(self, height: float) -> float:
        """Get dynamic pressure at specified height.
        
        Parameters
        ----------
        height : float
            Height above ground/water surface (m)
            
        Returns
        -------
        float
            Dynamic pressure (Pa)
        """
        speed = self.get_speed(height)
        return 0.5 * self.config.air_density * speed ** 2


class LogLawWindProfile(WindProfile):
    """Logarithmic law wind profile for neutral atmospheric stability.
    
    The log law is derived from the constant stress layer assumption and
    is valid for the lower 10% of the atmospheric boundary layer.
    
    Formula
    -------
    u(z) = (u*/κ) * ln(z/z0)
    
    where u* is calculated from reference conditions:
    u* = κ * U_ref / ln(z_ref/z0)
    
    Validation
    ----------
    The log law should satisfy:
    1. u(z_ref) = U_ref (reference condition)
    2. u(z0) = 0 (no-slip at roughness length)
    3. du/dz > 0 (wind speed increases with height)
    
    Error metric: |u(z_ref) - U_ref| / U_ref < 0.05 (5% tolerance)
    """
    
    def __init__(self, config: Optional[WindProfileConfig] = None):
        """Initialize log law wind profile.
        
        Parameters
        ----------
        config : WindProfileConfig, optional
            Configuration parameters
        """
        super().__init__(config)
        
        # Calculate friction velocity from reference conditions
        z_ref = self.config.reference_height
        z0 = self.config.roughness_length
        U_ref = self.config.reference_speed
        
        if z_ref <= z0:
            raise ValueError(f"Reference height ({z_ref} m) must be > roughness length ({z0} m)")
        
        # u* = κ * U_ref / ln(z_ref/z0)
        self.friction_velocity = VON_KARMAN_CONSTANT * U_ref / math.log(z_ref / z0)
    
    def get_speed(self, height: float) -> float:
        """Get wind speed at specified height using log law.
        
        Parameters
        ----------
        height : float
            Height above ground/water surface (m)
            
        Returns
        -------
        float
            Wind speed at height (m/s)
            
        Notes
        -----
        For heights below roughness length, returns 0 (within canopy).
        """
        z0 = self.config.roughness_length
        
        if height <= z0:
            return 0.0
        
        # u(z) = (u*/κ) * ln(z/z0)
        return (self.friction_velocity / VON_KARMAN_CONSTANT) * math.log(height / z0)
    
    def get_profile(self, heights: np.ndarray) -> np.ndarray:
        """Get wind speed profile at multiple heights.
        
        Parameters
        ----------
        heights : np.ndarray
            Array of heights (m)
            
        Returns
        -------
        np.ndarray
            Array of wind speeds (m/s)
        """
        return np.array([self.get_speed(h) for h in heights])
    
    def validate(self, tolerance: float = 0.05) -> dict:
        """Validate log law implementation against reference conditions.
        
        Parameters
        ----------
        tolerance : float
            Relative error tolerance (default 0.05 = 5%)
            
        Returns
        -------
        dict
            Validation results with error metrics
        """
        z_ref = self.config.reference_height
        U_ref = self.config.reference_speed
        
        # Check reference condition
        U_calc = self.get_speed(z_ref)
        error = abs(U_calc - U_ref) / U_ref if U_ref > 0 else 0.0
        
        # Check monotonicity
        heights = np.linspace(z_ref * 0.1, z_ref * 2, 10)
        speeds = self.get_profile(heights)
        is_monotonic = np.all(np.diff(speeds) >= 0)
        
        return {
            "reference_height_m": z_ref,
            "reference_speed_ms": U_ref,
            "calculated_speed_ms": U_calc,
            "relative_error": error,
            "is_valid": error < tolerance,
            "is_monotonic": is_monotonic,
            "friction_velocity_ms": self.friction_velocity,
        }


class PowerLawWindProfile(WindProfile):
    """Power law wind profile for engineering applications.
    
    The power law is an empirical approximation valid for the entire
    boundary layer, though less physically rigorous than the log law.
    
    Formula
    -------
    u(z) = U_ref * (z/z_ref)^(1/n)
    
    where n is the power law exponent:
    - n = 7 for neutral ABL over open terrain
    - n = 4-5 for urban areas
    - n = 10-12 for smooth water/ice
    
    References
    ----------
    - Davenport, A.G. (1960). The relationship of wind structure to wind loading.
    - API RP 2PS (2013). Planning, Designing, and Constructing Fixed Offshore Platforms.
    """
    
    def __init__(
        self,
        config: Optional[WindProfileConfig] = None,
        power_law_exponent: float = 7.0,
    ):
        """Initialize power law wind profile.
        
        Parameters
        ----------
        config : WindProfileConfig, optional
            Configuration parameters
        power_law_exponent : float
            Power law exponent n (default 7.0 for neutral ABL)
        """
        super().__init__(config)
        self.power_law_exponent = power_law_exponent
    
    def get_speed(self, height: float) -> float:
        """Get wind speed at specified height using power law.
        
        Parameters
        ----------
        height : float
            Height above ground/water surface (m)
            
        Returns
        -------
        float
            Wind speed at height (m/s)
        """
        z_ref = self.config.reference_height
        U_ref = self.config.reference_speed
        
        if height <= 0:
            return 0.0
        
        # u(z) = U_ref * (z/z_ref)^(1/n)
        return U_ref * (height / z_ref) ** (1.0 / self.power_law_exponent)
    
    def get_profile(self, heights: np.ndarray) -> np.ndarray:
        """Get wind speed profile at multiple heights.
        
        Parameters
        ----------
        heights : np.ndarray
            Array of heights (m)
            
        Returns
        -------
        np.ndarray
            Array of wind speeds (m/s)
        """
        return np.array([self.get_speed(h) for h in heights])
    
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
        z_ref = self.config.reference_height
        U_ref = self.config.reference_speed
        
        # Check reference condition
        U_calc = self.get_speed(z_ref)
        error = abs(U_calc - U_ref) / U_ref if U_ref > 0 else 0.0
        
        # Check monotonicity
        heights = np.linspace(z_ref * 0.1, z_ref * 2, 10)
        speeds = self.get_profile(heights)
        is_monotonic = np.all(np.diff(speeds) >= 0)
        
        return {
            "reference_height_m": z_ref,
            "reference_speed_ms": U_ref,
            "calculated_speed_ms": U_calc,
            "relative_error": error,
            "is_valid": error < tolerance,
            "is_monotonic": is_monotonic,
            "power_law_exponent": self.power_law_exponent,
        }
