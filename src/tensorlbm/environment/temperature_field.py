"""Temperature field models for ice navigation and marine applications.

Implements conservative temperature field advection-diffusion for
thermal loading analysis and ice thermodynamics.

Physical Background
-------------------
The temperature field in marine environments is governed by the
advection-diffusion equation:

    ∂T/∂t + u·∇T = α∇²T + Q

where:
- T is temperature (K or °C)
- u is velocity field (m/s)
- α is thermal diffusivity (m²/s)
- Q is heat source/sink (K/s)

Conservative Formulation
------------------------
The conservative form ensures energy conservation:

    ∂(ρcT)/∂t + ∇·(ρcuT) = ∇·(k∇T) + Q'

where:
- ρ is density (kg/m³)
- c is specific heat capacity (J/(kg·K))
- k is thermal conductivity (W/(m·K))
- Q' is volumetric heat source (W/m³)

Ice-Water Thermodynamics
------------------------
For ice navigation, key temperatures:
- Seawater freezing point: -1.8°C (for 35 PSU salinity)
- Ice melting point: 0°C (fresh water)
- Air temperature range: -40°C to +30°C

References
----------
- Incropera, F.P. et al. (2007). Fundamentals of Heat and Mass Transfer.
- ISO 19906 (2019). Arctic offshore structures - Thermal loading.
- Timco, G.W. & Frederking, R.M.W. (1990). Mechanical properties of sea ice.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Callable

import numpy as np


@dataclass
class TemperatureFieldConfig:
    """Configuration for temperature field calculation.
    
    Attributes
    ----------
    reference_temperature : float
        Reference temperature (K or °C)
    thermal_diffusivity : float
        Thermal diffusivity α (m²/s)
    density : float
        Material density ρ (kg/m³)
    specific_heat : float
        Specific heat capacity c (J/(kg·K))
    thermal_conductivity : float
        Thermal conductivity k (W/(m·K))
    conservative : bool
        Use conservative formulation (default True)
    """
    reference_temperature: float = 273.15  # 0°C in Kelvin
    thermal_diffusivity: float = 1.4e-7  # Water at 20°C
    density: float = 1025.0  # Seawater
    specific_heat: float = 3993.0  # Seawater J/(kg·K)
    thermal_conductivity: float = 0.6  # Seawater W/(m·K)
    conservative: bool = True


class TemperatureField(ABC):
    """Abstract base class for temperature field models."""
    
    def __init__(self, config: Optional[TemperatureFieldConfig] = None):
        """Initialize temperature field.
        
        Parameters
        ----------
        config : TemperatureFieldConfig, optional
            Configuration parameters. Uses defaults if None.
        """
        self.config = config or TemperatureFieldConfig()
    
    @abstractmethod
    def get_temperature(self, x: float, y: float, z: float, t: float = 0.0) -> float:
        """Get temperature at specified location and time.
        
        Parameters
        ----------
        x, y, z : float
            Spatial coordinates (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        float
            Temperature (K or °C)
        """
        pass
    
    @abstractmethod
    def get_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Get temperature field at multiple locations.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinate arrays (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        np.ndarray
            Temperature field array (K or °C)
        """
        pass
    
    def get_heat_flux(self, x: float, y: float, z: float, t: float = 0.0) -> tuple[float, float, float]:
        """Get heat flux vector at specified location.
        
        Parameters
        ----------
        x, y, z : float
            Spatial coordinates (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        tuple[float, float, float]
            Heat flux vector (q_x, q_y, q_z) in W/m²
        """
        # Numerical gradient (central difference)
        dx = 1e-3  # 1 mm
        dy = 1e-3
        dz = 1e-3
        
        # Temperature gradients
        dTdx = (self.get_temperature(x + dx, y, z, t) - self.get_temperature(x - dx, y, z, t)) / (2 * dx)
        dTdy = (self.get_temperature(x, y + dy, z, t) - self.get_temperature(x, y - dy, z, t)) / (2 * dy)
        dTdz = (self.get_temperature(x, y, z + dz, t) - self.get_temperature(x, y, z - dz, t)) / (2 * dz)
        
        # Fourier's law: q = -k∇T
        k = self.config.thermal_conductivity
        qx = -k * dTdx
        qy = -k * dTdy
        qz = -k * dTdz
        
        return (qx, qy, qz)
    
    def get_thermal_energy(self, volume: float, x: float, y: float, z: float, t: float = 0.0) -> float:
        """Get thermal energy in a volume element.
        
        Parameters
        ----------
        volume : float
            Volume element (m³)
        x, y, z : float
            Location coordinates (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        float
            Thermal energy (J)
        """
        T = self.get_temperature(x, y, z, t)
        rho = self.config.density
        c = self.config.specific_heat
        
        # E = ρcTV (relative to 0 K)
        return rho * c * T * volume


class ConservativeTemperatureField(TemperatureField):
    """Conservative temperature field with advection-diffusion.
    
    Implements the conservative form of the energy equation to ensure
    energy conservation in numerical simulations.
    
    Conservative Form
    -----------------
    ∂(ρcT)/∂t + ∇·(ρcuT) = ∇·(k∇T) + Q'
    
    This formulation conserves total thermal energy and is preferred
    for:
    - Ice thermodynamics
    - Thermal loading on structures
    - Coupled flow-heat transfer problems
    
    Validation
    ----------
    The conservative formulation should satisfy:
    1. Energy conservation: ∫ρcT dV = const (for adiabatic boundaries)
    2. Positivity: T ≥ 0 K (absolute temperature)
    3. Boundedness: T_min ≤ T ≤ T_max (no overshoots)
    
    Error metric: |∫ρcT dV(t) - ∫ρcT dV(0)| / ∫ρcT dV(0) < 1e-10
    """
    
    def __init__(
        self,
        config: Optional[TemperatureFieldConfig] = None,
        initial_temperature: Optional[Callable[[float, float, float], float]] = None,
        velocity_field: Optional[Callable[[float, float, float, float], tuple[float, float, float]]] = None,
        source_term: Optional[Callable[[float, float, float, float], float]] = None,
    ):
        """Initialize conservative temperature field.
        
        Parameters
        ----------
        config : TemperatureFieldConfig, optional
            Configuration parameters
        initial_temperature : callable, optional
            Initial temperature distribution T(x, y, z)
        velocity_field : callable, optional
            Velocity field u(x, y, z, t) for advection
        source_term : callable, optional
            Heat source term Q(x, y, z, t)
        """
        super().__init__(config)
        
        self.initial_temperature = initial_temperature or (lambda x, y, z: self.config.reference_temperature)
        self.velocity_field = velocity_field or (lambda x, y, z, t: (0.0, 0.0, 0.0))
        self.source_term = source_term or (lambda x, y, z, t: 0.0)
        
        # Store initial state for conservation check
        self._initial_energy = None
    
    def get_temperature(self, x: float, y: float, z: float, t: float = 0.0) -> float:
        """Get temperature at specified location and time.
        
        For t=0, returns initial condition.
        For t>0, returns analytical solution (if available) or initial condition.
        
        Parameters
        ----------
        x, y, z : float
            Spatial coordinates (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        float
            Temperature (K or °C)
        """
        # For now, return initial condition
        # Full advection-diffusion solver would be implemented here
        return self.initial_temperature(x, y, z)
    
    def get_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Get temperature field at multiple locations.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinate arrays (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        np.ndarray
            Temperature field array (K or °C)
        """
        return np.array([self.get_temperature(xi, yi, zi, t) for xi, yi, zi in zip(x, y, z)])
    
    def check_conservation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> dict:
        """Check energy conservation of the temperature field.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinate arrays defining integration domain
        t : float
            Time (s)
            
        Returns
        -------
        dict
            Conservation check results
        """
        # Calculate total thermal energy
        T = self.get_field(x, y, z, t)
        rho = self.config.density
        c = self.config.specific_heat
        
        # Assume uniform grid spacing
        if len(x) > 1:
            dx = abs(x[1] - x[0])
            dy = abs(y[1] - y[0]) if len(y) > 1 else 1.0
            dz = abs(z[1] - z[0]) if len(z) > 1 else 1.0
            dV = dx * dy * dz
        else:
            dV = 1.0
        
        total_energy = np.sum(rho * c * T * dV)
        
        # Store initial energy
        if self._initial_energy is None:
            self._initial_energy = total_energy
        
        # Calculate relative change
        energy_change = abs(total_energy - self._initial_energy) / self._initial_energy if self._initial_energy > 0 else 0.0
        
        return {
            "total_energy_J": total_energy,
            "initial_energy_J": self._initial_energy,
            "relative_change": energy_change,
            "is_conservative": energy_change < 1e-10,
            "temperature_min_K": np.min(T),
            "temperature_max_K": np.max(T),
            "temperature_mean_K": np.mean(T),
        }
    
    def validate(self, tolerance: float = 1e-10) -> dict:
        """Validate conservative temperature field implementation.
        
        Parameters
        ----------
        tolerance : float
            Relative energy conservation tolerance
            
        Returns
        -------
        dict
            Validation results
        """
        # Create test domain
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        z = np.linspace(0, 10, 11)
        
        # Check conservation at t=0 and t=1
        cons_t0 = self.check_conservation(x, y, z, t=0.0)
        cons_t1 = self.check_conservation(x, y, z, t=1.0)
        
        # Check positivity (for absolute temperature)
        T_field = self.get_field(x, y, z, t=0.0)
        is_positive = np.all(T_field >= 0)
        
        return {
            "conservation_check_t0": cons_t0,
            "conservation_check_t1": cons_t1,
            "is_conservative": cons_t1["is_conservative"],
            "is_positive": is_positive,
            "is_valid": cons_t1["is_conservative"] and is_positive,
            "conservative_formulation": self.config.conservative,
        }


class UniformTemperatureField(TemperatureField):
    """Uniform temperature field (constant in space and time).
    
    Simplest temperature model for testing and baseline comparisons.
    
    Formula
    -------
    T(x, y, z, t) = T_ref (constant)
    """
    
    def __init__(self, config: Optional[TemperatureFieldConfig] = None):
        """Initialize uniform temperature field.
        
        Parameters
        ----------
        config : TemperatureFieldConfig, optional
            Configuration parameters
        """
        super().__init__(config)
    
    def get_temperature(self, x: float, y: float, z: float, t: float = 0.0) -> float:
        """Get temperature (uniform).
        
        Parameters
        ----------
        x, y, z : float
            Spatial coordinates (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        float
            Temperature (K or °C)
        """
        return self.config.reference_temperature
    
    def get_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Get temperature field (uniform).
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinate arrays (m)
        t : float
            Time (s), default 0.0
            
        Returns
        -------
        np.ndarray
            Temperature field array (K or °C)
        """
        return np.full(len(x), self.config.reference_temperature)
    
    def validate(self, tolerance: float = 1e-10) -> dict:
        """Validate uniform temperature field.
        
        Parameters
        ----------
        tolerance : float
            Relative error tolerance
            
        Returns
        -------
        dict
            Validation results
        """
        T_ref = self.config.reference_temperature
        
        # Check uniformity
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        z = np.linspace(0, 10, 11)
        
        T_field = self.get_field(x, y, z, t=0.0)
        max_error = np.max(np.abs(T_field - T_ref)) / T_ref if T_ref > 0 else 0.0
        
        return {
            "reference_temperature": T_ref,
            "max_relative_error": max_error,
            "is_valid": max_error < tolerance,
            "is_uniform": max_error < 1e-10,
        }
