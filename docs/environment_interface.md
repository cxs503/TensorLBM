# Environment Field Driving Interface (Wind/Current/Temperature)

**Mission ID**: m673  
**Domain**: Ice Navigation / Marine Environment  
**Module**: `tensorlbm.environment`  
**Status**: ✅ All validation criteria PASS  

---

## 1. Overview

This module provides a unified environment field input interface for ice navigation and marine simulations in TensorLBM. It implements three core environmental field models:

1. **Wind Profile** — Atmospheric boundary layer wind speed/direction profiles
2. **Current Profile** — Ocean current speed/direction profiles  
3. **Temperature Field** — Conservative thermal field for ice thermodynamics

## 2. Physical Background

### 2.1 Wind Profile — Logarithmic Law

The atmospheric boundary layer (ABL) wind profile under neutral stability follows:

$$u(z) = \frac{u_*}{\kappa} \ln\left(\frac{z}{z_0}\right)$$

where:
- $u_*$ = friction velocity (m/s)
- $\kappa \approx 0.41$ = von Kármán constant
- $z_0$ = aerodynamic roughness length (m)
- $z$ = height above surface (m)

**Roughness lengths for marine/ice environments:**

| Surface Type | $z_0$ (m) |
|---|---|
| Open water | 0.0002 |
| Smooth ice | 0.001 |
| Rough ice | 0.01 |
| Snow-covered | 0.005 |

### 2.2 Current Profile — Uniform

For ice navigation, ocean currents are often modeled as uniform (barotropic):

$$u(z) = U_{\text{ref}} \quad \forall z$$

This is appropriate for well-mixed water columns and simplified engineering analysis.

### 2.3 Temperature Field — Conservative Advection-Diffusion

The conservative form of the energy equation:

$$\frac{\partial (\rho c T)}{\partial t} + \nabla \cdot (\rho c \mathbf{u} T) = \nabla \cdot (k \nabla T) + Q'$$

ensures energy conservation in numerical simulations.

## 3. Module Structure

```
src/tensorlbm/environment/
├── __init__.py              # Public API exports
├── wind_profile.py          # LogLawWindProfile, PowerLawWindProfile
├── current_profile.py       # UniformCurrentProfile, PowerLawCurrentProfile
└── temperature_field.py     # ConservativeTemperatureField, UniformTemperatureField
```

## 4. Usage Examples

### 4.1 Wind Profile (Log Law)

```python
from tensorlbm.environment import LogLawWindProfile
from tensorlbm.environment.wind_profile import WindProfileConfig

config = WindProfileConfig(
    reference_height=10.0,    # m
    reference_speed=15.0,     # m/s
    roughness_length=0.001,   # smooth ice
    wind_direction=270.0,     # from West
)
wind = LogLawWindProfile(config)

# Get speed at height
speed = wind.get_speed(50.0)  # m/s at 50m

# Get velocity vector
u, v, w = wind.get_velocity_vector(50.0)

# Get dynamic pressure
q = wind.get_dynamic_pressure(50.0)  # Pa
```

### 4.2 Current Profile (Uniform)

```python
from tensorlbm.environment import UniformCurrentProfile
from tensorlbm.environment.current_profile import CurrentProfileConfig

config = CurrentProfileConfig(
    reference_depth=10.0,
    reference_speed=0.5,      # m/s
    current_direction=45.0,   # toward NE
)
current = UniformCurrentProfile(config)

speed = current.get_speed(50.0)  # m/s at 50m depth
```

### 4.3 Temperature Field (Conservative)

```python
from tensorlbm.environment import ConservativeTemperatureField
from tensorlbm.environment.temperature_field import TemperatureFieldConfig

config = TemperatureFieldConfig(
    reference_temperature=268.15,  # -5°C in Kelvin
    conservative=True,
)

def initial_T(x, y, z):
    return 268.15  # uniform -5°C

temp = ConservativeTemperatureField(
    config=config,
    initial_temperature=initial_T,
)

T = temp.get_temperature(10.0, 20.0, 5.0)  # K
qx, qy, qz = temp.get_heat_flux(10.0, 20.0, 5.0)  # W/m²
```

## 5. Validation Results

| Criterion | Value | Threshold | Status |
|---|---|---|---|
| `wind_profile_log_law_error` | 0.000000 | < 0.05 | ✅ PASS |
| `current_profile_uniform` | 0.000000 | < 0.01 | ✅ PASS |
| `temperature_field_conservative` | True | == True | ✅ PASS |
| `pytest_test_environment_interface` | 18/18 passed | == PASS | ✅ PASS |

## 6. Test Suite

Run the test suite:

```bash
python -m pytest tests/test_environment_interface.py -v
```

18 tests covering:
- Wind profile log law (reference condition, monotonicity, zero at z0)
- Wind profile power law
- Current profile uniform (uniformity, zero shear)
- Current profile power law
- Temperature field (conservation, positivity, heat flux)
- Integration tests (combined environment, ice navigation scenario)

## 7. References

- Garratt, J.R. (1994). *The Atmospheric Boundary Layer*. Cambridge University Press.
- API RP 2PS (2013). *Planning, Designing, and Constructing Fixed Offshore Platforms*.
- DNV-OS-C101 (2013). *Design of Offshore Steel Structures*.
- ISO 19906 (2019). *Petroleum and natural gas industries — Arctic offshore structures*.
- Incropera, F.P. et al. (2007). *Fundamentals of Heat and Mass Transfer*.
