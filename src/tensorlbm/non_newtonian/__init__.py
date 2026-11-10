"""Non-Newtonian LBM solver package.

Provides Carreau-Yasuda / power-law / Cross rheology with Picard-iterated
local relaxation time, plus three benchmark solvers:
  - Lid-driven cavity (2D, D2Q9)
  - Pipe / channel flow (2D, D2Q9)
  - 4:1 planar contraction (2D, D2Q9)

This package consolidates the original non-Newtonian rheology models
(viscosity closures, strain-rate calculation, analytical Poiseuille solutions,
and BGK collision operators) with the new Picard-iterated solver framework
and benchmark validation drivers.
"""

# Re-export everything from the rheology module (backward compatibility)
from .rheology import (
    strain_rate_magnitude_2d,
    apparent_viscosity_power_law,
    apparent_viscosity_carreau_yasuda,
    apparent_viscosity_cross,
    apparent_viscosity_bingham,
    apparent_viscosity_herschel_bulkley,
    poiseuille_power_law_profile,
    poiseuille_newtonian_profile,
    poiseuille_max_velocity_power_law,
    poiseuille_mean_velocity_power_law,
    collide_power_law_bgk,
    collide_carreau_yasuda_bgk,
    collide_cross_bgk,
)

# New solver modules
from .picard import PicardNonNewtonianSolver, PicardConfig
from .cavity import CavityNonNewtonian, CavityConfig, run_cavity_benchmark
from .pipe import PipeNonNewtonian, PipeConfig, run_pipe_benchmark
from .contraction import Contraction4to1, ContractionConfig, run_contraction_benchmark

__all__ = [
    # Rheology (from rheology.py)
    "strain_rate_magnitude_2d",
    "apparent_viscosity_power_law",
    "apparent_viscosity_carreau_yasuda",
    "apparent_viscosity_cross",
    "apparent_viscosity_bingham",
    "apparent_viscosity_herschel_bulkley",
    "poiseuille_power_law_profile",
    "poiseuille_newtonian_profile",
    "poiseuille_max_velocity_power_law",
    "poiseuille_mean_velocity_power_law",
    "collide_power_law_bgk",
    "collide_carreau_yasuda_bgk",
    "collide_cross_bgk",
    # Picard solver
    "PicardNonNewtonianSolver",
    "PicardConfig",
    # Cavity
    "CavityNonNewtonian",
    "CavityConfig",
    "run_cavity_benchmark",
    # Pipe
    "PipeNonNewtonian",
    "PipeConfig",
    "run_pipe_benchmark",
    # Contraction
    "Contraction4to1",
    "ContractionConfig",
    "run_contraction_benchmark",
]
