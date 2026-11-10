"""Micro-bubble drag reduction in a Shan-Chen channel (campaign node IN67-N01).

Modules
-------
``eos``           lattice-EOS mechanical pressure + Maxwell coexistence, so the
                  void-fraction field is scaled from the model's own densities;
``channel``       the two-phase plane channel: seeded bubbles, phase-dependent
                  relaxation time, matched-mass-flux drive, momentum-exchange
                  wall shear and Reynolds statistics;
``decomposition`` the closed wall-shear budget ``tau_w = A - B`` and the
                  slip / viscosity / turbulence mechanism split;
``campaign``      the ``d+`` scan, the void-fraction negative control and the
                  artifacts written by this node.

References
----------
Fukagata, Iwamoto & Kasagi (2002) Phys. Fluids 14, L73 -- wall-shear identity.
Merkle & Deutsch (1992) AIAA-92-4573 -- surface-gas / slip drag reduction.
Madavan, Deutsch & Merkle (1984) Phys. Fluids 27, 356 -- micro-bubble drag
    reduction mechanism review and the three-way mechanism account used here.
Shan & Chen (1993) Phys. Rev. E 47, 1815 -- pseudopotential two-phase model.
"""

from .channel import MicrobubbleChannel2D, MicrobubbleConfig, single_phase_reference
from .decomposition import (
    HalfChannelProfiles,
    einstein_bubbly_viscosity,
    laminar_wall_shear,
    mechanism_split,
    slip_length_drag_reduction,
    slip_velocity,
    wall_shear_budget,
)
from .eos import (
    kinematic_viscosity,
    maxwell_coexistence,
    mechanical_pressure,
    spinodal_window,
    void_fraction,
)

__all__ = [
    "MicrobubbleChannel2D",
    "MicrobubbleConfig",
    "single_phase_reference",
    "HalfChannelProfiles",
    "mechanism_split",
    "wall_shear_budget",
    "slip_velocity",
    "slip_length_drag_reduction",
    "laminar_wall_shear",
    "einstein_bubbly_viscosity",
    "maxwell_coexistence",
    "spinodal_window",
    "mechanical_pressure",
    "void_fraction",
    "kinematic_viscosity",
]