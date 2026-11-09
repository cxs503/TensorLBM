"""Full-ship broken-ice resistance prediction chain (P66-W2-N04).

A first-principles (non-empirical, non-ML) prediction chain for the resistance of a
full ship form advancing through a broken-ice (brash / very-close-pack) field:

    DEM floe field  ->  porous-medium LBM flow  ->  two-way FSI  ->  resistance
    integration over the full hull form  ->  reference comparison / GCI audit

Modules
-------
floe_field : polydisperse floe field generation + concentration / D50 verification
lbm_porous : D3Q19 LBM water phase with Darcy-Forchheimer porous ice sink and
             moving-wall momentum-exchange hull forces
fsi        : two-way coupling (Eulerian porous sink <-> Lagrangian floe drag),
             momentum-exchange residual audit
dem_slice  : 3D floe DEM (buoyancy, water drag, ice-ice / ice-hull contact with
             Coulomb friction) in the ship frame, recycled inflow
energy     : resistance decomposition + power (energy) closure audit
riska      : reference ice-resistance models used for the acceptance comparison
chain      : campaign driver; produces the machine-checkable criteria report
"""

from __future__ import annotations

__all__ = [
    "floe_field",
    "lbm_porous",
    "fsi",
    "dem_slice",
    "energy",
    "riska",
    "chain",
]