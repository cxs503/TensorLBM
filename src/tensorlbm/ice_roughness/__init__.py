"""Ice-underside roughness -> ice/water drag closure (``C_w``).

Mission P103W2B-N02: replace the constant water-drag coefficient used in the
sea-ice drift momentum balance with a roughness-dependent closure
``C_w(k_s^+, lambda/k_s)`` whose three regimes (hydraulically smooth /
transitional / fully rough) come from wall-resolved lattice-Boltzmann
simulations of the turbulent boundary layer over ice-keel roughness elements.

Modules
-------
``geometry``
    Ice-ridge keel statistics -> geometric roughness height ``k_s`` and ridge
    spacing ``lambda``; voxelised rib geometry fed to the LBM channel.
``regime_law``
    The three-zone roughness function / drag closure, the fit of its onset
    parameter ``k0`` from LBM records, and the self-consistent
    ``C_w(u_rel, k_s, lambda)`` entry point used by the drift model.
``sweep``
    Case table, single-case runner (native D3Q19 rough channel) and the
    aggregation that evaluates the acceptance criteria C1-C4.
``freedrift``
    Free-drift momentum balance (wind stress + water drag + Coriolis) that
    turns a ``C_w`` into a Nansen drift angle, i.e. the back-fill that makes
    the closure testable against observations.
"""

from . import freedrift, geometry, regime_law, sweep

__all__ = ["freedrift", "geometry", "regime_law", "sweep"]