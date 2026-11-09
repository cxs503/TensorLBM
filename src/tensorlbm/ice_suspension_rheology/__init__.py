"""Effective-viscosity closures for ice-crystal (frazil) suspensions.

Frontier module (campaign IN13-W4-N02).  It supplies three things:

1. :mod:`tensorlbm.ice_suspension_rheology.closures` -- closed forms for the
   relative viscosity ``eta_r = eta_eff / eta_f`` (Einstein dilute limit,
   Batchelor semi-dilute, Krieger-Dougherty concentrated) plus a *contract*
   object that makes ``eta_r(phi)`` a plug-in module for the Bingham and
   mu(I) solvers shipped elsewhere in the repo.
2. :mod:`tensorlbm.ice_suspension_rheology.shapes` -- discretised particle
   geometry (spheres, near-spherical faceted crystals, spheroids) and
   non-overlapping placement for monodisperse / polydisperse populations.
3. :mod:`tensorlbm.ice_suspension_rheology.couette` -- a D3Q19 lattice
   Boltzmann shear cell (moving-wall Couette) that measures
   ``eta_eff = tau_wall / shear_rate`` for a rigid population of particles
   moving with the imposed shear (translation + half-vorticity rotation) so
   that each particle is force- and torque-free, which is the condition under
   which Einstein's ``[eta] = 5/2`` holds.

:mod:`tensorlbm.ice_suspension_rheology.calibrate` drives the calibration
campaign (drag pre-gate, dilute sweep, concentrated sweep, Re_p sweep) and
writes the evidence package.

Nothing in this package is a mock: every number is either an analytic
evaluation of a published closed form or the output of the LBM shear cell.
"""

from . import closures, couette, shapes  # noqa: F401

__all__ = ["closures", "shapes", "couette"]