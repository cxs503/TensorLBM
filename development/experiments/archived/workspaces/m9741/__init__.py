"""Invariance verification face: grid orientation and Galilean frame changes.

The physical claim this package exists to certify is:

    the same physical problem, discretised on a rotated lattice or observed from
    a Galilean-boosted frame, must give the same answer within a declared
    tolerance, and the residual must follow a *predictable* law in the Mach
    number (quadratic to leading order).

Contents
--------
``lattice``   D2Q9 BGK reference kernel (Guo forcing, BFL interpolated
              bounce-back, Zou-He inlet/outlet, specular free-slip, Ladd
              momentum exchange).
``geometry``  analytical SDF geometry (tilted periodic channel array, circle),
              lattice<->physical frame, sub-grid wall distance by bisection.
``galilean``  the boost operator, the boost of the boundary data, and the
              ``defect = c * Ma^p`` / ``Ma^2`` fitting machinery.
``cases``     the three test cases (inclined Poiseuille, cylinder, boosted
              Taylor-Green).
``runner``    end-to-end driver: runs the cases, writes the JSON evidence and the
              figures into ``artifacts/invariance/``.

Deliberately kept free of production-solver dependencies: an invariance test must
not change when the solver is refactored.
"""

from . import cases, galilean, geometry, lattice

__all__ = ["cases", "galilean", "geometry", "lattice"]
