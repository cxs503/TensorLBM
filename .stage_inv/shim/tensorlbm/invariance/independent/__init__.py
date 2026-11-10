"""Second, independent implementation of the invariance tests (mission P170W2B-N04).

Why a second implementation
--------------------------
:mod:`tensorlbm.invariance` already contains a D2Q9 solver and the three
invariance cases (node *P170W2-N08*).  A verification certificate that is
produced by a single code cannot distinguish "the physics is invariant" from
"this particular discretisation happens to be biased the same way at every
orientation".  This subpackage therefore re-implements the three cases with
deliberately *different* numerical choices and checks that the declared
tolerances are met by both codes:

============================  ==============================  ==============================
aspect                        ``tensorlbm.invariance``        ``...invariance.independent``
============================  ==============================  ==============================
collision                     BGK / TRT                       BGK
body force                    momentum controller (periodic)  Guo forcing (source term)
wall sub-grid distance        boundary cells                  analytical SDF ray bisection
open boundaries               none (periodic box)             Zou-He inlet / pressure outlet
cylinder drag                 momentum budget (periodic box)  Ladd momentum exchange (open box)
Galilean test                 integer-cell translation        continuous spectral phase drift
channel construction          single channel, oblique box     periodic tilted channel array
============================  ==============================  ==============================

Agreement between the two codes on the shared quantities is the actual claim of
this subpackage; where they disagree the disagreement is reported, not hidden.
"""

from . import cases, galilean, geometry, lattice, runner

__all__ = ["cases", "galilean", "geometry", "lattice", "runner"]
