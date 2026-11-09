"""Ice-keel form/skin drag decomposition on a periodic under-ice channel.

Mission IN101-N01.  The module answers one question with machine-checkable
criteria: given an ice-underside morphology (flat -> triangular keel array),
what is the total ice-water drag coefficient ``C_d`` and how does it split
into a skin part ``C_d^skin`` and a form part ``C_d^form``, and how do those
scale with the keel density ``lambda`` and the shape parameter
``h_keel / h_i``?

Layout
------
``cases``
    Case tables: a smooth-wall resolution ladder (for the discretisation
    uncertainty, GCI) and a keel sweep over ``(lambda, h_keel)``.
``campaign``
    Runs a case table through the lattice-Boltzmann under-ice channel kernel
    of :mod:`tensorlbm.underice_drag` and stores one JSON record per case.
``criteria``
    Turns records into the acceptance verdicts (a)-(e) of the mission node:
    smooth-wall ``Cf`` against the log law / Dean correlation, form-drag
    scaling exponent against the analytic frontal-area estimate, the periodic-
    channel momentum budget and the skin+form closure, the observational
    ``C_d`` band and the keel-off negative control.
``figures``
    Paper figures from the aggregated artifact.

Physical conventions inherited from :mod:`tensorlbm.underice_drag`: the drag
coefficient uses the observational convention ``tau = rho C_d U^2`` while the
channel skin-friction coefficient is ``Cf = 2 tau / (rho U^2)``, so
``C_d = Cf / 2``.  Nothing here hard-codes a target: the observational band is
only used to adjudicate the computed numbers.
"""

from . import campaign, cases, criteria, figures, report

__all__ = ["campaign", "cases", "criteria", "figures", "report"]