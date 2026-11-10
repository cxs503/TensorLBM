"""Rarefied gas / slip-flow verification ladder for TensorLBM (P218-N06).

Modules
-------
``knudsen``     analytic reference layer: BGK mean free path, Maxwell slip
                length, Navier-Stokes+slip profiles/flow rates, free-molecular
                flow-rate asymptote, Knudsen-number bookkeeping.
``bgk_slab``    deterministic linearized-BGK discrete-velocity (DVM) solver for
                plane Couette / force-driven Poiseuille flow between walls with
                Maxwell (diffuse+specular, TMAC = sigma) reflection.
``dsmc_slab``   independent stochastic reference: NTC direct simulation Monte
                Carlo in the same 1-D slab geometry.
``ladder``      the C1..C5 acceptance ladder: runs both solvers, extracts
                slip length / Knudsen-layer thickness / mass-flow-rate curve and
                writes a machine-readable JSON verdict.

Physical picture
----------------
The continuum (Kn -> 0) and free-molecular (Kn -> inf) ends are covered by
closed-form asymptotes; the transitional range is covered by two mutually
independent kinetic solvers (deterministic DVM-BGK and stochastic DSMC), so no
number in the ladder is self-referential to a single code.
"""

from __future__ import annotations

from .knudsen import (  # noqa: F401
    BGK_TAU_FACTOR,
    free_molecular_flow_rate,
    maxwell_slip_length,
    mean_free_path,
    ns_slip_couette_profile,
    ns_slip_flow_rate,
    ns_slip_poiseuille_profile,
    tau_from_kn,
)

__all__ = [
    "BGK_TAU_FACTOR",
    "free_molecular_flow_rate",
    "maxwell_slip_length",
    "mean_free_path",
    "ns_slip_couette_profile",
    "ns_slip_flow_rate",
    "ns_slip_poiseuille_profile",
    "tau_from_kn",
    "LinearizedBGKSlab",
    "SlabSolution",
    "run_dsmc_slab",
]

__all__.append("LinearizedBGKSlab")
__all__.append("SlabSolution")
__all__.append("run_dsmc_slab")


def __getattr__(name):  # lazy import so that `import tensorlbm.rarefied` stays light
    if name in ("LinearizedBGKSlab", "SlabSolution"):
        from . import bgk_slab

        return getattr(bgk_slab, name)
    if name == "run_dsmc_slab":
        from . import dsmc_slab

        return getattr(dsmc_slab, "run_dsmc_slab")
    raise AttributeError(name)