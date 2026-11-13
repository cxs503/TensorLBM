"""XVAL-LBM-CARTESIAN-MPI: Cross-validation of TensorLBM against pylbm's
cartesian-MPI domain decomposition approach.

This module implements the same cartesian domain decomposition strategy used
by pylbm/pylbm (MpiTopology + halo exchange) but emulated in a single process
for environments where mpi4py is unavailable.  The decomposition logic,
halo width, and subdomain indexing follow pylbm's source exactly.

Public API
----------
``run_campaign(spec, out_dir)``
    Run the Poiseuille benchmark with both monolithic and domain-decomposed
    TensorLBM, compare against the analytical solution, and write result.json.

``PoiseuilleSpec``
    Frozen case specification (geometry, Re, resolution, decomposition).

``CartesianDecomposition``
    Domain decomposition emulating pylbm's MpiTopology.
"""

from .case_spec import PoiseuilleSpec, DEFAULT_SPEC
from .domain_decomp import CartesianDecomposition
from .campaign import run_campaign

__all__ = [
    "PoiseuilleSpec",
    "DEFAULT_SPEC",
    "CartesianDecomposition",
    "run_campaign",
]
