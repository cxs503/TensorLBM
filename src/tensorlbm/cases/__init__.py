"""Named simulation cases for TensorLBM (lettuce ``ExtFlow`` pattern).

Importing this package registers the built-in benchmark-aligned cases:

* ``cavity``       — lid-driven cavity, spanwise-periodic 3D
                     (``benchmarks/verified/cavity/3d``);
* ``poiseuille``   — circular-pipe Hagen-Poiseuille
                     (``benchmarks/verified/poiseuille_3d_pipe``);
* ``suboff_n128``  — small SUBOFF bare-hull channel
                     (``examples/ai4s_export.py`` pilot scale).

::

    from tensorlbm.cases import get_case, list_cases, run_case

    print([c["name"] for c in list_cases()])
    result = run_case("cavity", steps=2000, resolution=96, re=400.0)

New cases are added with the :func:`register_case` decorator; see
``cases/base.py`` for the three hooks to implement.
"""

from __future__ import annotations

from .base import CaseBase, CaseUnits
from .cavity import LidCavityCase
from .poiseuille import PipePoiseuilleCase
from .registry import (
    case_registry,
    get_case,
    has_case,
    list_cases,
    register_case,
    unregister_case,
)
from .runner import CaseRunResult, ExportSpec, run_case
from .suboff import SuboffChannelCase

__all__ = [
    "CaseBase",
    "CaseUnits",
    "CaseRunResult",
    "ExportSpec",
    "LidCavityCase",
    "PipePoiseuilleCase",
    "SuboffChannelCase",
    "case_registry",
    "get_case",
    "has_case",
    "list_cases",
    "register_case",
    "run_case",
    "unregister_case",
]
