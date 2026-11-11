"""Cross-validation harness for the XVAL-LBM-TAYLORGREEN anchor.

Public entry point
------------------
``run_and_report(spec, out_dir)``
    run the case on every declared tool, compute the metrics against the exact
    Taylor-Green solution, and write ``result.json`` (``xval_table``,
    ``criteria``, ``bug_suspects``, ``frictions``) plus raw per-tool artefacts.

Support API
-----------
``TG2DSpec`` / ``DEFAULT_SPEC`` / ``boosted_spec``
    the single case specification shared by all tools (``result.case_spec``).
``exact_velocity`` / ``exact_K`` / ``fit_exp_rate`` / ``mode_phase_deg``
    analytic reference and metric primitives (used by the tests as well).
``run_tensorlbm`` / ``run_palabos`` / ``run_openfoam``
    individual tool adapters (each returns the same artefact schema).
"""

from .case_spec import DEFAULT_SPEC, TG2DSpec, boosted_spec
from .metrics import (
    exact_K,
    exact_phase_rate,
    exact_velocity,
    field_l2_error_pct,
    fit_exp_rate,
    mode_phase_deg,
    peak_position_fraction,
)
from .campaign import find_bug_suspects, make_figure, run_and_report, run_campaign
from .runners import run_openfoam, run_palabos, run_tensorlbm

__all__ = [
    "DEFAULT_SPEC",
    "TG2DSpec",
    "boosted_spec",
    "exact_K",
    "exact_phase_rate",
    "exact_velocity",
    "field_l2_error_pct",
    "find_bug_suspects",
    "fit_exp_rate",
    "make_figure",
    "mode_phase_deg",
    "peak_position_fraction",
    "run_and_report",
    "run_campaign",
    "run_openfoam",
    "run_palabos",
    "run_tensorlbm",
]