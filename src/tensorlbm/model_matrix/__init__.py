"""Model matrix CI: every collision model is a CI matrix key.

Public surface
--------------
``registry()`` / ``model_names()`` / ``get_model()`` / ``build_collider()``
    the model registry (model -> real production collision operator)
``resolve_collide_fn()``
    solver-facing resolver: a model name behaves like a raw ``collide_fn``
``run_matrix()`` / ``summarize()``
    the coverage driver (coverage fraction, gate verdict, ref-vs-got records)
``CASES`` / ``case_names()``
    the analytic verification cases

Transfer source: CFD-GO/TCLB ``model-matrix-ci-plus-coverage`` (CI keyed by
model, per-model coverage, separate CPU / compile-guard pipelines) and
``model-dir-as-build-and-ci-unit`` (model name = build and CI key).
"""

from __future__ import annotations

from .cases import CASES, CaseResult, Grid, case_names, run_case
from .coverage import guard_one, make_figures, run_matrix, summarize
from .registry import (
    ModelSpec,
    build_collider,
    get_model,
    lattice_ops,
    model_names,
    nu_from_tau,
    registry,
    resolve_collide_fn,
)

__all__ = [
    "CASES",
    "CaseResult",
    "Grid",
    "ModelSpec",
    "build_collider",
    "case_names",
    "get_model",
    "guard_one",
    "lattice_ops",
    "make_figures",
    "model_names",
    "nu_from_tau",
    "registry",
    "resolve_collide_fn",
    "run_case",
    "run_matrix",
    "summarize",
]

__version__ = "1.0.0"