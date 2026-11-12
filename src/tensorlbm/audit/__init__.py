"""Evidence-driven capability audits for TensorLBM.

The package exists because "the module is importable" and "the model is verified"
are different claims (owner 2026-11-12 maturity ruling).  Every audit in here must
bind each grade to a file on disk (test, benchmark result, convergence report) and
record the digest of the inputs it read, so a grade can never be self-declared.

Public API
----------
``intl_bench.run_audit(repo_root, out_dir)``
    Full GAP-LBM-INTL-BENCH audit: four-upstream (OpenLB / waLBerla / Palabos /
    Lettuce) citation check, four-tier module maturity, ten-dimension compete
    score, keep/refactor/delete disposition and gap->development task list.
``intl_bench.grade(repo_root, status, module_spec)``
    Grade a single module from the evidence on disk (importable for tests).
"""

from __future__ import annotations

from .intl_bench import (
    MODULES,
    UPSTREAM,
    check_citations,
    compete,
    grade,
    load_status,
    main,
    run_audit,
)

__all__ = [
    "MODULES",
    "UPSTREAM",
    "check_citations",
    "compete",
    "grade",
    "load_status",
    "main",
    "run_audit",
]