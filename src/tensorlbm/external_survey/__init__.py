"""External mature-technology survey layer (IN25-T6, 2026-11).

Absorb-then-exceed survey of five measures the ship-performance氏族 needs before
pushing higher-Re external-flow work on top of TensorLBM:

* :mod:`tensorlbm.external_survey.lit_anchors` — literature anchor table with
  offline-readable provenance (Crossref/arXiv records shipped in-package) and
  explicit ``ABSENT`` items for everything that could not be verified.
* :mod:`tensorlbm.external_survey.suboff_anchor` — the second resistance anchor
  lane for DARPA SUBOFF (AFF-1/AFF-8) rebuilt from the archived Liu & Huang
  Table 14 tow-tank forces, plus the cross-consistency arbitration against the
  ITTC line and the widely quoted ``CT ≈ 0.004``.
* :mod:`tensorlbm.external_survey.consistency` — friction-line agreement,
  archived grid-practice audit, published grid-scale audit, honest-FAIL record.
* :mod:`tensorlbm.external_survey.playbook` — the five executable PLAYBOOK
  entries distilled from the survey.
* :mod:`tensorlbm.external_survey.report` — acceptance report / anchor-table /
  playbook exporters (machine-readable, program-generated).
"""

from __future__ import annotations

from tensorlbm.external_survey import consistency, lit_anchors, playbook, report, suboff_anchor
from tensorlbm.external_survey.lit_anchors import (
    ABSENT_ITEMS,
    LITERATURE_ANCHORS,
    MEASURES,
    LitAnchor,
    validate_anchor_table,
)
from tensorlbm.external_survey.playbook import PLAYBOOK, validate_playbook
from tensorlbm.external_survey.report import (
    build_acceptance_report,
    write_acceptance_report,
    write_anchor_table,
    write_playbook,
)
from tensorlbm.external_survey.suboff_anchor import (
    aff1_ladder,
    aff8_ct_bracket,
    aff8_ladder,
    anchor_consistency,
    scorer_candidates,
)

__all__ = [
    "ABSENT_ITEMS",
    "LITERATURE_ANCHORS",
    "MEASURES",
    "PLAYBOOK",
    "LitAnchor",
    "aff1_ladder",
    "aff8_ct_bracket",
    "aff8_ladder",
    "anchor_consistency",
    "build_acceptance_report",
    "consistency",
    "lit_anchors",
    "playbook",
    "report",
    "scorer_candidates",
    "suboff_anchor",
    "validate_anchor_table",
    "validate_playbook",
    "write_acceptance_report",
    "write_anchor_table",
    "write_playbook",
]