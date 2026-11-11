"""Unit tests for the round-4 pre-registered model-selection rule.

The rule (``coexistence.select_model``) is what fixes ``G_12`` for the
acceptance matrix, so it must be testable without running a solver: the
admissibility predicates are pure functions of the calibration record.
"""
from __future__ import annotations

from cases.mcmp_stratified_poiseuille.coexistence import (
    CONTRAST_MIN, DRIFT_MAX_CELLS, W_MIN_CELLS, select_model,
)


def _rec(g12, w, contrast=0.7, drift=0.2, steady=True, admissible=None):
    if admissible is None:
        admissible = bool(steady and w >= W_MIN_CELLS and contrast >= CONTRAST_MIN
                          and drift < DRIFT_MAX_CELLS)
    return dict(config=dict(g12=g12), rho_A=0.87, rho_B=0.14, contrast=contrast,
                w_10_90=w, w_tanh=w / 2.1972, steady=steady, drift=drift,
                admissible=admissible)


def test_picks_strongest_coupling_among_admissible():
    recs = [_rec(-2.0, 40.0), _rec(-2.5, 4.6), _rec(-2.4, 6.0)]
    sel = select_model(recs)
    assert sel["selected"] == -2.5


def test_unresolved_or_mixed_states_are_not_admissible():
    recs = [_rec(-2.4, 52.0), _rec(-2.0, 1e9, contrast=0.05)]
    sel = select_model(recs)
    assert sel["selected"] is None and "no admissible" in sel["reason"]


def test_weak_stratification_rejected():
    assert select_model([_rec(-3.0, 4.0, contrast=0.3)])["selected"] is None
