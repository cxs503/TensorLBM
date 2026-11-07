"""IN25-T6 external-survey contract tests.

Guards the survey's evidence discipline, which is the whole point of the node:

* the anchor table cannot lose a measure, a provenance field, or gain an
  unsourced numeric claim;
* the SUBOFF second anchor lane must keep reproducing the *independent* AFF-1
  value (0.0031) and must keep refusing the unreproducible AFF-8 value (0.004);
* conventions must stay explicit — the CT sensitivity to wetted area and
  density is asserted, not documented;
* the five PLAYBOOK entries must stay executable (anchor present + a numeric
  acceptance threshold);
* the acceptance report must carry the criteria numbers and product paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm.external_survey import consistency as cons
from tensorlbm.external_survey import report as rpt
from tensorlbm.external_survey import suboff_anchor as sub
from tensorlbm.external_survey.lit_anchors import (
    ABSENT_ITEMS,
    LITERATURE_ANCHORS,
    MEASURES,
    ClaimKind,
    EvidenceSource,
    anchor_index,
    load_record_store,
    validate_anchor_table,
)
from tensorlbm.external_survey.playbook import PLAYBOOK, validate_playbook

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# anchor table
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_anchor_table_has_no_violations():
    table = validate_anchor_table()
    assert table.violations == ()
    assert table.ok


@pytest.mark.unit
def test_anchor_table_covers_five_measures_with_at_least_five_anchors():
    table = validate_anchor_table()
    assert table.n_anchors >= 5
    assert set(table.per_measure) == set(MEASURES)
    assert all(count >= 2 for count in table.per_measure.values()), table.per_measure


@pytest.mark.unit
def test_every_anchor_has_an_archived_bibliographic_record():
    store = load_record_store()
    for a in LITERATURE_ANCHORS:
        rec = store[a.anchor_id]
        assert rec["title"] and rec["authors"], a.anchor_id
        assert rec["retrieved_at"].endswith("Z"), a.anchor_id
        assert rec["endpoint"].startswith("http"), a.anchor_id
        assert rec["id_kind"] in {"doi", "arxiv"}


@pytest.mark.unit
def test_bibliographic_records_match_the_registry_text():
    """Spot-check that the packaged store really holds the registry strings."""
    store = load_record_store()
    walberla = store["10.1016/j.camwa.2020.01.007"]
    assert walberla["title"].startswith("waLBerla: A block-structured high-performance")
    assert walberla["year"] == 2021
    assert walberla["authors"][0].startswith("Bauer")
    musker = store["10.2514/3.61193"]
    assert musker["year"] == 1979
    assert "Explicit Expression for the Smooth Wall Velocity Distribution" in musker["title"]


@pytest.mark.unit
def test_numeric_claims_are_readable_offline():
    for a in LITERATURE_ANCHORS:
        if a.claim_kind is not ClaimKind.NUMERIC:
            continue
        assert a.quantities, a.anchor_id
        assert a.evidence in (EvidenceSource.ABSTRACT_OA, EvidenceSource.TITLE)
        for q in a.quantities:
            assert q.unit and q.value > 0.0


@pytest.mark.unit
def test_conditions_are_stated_for_every_anchor():
    for a in LITERATURE_ANCHORS:
        assert len(a.conditions) > 15, a.anchor_id
        assert a.citation.year is not None


@pytest.mark.unit
def test_absent_items_are_documented_and_excluded_from_scoring():
    assert len(ABSENT_ITEMS) >= 1
    for item in ABSENT_ITEMS:
        assert len(item.reason.strip()) > 10
        assert len(item.consequence.strip()) > 10
    # the scorer candidates must not depend on withheld material
    blocked = " ".join(i.item for i in ABSENT_ITEMS)
    assert "0.004" in blocked  # the unreproducible anchor is explicitly recorded


# --------------------------------------------------------------------------- #
# SUBOFF second anchor lane
# --------------------------------------------------------------------------- #
@pytest.mark.benchmark
def test_suboff_ladder_shape_and_monotonicity():
    for cfg, fn in ((1, sub.aff1_ladder), (8, sub.aff8_ladder)):
        pts = fn()
        assert len(pts) == 6
        assert all(p.configuration == cfg for p in pts)
        res = [p.re for p in pts]
        assert res == sorted(res)
        cts = [p.ct for p in pts]
        assert all(c > 0 for c in cts)
        assert all(b < a for a, b in zip(cts, cts[1:])), "CT must fall with Re"


@pytest.mark.benchmark
def test_aff1_reproduces_the_independent_published_value():
    target = min(sub.aff1_ladder(), key=lambda p: abs(p.re - sub.PUBLISHED_AFF1_RE))
    dev = 100.0 * (target.ct - sub.PUBLISHED_AFF1_CT) / sub.PUBLISHED_AFF1_CT
    assert abs(dev) <= 5.0, dev
    audit = {c.label: c for c in sub.anchor_consistency()}
    assert audit["SUBOFF AFF-1 CT"].verdict == "cross_consistent"
    assert audit["SUBOFF AFF-1 CT"].usable


@pytest.mark.benchmark
def test_aff8_published_value_is_withheld_as_unreproducible():
    audit = {c.label: c for c in sub.anchor_consistency()}
    rec = audit["SUBOFF AFF-8 CT"]
    assert rec.verdict == "withheld"
    assert not rec.usable
    lo, hi = rec.dev_pct_bracket
    assert hi < -5.0 and lo < 0.0  # Table 14 sits below 0.004 for every area convention
    sens = sub.convention_sensitivity()
    assert sens["implied_s_wet_for_published_ct_m2"] < sub.S_WET_AFF1
    assert sens["published_ct_physically_reachable"] is False


@pytest.mark.benchmark
def test_aff8_interval_brackets_the_ladder_point():
    bracket = sub.aff8_ct_bracket()
    lo, hi = bracket["interval"]
    assert lo < hi
    # smaller wetted area -> larger CT, so the interval low end uses the LARGER area
    assert lo == pytest.approx(bracket["ct_using_s_wet_hi"])
    assert hi == pytest.approx(bracket["ct_using_s_wet_lo"])
    assert bracket["ct_using_s_wet_lo"] > bracket["ct_using_s_wet_hi"]
    mid = min(sub.aff8_ladder(), key=lambda p: abs(p.re - sub.PUBLISHED_AFF8_RE))
    assert lo <= mid.ct * 1.06 and mid.ct * 0.94 <= hi


@pytest.mark.benchmark
def test_appendage_force_ratio_is_consistent_over_the_ladder():
    ratios = [r for _, r in sub.appendage_force_ratio()]
    assert len(ratios) == 6
    assert all(1.10 < r < 1.25 for r in ratios), ratios
    spread = max(ratios) - min(ratios)
    assert spread < 0.02, "appendage increment should be nearly speed-independent"


@pytest.mark.benchmark
def test_ittc_line_plus_one_form_factor_reproduces_the_ladder():
    for cfg in (1, 8):
        rows = sub.ittc_ladder_reproduction(cfg)
        max_err = max(abs(r["err_pct"]) for r in rows)
        assert max_err <= 15.0, (cfg, max_err)


@pytest.mark.benchmark
def test_convention_sensitivity_is_explicit():
    sens = sub.convention_sensitivity()
    assert 2.0 < sens["sensitivity_s_lo_pct"] < 10.0
    assert -10.0 < sens["sensitivity_s_hi_pct"] < -2.0
    assert -4.0 < sens["sensitivity_rho_sea_water_pct"] < -1.0


@pytest.mark.unit
def test_scorer_candidates_carry_conditions_sources_and_uncertainty():
    cands = sub.scorer_candidates()
    assert len(cands) == 2
    for c in cands:
        assert c["verdict"].startswith("admit_to_scorer")
        assert len(c["points"]) == 6
        assert c["conditions"]
        assert c["sources"]
        assert c["uncertainty"]
        assert "primary_point" in c


# --------------------------------------------------------------------------- #
# cross-consistency / audit layer
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_friction_lines_agree_in_the_suboff_band():
    ag = cons.friction_line_agreement()
    assert ag.verdict == "cross_consistent"
    assert ag.max_rel_diff_pct_model_band <= 3.0
    assert ag.max_rel_diff_pct <= 5.0


@pytest.mark.unit
def test_archived_grid_practice_is_second_order_like():
    gp = cons.archived_grid_practice()
    assert gp["available"], gp
    assert 1.5 < gp["observed_order_p"] < 2.5, gp["observed_order_p"]
    assert gp["err_vs_reference_pct"]["L80"] > gp["err_vs_reference_pct"]["L48"]  # both negative


@pytest.mark.unit
def test_published_grid_scale_audit_is_labelled_as_derived():
    audit = cons.published_grid_scale_audit()
    assert "DERIVED ESTIMATE" in audit["assumption"]
    assert 100.0 < audit["implied_cells_per_hull_length"] < 250.0


@pytest.mark.unit
def test_own_ladder_is_recorded_as_honest_fail():
    own = cons.own_ladder_status()
    assert own["delivered"] is False
    assert own["honest_fail"] is True
    assert own["reason"]
    assert own["what_would_close_it"]


# --------------------------------------------------------------------------- #
# playbook
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_playbook_has_five_executable_entries():
    book = validate_playbook()
    assert book.violations == ()
    assert book.n_entries >= 5
    assert set(book.measures_covered) == set(MEASURES)
    for entry in PLAYBOOK:
        assert entry.has_numeric_threshold, entry.measure
        assert entry.first_step.startswith(("python", "同一", "以", "量化", "固定", "用", "在"))


@pytest.mark.unit
def test_playbook_entries_cite_existing_anchors():
    idx = anchor_index()
    ladder = {c["id"] for c in sub.scorer_candidates()}
    for entry in PLAYBOOK:
        assert entry.anchors
        for a in entry.anchors:
            assert a in idx or a in ladder, (entry.measure, a)


# --------------------------------------------------------------------------- #
# report / products
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_acceptance_report_carries_criteria_numbers_and_paths():
    rep = rpt.build_acceptance_report()
    assert rep["mission"] == "IN25-T6"
    for key in (
        "seats_real_run",
        "literature_anchors_total",
        "measures_covered",
        "anchor_table_valid",
        "anchors_cross_consistent",
        "playbook_entries",
        "friction_lines_consistent",
        "honest_fail_recorded",
    ):
        assert key in rep["criteria"], key
        assert rep["criteria"][key]["required"], key
    assert rep["criteria"]["literature_anchors_total"]["value"] >= 5
    assert rep["criteria"]["playbook_entries"]["value"] >= 5
    # core package products must exist next to the tests
    for key in ("package", "lit_anchors", "suboff_anchor", "consistency",
                "playbook_module", "report_module", "record_store"):
        assert rep["products"][key]["exists"], (key, rep["products"][key])


def test_anchor_table_export_matches_module():
    path = REPO / "docs" / "survey" / "in25_t6_lit_anchor_table.json"
    if not path.exists():
        pytest.skip("anchor table not generated yet (run emit_survey_artifacts.py)")
    payload = json.loads(path.read_text())
    assert payload["n_anchors"] == len(LITERATURE_ANCHORS)
    assert payload["validation_violations"] == []
    ids = {row["id"] for row in payload["rows"]}
    assert ids == {a.anchor_id for a in LITERATURE_ANCHORS}
    assert {row["measure"] for row in payload["rows"]} == set(MEASURES)


def test_playbook_export_matches_module():
    path = REPO / "docs" / "survey" / "in25_t6_playbook.md"
    if not path.exists():
        pytest.skip("playbook not generated yet (run emit_survey_artifacts.py)")
    text = path.read_text()
    for entry in PLAYBOOK:
        assert entry.title in text
        assert entry.acceptance in text