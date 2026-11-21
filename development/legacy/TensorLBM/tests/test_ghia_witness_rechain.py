"""Gates for the Ghia (1982) witness re-chain.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE``.  These tests pin the claim
that the ``BOUND`` Ghia table used by the fused stream-collide acceptance is
re-derivable from its sha256-pinned OCR witness, and that the re-chain is
sensitive enough to notice a transcription defect.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_PKG, "benchmarks", "xval"))

import ghia_witness_rechain as R  # noqa: E402

RESULTS = {}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    if "r" not in RESULTS:
        out = tmp_path_factory.mktemp("rechain") / "rechain.json"
        RESULTS["r"] = R.run(False, str(out))
    return RESULTS["r"]


def test_witness_is_present_and_hash_matches(result):
    w = result["witness"]
    assert os.path.isfile(w["path"]), w["path"]
    assert w["bytes"] == 6697
    assert w["sha256"] == w["pinned"]
    assert w["match"] is True


def test_anchor_re100_cells_fully_rederived(result):
    """Every Re=100 cell (the table the acceptance scores against) must be
    found as a token of the OCR text layer."""
    s = result["magnitude_rechain_summary"]
    assert s["anchor_re100_missing"] == []
    for table in ("I", "II"):
        m = result["magnitude_rechain"][table]
        assert m["n_expected"] == 68
        assert m["n_matched"] >= 66


def test_residual_cells_are_ocr_corruptions_not_transcription_drift(result):
    s = result["magnitude_rechain_summary"]
    assert s["coverage"] >= 0.95
    # the text layer mangles these three cells ('4.38289' for '-0.38289',
    # '-0.51500' rendered next to the Re=400 block, etc.)
    assert set(s["missing_cells"]) <= {"I/Re1000/u[11]", "II/Re400/v[8]",
                                       "II/Re1000/v[5]"}
    assert s["residual_cells_corroborated"] is True


def test_known_provenance_defect_is_reported_not_patched(result):
    defects = result["magnitude_rechain_summary"]["known_provenance_defects_off_anchor"]
    assert [d["cell"] for d in defects] == ["II/Re1000/v[5]"]
    assert defects[0]["provenance_value"] == pytest.approx(-0.51550)
    # the witness and both independent copies print -0.51500
    assert defects[0]["value_reproduced_by"] == ["matlab"]


def test_signs_corroborated_by_two_independent_copies(result):
    s = result["sign_audit"]
    assert s["n_cells"] == 102
    assert s["n_cells_with_multiple_witnesses"] == 102
    assert s["n_disagreements"] == 0


def test_physical_invariants_of_the_anchor_table(result):
    assert result["invariant_gates"]["all_pass"] is True


def test_detection_power_measured(result):
    p = result["detection_power"]
    assert p["n_trials"] == 200
    assert p["catch_rate"] >= 2.0 / 3.0, p


def test_perturbation_is_detected():
    """Inject a defect into the provenance table: the re-chain must report it."""
    prov, _ = R.load_provenance()
    wit = R.check_witness(prov)
    sec = R._sections(open(wit["path"]).read())
    toks = R.tokens(sec["I"])
    mut = json.loads(json.dumps(prov))
    k = mut["reynolds_numbers"]["100"]["u"].index(-0.20581)
    mut["reynolds_numbers"]["100"]["u"][k] = -0.30581   # was -0.20581
    m = R.matched_sequence(R.expected_cells(mut, "I"), toks)
    assert m["n_missing"] >= 1
    assert any(x["cell"].endswith("u[%d]" % k) for x in m["misses"])


def test_acceptance_rescore_uses_no_reference_module(result):
    """The stored fields are re-scored with the witness-derived table only."""
    sc = result["rescore"]
    assert set(sc) == {"L32", "L128"}
    for key, v in sc.items():
        assert v["gate_15pct"] is True, (key, v)
        assert v["gate_3pct"] is True, (key, v)
    assert sc["L128"]["L2_u"] == pytest.approx(0.0067, abs=5e-4)
    assert sc["L128"]["L2_v"] == pytest.approx(0.0038, abs=5e-4)
    assert sc["L32"]["L2_u"] == pytest.approx(0.0261, abs=5e-4)
    assert sc["L32"]["L2_v"] == pytest.approx(0.0105, abs=5e-4)


def test_verdict_is_pass(result):
    assert result["verdict"]["pass"] is True, result["verdict"]