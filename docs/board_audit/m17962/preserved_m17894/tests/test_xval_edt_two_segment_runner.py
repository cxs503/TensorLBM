"""Machine gate for the XVAL runner: ref (MPLBM-UT transcription) vs got.

Runs the same comparison the campaign report quotes, at reduced grid size, and
asserts the owner criterion (<15%) plus bit-level tag agreement with the OSS
transcription.  Keeps `runs/xval_edt_two_segment` honest: if the library drifts,
this test fails before a report can claim otherwise.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = REPO_ROOT / "tools" / "xval_edt_two_segment.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("xval_edt_two_segment", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_runner_two_segment_matches_oss(runner):
    case = runner.case_two_segment_L(0.6)
    res = runner.compare_case(case)
    row = res["row"]
    # bit-identical Palabos geometry vs the OSS transcription
    assert row["tag_disagreement_count"] == 0
    assert row["tag_agreement"] == 1.0
    # every quoted check must satisfy its threshold
    failed = [c for c in row["checks"] if not c["pass"]]
    assert not failed, f"failed checks: {failed}"
    # the depth comparison must reproduce the analytic geometry within 15%
    assert row["analytic"]["dist_rel_err_mean_deep"] < 0.15
    assert row["analytic"]["class_agreement_vs_analytic"] > 0.98


@pytest.mark.unit
def test_runner_feature_path_matches_oss(runner):
    """Padded pressure-BC slices + non-wetting seeds: the full OSS entry path."""
    import numpy as np

    case = runner.case_two_segment_L(0.6)
    nw = np.zeros_like(case["solid"], dtype=bool)
    nw[:, :, 2:5] = ~case["solid"][:, :, 2:5]
    row = runner.compare_case(case, num_slices=2, nw_seeds=nw)["row"]
    assert row["tag_disagreement_count"] == 0
    assert row["counts"]["nw"] > 0
    assert row["got_geom_name"].startswith("two_segment_L_")


@pytest.mark.unit
def test_runner_capsule_volume_below_threshold(runner):
    case = runner.case_capsule(radius=8.0, length=12.0)
    row = runner.compare_case(case)["row"]
    assert row["volume"]["volume_rel_err"] < 0.15
    assert row["tag_disagreement_count"] == 0