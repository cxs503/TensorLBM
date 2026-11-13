"""Tests for the XVAL method-selection benchmark entry (TensorLBM).

These are integration tests: they call the published entry
``benchmarks/xval_method_selection_bench.py`` which calls the same arm runner
as the published matrix, and they assert invariants rather than merely
executing code:

  * both selections integrate to a finite state on a short budget,
  * the sampled drag is positive and O(1) (guards the post-stream force-sampling
    order — the pre-stream order reverses the sign),
  * the two arms do not disagree by more than 50 % at D=6 (guards against the
    harness manufacturing a method difference),
  * the arm table is complete and self-consistent (A1..A5 defined, no duplicates),
  * the executed-OSS-source kernel cross-check agrees to float32 round-off
    (skipped when the OSS checkout is absent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "xval" / "benchmark_method_selection"
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(XVAL))

import xval_method_selection_bench as bench  # noqa: E402
from run_arm import ARMS  # noqa: E402


def test_arm_table_defined():
    assert set(ARMS) == {"A1", "A2", "A3", "A4", "A5"}
    # A1 and A4 differ only in the sponge, A4 and A2 only in the collision model
    assert ARMS["A1"]["collision"] == ARMS["A4"]["collision"] == "mrt"
    assert ARMS["A1"]["sponge"] is True and ARMS["A4"]["sponge"] is False
    assert ARMS["A2"]["collision"] == "bgk" and ARMS["A2"]["sponge"] is False


def test_smoke_two_selections_finite_positive_drag(tmp_path):
    out = bench.smoke(arms=("A1", "A2"), out_root=tmp_path)
    recs = out["records"]
    assert set(recs) == {"A1", "A2"}
    for arm, rec in recs.items():
        assert rec["status"] == "ok", arm
        assert rec["stability_ok"] is True, arm
        assert rec["cd_mean"] > 0.0, f"{arm}: sign defect in force sampling"
        assert rec["ma"] < 0.1, arm
        assert rec["tau"] > 0.5, arm
        assert (tmp_path / f"{arm}_D6" / "result.json").exists()
    assert recs["A1"]["smoke_cd_spread_pct"] < 50.0


def test_smoke_writes_machine_readable_artifacts(tmp_path):
    out = bench.smoke(arms=("A4",), out_root=tmp_path)
    res = json.loads((tmp_path / "A4_D6" / "result.json").read_text())
    for key in ("cd_mean", "cl_amp", "st_fft", "st_zc", "total_steps", "tau", "u_in", "re"):
        assert key in res
    assert res["arm"] == "A4" and res["D"] == 6
    assert out["out_root"] == str(tmp_path)


@pytest.mark.skipif(not (Path.home() / "oss-cache" / "Maarten-vd-Sande_lbm" / "lbm").is_dir(),
                    reason="OSS reference checkout not available")
def test_oss_kernel_crosscheck_agrees():
    """ref = executed OSS source, got = TensorLBM kernel: must agree to float32 round-off."""
    import oss_crosscheck as oc

    k0 = oc.k0_direction_map()
    assert k0["all_agree"] is True
    k1 = oc.k1_collision(n=24)
    assert k1["rel_l2_collision"] < 1e-5, k1
    k2 = oc.k2_streaming(n=24)
    assert k2["rel_l2_streaming_interior"] < 1e-5, k2


def test_published_summary_is_consistent_when_present():
    """If the published matrix is in the tree, its summary must be self-consistent."""
    p = XVAL / "summary.json"
    if not p.exists():
        pytest.skip("matrix summary not published in this checkout")
    s = json.loads(p.read_text())
    table = s["table"]
    assert "A1_D10" in table
    for key, row in table.items():
        assert abs(row["st_crosscheck_pct"]) < s["thresholds"]["st_xcheck_pct"], key
        if row["arm"] == "A1":
            continue
        for kpi, delta in row.get("delta_vs_A1_pct", {}).items():
            assert abs(delta) < 1000.0, (key, kpi, delta)
        assert row["within_15pct"] == all(
            abs(v) < s["thresholds"]["accept_pct"] for v in row["delta_vs_A1_pct"].values()
        )
    for fail in s["failures"]:
        assert fail["status"] == "blowup"
        assert fail["step"] < fail["total_steps_planned"]