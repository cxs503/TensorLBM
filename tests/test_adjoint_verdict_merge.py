"""Tests for the sharded GAP-DIFFLBM-ADJOINT-VERDICT campaign merger.

The merge exists so that the paper-resolution campaign can be split over
several cluster jobs.  Its two jobs are (a) assemble the configs and (b) apply
the *benchmark's own* verdict logic -- no second implementation, no silently
mixed runs.  These tests pin both, on synthetic shards (no solver involved), so
they run in well under a second.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

from adjoint_verdict_merge import collect_fields, load_shards  # noqa: E402


def _rec(name: str, *, cos_frozen: float, los: float = 0.001) -> dict:
    """A config record with exactly the keys ``summarise`` reads."""
    return {
        "config": {"name": name, "reynolds": 40.0},
        "base": {"C_L": los, "C_D": -0.1},
        "fd": {"reference": -2.27e-3, "reference_vector": [-1e-9, -2.27e-3]},
        "ad": {
            "reference": -2.27e-3,
            "vector": [-1e-9, -2.27e-3],
            "n=100": {"rel_error_vs_fd": 1e-5},
        },
        "frozen": {"dJ_dy_c": 1.0e-3, "cos_vs_ad": cos_frozen, "cos_vs_ad_y_only": cos_frozen},
        "ship": {
            "dJ_dy_c": 1.1e-3,
            "cos_vs_ad": cos_frozen,
            "amplitude_ratio_vs_ad": 2.289,
            "optimization_path_eligible": False,
        },
        "trajectory": {"ship": {"improved": False}, "ad": {"improved": True}},
    }


def _shard(tmp_path, fname: str, configs: dict, *, predeclared=None, mode="paper"):
    payload = {
        "anchor": "GAP-DIFFLBM-ADJOINT-VERDICT",
        "schema": "tensorlbm/adjoint-verdict-diff-lbm/v1",
        "mode": mode,
        "generated_at": "2026-01-01T00:00:00Z",
        "environment": {"torch": "test"},
        "predeclared": predeclared or {"gate": "cos>=0.99", "max_steps": 30000},
        "configs": configs,
        "wall_seconds": 10.0,
    }
    path = tmp_path / fname
    path.write_text(json.dumps(payload))
    return str(path)


def test_merge_concatenates_configs_and_sums_wall_time(tmp_path):
    a = _shard(tmp_path, "R1.json", {"R1_yc050": _rec("R1_yc050", cos_frozen=-1.0)})
    b = _shard(tmp_path, "R2.json", {"R2_yc050": _rec("R2_yc050", cos_frozen=-0.998)})
    merged = load_shards([a, b])
    assert list(merged["configs"]) == ["R1_yc050", "R2_yc050"]
    assert merged["wall_seconds"] == pytest.approx(20.0)
    assert [s["configs"] for s in merged["shards"]] == [["R1_yc050"], ["R2_yc050"]]


def test_merge_applies_the_benchmark_verdict(tmp_path):
    # anti-parallel surrogate -> gate fails -> surrogate banned
    a = _shard(tmp_path, "R1.json", {"R1_yc050": _rec("R1_yc050", cos_frozen=-1.0)})
    b = _shard(tmp_path, "R2.json", {"R2_yc050": _rec("R2_yc050", cos_frozen=-0.998)})
    verdict = load_shards([a, b])["verdict"]
    assert verdict["frozen_field_surrogate_banned"] is True
    assert verdict["ad_matches_fd_all_configs"] is True

    # a surrogate that does pass the cosine gate must flip the verdict, so the
    # test cannot pass by the merge hard-coding the expected answer
    c = _shard(tmp_path, "R3.json", {"R3_yc050": _rec("R3_yc050", cos_frozen=+1.0)})
    assert load_shards([c])["verdict"]["frozen_field_surrogate_banned"] is False


def test_merge_rejects_mixed_runs(tmp_path):
    a = _shard(tmp_path, "R1.json", {"R1_yc050": _rec("R1_yc050", cos_frozen=-1.0)})
    b = _shard(
        tmp_path,
        "R2.json",
        {"R2_yc050": _rec("R2_yc050", cos_frozen=-1.0)},
        predeclared={"gate": "cos>=0.99", "max_steps": 700},
    )
    with pytest.raises(SystemExit, match="predeclared"):
        load_shards([a, b])

    c = _shard(tmp_path, "R3.json", {"R1_yc050": _rec("R1_yc050", cos_frozen=-1.0)})
    with pytest.raises(SystemExit, match="duplicate configs"):
        load_shards([a, c])


def test_collect_fields_copies_snapshots_next_to_the_merged_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    merged = {
        "configs": {"R1_yc050": _rec("R1_yc050", cos_frozen=-1.0)},
        "shards": [],
    }
    shard_dir = tmp_path / "artifacts" / "adjoint_verdict_diff_lbm_R1"
    shard_dir.mkdir(parents=True)
    (shard_dir / "fields_R1_yc050.npz").write_bytes(b"npz-bytes")
    out = str(tmp_path / "artifacts" / "adjoint_verdict_diff_lbm.json")
    copied = collect_fields(merged, out)
    dst = tmp_path / "artifacts" / "adjoint_verdict_diff_lbm" / "fields_R1_yc050.npz"
    assert copied == [str(dst)]
    assert dst.read_bytes() == b"npz-bytes"