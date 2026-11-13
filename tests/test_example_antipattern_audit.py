"""Tests for the example-per-main anti-pattern auditor.

The auditor is only useful if it *discriminates*: the planted anti-pattern
fixtures (three near-identical stand-alone mains) must score high, the planted
consolidated fixtures (three thin mains delegating to the shared runtime) must
score zero duplicated driver lines.  A detector that flags everything, or
nothing, fails these tests.
"""

from __future__ import annotations

import json
import os

from tensorlbm.audit.example_antipattern import audit_tree, format_summary, main

HERE = os.path.dirname(os.path.abspath(__file__))
XV = os.path.join(os.path.dirname(HERE), "xval", "XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN")


def test_detector_fires_on_planted_antipattern():
    rep = audit_tree(os.path.join(XV, "fixtures", "antipattern"))
    m1, m2, m3 = rep["M1_standalone_mains"], rep["M2_duplication"], rep["M3_thin_mains"]
    assert rep["python_files"] == 3
    assert m1["share_with_main"] == 1.0
    assert m1["n_files_own_step_loop"] == 3
    assert m2["n_dup_driver_blocks"] > 0
    assert m2["dup_driver_loc_ratio"] > 0.3
    assert m2["dup_loc_ratio"] <= 1.0  # distinct lines, never >100%
    assert m3["share"] == 0.0


def test_detector_silent_on_planted_consolidated():
    rep = audit_tree(os.path.join(XV, "fixtures", "consolidated"))
    m1, m2, m3 = rep["M1_standalone_mains"], rep["M2_duplication"], rep["M3_thin_mains"]
    assert rep["python_files"] == 3
    # thin mains still have a __main__, but no driver logic of their own
    assert m1["n_files_own_step_loop"] == 0
    assert m2["n_dup_driver_blocks"] == 0
    assert m2["dup_driver_loc"] == 0
    assert m3["share"] == 1.0


def test_discrimination_gap_is_material():
    ap = audit_tree(os.path.join(XV, "fixtures", "antipattern"))
    co = audit_tree(os.path.join(XV, "fixtures", "consolidated"))
    gap = ap["M2_duplication"]["dup_driver_loc_ratio"] - co["M2_duplication"]["dup_driver_loc_ratio"]
    assert gap > 0.2, f"detector cannot separate the two trees (gap={gap})"


def test_synthetic_consolidated_tree_is_clean(tmp_path):
    """A tree where every example imports the runtime must score 0 driver copies."""
    for i in range(4):
        (tmp_path / f"case_{i}.py").write_text(
            "import json\n"
            "from tensorlbm.example_harness import get_case, run_case\n"
            "\n"
            "\n"
            "def main():\n"
            "    record = run_case(get_case('channel_poiseuille', ny=16, steps=100))\n"
            "    print(json.dumps(record['kpis']))\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
            encoding="utf-8",
        )
    rep = audit_tree(str(tmp_path))
    assert rep["python_files"] == 4
    assert rep["M2_duplication"]["dup_driver_loc"] == 0
    assert rep["M3_thin_mains"]["share"] == 1.0


def test_single_file_and_parse_error_are_tolerated(tmp_path):
    good = tmp_path / "ok.py"
    good.write_text("import torch\n\ndef f(x):\n    y = x + 1\n    return y\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n    pass\n", encoding="utf-8")
    rep = audit_tree(str(tmp_path))
    assert rep["python_files"] == 2
    assert rep["parsed_files"] == 1
    assert len(rep["parse_errors"]) == 1
    assert "SyntaxError" in rep["parse_errors"][0]["error"]

    single = audit_tree(str(good))
    assert single["python_files"] == 1


def test_cli_writes_json(tmp_path, capsys):
    out = tmp_path / "ap.json"
    rc = main(["--root", os.path.join(XV, "fixtures", "antipattern"), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["python_files"] == 3
    assert "M2_duplication" in payload
    assert "M2 duplication" in capsys.readouterr().out


def test_summary_is_human_readable():
    rep = audit_tree(os.path.join(XV, "fixtures", "consolidated"))
    text = format_summary(rep)
    assert "M1 standalone mains" in text
    assert "M3 thin mains" in text