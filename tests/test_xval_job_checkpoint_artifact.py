"""Machine re-check of the delivered XVAL artifacts (no self-declaration).

These tests read the committed evidence bundle produced by
``benchmarks/xval_job_checkpoint/run_xval.py`` and assert the properties a
reviewer would otherwise have to eyeball: the reference-vs-got table exists and
passes, bit-exact rows are *exactly* zero, every status state is present in the
delivered ``job_status.txt`` files, and the two checkpoint generations exist for
the restart cases.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_job_checkpoint"
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.job_checkpoint import JOB_STATUSES  # noqa: E402


def _summary() -> dict:
    return json.loads((ART / "summary.json").read_text(encoding="utf-8"))


def test_ref_vs_got_bundle_exists_and_passes() -> None:
    table = json.loads((ART / "BENCHMARK_REF_VS_GOT.json").read_text(encoding="utf-8"))
    assert table["all_pass"] is True, [r for r in table["rows"] if not r["pass"]]
    assert len(table["rows"]) >= 8
    for row in table["rows"]:
        assert row["rel_err_pct"] <= max(row["tolerance_pct"], 15.0)
        assert row["ref_source"], "every row must cite the OSS source location"
    refs = {row["ref_source"] for row in table["rows"]}
    assert any("IO_multiphase.F90" in r for r in refs)
    assert any("Main_multiphase.F90" in r for r in refs)


def test_bit_exact_claims_are_exactly_zero() -> None:
    summary = _summary()
    for row in summary["restart_study"]["rows"]:
        assert row["max_abs_df_vs_continuous"] == 0.0, row
        assert row["bit_exact"] is True
        assert row["steps_recomputed"] == 0
    assert summary["overhead_study"]["auto_restart"]["max_abs_df_vs_continuous"] == 0.0
    assert summary["recovery"]["dual_backup"]["max_abs_df_vs_continuous"] == 0.0
    assert summary["recovery"]["dual_backup"]["resume_source"] == "backup"


def test_every_restart_arm_reaches_the_baseline_field() -> None:
    summary = _summary()
    digests = {row["final_digest"] for row in summary["restart_study"]["rows"]}
    assert digests == {summary["baseline"]["digest"]}
    assert summary["baseline"]["steps"] == summary["case"]["steps"]


def test_status_vocabulary_is_exercised_by_the_artifacts() -> None:
    statuses = set()
    for path in ART.glob("cases/*/job_status.txt"):
        with path.open(encoding="utf-8") as handle:
            head = handle.readline().strip()
        assert head in JOB_STATUSES, path
        statuses.add(head)
    assert {"continue", "done", "reached_max_step"} <= statuses, statuses


def test_dual_backup_generations_present() -> None:
    """``*.pt`` is gitignored, so the committed evidence is the digest manifest.

    Both generations must be recorded for the restart arms (primary + second
    backup), and where the files are still in the working tree they must match
    the manifest byte-for-byte.
    """
    manifest = json.loads((ART / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
    by_case = {c["case"]: c for c in manifest["cases"]}
    for case in ("restart_04", "recovery_dual"):
        files = by_case[case]["files"]
        assert "checkpoint.pt" in files and "checkpoint_bak.pt" in files, files
        assert files["checkpoint.pt"]["bytes"] > 0
        assert files["checkpoint_bak.pt"]["bytes"] > 0
        # recovery_dual's primary was deliberately truncated by the experiment;
        # the other arm's two generations are complete files of the same schema.
        if case == "restart_04":
            assert files["checkpoint_bak.pt"]["bytes"] == files["checkpoint.pt"]["bytes"]
        primary = ART / "cases" / case / "checkpoint.pt"
        if primary.exists():  # present in a working tree, not in a fresh clone
            import hashlib

            assert hashlib.sha256(primary.read_bytes()).hexdigest() == files["checkpoint.pt"]["sha256"]


def test_figures_and_repro_committed() -> None:
    figs = sorted((ART / "figures").glob("*.png"))
    assert len(figs) >= 4
    assert all(f.stat().st_size > 5_000 for f in figs)
    repro = (ART / "repro.sh").read_text(encoding="utf-8")
    assert "benchmarks/xval_job_checkpoint/run_xval.py" in repro
    docs = ROOT / "docs" / "xval" / "XVAL-LBM-JOB-STATUS-CHECKPOINT.md"
    assert docs.exists()
    text = docs.read_text(encoding="utf-8")
    for section in ("①", "②", "③", "④", "⑤", "⑥"):
        assert section in text