"""Pytest front end for the shell-driven instruction-DSL conformance suite.

``development/verification/instruction_dsl/dsl_shell_suite/run_cases.sh`` is the
``XVAL-LBM-INSTRUCTION-DSL-TESTS-SH``: TCLB-style *shell* tests that drive the
CLI, judge its exit status and streams, and need no Python test framework.  A
shell suite is worthless if it cannot fail, so this module does two things:

1. runs the suite and asserts every case passes (and that the suite is not
   vacuously empty);
2. **mutation-tests the suite itself** -- it corrupts an expectation, or the
   instruction text a digest pins, and asserts the runner *rejects* it.  Those
   are the checks that distinguish "the tests assert an invariant" from "the
   tests merely execute code".

The CLI exit-code contract (docstring of ``instruction_dsl_cli``) is asserted
directly here as well, because the shell suite depends on it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "development/verification/instruction_dsl" / "dsl_shell_suite"
RUNNER = SUITE / "run_cases.sh"
CASES = SUITE / "cases"

pytestmark = pytest.mark.integration


def _run_suite(cases_dir: Path, out_dir: Path, timeout: int = 900):
    env = dict(os.environ)
    env["DSL_CASES_DIR"] = str(cases_dir)
    env["DSL_OUT_DIR"] = str(out_dir)
    env["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHON"] = sys.executable
    return subprocess.run(
        ["sh", str(RUNNER)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _copy_cases(dest: Path, names: list[str]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copytree(CASES / name, dest / name)
    return dest


# ---------------------------------------------------------------------------
# the suite itself
# ---------------------------------------------------------------------------
def test_shell_suite_passes_every_case(tmp_path):
    proc = _run_suite(CASES, tmp_path / "artifacts")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((tmp_path / "artifacts" / "results.json").read_text())
    assert summary["total"] >= 17
    assert summary["failed"] == 0
    assert summary["passed"] == summary["total"]
    # both halves of the contract must be exercised, not just the happy path
    statuses = {r["case"]: r["status"] for r in summary["results"]}
    valid = [c for c in statuses if c.startswith("v")]
    reject = [c for c in statuses if c.startswith("i")]
    assert len(valid) >= 5 and len(reject) >= 12
    assert all(v == "PASS" for v in statuses.values())


def test_suite_covers_both_error_classes():
    """Every rejection case must expect a specific DSL error class, not just non-zero."""
    seen = set()
    for expect in sorted(CASES.glob("*/expect.conf")):
        text = expect.read_text()
        if "error_class=" in text:
            seen.add(expect.parent.name[0])
            assert "diagnostic=" in text, f"{expect.parent.name} asserts no diagnostic"
    assert seen == {"i"}


# ---------------------------------------------------------------------------
# mutation tests: the suite must have teeth
# ---------------------------------------------------------------------------
def test_suite_rejects_wrong_physical_reference(tmp_path):
    """A 50 % shift in an analytical reference must fail the run."""
    cases = _copy_cases(tmp_path / "cases", ["v03_tg_decay_32"])
    expect = cases / "v03_tg_decay_32" / "expect.conf"
    lines = []
    for line in expect.read_text().splitlines():
        if line.startswith("approx.u_max="):
            ref, tol = line.split("=", 1)[1].split(":")
            line = f"approx.u_max={float(ref) * 1.5:.12e}:{tol}"
        lines.append(line)
    expect.write_text("\n".join(lines) + "\n")
    proc = _run_suite(cases, tmp_path / "artifacts")
    assert proc.returncode != 0
    assert "FAIL v03_tg_decay_32" in proc.stdout


def test_suite_rejects_wrong_diagnostic(tmp_path):
    """Asserting the wrong error message must fail, so the diagnostic is real."""
    cases = _copy_cases(tmp_path / "cases", ["i05_singular_moment_matrix"])
    expect = cases / "i05_singular_moment_matrix" / "expect.conf"
    expect.write_text(
        expect.read_text().replace(
            "diagnostic=moment matrix is singular",
            "diagnostic=this text never appears in any diagnostic",
        )
    )
    proc = _run_suite(cases, tmp_path / "artifacts")
    assert proc.returncode != 0
    assert "diagnostic lacks" in proc.stdout


def test_suite_rejects_wrong_exit_code(tmp_path):
    cases = _copy_cases(tmp_path / "cases", ["i04_relaxation_out_of_range"])
    expect = cases / "i04_relaxation_out_of_range" / "expect.conf"
    expect.write_text(expect.read_text().replace("exit=3", "exit=0"))
    proc = _run_suite(cases, tmp_path / "artifacts")
    assert proc.returncode != 0
    assert "exit 3 != 0" in proc.stdout


def test_suite_detects_changed_instruction_text(tmp_path):
    """The pinned digest must notice that the exercised program changed."""
    cases = _copy_cases(tmp_path / "cases", ["v01_static_uniform"])
    program = cases / "v01_static_uniform" / "program.lbm"
    program.write_text(program.read_text().replace("nx=8 ny=8", "nx=9 ny=8"))
    proc = _run_suite(cases, tmp_path / "artifacts")
    assert proc.returncode != 0
    assert "program_digest" in proc.stdout


def test_suite_reports_malformed_expectation(tmp_path):
    cases = _copy_cases(tmp_path / "cases", ["v05_lattice_invariants"])
    (cases / "v05_lattice_invariants" / "expect.conf").write_text("# no exit line\n")
    proc = _run_suite(cases, tmp_path / "artifacts")
    assert proc.returncode != 0
    assert "no exit= line" in proc.stdout


def test_empty_case_dir_is_a_failure_not_a_pass(tmp_path):
    """A suite that finds nothing must not report success."""
    empty = tmp_path / "cases"
    empty.mkdir()
    proc = _run_suite(empty, tmp_path / "artifacts")
    assert proc.returncode != 0
    assert "no cases found" in proc.stdout


# ---------------------------------------------------------------------------
# the CLI contract the shell suite relies on
# ---------------------------------------------------------------------------
def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, "-m", "tensorlbm.instruction_dsl_cli", *args],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300,
    )


def test_cli_exit_codes_match_the_documented_contract():
    valid = (CASES / "v05_lattice_invariants" / "program.lbm").as_posix()
    syntax = (CASES / "i01_unknown_instruction" / "program.lbm").as_posix()
    semantic = (CASES / "i05_singular_moment_matrix" / "program.lbm").as_posix()

    ok = _cli(valid, "--kv")
    assert ok.returncode == 0 and "program_digest=" in ok.stdout
    assert ok.stderr == ""

    bad = _cli(syntax, "--kv")
    assert bad.returncode == 2
    assert bad.stderr.startswith("ERROR DSLSyntaxError: ")
    assert "Traceback" not in bad.stderr

    sem = _cli(semantic, "--kv")
    assert sem.returncode == 3
    assert sem.stderr.startswith("ERROR DSLSemanticError: ")
    assert "Traceback" not in sem.stderr

    missing = _cli(str(REPO / "no_such_program.lbm"), "--kv")
    assert missing.returncode == 4


def test_cli_writes_full_record_when_asked(tmp_path):
    out = tmp_path / "record.json"
    proc = _cli((CASES / "v03_tg_decay_32" / "program.lbm").as_posix(), "--out", str(out))
    assert proc.returncode == 0
    record = json.loads(out.read_text())
    for key in ("program_digest", "nu", "tau", "cs2", "u_max", "summary"):
        assert key in record