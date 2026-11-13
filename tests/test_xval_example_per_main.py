"""Tests for the example-per-main antipattern audit and single dispatch entry.

Anchor: XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN.

The audit tooling is static by design (no example is imported), so the unit
tests build synthetic trees in ``tmp_path`` and only the integration test
touches the real ``examples/`` directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "xval"))

from example_main_audit import (  # noqa: E402
    duplicate_line_ratio,
    fortran_file_info,
    python_file_info,
    scan_tree,
)
from example_runner import build_index, resolve, run_example  # noqa: E402

MAIN_GUARD = '''

if __name__ == "__main__":
    main()
'''


def _py_example(path: Path, body: str, header: str = "", cli: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'"""{header or path.stem} example."""', ""]
    if cli:
        lines += ["import argparse", ""]
    lines += ["def main() -> None:", f"    {body}", "", "    parser = None"]
    if cli:
        lines += [
            "    parser = argparse.ArgumentParser()",
            '    parser.add_argument("--nx", type=int, default=8)',
            "    parser.parse_args()",
        ]
    lines += ["", MAIN_GUARD.strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# metrics primitives
# --------------------------------------------------------------------------- #
def test_duplicate_line_ratio_distinguishes_copied_headers():
    shared = ["! license line one", "! license line two", "! license line three"]
    body_a = ["program a", "  call init()", "end program a"]
    body_b = ["program b", "  call step()", "end program b"]
    a = "\n".join(shared + body_a)
    b = "\n".join(shared + body_b)
    ratio, dup, total = duplicate_line_ratio([a, b], block_len=3)
    assert total == 12
    assert dup == 6
    assert ratio == pytest.approx(0.5)
    # unique bodies must not be counted
    unique_ratio, _, _ = duplicate_line_ratio(["\n".join(body_a)], block_len=3)
    assert unique_ratio == 0.0


def test_duplicate_line_ratio_block_len_is_monotone_upper_bound_free_but_valid():
    texts = ["a\nb\nc\nd", "a\nb\nc\ne"]
    r2, _, _ = duplicate_line_ratio(texts, block_len=2)
    r3, _, _ = duplicate_line_ratio(texts, block_len=3)
    r1, _, _ = duplicate_line_ratio(texts, block_len=1)
    assert 0.0 <= r3 <= r2 <= r1 <= 1.0
    with pytest.raises(ValueError):
        duplicate_line_ratio(texts, block_len=0)


def test_python_file_info_detects_entry_cli_and_import_side_effects(tmp_path):
    cli_file = tmp_path / "cli_example.py"
    _py_example(cli_file, "print('run')", cli=True)
    info = python_file_info(cli_file)
    assert info["has_entry"] is True
    assert info["entry_kind"] == "main_guard"
    assert info["has_cli"] is True
    assert "--nx" in info["cli_flags"]
    assert info["import_side_effects"] == 0

    side_effect_file = tmp_path / "side_effect.py"
    side_effect_file.write_text(
        '"""doc."""\n\nrun()\n\ndef helper():\n    return 1\n', encoding="utf-8"
    )
    side = python_file_info(side_effect_file)
    assert side["import_side_effects"] == 1
    assert side["has_entry"] is False


def test_fortran_file_info_detects_program_and_hardcoded_params(tmp_path):
    f90 = tmp_path / "main.f90"
    f90.write_text(
        "!============================== License GPLv3 =================\n"
        "! header\n"
        "program main\n"
        "    integer :: xlen=400,ylen=100\n"
        "    real :: tau\n"
        "    tau = 0.6\n"
        "end program main\n",
        encoding="utf-8",
    )
    info = fortran_file_info(f90)
    assert info["has_entry"] is True
    assert info["entry_kind"] == "program main"
    assert info["hardcoded_params"] == 1
    assert info["param_declarations"] == 2
    assert info["has_cli"] is False


# --------------------------------------------------------------------------- #
# tree scanner
# --------------------------------------------------------------------------- #
def _build_python_tree(root: Path, n: int = 3, cli_every: int = 0) -> Path:
    examples = root / "examples"
    for i in range(n):
        _py_example(
            examples / f"case_{i}.py",
            f"print({i})",
            header=f"Case {i}: shared one-command example",
            cli=(cli_every and i % cli_every == 0),
        )
    return examples


def test_scan_tree_python_signature_and_registry_detection(tmp_path):
    examples = _build_python_tree(tmp_path, n=3, cli_every=2)
    report = scan_tree(examples, relative_to=tmp_path)
    assert report["n_files"] == 3
    assert report["n_entry_units"] == 3
    assert report["entry_ratio"] == 1.0
    assert report["cli_adoption"] == pytest.approx(2 / 3)
    assert report["has_programmatic_index"] is False
    assert report["registry_files"] == []
    assert report["dup_line_ratio"] > 0.0  # identical scaffolding lines

    # a dispatch script elsewhere in the repo flips the index flag
    tools = tmp_path / "tools" / "xval"
    tools.mkdir(parents=True)
    (tools / "example_runner.py").write_text("print('index')\n", encoding="utf-8")
    assert scan_tree(examples, relative_to=tmp_path)["has_programmatic_index"] is True
    # ... and the masked-out baseline still reports the pre-change state
    masked = scan_tree(
        examples, relative_to=tmp_path, ignore=("tools/xval/example_runner.py",)
    )
    assert masked["has_programmatic_index"] is False


def test_scan_tree_detects_single_example_build_selector(tmp_path):
    examples = _build_python_tree(tmp_path, n=2)
    (tmp_path / "CMakeLists.txt").write_text(
        "set(example_path ../examples/2d/von_karman_vortex/main.f90)\n", encoding="utf-8"
    )
    report = scan_tree(examples, relative_to=tmp_path)
    assert report["build_selectors"], "share an 'edit the build to pick a case' selector"
    assert report["build_selectors"][0]["distinct_example_refs"] == 1


def test_scan_tree_is_deterministic(tmp_path):
    examples = _build_python_tree(tmp_path, n=4, cli_every=2)
    a = scan_tree(examples, relative_to=tmp_path)
    b = scan_tree(examples, relative_to=tmp_path)
    assert a == b


def test_scan_tree_rejects_missing_root(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_tree(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# real tree (integration)
# --------------------------------------------------------------------------- #
def test_real_examples_tree_signature():
    report = scan_tree(REPO_ROOT / "examples", relative_to=REPO_ROOT)
    assert report["n_files"] >= 150
    assert report["entry_ratio"] > 0.85
    assert report["cli_adoption"] > 0.5
    assert report["has_programmatic_index"] is True  # after this mission's dispatch layer
    assert all("examples/" in u["path"] for u in report["units"])


def test_real_index_ids_unique_and_files_exist():
    entries = build_index(REPO_ROOT / "examples", REPO_ROOT)
    assert len(entries) >= 150
    ids = [e.id for e in entries]
    assert len(set(ids)) == len(ids)
    for entry in entries:
        assert (REPO_ROOT / entry.path).is_file()
        assert len(entry.sha256) == 64
    by_id = {e.id: e for e in entries}
    assert by_id["minimal_d2q9"].has_entry is True
    assert resolve(entries, "minimal_d2q9").id == "minimal_d2q9"
    with pytest.raises(KeyError):
        resolve(entries, "definitely-not-an-example")


# --------------------------------------------------------------------------- #
# dispatch transparency (control: hand-typed subprocess)
# --------------------------------------------------------------------------- #
def test_runner_is_transparent_on_synthetic_example(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    script = examples / "demo.py"
    script.write_text(
        '"""demo."""\nimport sys\nprint("hello", sys.argv[1:])\n'
        "print('line2')\n",
        encoding="utf-8",
    )
    entries = build_index(examples, tmp_path)
    assert [e.id for e in entries] == ["demo"]
    manifest = tmp_path / "manifest.json"

    record = run_example(entries[0], root=tmp_path, timeout=120,
                         manifest=manifest, extra_args=["--nx", "4"])

    direct = subprocess.run(
        [sys.executable, "examples/demo.py", "--nx", "4"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    import hashlib

    assert record["rc"] == direct.returncode == 0
    assert record["stdout_sha256"] == hashlib.sha256(direct.stdout.encode()).hexdigest()
    assert record["stderr_sha256"] == hashlib.sha256(direct.stderr.encode()).hexdigest()
    saved = json.loads(manifest.read_text())
    assert saved["stdout_sha256"] == record["stdout_sha256"]
    assert saved["example_sha256"] == record["source_sha256_actual"]


def test_runner_reports_failure_rc(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "boom.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    entries = build_index(examples, tmp_path)
    record = run_example(entries[0], root=tmp_path, timeout=120)
    assert record["rc"] == 3


def test_runner_refuses_non_entry_example(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "lib.py").write_text('"""helper."""\n\nX = 1\n', encoding="utf-8")
    entries = build_index(examples, tmp_path)
    assert entries[0].has_entry is False


@pytest.mark.slow
def test_runner_runs_real_minimal_example(tmp_path):
    entries = build_index(REPO_ROOT / "examples", REPO_ROOT)
    entry = resolve(entries, "minimal_d2q9")
    record = run_example(entry, root=REPO_ROOT, timeout=600,
                         manifest=tmp_path / "run.json")
    assert record["rc"] == 0
    assert any("Simulation complete" in line for line in record["stdout_tail"])
    assert record["example_sha256"] == record["source_sha256_actual"]