"""Tests for the numerical-baseline reference harness.

The harness ports the pytest reference practice of ``pylbm/pylbm``
(``tests/conftest.py``: deterministic test-id -> file name, atol/rtol compare
against ``tests/reference/*``, hard error when a baseline is missing, generate
mode that writes the reference and skips).  These tests pin the *semantics* of
that port, so a refactor cannot turn a failing comparison into a silent pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.baseline_io import payload_digest, read_block, write_block
from tensorlbm.verification.baseline_reference import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    BaselineStore,
    ReferenceNotFound,
    _cli,
    baseline_name,
    compare_fields,
)


@pytest.fixture()
def store(tmp_path: Path) -> BaselineStore:
    ref = tmp_path / "reference"
    ref.mkdir()
    return BaselineStore(ref)


def _fields(scale: float = 1.0) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, 17)
    return {"ux": scale * np.sin(x), "uy": scale * np.cos(x), "mass": np.asarray(1.0)}


# ---------------------------------------------------------------------------
# Naming (port of conftest.py:318-335)
# ---------------------------------------------------------------------------


def test_baseline_name_brackets_become_underscores() -> None:
    assert baseline_name("tests/test_x.py::test_cavity[33-100]") == "test_cavity_33-100.npz"


def test_baseline_name_single_reference_drops_parametrisation() -> None:
    assert baseline_name("tests/test_x.py::test_cavity[33]", single_reference=True) == "test_cavity.npz"


def test_baseline_name_explicit_filename_wins() -> None:
    assert baseline_name("t::a[1]", filename="fixed.npz") == "fixed.npz"


# ---------------------------------------------------------------------------
# Block IO / provenance
# ---------------------------------------------------------------------------


def test_block_roundtrip_preserves_arrays_and_meta(tmp_path: Path) -> None:
    path = write_block(tmp_path / "b.npz", _fields(), {"solver": "unit-test"})
    fields, meta = read_block(path)
    assert set(fields) == {"ux", "uy", "mass"}
    np.testing.assert_array_equal(fields["ux"], _fields()["ux"])
    assert meta["solver"] == "unit-test"
    assert meta["sha256"] == payload_digest(_fields())
    assert meta["format"].startswith("tensorlbm-baseline-block/")


def test_block_rejects_non_numeric_fields(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_block(tmp_path / "b.npz", {"bad": np.asarray(["a", "b"])})


def test_block_rejects_empty_field_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_block(tmp_path / "b.npz", {})


# ---------------------------------------------------------------------------
# Comparison semantics (pylbm: pytest.approx(rel=rtol, abs=atol))
# ---------------------------------------------------------------------------


def test_compare_identical_passes(store: BaselineStore) -> None:
    store.write("case.npz", _fields(), {"solver": "unit-test"})
    result = store.compare("case.npz", _fields())
    assert result.ok, result.message
    assert result.atol == DEFAULT_ATOL and result.rtol == DEFAULT_RTOL
    assert all(d.max_abs == 0.0 for d in result.fields.values())


def test_compare_detects_violation_and_reports_magnitude(store: BaselineStore) -> None:
    store.write("case.npz", _fields(), {})
    perturbed = _fields()
    perturbed["ux"] = perturbed["ux"] + 1e-3
    result = store.compare("case.npz", perturbed)
    assert not result.ok
    assert result.fields["ux"].max_abs == pytest.approx(1e-3, rel=1e-6)
    assert result.fields["ux"].n_exceeded == perturbed["ux"].size
    assert result.fields["uy"].ok
    assert "FAIL" in result.message


def test_compare_accepts_perturbation_below_atol(store: BaselineStore) -> None:
    store.write("case.npz", _fields(), {})
    perturbed = _fields()
    perturbed["ux"] = perturbed["ux"] + DEFAULT_ATOL / 10.0
    assert store.compare("case.npz", perturbed).ok


def test_missing_baseline_is_a_hard_error(store: BaselineStore) -> None:
    with pytest.raises(ReferenceNotFound) as exc:
        store.compare("absent.npz", _fields())
    assert "expected for new tests" in str(exc.value)


def test_field_set_mismatch_raises(store: BaselineStore) -> None:
    store.write("case.npz", _fields(), {})
    bad = _fields()
    bad.pop("uy")
    with pytest.raises(ValueError, match="field set mismatch"):
        store.compare("case.npz", bad)


def test_shape_mismatch_raises(store: BaselineStore) -> None:
    store.write("case.npz", _fields(), {})
    with pytest.raises(ValueError, match="shape"):
        compare_fields({"a": np.zeros(3)}, {"a": np.zeros(4)})


def test_hand_edited_baseline_is_detected(tmp_path: Path) -> None:
    path = write_block(tmp_path / "case.npz", _fields(), {})
    _, meta = read_block(path)
    tampered = _fields()
    tampered["ux"] = tampered["ux"] * 2.0
    # Rewrite the same file with the stale metadata record: the payload digest
    # no longer matches, which must fail even though the comparison is vacuous.
    np.savez_compressed(
        path,
        __meta__=np.asarray(json.dumps(meta)),
        **{f"field__{k}": v for k, v in tampered.items()},
    )
    result = BaselineStore(tmp_path).compare("case.npz", tampered)
    assert not result.ok
    assert "digest mismatch" in result.message


# ---------------------------------------------------------------------------
# Generate mode and registry
# ---------------------------------------------------------------------------


def test_generate_mode_writes_into_generate_dir(tmp_path: Path) -> None:
    ref = tmp_path / "reference"
    gen = tmp_path / "generated"
    store = BaselineStore(ref, generate_dir=gen)
    assert store.generating
    path = store.write("case.npz", _fields(), {"solver": "unit-test"})
    assert path.parent == gen
    np.testing.assert_array_equal(read_block(path)[0]["ux"], _fields()["ux"])
    assert store.list() == []  # nothing registered yet


def test_register_copies_block_into_reference_dir(store: BaselineStore, tmp_path: Path) -> None:
    candidate = write_block(tmp_path / "candidate.npz", _fields(), {"solver": "unit-test"})
    target = store.register("case.npz", candidate)
    assert target == store.reference_dir / "case.npz"
    assert store.list() == ["case.npz"]
    assert store.compare("case.npz", candidate).ok


def test_environment_override_sets_reference_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLBM_BASELINE_REFERENCE_DIR", str(tmp_path / "envref"))
    assert BaselineStore().reference_dir == tmp_path / "envref"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_cli_list_register_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref = tmp_path / "reference"
    candidate = write_block(tmp_path / "candidate.npz", _fields(), {})
    assert _cli(["--reference-dir", str(ref), "register", "case.npz", str(candidate)]) == 0
    assert _cli(["--reference-dir", str(ref), "list"]) == 0
    assert "case.npz" in capsys.readouterr().out

    bad = write_block(tmp_path / "bad.npz", {**_fields(), "mass": np.asarray(2.0)}, {})
    json_out = tmp_path / "result.json"
    assert _cli(["--reference-dir", str(ref), "check", "case.npz", str(bad), "--json-out", str(json_out)]) == 1
    payload = json.loads(json_out.read_text())
    assert payload["ok"] is False and payload["fields"]["mass"]["n_exceeded"] == 1


def test_module_entry_point_runs(tmp_path: Path) -> None:
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-m", "tensorlbm.verification.baseline_reference", "--reference-dir",
         str(tmp_path), "list"],
        capture_output=True, text=True, env=env, check=False,
    )
    assert out.returncode == 0, out.stderr
