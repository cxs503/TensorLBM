"""Manifest-kind coverage and probe provenance (XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT).

These tests pin the two properties the CI-unit convention is supposed to buy and
that a mature code base drifts away from:

1. **No dialect escapes the matrix.**  The repository carries two manifest
   dialects for the same convention (``models/lbm/*/model.toml`` and
   ``models/units/*/model.json``).  The generated matrix must contain a cell for
   *every* model directory, so adding a directory -- in either dialect -- is
   still the only edit needed, and a directory that cannot be validated fails a
   job of its own.  TCLB's own drift (``models.R`` enumerates 61 directories,
   ``cpu_test.yml`` lists 16) is the counterexample being excluded here.

2. **The build artefact describes this checkout.**  The per-unit probe runs in
   a child interpreter; if a stale editable install or a ``PYTHONPATH`` entry
   pointing at another worktree wins the import search, the fingerprint and the
   CI verdict become deterministic statements about the wrong code.  The probe
   therefore pins ``<repo>/src`` first and refuses to report a foreign package.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

import pytest

from tensorlbm.model_units import coverage as cov
from tensorlbm.model_units import discover, doctor
from tensorlbm.model_units.coverage import (
    CoverageError,
    ModelDirectory,
    coverage,
    coverage_report,
    matrix_cells,
)

# the module, not the re-exported ``tensorlbm.model_units.build`` function
build_mod = importlib.import_module("tensorlbm.model_units.build")

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "benchmarks" / "model_units"
TOML_ROOT = MODELS / "lbm"
JSON_UNIT = MODELS / "units" / "d2q9_bgk_shear_wave"  # cheapest json unit


# --------------------------------------------------------------------------- #
# coverage: every model directory is a CI unit, whatever its manifest dialect
# --------------------------------------------------------------------------- #
def test_every_model_directory_is_on_the_generated_matrix():
    report = coverage_report(models_root=MODELS, repo_root=REPO)
    assert report["ok"], report
    assert report["uncovered"] == [] and report["orphans"] == [] and report["duplicates"] == []
    assert report["directories"] == report["cells"]
    # both dialects are really present, otherwise this test proves nothing
    assert report["by_kind"] == {"unit_json": 4, "unit_toml": 3} or (
        set(report["by_kind"]) == {cov.JSON_KIND, cov.TOML_KIND}
    )
    for cell in report["cells_detail"]:
        assert (REPO / cell["dir"]).is_dir(), cell
        assert cell["cmd"].strip()


def test_each_dialect_gets_its_own_runner():
    report = coverage_report(models_root=MODELS, repo_root=REPO)
    for cell in report["cells_detail"]:
        if cell["kind"] == cov.TOML_KIND:
            assert "tensorlbm.model_units build-and-test" in cell["cmd"]
            assert cell["cmd"].endswith(f"--unit {cell['unit']}")
            assert "--only" not in cell["cmd"]
        else:
            assert "tensorlbm.models.ci_unit ci" in cell["cmd"]
            assert f"--only {cell['unit']}" in cell["cmd"]
            assert "build-and-test" not in cell["cmd"]


def test_new_directory_needs_no_edit_elsewhere(tmp_path):
    """Adding a json model == adding a directory (no registry edit anywhere)."""
    (tmp_path / "models" / "units").mkdir(parents=True)
    target = tmp_path / "models" / "units" / "extra_shear_wave"
    shutil.copytree(JSON_UNIT, target)
    manifest = json.loads((target / "model.json").read_text())
    manifest["unit_id"] = "extra_shear_wave"
    (target / "model.json").write_text(json.dumps(manifest, indent=2))

    report = coverage_report(models_root=tmp_path / "models", repo_root=tmp_path)
    assert report["ok"], report
    assert report["cells"] == 1
    cell = report["cells_detail"][0]
    assert cell["unit"] == "extra_shear_wave"
    assert cell["kind"] == cov.JSON_KIND
    assert "--only extra_shear_wave" in cell["cmd"]

    shutil.rmtree(target)
    empty = coverage_report(models_root=tmp_path / "models", repo_root=tmp_path)
    assert empty["cells"] == 0 and empty["ok"]


def test_hand_maintained_matrix_missing_a_unit_fails_the_gate():
    """The TCLB drift mode: a directory exists, the matrix does not list it."""
    directories = cov.discover_directories(MODELS, repo_root=REPO)
    cells = matrix_cells(directories, models_root=MODELS, repo_root=REPO)
    dropped = cells.pop()
    report = cov.coverage(directories, cells, models_root=MODELS, repo_root=REPO)
    assert report["ok"] is False
    assert dropped["dir"] in report["uncovered"]


def test_cell_without_a_directory_fails_the_gate():
    directories = cov.discover_directories(MODELS, repo_root=REPO)
    cells = matrix_cells(directories, models_root=MODELS, repo_root=REPO)
    cells.append(
        {
            "unit": "ghost_unit",
            "dir": "models/lbm/ghost_unit",
            "kind": cov.TOML_KIND,
            "cmd": "true",
        }
    )
    report = cov.coverage(directories, cells, models_root=MODELS, repo_root=REPO)
    assert report["ok"] is False
    assert "models/lbm/ghost_unit" in report["orphans"]


def test_duplicate_unit_id_across_dialects_is_reported():
    one = ModelDirectory(
        path=Path("/r/models/lbm/a"),
        manifest=Path("/r/models/lbm/a/model.toml"),
        kind=cov.TOML_KIND,
        unit_id="same_id",
        rel_dir="models/lbm/a",
    )
    two = ModelDirectory(
        path=Path("/r/models/units/b"),
        manifest=Path("/r/models/units/b/model.json"),
        kind=cov.JSON_KIND,
        unit_id="same_id",
        rel_dir="models/units/b",
    )
    cells = [
        {"unit": "same_id", "dir": "models/lbm/a", "kind": cov.TOML_KIND},
        {"unit": "same_id", "dir": "models/units/b", "kind": cov.JSON_KIND},
    ]
    report = coverage([one, two], cells, models_root=Path("/r/models"), repo_root=Path("/r"))
    assert report["ok"] is False
    assert report["duplicates"]


def test_unknown_json_schema_is_rejected_not_ignored(tmp_path):
    root = tmp_path / "models" / "units"
    root.mkdir(parents=True)
    bad = root / "bogus"
    bad.mkdir()
    (bad / "model.json").write_text(json.dumps({"schema": "something/v0", "unit_id": "bogus"}))
    with pytest.raises(CoverageError):
        cov.discover_directories(tmp_path / "models", repo_root=tmp_path)


def test_makefile_fragment_covers_both_dialects():
    from tensorlbm.model_units import makefile_targets

    units = discover(TOML_ROOT, repo_root=REPO)
    directories = cov.discover_directories(MODELS, repo_root=REPO)
    extra = [
        cell
        for cell in matrix_cells(directories, models_root=MODELS, repo_root=REPO)
        if cell["kind"] != cov.TOML_KIND
    ]
    text = makefile_targets(units, extra_cells=extra)
    for cell in extra:
        assert f"model-{cell['unit']}:" in text
        assert cell["cmd"] in text


# --------------------------------------------------------------------------- #
# probe provenance: the build artefact must describe this checkout
# --------------------------------------------------------------------------- #
def test_probe_env_puts_this_checkout_first():
    path = build_mod._probe_env()["PYTHONPATH"].split(os.pathsep)
    assert path[0] == str(build_mod.REPO_SRC)
    assert path.count(str(build_mod.REPO_SRC)) == 1


def test_probe_not_shadowed_by_a_foreign_tensorlbm(tmp_path, monkeypatch):
    """A stale ``tensorlbm`` earlier on PYTHONPATH must not be the one measured."""
    fake = tmp_path / "fake_site"
    (fake / "tensorlbm").mkdir(parents=True)
    (fake / "tensorlbm" / "__init__.py").write_text("raise ImportError('foreign tensorlbm')\n")
    monkeypatch.setenv("PYTHONPATH", str(fake))

    unit = discover(TOML_ROOT, repo_root=REPO)[0]
    probe = build_mod._probe(unit)  # would explode on the fake package
    assert Path(probe["tensorlbm_root"]) == build_mod.REPO_SRC / "tensorlbm"
    assert probe["finite"] is True


def test_build_artifact_records_package_origin(tmp_path):
    unit = discover(TOML_ROOT, repo_root=REPO)[0]
    record = build_mod.build(unit, out_root=tmp_path)
    assert Path(record["tensorlbm_root"]) == build_mod.REPO_SRC / "tensorlbm"
    artifact = json.loads((tmp_path / f"{unit.name}.build.json").read_text())
    assert Path(artifact["tensorlbm_root"]) == build_mod.REPO_SRC / "tensorlbm"
    assert artifact["unit"]["fingerprint"] == unit.fingerprint()


def test_doctor_gates_coverage_and_provenance():
    report = doctor(TOML_ROOT, repo_root=REPO, models_root=MODELS)
    checks = {c["check"]: c for c in report["checks"]}
    assert checks["manifest_kind_coverage"]["ok"], checks["manifest_kind_coverage"]
    assert checks["runtime_package_origin"]["ok"], checks["runtime_package_origin"]
    assert checks["probe_env_isolated"]["ok"], checks["probe_env_isolated"]
    assert report["coverage"]["ok"] is True
    assert report["ok"] is True