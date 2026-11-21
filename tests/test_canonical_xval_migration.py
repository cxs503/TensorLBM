"""Guard the XVAL package promotion without asserting scientific acceptance."""

from __future__ import annotations

import csv
import json
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np

from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/architecture/migrations/canonical-xval-2026-11-21.json"


def git_blob(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()


def test_mapped_files_and_historical_objects_are_retained() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger["base_revision"] == "2822dddb763c04b8dbe34572056329057bbe55e4"
    assert len(ledger["path_mappings"]) == ledger["compatibility"]["mapped_files"]
    for row in ledger["path_mappings"]:
        old = ROOT / "development" / "legacy" / row["historical_path"]
        new = ROOT / row["canonical_path"]
        assert old.is_file() and new.is_file()
        assert git_blob(old) == row["historical_git_blob"]
        assert git_blob(new) == row["candidate_git_blob"]
        if row["byte_identical"]:
            assert old.read_bytes() == new.read_bytes()


def test_packaged_reference_is_the_recorded_fixture() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    package = resources.files("tensorlbm.verification.xval")
    manifest = json.loads(package.joinpath("refdata/manifest.json").read_text())
    fixture = package.joinpath("refdata/poiseuille_2d_H16_v1.npz")
    assert fixture.is_file()
    with np.load(fixture) as archive:
        assert manifest["cases"]["poiseuille_2d_H16"]["sha256_first_array"] == compute_state_hash(
            archive["arr_u_profile"]
        )
    assert ledger["source_and_wheel_replay"]["comparison"].endswith("byte-identical.")


def test_migrated_entrypoints_select_the_canonical_tree() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    for row in ledger["active_entrypoints"]:
        text = (ROOT / row["path"]).read_text(encoding="utf-8")
        assert 'REPO / "TensorLBM" / "src"' not in text
        assert 'REPO / "src"' in text


def test_migration_is_registered_once() -> None:
    with (ROOT / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row for row in rows
        if row["verification"] == "docs/architecture/migrations/canonical-xval-2026-11-21.json"
    ]
    assert len(matches) == 1
