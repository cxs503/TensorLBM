"""Model directory as build-and-CI unit (``models/units/<dir>/model.json``).

These tests pin the properties the directory-as-unit convention is supposed to
buy, not just that the code runs:

* discovery comes from the filesystem and an incomplete directory is an error
  (a broken model can never silently drop out of CI);
* the manifest is strict — unknown keys, unknown collisions, out-of-range
  parameters and composition/kernel mismatches are rejected at load time;
* the build is deterministic and yields the same fingerprint twice;
* the CI verdict is computed from the analytic reference, and a gate that
  cannot be evaluated fails rather than being skipped;
* the public-API facade (``tensorlbm.api.run_model_unit_ci``) really calls the
  directory runner.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tensorlbm.models.ci_unit import (
    ManifestError,
    build_unit,
    discover_units,
    load_manifest,
    run_ci,
    run_unit,
    write_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_ROOT = REPO_ROOT / "models" / "units"

# The cheapest unit in the repository (periodic box, no walls); the wall-bounded
# units cost ~30 s each, which belongs in the CI target, not in the fast suite.
CHEAP_UNIT = "d2q9_bgk_shear_wave"


def _manifest(**overrides) -> dict:
    payload = json.loads((UNIT_ROOT / CHEAP_UNIT / "model.json").read_text())
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, unit_id: str, payload: dict) -> Path:
    unit_dir = tmp_path / unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    manifest = unit_dir / "model.json"
    manifest.write_text(json.dumps(payload))
    return manifest


# ---------------------------------------------------------------------------
# repository state
# ---------------------------------------------------------------------------


def test_repository_units_are_discoverable_and_valid() -> None:
    units = discover_units(UNIT_ROOT)
    assert [u.unit_id for u in units] == sorted(u.unit_id for u in units)
    assert len(units) >= 4
    for unit in units:
        assert unit.unit_id == unit.path.parent.name
        assert unit.ci.gates, f"{unit.unit_id} declares no gate"
        assert unit.composition.lattice == "D2Q9"


def test_discovery_requires_a_manifest_in_every_directory(tmp_path: Path) -> None:
    _write(tmp_path, CHEAP_UNIT, _manifest())
    (tmp_path / "half_added_model").mkdir()
    with pytest.raises(ManifestError, match="must be a buildable model unit"):
        discover_units(tmp_path)


def test_directory_name_is_the_unit_id(tmp_path: Path) -> None:
    manifest = _write(tmp_path, "some_other_dir", _manifest())
    with pytest.raises(ManifestError, match="must equal the directory name"):
        load_manifest(manifest)


# ---------------------------------------------------------------------------
# strict manifest validation
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    payload = _manifest()
    payload["ci_gates"] = []  # typo'd key must not silently weaken CI
    with pytest.raises(ManifestError, match="unknown key"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))


def test_unknown_collision_is_rejected(tmp_path: Path) -> None:
    payload = _manifest()
    payload["composition"]["collision"] = "cumulant"
    with pytest.raises(ManifestError, match="unsupported collision"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))


def test_kernel_and_composition_must_agree(tmp_path: Path) -> None:
    payload = _manifest()
    payload["composition"]["forcing"] = ["edm"]
    with pytest.raises(ManifestError, match="unforced kernel"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))


def test_out_of_range_parameters_are_rejected(tmp_path: Path) -> None:
    payload = _manifest()
    payload["ci"]["params"]["tau"] = 0.5
    with pytest.raises(ManifestError, match="tau must be > 0.5"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))

    payload = _manifest()
    payload["ci"]["params"]["u0"] = 0.4
    with pytest.raises(ManifestError, match="low-Mach"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))

    payload = _manifest()
    payload["ci"]["params"]["bogus"] = 1
    with pytest.raises(ManifestError, match="unknown kernel param"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))


def test_duplicate_gate_metric_is_rejected(tmp_path: Path) -> None:
    payload = _manifest()
    payload["ci"]["gates"].append(payload["ci"]["gates"][0])
    with pytest.raises(ManifestError, match="duplicate metric"):
        load_manifest(_write(tmp_path, CHEAP_UNIT, payload))


def test_missing_manifest_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="missing manifest"):
        load_manifest(tmp_path / "nope" / "model.json")


# ---------------------------------------------------------------------------
# build / CI behaviour on the cheapest real unit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cheap_unit():
    return load_manifest(UNIT_ROOT / CHEAP_UNIT / "model.json")


def test_build_is_deterministic(cheap_unit) -> None:
    record = build_unit(cheap_unit)
    assert record.ok, record.error
    assert record.fingerprint == record.fingerprint_repeat
    assert record.determinism is True
    assert len(record.fingerprint) == 64
    assert record.manifest_hash == cheap_unit.manifest_hash


def test_ci_verdict_is_bound_to_the_analytic_reference(cheap_unit) -> None:
    build = build_unit(cheap_unit)
    record = run_ci(cheap_unit, build)
    assert record.ok, record.error
    assert record.build_id and build.fingerprint
    # the reference is the analytic nu, not a number copied from a previous run
    assert record.reference["nu"] == pytest.approx((0.65 - 0.5) / 3.0)
    assert record.metrics["nu_rel_err"] < 1e-2
    assert {g["metric"] for g in record.gates} == {g.metric for g in cheap_unit.ci.gates}


def test_a_failing_gate_fails_the_unit(cheap_unit, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from tensorlbm.models.ci_unit import Gate

    impossible = replace(
        cheap_unit,
        ci=replace(cheap_unit.ci, gates=(Gate(metric="nu_rel_err", max=1e-12),)),
    )
    record = run_ci(impossible)
    assert record.ok is False
    assert record.gates[0]["passed"] is False


def test_missing_gated_metric_is_a_failure_not_a_skip(cheap_unit) -> None:
    from dataclasses import replace

    from tensorlbm.models.ci_unit import Gate

    bogus = replace(
        cheap_unit,
        ci=replace(cheap_unit.ci, gates=(Gate(metric="not_a_metric", max=1.0),)),
    )
    record = run_ci(bogus)
    assert record.ok is False
    assert "did not produce gated metric" in record.error


def test_artifacts_carry_build_ids_and_the_repro_command(tmp_path: Path) -> None:
    run = run_unit(load_manifest(UNIT_ROOT / CHEAP_UNIT / "model.json"))
    assert run.ok
    summary = write_artifacts([run], tmp_path, repro_cmd="python -m tensorlbm.models.ci_unit all")
    assert summary["all_ok"] is True
    unit_out = json.loads((tmp_path / "ci" / f"{CHEAP_UNIT}.json").read_text())
    assert unit_out["ok"] is True
    assert unit_out["build_id"] == run.ci.build_id
    build_out = json.loads((tmp_path / "builds" / f"{CHEAP_UNIT}.json").read_text())
    assert build_out["deterministic"] is True
    text = (tmp_path / "SUMMARY.md").read_text()
    assert "python -m tensorlbm.models.ci_unit all" in text
    assert CHEAP_UNIT in text


# ---------------------------------------------------------------------------
# integration: the public API facade is a real caller
# ---------------------------------------------------------------------------


def test_public_api_facade_runs_a_directory(tmp_path: Path) -> None:
    import tensorlbm.api as stable_api

    units = stable_api.list_model_units(UNIT_ROOT)
    assert CHEAP_UNIT in {u.unit_id for u in units}
    unit_root = tmp_path / "units"
    shutil.copytree(UNIT_ROOT / CHEAP_UNIT, unit_root / CHEAP_UNIT)
    runs = stable_api.run_model_unit_ci(unit_root)
    assert len(runs) == 1 and runs[0].ok
    summary = stable_api.write_model_unit_artifacts(runs, tmp_path / "out", repro_cmd="pytest")
    assert summary["all_ok"] is True
    assert stable_api.model_unit_build(units[0]).ok
