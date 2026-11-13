"""Model-dir-as-build-and-CI-unit: registry, generation and per-unit verification.

Cross-validation target: CFD-GO/TCLB.  The properties asserted here are the
TCLB invariants ported to this repository, not implementation details:

* the model list is *enumerated* (``find``/``git ls-files``), so a new directory
  is on the build and CI path with no edit anywhere else;
* the CI matrix and the per-model Makefile targets are *generated* from the
  manifests (TCLB regenerates ``makefile.main`` from ``models/*``);
* every unit is independently buildable and carries its own reference + tolerance;
* the workflow must not contain a hand-written unit list (TCLB's drift mode: its
  ``models.R`` enumerates 61 model directories while ``cpu_test.yml`` lists 16).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from tensorlbm.model_units import build, ci_matrix, discover, doctor, makefile_targets
from tensorlbm.model_units import test as run_unit_case
from tensorlbm.model_units.manifest import ManifestError, parse_manifest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "models" / "lbm"


@pytest.fixture(scope="module")
def units():
    found = discover(ROOT, repo_root=REPO)
    assert found, "no model units discovered"
    return found


def test_discovery_matches_directories(units):
    dirs = sorted(p.parent.name for p in ROOT.glob("*/model.toml"))
    assert sorted(u.name for u in units) == dirs
    for unit in units:
        assert unit.name == unit.path.name  # directory *is* the unit name
        assert unit.fingerprint()  # manifest + case + kernel hashes are populated


def test_units_span_more_than_one_lattice(units):
    lattices = {u.lattice for u in units}
    assert {"D2Q9", "D3Q19"} <= lattices
    assert {u.dimension for u in units} == {2, 3}


def test_ci_matrix_is_generated_not_listed(units):
    matrix = ci_matrix(units)
    expected = sorted(u.name for u in units if u.ci)
    assert sorted(cell["unit"] for cell in matrix["include"]) == expected
    for cell in matrix["include"]:
        assert cell["cmd"].endswith(f"--unit {cell['unit']}")
        assert cell["lattice"] and cell["dimension"] and cell["tier"]


def test_makefile_targets_one_per_unit(units):
    text = makefile_targets(units)
    for unit in units:
        assert f"model-{unit.name}:" in text
        assert f"--unit {unit.name}" in text


def test_new_directory_needs_no_edit_elsewhere(tmp_path):
    """Adding a model == adding a directory (TCLB's unit-of-work invariant)."""
    source = ROOT / "tgv2d_d2q9_bgk"
    target = tmp_path / "models" / "lbm" / "extra_tgv2d_d2q9_bgk"
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    manifest = (
        (target / "model.toml")
        .read_text()
        .replace('name = "tgv2d_d2q9_bgk"', 'name = "extra_tgv2d_d2q9_bgk"')
    )
    (target / "model.toml").write_text(manifest)

    units = discover(target.parent, repo_root=tmp_path)  # git is absent here -> filesystem path
    assert [u.name for u in units] == ["extra_tgv2d_d2q9_bgk"]
    matrix = ci_matrix(units)
    assert [c["unit"] for c in matrix["include"]] == ["extra_tgv2d_d2q9_bgk"]


def test_manifests_reject_broken_units(tmp_path):
    good = (ROOT / "tgv2d_d2q9_bgk" / "model.toml").read_text()

    def write(dirname: str, text: str, *, keep_name: bool = False) -> Path:
        d = tmp_path / dirname
        d.mkdir()
        (d / "case.py").write_text("def run(params):\n    return {'kpis': {}}\n")
        if not keep_name:  # keep the directory-name invariant satisfied
            text = re.sub(r'^name = ".*"$', f'name = "{dirname}"', text, count=1, flags=re.M)
        (d / "model.toml").write_text(text)
        return d / "model.toml"

    # directory name and unit.name must agree (the directory is the unit id)
    with pytest.raises(ManifestError, match="must equal the directory name"):
        parse_manifest(write("wrong_name", good, keep_name=True))
    # unknown lattice
    with pytest.raises(ManifestError, match="unsupported lattice"):
        parse_manifest(write("bad_lattice", good.replace('"D2Q9"', '"D2Q37"')))
    # 2D/3D consistency
    with pytest.raises(ManifestError, match="is not 3D"):
        parse_manifest(write("bad_dim", good.replace("dimension = 2", "dimension = 3")))
    # missing case script
    bad = good.replace('script = "case.py"', 'script = "missing.py"')
    with pytest.raises(ManifestError, match="case script"):
        parse_manifest(write("missing_case", bad))
    # non-relative tolerance
    bad = good.replace("threshold = 0.15", "threshold = 15.0")
    with pytest.raises(ManifestError, match="relative tolerance"):
        parse_manifest(write("bad_threshold", bad))


def test_workflow_has_no_hand_written_unit_list(units):
    text = (REPO / ".github" / "workflows" / "model-units.yml").read_text()
    assert "fromJSON(needs.plan.outputs.matrix)" in text
    hand_listed = [u.name for u in units if u.name in text]
    assert hand_listed == [], f"unit names must not be written in the workflow: {hand_listed}"


def test_doctor_passes(units):
    report = doctor(ROOT, repo_root=REPO)
    failures = [c for c in report["checks"] if not c["ok"]]
    assert report["ok"], failures
    assert {c["check"] for c in report["checks"]} >= {
        "discovery_union_complete",
        "every_ci_unit_has_case",
        "workflow_matrix_not_hand_written",
        "workflow_uses_generated_matrix",
    }


@pytest.mark.slow
def test_build_is_isolated_and_deterministic(units, tmp_path):
    """A unit's build must import its kernel, not the whole library."""
    unit = next(u for u in units if u.name == "tgv2d_d2q9_bgk")
    first = build(unit, out_root=tmp_path)
    second = build(unit, out_root=tmp_path)
    closure = set(first["kernel_import"]["closure"])
    assert first["kernel_import"]["finite"] is True
    assert first["kernel_import"]["feq_checksum"] == second["kernel_import"]["feq_checksum"]
    # the unit's own kernel is in; the build must not drag in the heavy,
    # unrelated parts of the library (AI/apps/backends)
    assert unit.kernel_source in closure
    forbidden = ("tensorlbm.ai", "tensorlbm.apps", "tensorlbm.backends")
    assert not [m for m in closure if m.startswith(forbidden)]
    assert first["kernel_import"]["n_modules"] <= 20
    assert first["library_surface_modules"] > 10 * first["kernel_import"]["n_modules"]
    artifact = json.loads(Path(first["artifact"]).read_text())
    assert artifact["unit"]["unit"] == unit.unit_id
    assert artifact["unit"]["fingerprint"] == unit.fingerprint()


@pytest.mark.slow
def test_every_unit_meets_its_declared_reference(units, tmp_path):
    """Each model directory is judged by its own case and its own threshold."""
    failures = []
    for unit in units:
        record = run_unit_case(unit, out_root=tmp_path)
        threshold = float(unit.acceptance["threshold"])
        assert threshold == 0.15  # owner accuracy standard: >15% is not acceptable
        if record["status"] != "passed":
            failures.append((unit.unit_id, record["kpi"], record["value"]))
    assert failures == [], failures
    # the unit results must carry provenance, not just a verdict
    for unit in units:
        payload = json.loads((tmp_path / unit.name / "result.json").read_text())
        assert payload["unit"]["fingerprint"] == unit.fingerprint()
        assert payload["kpi"] in payload["kpis"]
        assert payload["curves"] or payload["metrics"]
