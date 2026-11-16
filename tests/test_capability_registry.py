"""Contract tests for :mod:`tensorlbm.registry` (capability registry pilot).

The tests pin the properties the platform dashboard depends on:

* the static AST dump is a valid JSON contract (``source_commit`` + ``capabilities``)
  and needs neither the package import nor torch;
* import-time registration and static scanning agree field for field;
* the maturity discipline is enforced in-process (a false ``mature`` cannot even
  be registered, mirroring the server's 422 gate);
* every ``mature`` row rests on a measured error band that the run artifact in
  ``docs/m23186`` reproduces.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "tensorlbm"
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import registry as reg  # noqa: E402


@pytest.fixture(scope="module")
def static_dump():
    caps, problems = reg.scan_static(PKG_ROOT, repo_root=ROOT)
    return caps, problems


def test_static_scan_is_clean_and_populated(static_dump):
    caps, problems = static_dump
    errors = [p for p in problems if p["severity"] == "error"]
    assert errors == [], errors
    assert len(caps) >= 20, f"only {len(caps)} capabilities registered"
    ids = [c.id for c in caps]
    assert len(ids) == len(set(ids)), "duplicate capability ids in the static scan"
    # the anchor families of the pilot must all be present
    for required in (
        "lbm.collision.bgk.d3q19",
        "lbm.collision.trt.d3q19",
        "lbm.collision.rlbm.d3q19",
        "lbm.collision.mrt.d3q19",
        "lbm.collision.kbc.d3q19",
        "lbm.collision.cascaded.d3q19",
        "lbm.multiphase.shan_chen_force.d3q19",
        "lbm.phasefield.cac_hch",
        "lbm.free_surface.vof_sweep",
        "lbm.acoustics.fwh_farfield",
        "lbm.icing.rime_density",
        "lbm.amr.reflux_restrict",
        "lbm.backend.triton.suboff_step",
        "lbm.force.momentum_exchange",
        "lbm.differentiable.closure_calibration",
    ):
        assert required in ids, f"missing capability {required}"


def test_static_mode_runs_without_importing_the_package(tmp_path):
    """``python registry.py --static`` must work with torch unavailable."""
    blocker = tmp_path / "torch"
    blocker.mkdir()
    (blocker / "__init__.py").write_text(
        "raise ImportError('torch must not be imported in static mode')\n", encoding="utf-8")
    env = {"PYTHONPATH": str(blocker), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(PKG_ROOT / "registry.py"), "--static"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "static"
    assert payload["source_commit"]
    assert payload["count"] == len(payload["capabilities"]) >= 20
    assert "torch" not in proc.stderr


def test_import_mode_matches_static_mode(static_dump):
    caps, _ = static_dump
    runtime, meta = reg.import_registry(pkg_root=PKG_ROOT)
    assert meta["failed"] == [], meta["failed"]
    static = {c.id: c.to_payload() for c in caps}
    live = {c.id: c.to_payload() for c in runtime.capabilities()}
    assert sorted(static) == sorted(live), (
        f"static-only={sorted(set(static) - set(live))} runtime-only={sorted(set(live) - set(static))}")
    for cid in static:
        assert static[cid] == live[cid], f"field mismatch for {cid}"


def test_mature_requires_verified_and_wired():
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.bad", kind="collision", maturity="mature",
            module_path="tensorlbm.solver3d", entry="collide_bgk3d",
            wired=False, verified=True))


def test_planned_requires_feature_ledger_anchor():
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.roadmap", maturity="planned", module_path="tensorlbm.solver3d",
            evidence={"grade": "NONE"}))
    cap = reg.Registry().register(reg.Capability(
        id="probe.roadmap", maturity="planned", module_path="tensorlbm.solver3d",
        evidence={"grade": "NONE", "feature_ledger": "GAP-PROBE"}))
    assert cap.maturity == "planned"


def test_duplicate_ids_are_rejected():
    table = reg.Registry()
    cap = reg.Capability(id="probe.dup", maturity="developing",
                         module_path="tensorlbm.solver3d")
    table.register(cap)
    with pytest.raises(reg.DuplicateCapabilityError):
        table.register(cap)


def test_endpoint_must_be_a_wired_callable_contract():
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.endpoint", maturity="developing", module_path="tensorlbm.solver3d",
            wired=False, endpoint="api:tensorlbm.solver3d.collide_bgk3d"))
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.endpoint", maturity="developing", module_path="tensorlbm.solver3d",
            wired=True, endpoint="collide_bgk3d"))


def test_static_scan_resolves_pointers_of_a_synthetic_package(tmp_path):
    pkg = tmp_path / "probe_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(
        "def capability(*a, **k):\n"
        "    return lambda f: f\n\n"
        "from .registry_stub import noop\n" if False else
        "def collide(f):\n    return f\n\n\n"
        "def capacity_anchor():\n    pass\n", encoding="utf-8")
    src = (
        "from ._reg import capability\n\n\n"
        "@capability(id='probe.ok', kind='collision', maturity='developing',\n"
        "            wired=True, tested=False, evidence={'grade': 'MECHANICS'})\n"
        "def collide(f):\n    return f\n\n\n"
        "@capability(id='probe.ghost', kind='collision', maturity='developing',\n"
        "            entry='no_such_entry')\n"
        "def other(f):\n    return f\n"
    )
    (pkg / "mod.py").write_text(src, encoding="utf-8")
    caps, problems = reg.scan_static(pkg, package="probe_pkg", repo_root=tmp_path)
    ids = {c.id for c in caps}
    assert "probe.ok" in ids and "probe.ghost" in ids
    codes = {p["code"] for p in problems if p["severity"] == "error"}
    assert "ENTRY_UNRESOLVED" in codes, codes


def test_mature_entries_carry_a_reproducible_error_band(static_dump):
    caps, _ = static_dump
    artifact = ROOT / "docs" / "m23186" / "collision_family_verification.json"
    measured = {}
    if artifact.is_file():
        data = json.loads(artifact.read_text(encoding="utf-8"))
        measured = {f"lbm.collision.{f['collision']}.d3q19": f["max_error_pct"]
                    for f in data["families"]}
    for cap in caps:
        if cap.maturity != "mature":
            continue
        assert cap.wired and cap.verified and cap.tested, cap.id
        assert cap.evidence.get("grade") in reg.VERIFYING_GRADES, cap.id
        err = float(cap.evidence["max_error_pct"])
        assert 0.0 <= err < 15.0, f"{cap.id}: error band {err}% is not acceptable"
        if cap.id in measured:
            assert abs(err - measured[cap.id]) < 5e-3, (
                f"{cap.id}: declared {err} vs artifact {measured[cap.id]}")


def test_mrt_is_not_a_bgk_alias():
    """Anti-mock probe: MRT must respond to the s_e relaxation rate."""
    from tensorlbm.solver3d import (
        collide_bgk3d, collide_mrt3d, equilibrium3d)

    torch.manual_seed(11)
    zero = torch.zeros(2, 6, 6)
    f = equilibrium3d(torch.ones(2, 6, 6), zero, zero, zero)
    f = f + 0.05 * torch.randn_like(f)
    ref = collide_bgk3d(f, 0.8)
    a = collide_mrt3d(f, 0.8, s_e=0.5)
    b = collide_mrt3d(f, 0.8, s_e=1.9)
    scale = ref.abs().max()
    assert float((a - b).abs().max() / scale) > 1e-3, "MRT ignores s_e -> alias of BGK"
    assert float((a - ref).abs().max() / scale) > 1e-3, "MRT equals BGK -> alias"
