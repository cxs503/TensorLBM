"""Tests: YAML-driven XML template pipeline wired to the REAL TensorLBM solver.

Anchor: ``XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE``.

The pipeline's legacy built-in solvers are demonstration stubs (synthesised
metrics, no lattice).  These tests pin the real-solver wiring:

1. YAML -> ChannelCase mapping is exact (geometry/physics/boundary/solver).
2. End-to-end pipeline run with ``reference.solution_fn: poiseuille_zou_he_2d``
   executes a genuine D2Q9 BGK + Zou-He channel simulation and the verdict is
   computed from the converged field vs the exact plane-Poiseuille solution
   (15 % acceptance bound, project standard).
3. The verdict/result artifacts record which solver produced the metrics, so a
   stub verdict can never be mistaken for a real run.
4. Cases without a registered ``solution_fn`` still fall back to the legacy
   stub path (backward compatibility).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tensorlbm.xval.pipeline import YamlXmlPipeline
from tensorlbm.xval.pipeline.real_solvers import (
    REAL_SOLVER_REGISTRY,
    _channel_case_from_yaml,
    resolve_real_solver,
)
from tensorlbm.xval.pipeline.yaml_schema import XValYAMLCase, load_yaml_case

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_YAML = REPO_ROOT / "cases" / "xval" / "poiseuille_zou_he_2d.yaml"


def _small_case_dict() -> dict:
    """Small-grid real-solver case (fast CI run)."""
    return {
        "schema_version": "1.0",
        "anchor": "TEST-PIPELINE-REAL-POISEUILLE",
        "title": "Pipeline real-solver smoke case",
        "geometry": {"type": "channel", "dimensions": "2D", "nx": 48, "ny": 14},
        "physics": {"model": "bgk", "lattice": "D2Q9", "viscosity": 0.05},
        "boundary": {
            "walls": [
                {"location": "top", "type": "bounce_back"},
                {"location": "bottom", "type": "bounce_back"},
            ],
            "inlet": {"type": "pressure", "value": 1.002},
            "outlet": {"type": "pressure", "value": 0.998},
        },
        "solver": {
            "max_steps": 8000,
            "convergence_threshold": 1.0e-5,
            "burn_in": 1000,
            "ramp_steps": 1000,
            "avg_window": 200,
            "check_every": 200,
            "seed": 42,
            "precision": "float64",
        },
        "reference": {
            "type": "analytical",
            "citation": "exact plane Poiseuille",
            "solution_fn": "poiseuille_zou_he_2d",
        },
        "acceptance": {"metrics": {"eps_inf": 0.15, "eps_l2": 0.15}},
    }


def _write_yaml(tmp_path: Path, d: dict) -> Path:
    p = tmp_path / "case.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


# ---------------------------------------------------------------------------
# 1. YAML -> ChannelCase mapping
# ---------------------------------------------------------------------------

def test_yaml_to_channel_case_mapping(tmp_path):
    d = _small_case_dict()
    case = XValYAMLCase.from_dict(d)
    cc = _channel_case_from_yaml(case)
    assert cc.height == 14 - 2          # ny includes two solid wall rows
    assert cc.length == 48
    assert cc.tau == pytest.approx(3.0 * 0.05 + 0.5)
    assert cc.drho == pytest.approx(1.002 - 0.998)
    assert cc.arm == "pressure-pressure"
    assert cc.max_steps == 8000
    assert cc.tol == pytest.approx(1.0e-5)
    assert cc.burn_in == 1000
    assert cc.ramp_steps == 1000
    assert cc.avg_window == 200
    assert cc.check_every == 200


def test_yaml_to_channel_case_rejects_bad_geometry(tmp_path):
    d = _small_case_dict()
    d["geometry"]["ny"] = 4            # height 2 < 4
    case = XValYAMLCase.from_dict(d)
    with pytest.raises(ValueError, match="too small"):
        _channel_case_from_yaml(case)


def test_yaml_to_channel_case_rejects_non_bgk_d2q9(tmp_path):
    d = _small_case_dict()
    d["physics"]["lattice"] = "D3Q19"
    case = XValYAMLCase.from_dict(d)
    with pytest.raises(ValueError, match="bgk"):
        _channel_case_from_yaml(case)


def test_resolve_real_solver_selection():
    d = _small_case_dict()
    case = XValYAMLCase.from_dict(d)
    fn = resolve_real_solver(case)
    assert fn is not None
    assert "poiseuille_zou_he_2d" in REAL_SOLVER_REGISTRY

    d2 = _small_case_dict()
    d2["reference"]["solution_fn"] = "unknown_fn"
    assert resolve_real_solver(XValYAMLCase.from_dict(d2)) is None

    d3 = _small_case_dict()
    d3["reference"]["solution_fn"] = None
    assert resolve_real_solver(XValYAMLCase.from_dict(d3)) is None


# ---------------------------------------------------------------------------
# 2. End-to-end pipeline run with the REAL solver (small grid, fast)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pipeline_result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline_real")
    yaml_path = _write_yaml(tmp, _small_case_dict())
    pipeline = YamlXmlPipeline(output_dir=tmp / "out")
    result = pipeline.run_from_yaml(yaml_path)
    return tmp, yaml_path, result


def test_real_pipeline_runs_genuine_solver(real_pipeline_result):
    _, _, result = real_pipeline_result
    assert result.errors == [], result.errors
    assert result.details["solver"] == "real:zou_he_channel"
    # A real run must have actually marched the lattice.
    assert result.details["steps"] > 0
    # The soft Zou-He pressure closure leaves a small acoustic ripple, so the
    # pointwise drift criterion is telemetry, not the verdict basis; the mean
    # field must nonetheless be settled (drift between consecutive blocks tiny).
    assert result.details["drift_last"] < 1.0e-2
    # Gradient consistency: measured dp/dx within 5 % of the nominal value.
    assert result.details["g_rel_dev"] < 0.05


def test_real_pipeline_metrics_vs_analytic(real_pipeline_result):
    _, _, result = real_pipeline_result
    m = result.metrics
    assert m["eps_inf"] < 0.15, m      # owner acceptance bound
    assert m["eps_l2"] < 0.15, m
    assert m["rel_umax"] < 0.15, m
    assert result.passed is True


def test_real_pipeline_profile_is_parabolic(real_pipeline_result):
    _, _, result = real_pipeline_result
    prof = result.details["profiles"]
    u = prof["u_sim"]
    n = len(u)
    # Parabolic shape: centre well above the near-wall values.
    assert u[n // 2] > 2.0 * max(abs(u[0]), abs(u[-1]), 1e-12)
    # And the reference profile is the exact parabola: unimodal, symmetric,
    # with the maximum at the two central rows (even fluid-row count).
    uref = prof["u_ref"]
    assert max(uref) == max(uref[n // 2 - 1], uref[n // 2])
    assert all(uref[i] < uref[i + 1] for i in range(n // 2 - 1))
    assert all(uref[i] > uref[i + 1] for i in range(n // 2, n - 1))


def test_real_pipeline_artifacts_record_solver(real_pipeline_result):
    tmp, _, result = real_pipeline_result
    art = tmp / "out" / "artifacts"
    res_json = json.loads((art / f"{result.anchor}_result.json").read_text())
    ver_json = json.loads((art / f"{result.anchor}_verdict.json").read_text())
    assert res_json["details"]["solver"] == "real:zou_he_channel"
    assert ver_json["verdict"] == "PASS"
    prof_path = art / f"{result.anchor}_profiles.json"
    assert prof_path.exists()
    prof = json.loads(prof_path.read_text())
    assert set(prof) == {"y", "u_sim", "u_ref"}
    assert len(prof["y"]) == 12        # ny - 2 fluid rows


def test_stub_fallback_still_available(tmp_path):
    """Cases without a registered solution_fn keep the legacy stub path."""
    d = _small_case_dict()
    d["reference"]["solution_fn"] = None
    d["anchor"] = "TEST-PIPELINE-STUB-FALLBACK"
    yaml_path = _write_yaml(tmp_path, d)
    pipeline = YamlXmlPipeline(output_dir=tmp_path / "out")
    result = pipeline.run_from_yaml(yaml_path)
    assert result.details["solver"] == "stub"
    assert "eps_inf" in result.metrics
