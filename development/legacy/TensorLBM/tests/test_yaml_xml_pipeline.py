"""Tests for the YAML-driven XML template pipeline.

Covers:
1. YAML schema loading and validation
2. XML template generation
3. Pipeline execution (YAML→XML→run→verify)
4. End-to-end integration with example cases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from tensorlbm.xval.pipeline.yaml_schema import (
    XValYAMLCase,
    load_yaml_case,
    validate_yaml_case,
    GeometryConfig,
    PhysicsConfig,
    SolverConfig,
    AcceptanceCriteria,
    ReferenceConfig,
    BoundaryConfig,
    ResolutionScan,
    SCHEMA_VERSION,
)
from tensorlbm.xval.pipeline.xml_template import (
    XMLTemplateRegistry,
    generate_xml_template,
    xml_to_string,
    save_xml_template,
    get_default_registry,
)
from tensorlbm.xval.pipeline.runner import (
    YamlXmlPipeline,
    PipelineResult,
    run_pipeline,
    _stub_poiseuille_2d,
    _stub_lid_cavity,
    _stub_couette_2d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_yaml_dict():
    """Minimal valid YAML case dictionary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "anchor": "TEST-POISEUILLE-2D",
        "title": "2D Poiseuille Flow Validation",
        "description": "Plane Poiseuille flow with analytical reference",
        "category": "analytical",
        "geometry": {
            "type": "channel",
            "dimensions": "2D",
            "nx": 64,
            "ny": 32,
            "lx": 4.0,
            "ly": 1.0,
        },
        "physics": {
            "model": "bgk",
            "lattice": "D2Q9",
            "reynolds": 100.0,
            "density": 1.0,
        },
        "boundary": {
            "walls": [
                {"location": "top", "type": "bounce_back"},
                {"location": "bottom", "type": "bounce_back"},
            ],
            "inlet": {"type": "pressure", "value": 1.0},
            "outlet": {"type": "pressure", "value": 0.9},
        },
        "solver": {
            "max_steps": 5000,
            "convergence_threshold": 1e-6,
            "output_interval": 100,
            "seed": 42,
            "precision": "float64",
        },
        "reference": {
            "type": "analytical",
            "citation": "Schlichting & Gersten, Boundary-Layer Theory, §III.2.1",
            "doi": "",
            "solution_fn": "poiseuille_2d",
        },
        "acceptance": {
            "metrics": {
                "eps_inf": 0.02,
                "eps_l2": 0.02,
            },
            "stopping_rule": "Both error norms must be below threshold",
        },
        "tags": ["poiseuille", "analytical", "2d"],
        "external_anchor": "Schlichting & Gersten (2017)",
    }


@pytest.fixture
def sample_yaml_file(sample_yaml_dict, tmp_path):
    """Write sample YAML dict to a temp file."""
    yaml_path = tmp_path / "test_case.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(sample_yaml_dict, f)
    return yaml_path


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline with temp output dir."""
    return YamlXmlPipeline(output_dir=tmp_path / "output")


# ---------------------------------------------------------------------------
# YAML Schema Tests
# ---------------------------------------------------------------------------

class TestYAMLSchema:
    """Tests for YAML schema loading and validation."""

    def test_validate_valid_case(self, sample_yaml_dict):
        errors = validate_yaml_case(sample_yaml_dict)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_validate_missing_required_key(self, sample_yaml_dict):
        del sample_yaml_dict["geometry"]
        errors = validate_yaml_case(sample_yaml_dict)
        assert any("geometry" in e for e in errors)

    def test_validate_invalid_dimensions(self, sample_yaml_dict):
        sample_yaml_dict["geometry"]["dimensions"] = "1D"
        errors = validate_yaml_case(sample_yaml_dict)
        assert any("dimensions" in e for e in errors)

    def test_validate_invalid_lattice(self, sample_yaml_dict):
        sample_yaml_dict["physics"]["lattice"] = "D2Q5"
        errors = validate_yaml_case(sample_yaml_dict)
        assert any("lattice" in e for e in errors)

    def test_validate_negative_threshold(self, sample_yaml_dict):
        sample_yaml_dict["acceptance"]["metrics"]["eps_inf"] = -0.01
        errors = validate_yaml_case(sample_yaml_dict)
        assert any("eps_inf" in e for e in errors)

    def test_load_yaml_case(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        assert isinstance(case, XValYAMLCase)
        assert case.anchor == "TEST-POISEUILLE-2D"
        assert case.geometry.nx == 64
        assert case.geometry.ny == 32
        assert case.physics.model == "bgk"
        assert case.physics.lattice == "D2Q9"
        assert case.solver.max_steps == 5000
        assert case.acceptance.metrics["eps_inf"] == 0.02

    def test_load_yaml_case_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_case("/nonexistent/path/case.yaml")

    def test_load_yaml_case_invalid_yaml(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("not: a: valid: yaml: [")
        with pytest.raises(Exception):
            load_yaml_case(bad_yaml)

    def test_xval_yaml_case_to_dict(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        d = case.to_dict()
        assert d["anchor"] == "TEST-POISEUILLE-2D"
        assert d["geometry"]["nx"] == 64

    def test_xval_yaml_case_to_yaml(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        yaml_str = case.to_yaml()
        assert "TEST-POISEUILLE-2D" in yaml_str
        assert "bgk" in yaml_str

    def test_resolution_scan(self, sample_yaml_dict):
        sample_yaml_dict["resolution_scan"] = {
            "variable": "nx",
            "values": [32, 64, 128, 256],
        }
        errors = validate_yaml_case(sample_yaml_dict)
        assert errors == []
        case = XValYAMLCase.from_dict(sample_yaml_dict)
        assert case.resolution_scan.variable == "nx"
        assert case.resolution_scan.values == [32, 64, 128, 256]


# ---------------------------------------------------------------------------
# XML Template Tests
# ---------------------------------------------------------------------------

class TestXMLTemplate:
    """Tests for XML template generation."""

    def test_generate_bgk_d2q9_template(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        assert root.tag == "lbm_case"
        assert root.get("version") == "1.0"
        assert root.get("generator") == "yaml-xml-pipeline"

    def test_xml_contains_metadata(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        meta = root.find("metadata")
        assert meta is not None
        assert meta.find("anchor").text == "TEST-POISEUILLE-2D"
        assert meta.find("title").text == "2D Poiseuille Flow Validation"

    def test_xml_contains_geometry(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        geo = root.find("geometry")
        assert geo is not None
        assert geo.get("type") == "channel"
        assert geo.get("dimensions") == "2D"
        assert geo.find("resolution").text == "64x32"

    def test_xml_contains_physics(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        phys = root.find("physics")
        assert phys is not None
        assert phys.get("model") == "bgk"
        assert phys.get("lattice") == "D2Q9"
        assert phys.find("reynolds").text == "100.0"

    def test_xml_contains_solver(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        solv = root.find("solver")
        assert solv is not None
        assert solv.get("max_steps") == "5000"
        assert solv.get("precision") == "float64"

    def test_xml_contains_acceptance(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        acc = root.find("acceptance")
        assert acc is not None
        metrics = acc.findall("metric")
        assert len(metrics) == 2

    def test_xml_to_string(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        root = generate_xml_template(case)
        xml_str = xml_to_string(root)
        assert "<?xml" in xml_str
        assert "lbm_case" in xml_str
        assert "TEST-POISEUILLE-2D" in xml_str

    def test_save_xml_template(self, sample_yaml_file, tmp_path):
        case = load_yaml_case(sample_yaml_file)
        out_path = tmp_path / "output" / "test.xml"
        save_xml_template(case, out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "TEST-POISEUILLE-2D" in content

    def test_no_template_for_unknown_model(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        case.physics.model = "unknown_model"
        registry = XMLTemplateRegistry()
        with pytest.raises(ValueError, match="No XML template registered"):
            generate_xml_template(case, registry)

    def test_default_registry_has_templates(self):
        reg = get_default_registry()
        assert reg.has_template("bgk", "D2Q9")
        assert reg.has_template("bgk", "D3Q19")
        assert reg.has_template("mrt", "D3Q19")
        assert not reg.has_template("unknown", "D2Q9")

    def test_register_custom_template(self):
        import xml.etree.ElementTree as ET
        reg = XMLTemplateRegistry()

        def my_template(case):
            root = ET.Element("custom_case")
            ET.SubElement(root, "name").text = case.anchor
            return root

        reg.register("custom", "D2Q9", my_template)
        assert reg.has_template("custom", "D2Q9")

        case = XValYAMLCase(
            anchor="TEST-CUSTOM",
            title="Custom",
            geometry=GeometryConfig(type="channel", dimensions="2D", nx=32, ny=32),
            physics=PhysicsConfig(model="custom", lattice="D2Q9"),
            solver=SolverConfig(max_steps=100),
        )
        root = generate_xml_template(case, reg)
        assert root.tag == "custom_case"


# ---------------------------------------------------------------------------
# Pipeline Runner Tests
# ---------------------------------------------------------------------------

class TestPipelineRunner:
    """Tests for the pipeline runner."""

    def test_run_pipeline_from_yaml(self, sample_yaml_file, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(sample_yaml_file, output_dir=output_dir)
        assert isinstance(result, PipelineResult)
        assert result.anchor == "TEST-POISEUILLE-2D"
        assert result.xml_path != ""
        assert Path(result.xml_path).exists()
        assert "eps_inf" in result.metrics
        assert "eps_l2" in result.metrics

    def test_pipeline_result_passed(self, sample_yaml_file, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(sample_yaml_file, output_dir=output_dir)
        # With nx=64, ny=32, the stub solver should produce errors < 0.02
        assert result.passed, f"Expected PASS but got FAIL. Metrics: {result.metrics}"

    def test_pipeline_saves_artifacts(self, sample_yaml_file, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(sample_yaml_file, output_dir=output_dir)
        artifacts_dir = output_dir / "artifacts"
        assert artifacts_dir.exists()
        assert (artifacts_dir / "TEST-POISEUILLE-2D_result.json").exists()
        assert (artifacts_dir / "TEST-POISEUILLE-2D_verdict.json").exists()

    def test_pipeline_result_json_serializable(self, sample_yaml_file, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(sample_yaml_file, output_dir=output_dir)
        d = result.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert "TEST-POISEUILLE-2D" in json_str

    def test_pipeline_invalid_yaml(self, tmp_path):
        """Test pipeline with YAML that fails schema validation (valid YAML syntax)."""
        bad_yaml = tmp_path / "bad.yaml"
        # Valid YAML but missing required keys
        bad_yaml.write_text("foo: bar\nbaz: 123\n")
        output_dir = tmp_path / "output"
        result = run_pipeline(bad_yaml, output_dir=output_dir)
        assert not result.passed
        assert len(result.errors) > 0

    def test_pipeline_missing_yaml(self, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(tmp_path / "nonexistent.yaml", output_dir=output_dir)
        assert not result.passed
        assert len(result.errors) > 0

    def test_stub_poiseuille_2d(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        result = _stub_poiseuille_2d(case)
        assert "metrics" in result
        assert "eps_inf" in result["metrics"]
        assert "eps_l2" in result["metrics"]
        assert result["metrics"]["eps_inf"] < 0.02
        assert result["metrics"]["eps_l2"] < 0.02

    def test_stub_lid_cavity(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        case.geometry.nx = 65
        case.geometry.ny = 65
        case.physics.reynolds = 100.0
        result = _stub_lid_cavity(case)
        assert "metrics" in result
        assert "l2_u" in result["metrics"]
        assert "l2_v" in result["metrics"]

    def test_stub_couette_2d(self, sample_yaml_file):
        case = load_yaml_case(sample_yaml_file)
        case.geometry.ny = 32
        result = _stub_couette_2d(case)
        assert "metrics" in result
        # Couette should be nearly exact (machine epsilon level)
        assert result["metrics"]["eps_inf"] < 1e-7
        assert result["metrics"]["eps_l2"] < 1e-7

    def test_pipeline_summary_line(self, sample_yaml_file, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(sample_yaml_file, output_dir=output_dir)
        line = result.summary_line()
        assert "TEST-POISEUILLE-2D" in line
        assert "PASS" in line or "FAIL" in line

    def test_pipeline_elapsed_time(self, sample_yaml_file, tmp_path):
        output_dir = tmp_path / "output"
        result = run_pipeline(sample_yaml_file, output_dir=output_dir)
        assert result.elapsed_seconds > 0
        assert result.elapsed_seconds < 10.0  # Should be fast

    def test_pipeline_with_custom_solver(self, sample_yaml_file, tmp_path):
        def custom_solver(case):
            return {"metrics": {"eps_inf": 0.001, "eps_l2": 0.0005}}

        output_dir = tmp_path / "output"
        pipeline = YamlXmlPipeline(
            output_dir=output_dir,
            solver_fn=custom_solver,
        )
        result = pipeline.run_from_yaml(sample_yaml_file)
        assert result.passed
        assert result.metrics["eps_inf"] == 0.001


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_roundtrip(self, sample_yaml_dict, tmp_path):
        """Write YAML → load → generate XML → run pipeline → check results."""
        yaml_path = tmp_path / "case.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(sample_yaml_dict, f)

        output_dir = tmp_path / "output"
        result = run_pipeline(yaml_path, output_dir=output_dir)

        # Check XML was generated
        assert Path(result.xml_path).exists()
        xml_content = Path(result.xml_path).read_text()
        assert "TEST-POISEUILLE-2D" in xml_content

        # Check result artifacts
        assert (output_dir / "artifacts" / "TEST-POISEUILLE-2D_result.json").exists()
        assert (output_dir / "artifacts" / "TEST-POISEUILLE-2D_verdict.json").exists()

        # Check verdict
        with open(output_dir / "artifacts" / "TEST-POISEUILLE-2D_verdict.json") as f:
            verdict = json.load(f)
        assert verdict["anchor"] == "TEST-POISEUILLE-2D"
        assert verdict["verdict"] in ("PASS", "FAIL")

    def test_multiple_cases(self, tmp_path):
        """Test running pipeline on multiple YAML files."""
        cases = [
            {
                "schema_version": SCHEMA_VERSION,
                "anchor": "TEST-CASE-A",
                "title": "Case A",
                "geometry": {"type": "channel", "dimensions": "2D", "nx": 32, "ny": 32},
                "physics": {"model": "bgk", "lattice": "D2Q9", "reynolds": 50.0},
                "solver": {"max_steps": 1000, "seed": 42},
                "acceptance": {"metrics": {"eps_inf": 0.05}},
            },
            {
                "schema_version": SCHEMA_VERSION,
                "anchor": "TEST-CASE-B",
                "title": "Case B",
                "geometry": {"type": "cavity", "dimensions": "2D", "nx": 64, "ny": 64},
                "physics": {"model": "bgk", "lattice": "D2Q9", "reynolds": 100.0},
                "solver": {"max_steps": 2000, "seed": 42},
                "acceptance": {"metrics": {"l2_u": 0.15, "l2_v": 0.15}},
            },
        ]

        output_dir = tmp_path / "output"
        results = []
        for i, case_dict in enumerate(cases):
            yaml_path = tmp_path / f"case_{i}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(case_dict, f)
            result = run_pipeline(yaml_path, output_dir=output_dir)
            results.append(result)

        assert len(results) == 2
        assert results[0].anchor == "TEST-CASE-A"
        assert results[1].anchor == "TEST-CASE-B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
