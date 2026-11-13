"""Tests for the YAML-driven XML template pipeline (XVAL vs MPLBM-UT).

Covers the three failure modes that the reference OSS pipeline leaves open
(f-string splicing, silent schema drift, no numeric gate):

1. ``spec -> xml -> spec`` identity, including XML metacharacters;
2. schema validation rejects unphysical / malformed cases;
3. the rendered document drives a solver whose permeability matches the exact
   plane-Poiseuille ``K = H**2/12`` inside the 15 % acceptance band, and is
   independent of viscosity and of forcing amplitude (Darcy linearity).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import yml_xml_pipeline as pipe

TOL = 0.15


def _case(ny: int = 16, omega: float = 1.7, force_x: float = 2e-5, name: str = "t") -> dict:
    return {
        "geometry": {"size": [4, ny], "period": [True, False], "solid_rows": [0, ny - 1]},
        "init": {"rho": 1.0},
        "fluids": {"omega": omega, "force_x": force_x},
        "output": {"out_folder": name, "it_max": 200000, "convergence": 1e-9},
    }


# --------------------------------------------------------------------------
# 1. round trip / escaping
# --------------------------------------------------------------------------
def test_roundtrip_is_identity():
    spec = pipe.validate_spec(_case())
    assert pipe.parse_xml(pipe.render_xml(spec)) == spec


def test_roundtrip_preserves_xml_metacharacters():
    spec = _case(name='weird_&_<value>"x\'')
    xml_text = pipe.render_xml(spec)
    assert "&amp;" in xml_text and "&lt;" in xml_text
    recovered = pipe.parse_xml(xml_text)
    assert recovered["output"]["out_folder"] == 'weird_&_<value>"x\''


def test_roundtrip_is_idempotent_under_repeated_rendering():
    spec = pipe.validate_spec(_case())
    once = pipe.render_xml(spec)
    twice = pipe.render_xml(pipe.parse_xml(once))
    assert once == twice


def test_parse_rejects_foreign_document():
    with pytest.raises(pipe.SpecError):
        pipe.parse_xml("<palabos><geometry/></palabos>")


# --------------------------------------------------------------------------
# 2. schema validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.pop("fluids"), "missing required section"),
        (lambda s: s.__setitem__("extra", {}), "unknown section"),
        (lambda s: s["fluids"].__setitem__("omega", 2.5), r"omega"),
        (lambda s: s["fluids"].__setitem__("omega", 0.0), r"omega"),
        (lambda s: s["geometry"].__setitem__("size", [4]), r"size"),
        (lambda s: s["geometry"].__setitem__("size", [4, 4]), r"ny>=5"),
        (lambda s: s["geometry"].__setitem__("solid_rows", [0, 16]), r"solid_rows"),
        (lambda s: s["geometry"].__setitem__("solid_rows", [0, 0]), r"duplicated"),
        (lambda s: (s["geometry"].__setitem__("size", [4, 5]), s["geometry"].__setitem__("solid_rows", [0, 1, 2])), r"fluid rows"),
        (lambda s: s["init"].__setitem__("rho", -1.0), r"rho"),
        (lambda s: s["output"].__setitem__("convergence", 1.5), r"convergence"),
        (lambda s: s["output"].__setitem__("out_folder", "a/b"), r"out_folder"),
        (lambda s: s["fluids"].__setitem__("force_x", 1.0), r"Mach"),
    ],
)
def test_validate_rejects_bad_spec(mutate, match):
    spec = _case()
    mutate(spec)
    with pytest.raises(pipe.SpecError, match=match):
        pipe.validate_spec(spec)


def test_validate_normalises_types():
    spec = _case()
    spec["geometry"]["size"] = ["4", "16"]
    spec["output"]["it_max"] = "2000"
    norm = pipe.validate_spec(spec)
    assert norm["geometry"]["size"] == [4, 16]
    assert norm["output"]["it_max"] == 2000
    assert isinstance(norm["init"]["rho"], float)


# --------------------------------------------------------------------------
# 3. numeric gate against the exact solution
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ny", [16, 24])
def test_permeability_matches_plane_poiseuille(ny):
    spec = pipe.validate_spec(_case(ny=ny, force_x=1e-5))
    res = pipe.run_channel(spec)
    k_ref = pipe.analytic_permeability(res["H"])
    err = abs(res["permeability"] - k_ref) / k_ref
    assert err < TOL, f"H={res['H']}: K={res['permeability']:.5f} vs ref {k_ref:.5f} ({err:.2%})"

    _, u_ref = pipe.analytic_profile(res["H"], res["nu"], spec["fluids"]["force_x"])
    u = res["u_profile"][1 : res["ny"] - 1]
    l2 = np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)
    assert l2 < TOL, f"L2 profile error {l2:.2%}"


def test_permeability_is_viscosity_independent():
    """K depends on geometry only: omega sweep must stay inside 2 %."""
    ks = []
    for omega in (1.2, 1.7):
        spec = pipe.validate_spec(_case(ny=16, omega=omega, force_x=1e-5))
        ks.append(pipe.run_channel(spec)["permeability"])
    assert abs(ks[0] - ks[1]) / ks[0] < 0.02


def test_darcy_linearity_in_forcing():
    """Doubling the body force doubles the flow rate at fixed geometry."""
    q = []
    for scale in (1.0, 2.0):
        spec = pipe.validate_spec(_case(ny=16, force_x=1e-5 * scale))
        res = pipe.run_channel(spec)
        q.append(res["u_mean"] / spec["fluids"]["force_x"])  # K = u_mean nu / g
        q[-1] *= res["nu"]
    assert abs(q[0] - q[1]) / q[0] < 0.02


# --------------------------------------------------------------------------
# 4. fixed text markers (MPLBM-UT post-processing contract)
# --------------------------------------------------------------------------
def test_run_log_markers_roundtrip():
    text = pipe.format_run_log({"iters": 1234, "permeability": 40.25, "u_peak": 0.0204})
    parsed = pipe.parse_run_log(text)
    assert parsed == {"iters": 1234.0, "permeability": 40.25, "u_peak": 0.0204}


def test_run_log_missing_marker_raises():
    with pytest.raises(pipe.SpecError, match="missing marker"):
        pipe.parse_run_log("# iters = 5\n")


def test_yaml_driven_end_to_end(tmp_path):
    """input.yml -> xml -> solver -> log -> metrics, the whole pipeline."""
    yaml = pytest.importorskip("yaml")
    spec = _case(ny=16, force_x=1e-5, name="e2e")
    yml_path = tmp_path / "input.yml"
    yml_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    loaded = pipe.load_spec(yml_path)
    xml_path = tmp_path / "solver.xml"
    xml_path.write_text(pipe.render_xml(loaded), encoding="utf-8")
    parsed_spec = pipe.parse_xml(xml_path.read_text(encoding="utf-8"))
    assert parsed_spec == loaded

    res = pipe.run_channel(parsed_spec)
    metrics = pipe.parse_run_log(pipe.format_run_log(res))
    assert math.isclose(metrics["permeability"], res["permeability"], rel_tol=1e-12)
    k_ref = pipe.analytic_permeability(res["H"])
    assert abs(metrics["permeability"] - k_ref) / k_ref < TOL


def test_runner_script_is_valid_shell(tmp_path):
    import subprocess

    spec = pipe.validate_spec(_case())
    script = tmp_path / "run.sh"
    script.write_text(pipe.render_runner_script(spec, str(tmp_path / "in.xml")), encoding="utf-8")
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr