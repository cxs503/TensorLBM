"""Mission m183 regression tests for the laplace_droplet sweep driver/artifacts.

These tests do **not** re-run the physics benchmark (that is minutes-scale); they
pin the parts that made mission 25's failure silent and that must not rot:

1. ``test_measure_parity`` — the mission's band-parameterised measurement must
   reproduce the case's own ``measure()`` exactly at the canonical bands
   (guards against criterion drift between driver and case);
2. ``test_fit_laplace_reference`` — the case's ``fit_laplace`` applied to the
   *stored verified* rows must return the stored sigma_eff (guards the analysis
   path used to judge the sweep);
3. ``test_sweep_artifacts`` — the produced artifacts must exist, be internally
   consistent, and the 2D sweep must reproduce the stored verified baseline
   within the 3% gate (skipped when the artifacts are absent, e.g. in a clean
   checkout without the sweeps).

Run: python -m pytest tests/test_m183_laplace_droplet_sweep.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "laplace_droplet"
ART = REPO / "artifacts" / "m183_laplace_droplet"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def case():
    assert (CASE_DIR / "run.py").exists(), f"missing case run.py in {CASE_DIR}"
    return _load("laplace_case_for_test", CASE_DIR / "run.py")


@pytest.fixture(scope="module")
def sweep():
    assert (CASE_DIR / "sweep_m183.py").exists(), "missing sweep_m183.py"
    return _load("m183_sweep_for_test", CASE_DIR / "sweep_m183.py")


def test_measure_parity(case, sweep):
    """Our band-parameterised measure == the case's measure at 0.5/1.5 bands."""
    torch.set_num_threads(2)
    dim, L, R = 2, 32, 8
    rho = case.init_rho(dim, L, R, torch.device("cpu"))
    z = torch.zeros_like(rho)
    f = case.equilibrium(rho, z, z)
    for _ in range(20):
        f = case.stream(case.collide_sc_single_component(
            f, G=case.G_LIB, tau=case.TAU, psi_fn=case.psi_exp))
    rr = case._radius_field(dim, L, torch.device("cpu"))
    ours = sweep.measure_bands(dim, f, R, rr)          # canonical 0.5/1.5
    ref = case.measure(dim, f, R, rr, torch.device("cpu"))
    for key in ("dp", "R_eq", "p_in", "p_out", "rho_in", "rho_out", "max_u", "mass"):
        assert ours[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key


def test_fit_laplace_reference(case):
    """fit_laplace on the stored verified rows reproduces the stored sigma_eff."""
    stored = json.loads((CASE_DIR / "result.json").read_text(encoding="utf-8"))
    for tag, dim in (("2d", 2), ("3d", 3)):
        base = stored["dims"][tag]
        rows = [dict(p) for p in base["per_radius"]]
        fit = case.fit_laplace(rows, dim)
        assert fit["sigma_eff_fit"] == pytest.approx(base["sigma_eff_fit"], rel=1e-9)
        assert fit["max_dev_sigma_pct"] == pytest.approx(base["max_dev_sigma_pct"], rel=1e-9)
        assert fit["pass_laplace"] is True


@pytest.mark.skipif(not (ART / "sweep_result.json").exists(),
                    reason="m183 sweep artifacts not present in this checkout")
def test_sweep_artifacts():
    res = json.loads((ART / "sweep_result.json").read_text(encoding="utf-8"))
    # 1) 2D must pass the case criterion and reproduce the stored baseline
    d2 = res["dims"]["2d"]
    assert d2["pass_laplace"] is True
    assert d2["max_dev_sigma_pct"] < 3.0
    assert d2["R2"] >= 0.999
    assert d2["vs_stored_result_json"]["sigma_eff_rel_diff_pct"] < 3.0
    for row in d2["vs_stored_result_json"]["matched_radii"]:
        assert row["rel_diff_pct"] < 3.0, row
    # 2) every fitted run must be flagged converged, and nothing unconverged may
    #    silently enter the regression
    for tag in ("2d", "3d"):
        d = res["dims"].get(tag, {})
        for p in d.get("per_radius", []):
            assert p["converged"] is True, (tag, p["R_init"])
        assert d.get("n_radii", 0) == len(d.get("per_radius", []))
    # 3) a Laplace-linearity pass is only claimed with >= 3 radii
    for tag, v in res["verdict"].items():
        if v.get("pass_laplace_strict"):
            assert v["n_radii"] >= 3
            assert v["max_dev_sigma_pct"] < 3.0
    # 4) the mechanical-route cross-check is recorded as an honest negative
    mech = json.loads((ART / "mechroute_variants.json").read_text(encoding="utf-8"))
    assert mech["verdict"].startswith("FAIL")
    assert all(r["v1_rel_diff_pct"] > 3.0 for r in mech["rows"])
    # 5) measurement self-check between driver and case must be (numerically) exact
    for p in res["per_run"]:
        if p["selfcheck_vs_case_measure_relerr"] is not None:
            assert p["selfcheck_vs_case_measure_relerr"] < 1e-6