"""Tests for the xval module: LbtStyleConverter, CaseSchema, and cross-validation.

These tests verify:
1. LbtStyleConverter produces identical results to LatticeBridge
2. CaseSchema JSON roundtrip preserves all parameters
3. bootstrap_from_schema matches direct bootstrap_bridge
4. Anchor cross-validation passes at machine precision
5. The full xval benchmark passes
"""

import json
import math
import pytest
from pathlib import Path

from tensorlbm.units.bridge import LatticeBridge, bootstrap_bridge
from tensorlbm.units.anchors import ANCHORS, U_LB, build_case, evaluate
from tensorlbm.xval.converter import LbtStyleConverter
from tensorlbm.xval.case_schema import CaseSchema, Physics, Discretisation, bootstrap_from_schema
from tensorlbm.xval.xval_benchmark import (
    XVAL_CASES,
    _make_bridge,
    _make_lbt_converter,
    _compare_derived_quantities,
    _compare_roundtrip,
    _validate_schema_roundtrip,
    _anchor_xval,
    _check_case_viability,
    run_xval,
)


# ---------------------------------------------------------------------------
# LbtStyleConverter tests
# ---------------------------------------------------------------------------

class TestLbtStyleConverter:
    """Test the LbtStyleConverter against LatticeBridge."""

    @pytest.fixture
    def water_channel(self):
        return XVAL_CASES["poiseuille_water"]

    @pytest.fixture
    def bridge_and_lbt(self, water_channel):
        bridge = _make_bridge(water_channel)
        lbt = _make_lbt_converter(water_channel, bridge)
        return bridge, lbt

    def test_dx_matches(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        assert lbt.physical_length / lbt.equivalent_lbm_length == pytest.approx(bridge.dx, rel=1e-15)

    def test_dt_matches(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        # to_lbm_time = to_lbm_length / to_lbm_velocity = dx / (u_ref/u_lb) = dt
        assert lbt.to_lbm_time == pytest.approx(bridge.dt, rel=1e-15)

    def test_velocity_roundtrip(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        u_phys = 2.5e-3
        u_lb_bridge = bridge.velocity_to_lattice(u_phys)
        u_lb_lbt = lbt.velocity_to_lbm(u_phys)
        assert u_lb_lbt == pytest.approx(u_lb_bridge, rel=1e-15)
        rt_bridge = bridge.velocity_to_phys(u_lb_bridge)
        rt_lbt = lbt.velocity_to_phys(u_lb_lbt)
        assert rt_lbt == pytest.approx(rt_bridge, rel=1e-15)

    def test_length_roundtrip(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        x_phys = 1.0e-3
        x_lb_bridge = bridge.length_to_lattice(x_phys)
        x_lb_lbt = lbt.length_to_lbm(x_phys)
        assert x_lb_lbt == pytest.approx(x_lb_bridge, rel=1e-15)

    def test_viscosity_roundtrip(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        nu_phys = 1.0e-6
        nu_lb_bridge = bridge.viscosity_to_lattice(nu_phys)
        nu_lb_lbt = lbt.nu_to_lbm(nu_phys)
        assert nu_lb_lbt == pytest.approx(nu_lb_bridge, rel=1e-15)

    def test_time_roundtrip(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        t_phys = 1.0
        t_lb_bridge = bridge.time_to_lattice(t_phys)
        t_lb_lbt = lbt.time_to_lbm(t_phys)
        assert t_lb_lbt == pytest.approx(t_lb_bridge, rel=1e-15)

    def test_density_roundtrip(self, bridge_and_lbt):
        bridge, lbt = bridge_and_lbt
        rho_phys = 1000.0
        rho_lb_bridge = bridge.density_to_lattice(rho_phys)
        rho_lb_lbt = lbt.density_to_lbm(rho_phys)
        assert rho_lb_lbt == pytest.approx(rho_lb_bridge, rel=1e-15)

    def test_reynolds_invariant(self, bridge_and_lbt, water_channel):
        bridge, lbt = bridge_and_lbt
        re_bridge = bridge.reynolds_lattice(
            water_channel["u_lb"], float(water_channel["nx"]),
            bridge.viscosity_to_lattice(water_channel["nu"]),
        )
        re_lbt = lbt.reynolds(water_channel["u_ref"], water_channel["L"], water_channel["nu"])
        assert re_lbt == pytest.approx(re_bridge, rel=1e-15)

    def test_mach_matches(self, bridge_and_lbt, water_channel):
        bridge, lbt = bridge_and_lbt
        ma_bridge = bridge.mach(water_channel["u_lb"])
        ma_lbt = lbt.mach(water_channel["u_lb"])
        assert ma_lbt == pytest.approx(ma_bridge, rel=1e-15)

    def test_tau_matches(self, bridge_and_lbt, water_channel):
        bridge, lbt = bridge_and_lbt
        nu_lb = bridge.viscosity_to_lattice(water_channel["nu"])
        tau_bridge = bridge.tau_from_viscosity(nu_lb)
        tau_lbt = lbt.tau_from_nu_lb(nu_lb)
        assert tau_lbt == pytest.approx(tau_bridge, rel=1e-15)

    def test_summary_produces_output(self, bridge_and_lbt):
        _, lbt = bridge_and_lbt
        s = lbt.summary()
        assert "LBT-Style Unit Converter" in s
        assert "to_lbm_length" in s

    def test_all_cases_viable(self):
        """All XVAL_CASES must satisfy Ma < 0.1 and tau in [0.51, 2.0]."""
        for name, case in XVAL_CASES.items():
            info = _check_case_viability(case)
            assert info["tau_ok"], f"{name}: tau={info['tau']:.3f} outside [0.51, 2.0]"
            assert info["ma_ok"], f"{name}: Ma={info['Ma']:.4f} >= 0.1"

    def test_derived_quantities_match_for_all_cases(self):
        """All derived quantities must match at machine precision for all cases."""
        for name, case in XVAL_CASES.items():
            bridge = _make_bridge(case)
            lbt = _make_lbt_converter(case, bridge)
            derived = _compare_derived_quantities(bridge, lbt, case)
            for qty, vals in derived.items():
                assert vals["rel_diff"] < 1e-12, (
                    f"{name}/{qty}: rel_diff={vals['rel_diff']:.2e}"
                )


# ---------------------------------------------------------------------------
# CaseSchema tests
# ---------------------------------------------------------------------------

class TestCaseSchema:
    """Test the CaseSchema JSON roundtrip and bootstrap."""

    @pytest.fixture
    def sample_schema(self):
        return CaseSchema(
            physics=Physics(density=998.0, kinematic_viscosity=1.0e-6),
            discretisation=Discretisation(NX=48),
            characteristic_length=1.0e-3,
            characteristic_velocity=2.5e-3,
            u_lb=0.02,
        )

    def test_json_roundtrip(self, sample_schema):
        json_dict = sample_schema.to_json()
        reloaded = CaseSchema.from_json(json_dict)
        assert reloaded.physics.density == sample_schema.physics.density
        assert reloaded.physics.kinematic_viscosity == sample_schema.physics.kinematic_viscosity
        assert reloaded.discretisation.NX == sample_schema.discretisation.NX
        assert reloaded.characteristic_length == sample_schema.characteristic_length
        assert reloaded.characteristic_velocity == sample_schema.characteristic_velocity
        assert reloaded.u_lb == sample_schema.u_lb

    def test_json_file_roundtrip(self, sample_schema, tmp_path):
        path = tmp_path / "test_case.json"
        sample_schema.to_json_file(path)
        reloaded = CaseSchema.from_json_file(path)
        assert reloaded.physics.density == sample_schema.physics.density
        assert reloaded.discretisation.NX == sample_schema.discretisation.NX

    def test_bootstrap_matches_direct(self, sample_schema):
        bridge_schema = bootstrap_from_schema(sample_schema)
        bridge_direct = bootstrap_bridge(
            length_phys=1.0e-3,
            velocity_phys=2.5e-3,
            nu_phys=1.0e-6,
            nx=48,
            rho0=998.0,
            u_lb=0.02,
        )
        assert bridge_schema.dx == pytest.approx(bridge_direct.dx, rel=1e-15)
        assert bridge_schema.dt == pytest.approx(bridge_direct.dt, rel=1e-15)

    def test_schema_validation_passes_for_all_cases(self):
        for name, case in XVAL_CASES.items():
            result = _validate_schema_roundtrip(name, case)
            assert result["passed"], f"{name}: schema validation failed"

    def test_lb_t_style_json_keys(self, sample_schema):
        """Verify the JSON output matches LB-t settings.json key naming."""
        j = sample_schema.to_json()
        assert "physics" in j
        assert "kinematicViscosity" in j["physics"]
        assert "density" in j["physics"]
        assert "discretisation" in j
        assert "NX" in j["discretisation"]
        assert "boundaryConditions" in j
        assert "initialConditions" in j
        assert "lbmSettings" in j
        assert "output" in j
        assert "times" in j


# ---------------------------------------------------------------------------
# Anchor cross-validation tests
# ---------------------------------------------------------------------------

class TestAnchorXval:
    """Test that anchor dimensionalisation matches between converters."""

    @pytest.mark.parametrize("anchor_name", list(ANCHORS))
    def test_anchor_converter_match(self, anchor_name):
        result = _anchor_xval(anchor_name, grid=32)
        assert result["passed"], (
            f"{anchor_name}: converter_rel_diff="
            f"{result.get('converter_rel_diff', result.get('rate_converter_rel_diff', 'N/A')):.2e}"
        )

    @pytest.mark.parametrize("anchor_name", list(ANCHORS))
    def test_anchor_sim_error_within_tolerance(self, anchor_name):
        """Anchor simulation errors should be within predeclared tolerances."""
        result = _anchor_xval(anchor_name, grid=32)
        from tensorlbm.units.anchors import TOLERANCES
        assert result["anchor_sim_error"] <= TOLERANCES[anchor_name], (
            f"{anchor_name}: sim_error={result['anchor_sim_error']:.2e} "
            f"> tolerance={TOLERANCES[anchor_name]}"
        )


# ---------------------------------------------------------------------------
# Full benchmark test
# ---------------------------------------------------------------------------

class TestFullXval:
    """Test the full cross-validation benchmark."""

    def test_run_xval_passes(self, tmp_path):
        report = run_xval(out_dir=tmp_path / "xval_out")
        assert report["summary"]["overall_passed"]
        assert report["summary"]["max_cross_rel_diff"] < 0.15

    def test_report_is_json_serialisable(self, tmp_path):
        report = run_xval(out_dir=tmp_path / "xval_out")
        # Should not raise
        text = json.dumps(report, sort_keys=True, default=float)
        reloaded = json.loads(text)
        assert reloaded["summary"]["overall_passed"]

    def test_artifacts_written(self, tmp_path):
        out = tmp_path / "xval_out"
        run_xval(out_dir=out)
        assert (out / "xval_report.json").exists()
        for name in XVAL_CASES:
            assert (out / f"case_{name}.json").exists()

    def test_case_schema_files_valid(self, tmp_path):
        out = tmp_path / "xval_out"
        run_xval(out_dir=out)
        for name in XVAL_CASES:
            path = out / f"case_{name}.json"
            data = json.loads(path.read_text())
            schema = CaseSchema.from_json(data)
            assert schema.physics.density > 0
            assert schema.discretisation.NX > 0
