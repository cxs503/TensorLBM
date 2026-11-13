"""Tests for the continuous-error-vs-DOI-refs cross-validation framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure TensorLBM is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.verification.xval.continuous_error import (
    DOIReference,
    ContinuousErrorResult,
    compute_continuous_errors,
    print_error_report,
    save_artifacts,
)


class TestComputeContinuousErrors:
    """Tests for compute_continuous_errors."""

    def test_identical_solutions(self):
        """Identical solutions should have zero error."""
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        l2_rel, linf_rel, l2_abs, linf_abs, n, passed = compute_continuous_errors(u, u, 0.15)
        assert l2_rel == 0.0
        assert linf_rel == 0.0
        assert l2_abs == 0.0
        assert linf_abs == 0.0
        assert n == 5
        assert passed

    def test_small_error_passes(self):
        """Small errors should pass the threshold."""
        u_num = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u_ref = np.array([1.01, 2.02, 2.98, 4.01, 4.99])
        l2_rel, linf_rel, l2_abs, linf_abs, n, passed = compute_continuous_errors(
            u_num, u_ref, 0.15
        )
        assert l2_rel < 0.01  # ~0.4%
        assert linf_rel < 0.01
        assert passed

    def test_large_error_fails(self):
        """Large errors should fail the threshold."""
        u_num = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u_ref = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # 2x
        l2_rel, linf_rel, l2_abs, linf_abs, n, passed = compute_continuous_errors(
            u_num, u_ref, 0.15
        )
        assert l2_rel > 0.5  # ~50% error
        assert linf_rel > 0.5
        assert not passed

    def test_shape_mismatch_raises(self):
        """Shape mismatch should raise AssertionError."""
        u_num = np.array([1.0, 2.0, 3.0])
        u_ref = np.array([1.0, 2.0])
        with pytest.raises(AssertionError):
            compute_continuous_errors(u_num, u_ref)

    def test_zero_reference(self):
        """Zero reference should use absolute error."""
        u_num = np.array([0.01, 0.02, 0.03])
        u_ref = np.zeros(3)
        l2_rel, linf_rel, l2_abs, linf_abs, n, passed = compute_continuous_errors(
            u_num, u_ref, 0.15
        )
        # When reference is zero, relative error = absolute error
        assert l2_rel == l2_abs
        assert linf_rel == linf_abs


class TestDOIReference:
    """Tests for DOIReference."""

    def test_creation(self):
        ref = DOIReference(
            doi="10.1234/test",
            citation="Test citation",
            solution_type="analytical",
        )
        assert ref.doi == "10.1234/test"
        assert ref.citation == "Test citation"
        assert ref.solution_type == "analytical"

    def test_default_solution_type(self):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        assert ref.solution_type == "analytical"


class TestContinuousErrorResult:
    """Tests for ContinuousErrorResult."""

    def test_summary_line_pass(self):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        r = ContinuousErrorResult(
            case_name="test_case",
            doi_ref=ref,
            l2_relative=0.01,
            linf_relative=0.02,
            n_points=10,
            passed=True,
        )
        line = r.summary_line()
        assert "[PASS]" in line
        assert "test_case" in line
        assert "0.010000" in line

    def test_summary_line_fail(self):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        r = ContinuousErrorResult(
            case_name="test_case",
            doi_ref=ref,
            l2_relative=0.5,
            linf_relative=0.6,
            n_points=10,
            passed=False,
        )
        line = r.summary_line()
        assert "[FAIL]" in line

    def test_to_dict(self):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        r = ContinuousErrorResult(
            case_name="test_case",
            doi_ref=ref,
            l2_relative=0.01,
            linf_relative=0.02,
            n_points=10,
            passed=True,
            params={"key": "value"},
        )
        d = r.to_dict()
        assert d["case_name"] == "test_case"
        assert d["doi"] == "10.1234/test"
        assert d["l2_relative"] == 0.01
        assert d["params"] == {"key": "value"}


class TestPrintErrorReport:
    """Tests for print_error_report."""

    def test_report_format(self):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        r = ContinuousErrorResult(
            case_name="test_case",
            doi_ref=ref,
            l2_relative=0.01,
            linf_relative=0.02,
            n_points=10,
            passed=True,
        )
        report = print_error_report([r])
        assert "Continuous Error Report" in report
        assert "PASS: 1" in report
        assert "test_case" in report


class TestSaveArtifacts:
    """Tests for save_artifacts."""

    def test_save_creates_files(self, tmp_path):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        r = ContinuousErrorResult(
            case_name="test_case",
            doi_ref=ref,
            l2_relative=0.01,
            linf_relative=0.02,
            n_points=10,
            passed=True,
        )
        artifacts = save_artifacts([r], tmp_path, label="test_xval")
        assert Path(artifacts["json"]).exists()
        assert Path(artifacts["csv"]).exists()
        assert Path(artifacts["sha256"]).exists()

    def test_json_content(self, tmp_path):
        ref = DOIReference(doi="10.1234/test", citation="Test")
        r = ContinuousErrorResult(
            case_name="test_case",
            doi_ref=ref,
            l2_relative=0.01,
            linf_relative=0.02,
            n_points=10,
            passed=True,
        )
        artifacts = save_artifacts([r], tmp_path, label="test_xval")
        data = json.loads(Path(artifacts["json"]).read_text())
        assert data["schema"] == "tensorlbm/continuous_error_xval/1"
        assert data["n_cases"] == 1
        assert data["n_pass"] == 1


class TestPoiseuilleIntegration:
    """Integration test: Poiseuille flow vs analytical solution."""

    def test_poiseuille_passes(self):
        """Poiseuille flow should match analytical solution within 15%."""
        from tensorlbm.ice_hydro.lbm_channel import (
            CX, CY, WT, ChannelConfig, D2Q9Channel,
        )

        G = 1.0e-5
        TAU = 0.8
        H = 16
        ny = H + 4
        steps = 3000

        cfg = ChannelConfig(
            nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
            cover_amplitude=0.0, cover_depth=float(ny - 3),
            bc="no_slip", forcing="guo",
        )
        ch = D2Q9Channel(cfg)

        rows = np.where(ch.fluid[0, :])[0]
        n_fluid = rows.size
        nu = (TAU - 0.5) / 3.0
        y = rows - 0.5
        u_ana = G / (2.0 * nu) * y * (n_fluid - y)

        u0 = np.zeros(ny)
        u0[rows] = u_ana
        ch.init_equilibrium(u0)

        acc = np.zeros(n_fluid)
        n_win = max(steps // 4, 1)
        n_acc = 0

        for step in range(1, steps + 1):
            ch.f = ch._stream_from(ch._collide(ch.f))
            if step > steps - n_win:
                rho = ch.f.sum(axis=0)
                ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
                acc += ux[0, rows]
                n_acc += 1

        u_num = acc / n_acc
        l2_rel, linf_rel, l2_abs, linf_abs, n_pts, passed = compute_continuous_errors(
            u_num, u_ana, threshold=0.15
        )

        assert passed, f"Poiseuille failed: L2_rel={l2_rel:.6f}, L∞_rel={linf_rel:.6f}"
        assert l2_rel < 0.01  # Should be well under 1%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
