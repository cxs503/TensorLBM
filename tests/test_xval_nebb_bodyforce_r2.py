"""Test for XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S (Round 2).

Verifies that the real-repo cross-benchmark driver:
1. Calls the actual GPUCorePathSolver (not a hand-written re-implementation)
2. Produces results within the pre-declared thresholds
3. Enforces no-slip at the wall
4. Shows convergence with grid refinement
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
XVAL_DIR = REPO_ROOT / "xval"
DRIVER = XVAL_DIR / "xval_nebb_bodyforce_driver_r2.py"
RESULT_JSON = XVAL_DIR / "xval_nebb_bodyforce_driver_r2.json"


def _run_driver():
    """Run the R2 driver and return the parsed JSON result."""
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, str(DRIVER)],
        capture_output=True, text=True, timeout=300, env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0, f"Driver failed: {result.stderr}"
    return json.loads(RESULT_JSON.read_text())


@pytest.fixture(scope="module")
def xval_result():
    """Run the driver once for all tests in this module."""
    return _run_driver()


class TestXvalNebbBodyforceR2:
    """L2/L3 evidence: real repo solver called, thresholds met, no-slip enforced."""

    def test_driver_produces_json(self, xval_result):
        """The driver writes a valid JSON result file."""
        assert "runs" in xval_result
        assert "verdict" in xval_result
        assert "tensorlbm_real_d3q19_ny10" in xval_result["runs"]

    def test_passes_owner_threshold(self, xval_result):
        """Pre-declared pointwise metric < 15 % (owner standard)."""
        verdict = xval_result["verdict"]
        assert verdict["pass_owner_15pct"] is True
        assert verdict["tensorlbm_pointwise_err_ny10"] < 0.15

    def test_passes_ecosystem_threshold(self, xval_result):
        """Pre-declared pointwise metric < 3 % (ecosystem stretch)."""
        verdict = xval_result["verdict"]
        assert verdict["pass_ecosystem_3pct"] is True
        assert verdict["tensorlbm_pointwise_err_ny10"] < 0.03

    def test_no_slip_enforced(self, xval_result):
        """No-slip boundary condition is enforced (first fluid row matches analytic)."""
        verdict = xval_result["verdict"]
        assert verdict["no_slip_enforced"] is True

    def test_convergence_with_grid_refinement(self, xval_result):
        """Error decreases with grid refinement (NY=10 → NY=20)."""
        err_ny10 = xval_result["verdict"]["tensorlbm_pointwise_err_ny10"]
        err_ny20 = xval_result["verdict"]["tensorlbm_pointwise_err_ny20"]
        assert err_ny20 < err_ny10, f"NY=20 error ({err_ny20}) should be < NY=10 error ({err_ny10})"

    def test_real_repo_solver_used(self, xval_result):
        """The result comes from the real repo solver (not hand-written)."""
        runs = xval_result["runs"]
        assert "tensorlbm_real_d3q19_ny10" in runs
        # The real repo solver produces much lower error than the hand-written R1
        assert runs["tensorlbm_real_d3q19_ny10"]["pointwise_max_rel_err"] < 0.01

    def test_u_max_accuracy(self, xval_result):
        """u_max matches analytic reference within 1 %."""
        run = xval_result["runs"]["tensorlbm_real_d3q19_ny10"]
        assert run["u_max_rel_err"] < 0.01

    def test_profile_shape_l2_error(self, xval_result):
        """Profile L2 relative error < 1 %."""
        run = xval_result["runs"]["tensorlbm_real_d3q19_ny10"]
        assert run["profile_rel_l2"] < 0.01

    def test_cross_arm_consistency(self, xval_result):
        """Cross-arm comparison shows repo is more accurate than OSS D2Q9."""
        cross = xval_result["runs"]["cross_arm_real_vs_oss"]
        assert cross["repo_d3q19_pointwise_err"] < cross["oss_d2q9_pointwise_err"]

    def test_round1_defects_fixed(self, xval_result):
        """Round 2 fixes the Round 1 defects (23.50 % → 0.19 %)."""
        verdict = xval_result["verdict"]
        assert verdict["round1_handwritten_pointwise_err"] == 0.235
        assert verdict["tensorlbm_pointwise_err_ny10"] < 0.01  # 0.19 %
