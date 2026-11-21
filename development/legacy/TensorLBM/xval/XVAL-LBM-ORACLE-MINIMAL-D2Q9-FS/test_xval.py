"""Test: XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS cross-validation.

Verifies that the Maarten-vd-Sande/lbm D2Q9 oracle and TensorLBM produce
consistent results at three levels: lattice constants, single-step identity,
and physical Poiseuille flow benchmark.
"""

import os
import sys
import json

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from run_xval import (
    check_lattice_constants,
    check_equilibrium,
    check_collision,
    oracle_poiseuille,
    tlbm_poiseuille,
    M_TO_TLBM,
    M_V, M_T, M_V_INV,
)


class TestLatticeConstants:
    """Level 1: D2Q9 lattice constant identity."""

    def test_mapping_complete(self):
        assert len(M_TO_TLBM) == 9

    def test_weights_match(self):
        from tensorlbm.d2q9 import C, W
        for i in range(9):
            j = M_TO_TLBM[i]
            assert abs(M_T[i] - float(W[j].item())) / max(abs(M_T[i]), 1e-30) < 1e-6

    def test_opposites_match(self):
        from tensorlbm.d2q9 import C, OPPOSITE
        for i in range(9):
            j = M_TO_TLBM[i]
            mapped_opp = M_TO_TLBM[M_V_INV[i]]
            assert mapped_opp == int(OPPOSITE[j].item())

    def test_lattice_check_function(self):
        result = check_lattice_constants()
        assert result["pass"]


class TestSingleStepIdentity:
    """Level 2: Single-step numerical identity."""

    def test_equilibrium_identity(self):
        result = check_equilibrium()
        assert result["max_abs_diff"] < 1e-12

    def test_collision_identity(self):
        result = check_collision()
        assert result["max_abs_diff"] < 1e-12


class TestPoiseuilleFlow:
    """Level 3: Plane Poiseuille flow benchmark."""

    @pytest.fixture(params=[
        {"ny": 16, "nx": 32, "tau": 0.8, "g_accel": 1e-5, "n_steps": 4000},
    ])
    def poiseuille_params(self, request):
        return request.param

    def test_oracle_vs_analytic(self, poiseuille_params):
        result = oracle_poiseuille(**poiseuille_params)
        # Owner precision standard: <15% error
        assert result["eps_inf"] < 0.15, (
            f"Oracle eps_inf={result['eps_inf']:.4e} exceeds 15% threshold"
        )

    def test_tlbm_vs_analytic(self, poiseuille_params):
        result = tlbm_poiseuille(**poiseuille_params)
        assert result["eps_inf"] < 0.15, (
            f"TensorLBM eps_inf={result['eps_inf']:.4e} exceeds 15% threshold"
        )

    def test_cross_code_agreement(self, poiseuille_params):
        ro = oracle_poiseuille(**poiseuille_params)
        rt = tlbm_poiseuille(**poiseuille_params)
        u1, u2 = ro["u"], rt["u"]
        scale = 0.5 * (float(np.max(np.abs(u1))) + float(np.max(np.abs(u2))))
        tv_rel = float(np.max(np.abs(u1 - u2)) / scale) if scale > 0 else float("inf")
        # Cross-code agreement should be < 1%
        assert tv_rel < 0.01, (
            f"Cross-code tv_rel={tv_rel:.4e} exceeds 1% threshold"
        )


class TestResultsArtifact:
    """Verify that results.json was written."""

    def test_results_file_exists(self):
        results_path = os.path.join(HERE, "results.json")
        assert os.path.exists(results_path), "results.json not found"
        with open(results_path) as f:
            data = json.load(f)
        assert "overall_pass" in data
        assert data["overall_pass"] is True
