"""Tests for XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE.

The tests re-run the fast arms (analytic modes, invariants, momentum, mutation
controls; a few seconds) and, when ``artifacts/summary.json`` exists, also
assert the recorded slab-arm verdicts.  They assert *invariants with numbers*
(a wrong sign, a wrong weight table, a wrong gather direction or a missing
``cs^2`` must break them), not merely that the code runs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
XV = ROOT / "xval" / "XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE"
sys.path.insert(0, str(XV))
sys.path.insert(0, str(ROOT / "src"))

import xval_lib as L  # noqa: E402


@pytest.fixture(scope="module")
def arms():
    import run_xval as R

    return R


def test_symbol_closed_form_matches_expansion(arms):
    """S(k) -> 1 - k^2/6 for small k (second-order consistency of the stencil)."""
    for k in (1e-2, 5e-2):
        s = L.symbol_ratio((k, 0.0))
        assert abs(s - (1.0 - k * k / 6.0)) < 1e-5
    # axis direction has the textbook closed form sin(k)/k
    for m in (1, 4, 9):
        k = 2.0 * math.pi * m / 64.0
        assert abs(L.symbol_ratio((k, 0.0)) - math.sin(k) / k) < 1e-12


def test_kernel_equals_oss_stencil(arms):
    """Library reverse-gather kernel == OSS forward-gather stencil, g_oss = -G_12."""
    rho1, rho2 = L.sloshing_fields(48)
    fx, fy = arms.kernel_force(rho1, rho2, -2.5)[:2]
    ox, oy = L.oss_force(rho1, rho2, g_oss=2.5)
    assert arms.rel_l2(fx, ox) < 1e-6
    assert arms.rel_l2(fy, oy) < 1e-6
    # the unmapped convention choice must be wrong by O(1), otherwise the map is untested
    bx, by = L.oss_force(rho1, rho2, g_oss=-2.5)
    assert arms.rel_l2(fx, bx) > 1.0


def test_uniform_field_gives_nil_force(arms):
    inv = arms.arm_invariants(32)
    assert inv["uniform_nil"] == 0.0
    assert inv["spectral_nil_uniform"] == 0.0


def test_d2q9_symmetry_equivariance(arms):
    inv = arms.arm_invariants(32)
    assert inv["symmetry_rot90"] < 1e-12
    assert inv["symmetry_transpose"] < 1e-12


def test_measured_error_equals_closed_form(arms):
    sweep = arms.arm_mode_sweep(32)
    assert sweep["closed_form_identity_max"] < 1e-6
    assert sweep["symbol_identity_max"] < 1e-6


def test_second_order_convergence(arms):
    ref = arms.arm_symbol_refinement()
    assert abs(ref["mean_order"] - 2.0) < 0.05
    errs = [r["err"] for r in ref["rows"]]
    assert errs == sorted(errs, reverse=True)


def test_total_momentum_conserved_by_both_forms(arms):
    """Both force forms telescope to zero net force on a periodic domain."""
    mom = arms.arm_momentum(32)
    assert mom["standard"]["defect_rel"] < 1e-13
    assert mom["edm"]["defect_rel"] < 1e-13


def test_mutations_are_detected(arms):
    mut = arms.arm_mutation(32)
    base = mut["baseline_err"]
    assert base < 0.15  # baseline mode m=4 on n=32 sits at k=0.785 -> ~10% truncation
    # five of six planted defects exceed the 15% gate, all of them exceed the
    # measurement floor by >= 4x (the psi = sqrt(rho) defect is the weakest one)
    assert sum(v >= 0.15 for v in mut["mutations"].values()) >= 5
    assert mut["min_mutation_err"] > 0.10  # weakest control still 10x the 1% floor


def test_slab_verdicts_recorded(arms):
    summary_path = XV / "artifacts" / "summary.json"
    if not summary_path.exists():
        pytest.skip("run_xval.py has not been executed in this checkout")
    s = json.loads(summary_path.read_text())
    cases = s["C_slab"]["cases"]
    assert len(cases) >= 3
    assert {c["n"] for c in cases} >= {64, 96}
    assert {c["G12"] for c in cases} == {-2.5, -3.0}
    for c in cases:
        if c["converged"]:
            assert c["u_max"] < 0.02
        # acceptance: <15% against the exact spectral gradient
        assert c["standard"]["err_vs_spectral"] < 0.15
        assert c["edm"]["err_vs_spectral"] < 0.15
    assert s["verdicts"]["checks"]["primary_slab_err_le_15pct"] is True
    assert s["verdicts"]["checks"]["secondary_resolved_band_le_15pct"] is True