"""Regression gates for the B14-ellipse DIGITAL-GEOMETRY reference audit.

`benchmarks/pending/poiseuille_3d_ellipse/digital_reference.py` supplies an
independent yardstick for the pending 3D elliptic-pipe Poiseuille debt: the
machine-precision Stokes solution of the *digital* cross-section with the
half-way (mid-point, 0.5 cell) no-slip closure, i.e. the geometry the LBM
actually integrates.

The five gates below lock (1) faithfulness to the recorded case data, (2) the
validation of the reference solver itself against the analytic ellipse,
(3) the falsification result (compensating error: the recorded 5.107% is NOT an
upper bound on the lattice's deviation from its own geometry), and (4) the
effective wall offset ordering.  They are cheap (~3 s, sparse direct solves on
43x23 and 83x43 masks); no LBM run is performed, the LBM profiles are read from
`case_a20_b10.json` / `case_a40_b20.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "benchmarks" / "pending" / "poiseuille_3d_ellipse"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CASE_DIR))

from digital_reference import (  # noqa: E402
    CASES,
    NU,
    analyse_case,
    ellipse_mask,
    solve_stokes_digital,
)

U_IN = 0.02


@pytest.fixture(scope="module")
def audits():
    return {a: analyse_case(a, b, CASE_DIR / name) for a, b, name in CASES}


def test_mask_matches_recorded_binning():
    """The digital mask + binning must reproduce the recorded per-bin cell counts.

    If this drifts, every comparison below is against the wrong geometry.
    """
    for a, b, name in CASES:
        rec = json.loads((CASE_DIR / name).read_text(encoding="utf-8"))
        lam, fluid = ellipse_mask(a, b)
        assert lam.shape == (2 * b + 3, 2 * a + 3)
        assert int(fluid.sum()) == int(rec["A_digital"])
        audited = analyse_case(a, b, CASE_DIR / name)
        assert audited["cells_match_recorded"] is True
        assert audited["cells_ana_match_recorded"] is True


def test_reproduces_recorded_primary_metric(audits):
    """theta_total must reproduce the case file's s^Q metric (pipeline gate)."""
    for a, _, _ in CASES:
        g = audits[a]
        rec = g["theta_total_recorded_pct"]
        assert g["theta_total_pct"] == pytest.approx(rec, abs=0.01), (
            f"a={a}: reproduced {g['theta_total_pct']:.3f}% vs recorded {rec:.3f}%"
        )


def test_digital_reference_validated_against_analytic_ellipse(audits):
    """The digital solver must converge to the continuous-ellipse Poiseuille flow.

    Peak velocity of the digital solution vs U_c = G a^2 b^2 / (2 nu (a^2+b^2)):
    O(1/a) staircase error, < 4% at a=20 and strictly smaller at a=40.
    """
    errs = {a: abs(audits[a]["u_c_digital_ref_analytic_pct"]) for a, _, _ in CASES}
    assert errs[20] < 4.0, errs
    assert errs[40] < errs[20], errs  # refinement improves the reference
    # analytic peak velocity sanity: digital solve is a real Poisson solution
    lam, fluid = ellipse_mask(20, 10)
    U = solve_stokes_digital(fluid, nu=NU)
    u_c = float(np.nanmax(U))
    assert u_c == pytest.approx(400.0, rel=0.05)  # G a^2 b^2 /(2 nu (a^2+b^2)) = 400


def test_reference_geometry_contributes_but_does_not_explain_the_miss(audits):
    """Falsification gate: theta_num > theta_total at BOTH grids.

    The pending README attributed the 5.107% miss to reference geometry alone
    ("剖面本身是精确椭圆抛物面").  Against the digital geometry the LBM deviates
    *more* (6.5% / 4.8%) than against the continuous ellipse -- the two errors
    have opposite sign and partially cancel.  A change that makes theta_num drop
    below theta_total would mean the lattice really does track its own geometry
    better than this audit says: re-open the verdict rather than edit the test.
    """
    for a, _, _ in CASES:
        g = audits[a]
        assert 2.0 < g["theta_geom_pct"] < 4.5, g  # reference-geometry term is real
        assert g["theta_num_pct"] > g["theta_total_pct"], g
        assert g["theta_num_pct"] < 10.0, g  # bounded, not a blow-up


def test_effective_wall_offset_ordering(audits):
    """Effective no-slip locus: 0 < delta_LBM < 0.5 cell, and the digital
    reference sits closer to the continuous ellipse than the lattice does."""
    for a, _, _ in CASES:
        g = audits[a]
        assert 0.1 < g["delta_eff_lbm_cells"] < 0.5, g
        assert g["delta_eff_digital_cells"] < g["delta_eff_lbm_cells"], g
        assert g["delta_eff_lbm_rms_pct"] < 5.0, g