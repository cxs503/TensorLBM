"""Tests for the IN37-N05 ice-resistance factor certification.

The tests are deliberately split into

* solver verification -- the LBM must reproduce the exact plane-Poiseuille
  solution of the *empty* channel (velocity field and wall friction) before any
  obstructed result can be trusted;
* estimator verification -- the certified force estimator (global momentum
  budget) must return zero drag for the empty channel, exactly the analytic
  wall friction, and it must not depend on the ice at all in that limit;
* GCI machinery -- observed order and Richardson extrapolation must recover a
  synthetic power law to machine precision, and must *refuse* (degrade) a
  non-monotone / divergent triplet instead of inventing an order;
* geometry -- the staircase obstruction must cover exactly ``D x D`` cells at
  every refinement level, with blockage ``D/H = 0.25``;
* the certification analysis -- the three-way decomposition of the inherited
  FVM/LBM gap must sum back to the gap exactly, and the 2x2 variance shares
  must sum to one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.ice_resistance_factor import (
    GRID_LEVELS,
    M5092_FVM_ANCHOR,
    M5092_LBM_ANCHOR,
    OPERATORS,
    IceChannelLBM,
    LevelIceCase,
    gci_fine,
    observed_order_ls,
    richardson_triple,
    run_single,
)
from tensorlbm.ice_resistance_factor_certify import build_plan, certify


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("D", [4, 9, 12, 15, 30, 60])
def test_blockage_resolved_by_exactly_D_cells(D):
    case = LevelIceCase(D=D)
    mask = case.solid_mask()
    assert mask.sum() == D * D, "obstruction must cover exactly D x D cells"
    assert case.ny == 4 * D + 2
    assert case.nx == 8 * D
    assert math.isclose(case.D / case.H, 0.25)


def test_eroded_staircase_shrinks_the_obstruction_by_one_ring():
    case = LevelIceCase(D=12)
    full = case.solid_mask()
    eroded = case.solid_mask(erode=1)
    assert eroded.sum() < full.sum()
    assert eroded.sum() == (12 - 2) ** 2


# ---------------------------------------------------------------------------
# solver verification against the exact empty-channel solution
# ---------------------------------------------------------------------------
def test_lbm_reproduces_plane_poiseuille_bulk_velocity():
    case = LevelIceCase(D=8)
    solver = IceChannelLBM(case, with_body=False)
    for _ in range(4000):
        solver.step()
    ubar = solver.bulk_velocity()
    # g = 12 nu u0 / H**2 is chosen so that the empty channel carries u0
    assert abs(ubar - case.u0) / case.u0 < 5e-3


def test_lbm_poiseuille_profile_shape():
    case = LevelIceCase(D=8)
    solver = IceChannelLBM(case, with_body=False)
    for _ in range(3000):
        solver.step()
    _rho, ux, _uy = solver.moments()
    prof = ux[1:-1].mean(axis=1)
    ana = case.poiseuille()[1:-1]
    assert np.max(np.abs(prof - ana)) / ana.max() < 0.02


def test_empty_channel_wall_friction_matches_the_momentum_budget():
    """The momentum-exchange wall force must equal the imposed drive exactly."""
    case = LevelIceCase(D=8)
    solver = IceChannelLBM(case, with_body=False)
    out = {}
    for _ in range(3000):
        out = solver.step()
    drive = solver.g * solver.n_open
    assert abs(out["f_wall_x_me"] - drive) / drive < 1e-3


def test_certified_estimator_gives_zero_drag_without_ice():
    """Global momentum budget: no ice -> no drag (the empty channel check)."""
    case = LevelIceCase(D=8)
    solver = IceChannelLBM(case, with_body=False)
    out = {}
    for _ in range(3000):
        out = solver.step()
    drive = solver.g * solver.n_open
    assert abs(out["cd_balance"]) / drive < 1e-3


# ---------------------------------------------------------------------------
# GCI / Richardson machinery
# ---------------------------------------------------------------------------
def test_observed_order_recovers_a_synthetic_second_order_law():
    D = np.array([9.0, 12.0, 15.0, 30.0, 60.0])
    f = 10.0 + 3.0 / D ** 2
    fit = observed_order_ls(1.0 / D, f)
    assert abs(fit["p"] - 2.0) < 0.02
    assert abs(fit["f_ext"] - 10.0) < 1e-6


def test_observed_order_recovers_a_synthetic_first_order_law():
    D = np.array([6.0, 12.0, 24.0, 48.0])
    f = 2.0 - 0.7 / D
    fit = observed_order_ls(1.0 / D, f)
    assert abs(fit["p"] - 1.0) < 0.02
    assert abs(fit["f_ext"] - 2.0) < 1e-6


def test_richardson_triple_uniform_ratio():
    h = 1.0 / np.array([40.0, 20.0, 10.0])
    f = 5.0 + 0.8 * h ** 2
    out = richardson_triple(f[0], f[1], f[2], 2.0, 2.0)
    assert out["certifiable"] is True
    assert abs(out["p"] - 2.0) < 1e-6
    assert abs(out["f_ext"] - 5.0) < 1e-9
    assert out["gci_fine_pct"] > 0.0


def test_richardson_triple_degrades_on_divergent_triplet():
    out = richardson_triple(1.0, 2.0, 1.5, 1.3333, 1.25)
    assert out["certifiable"] is False
    assert out["degradation"] is not None
    assert out["p"] is None


def test_gci_fine_is_none_without_an_order():
    assert gci_fine(1.0, 0.9, None) is None
    assert gci_fine(1.0, 0.9, 2.0) == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# certification analysis
# ---------------------------------------------------------------------------
def _write_fake_raw(tmp_path: Path) -> Path:
    """Synthetic raw runs arranged so that every certificate can be derived."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    # CD(D) = 15.5 - 12/D ; O1 baseline; operators O2..O4 offset by 1 %, 2 %, 4 %
    for i, D in enumerate(GRID_LEVELS):
        cd = 15.5 - 12.0 / D
        rec = {
            "tag": f"O1_sbb_halfway_D{D}",
            "D": D,
            "wall": "sbb",
            "force": "balance",
            "erode": 0,
            "dilate": 0,
            "cd": cd,
            "cd_balance": cd,
            "cd_me": cd / 2,
            "cd_stress": float("nan"),
            "cd_ibm": float("nan"),
            "steps": 1000,
            "converged": True,
            "drift": 1e-7,
        }
        (raw / f"O1_sbb_halfway_D{D}.json").write_text(json.dumps(rec))
        if D in (12, 30, 60):
            for name, off in (("O2_dirichlet_node", 0.01), ("O3_ibm_diffuse", 0.02), ("O4_sbb_eroded", 0.04)):
                spec = OPERATORS[name]
                r = dict(rec)
                r.update(
                    tag=f"{name}_D{D}",
                    wall=spec["wall"],
                    erode=int(spec["erode"]),
                    cd=cd * (1.0 + off),
                    cd_balance=cd * (1.0 + off),
                )
                (raw / f"{name}_D{D}.json").write_text(json.dumps(r))
    return tmp_path


def test_decomposition_sums_back_to_the_gap(tmp_path):
    cert = certify(str(_write_fake_raw(tmp_path)))
    dec = cert["decomposition"]
    assert dec["sum_check_abs"] is not None
    assert abs(dec["sum_check_abs"] - dec["gap_abs"]) < 1e-9
    shares = (
        dec["grid_share_pct_of_gap"]
        + dec["operator_share_pct_of_gap"]
        + dec["model_form_share_pct_of_gap"]
    )
    assert abs(shares - 100.0) < 1e-9


def test_two_by_two_shares_sum_to_one(tmp_path):
    cert = certify(str(_write_fake_raw(tmp_path)))
    d = cert["acceptance_criteria"]["d_two_by_two_factor_adjudication"]["design"]
    total = d["ss_share_grid"] + d["ss_share_operator"] + d["ss_share_interaction"]
    assert abs(total - 1.0) < 1e-9
    assert d["main_effect_grid"] != 0.0


def test_power_law_is_recovered_on_the_benchmark_grid(tmp_path):
    cert = certify(str(_write_fake_raw(tmp_path)))
    assert abs(cert["resolution_axis"]["observed_order_ls"]["p"] - 1.0) < 0.05
    assert abs(cert["resolution_axis"]["observed_order_ls"]["f_ext"] - 15.5) < 1e-3


def test_plan_covers_both_axes():
    plan = build_plan()
    grids = {j["D"] for j in plan}
    assert {9, 12, 15, 30, 60} <= grids
    walls = {j["wall"] for j in plan}
    assert {"sbb", "dirichlet", "ibm"} <= walls
    # the operator axis must be complete at the finest resolution
    fine = [j for j in plan if j["D"] == max(GRID_LEVELS)]
    assert len(fine) >= 4


def test_run_single_smoke():
    r = run_single(9, wall="sbb", force="balance", max_steps=1200, tol=1e-12)
    assert np.isfinite(r["cd_balance"])
    assert r["cd_balance"] > 0.0
    assert r["steps"] == 1200
    assert 0.0 < r["u_bar"] < 0.06