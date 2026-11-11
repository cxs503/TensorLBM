"""Tests for :mod:`tensorlbm.cavitation_lcav_attribution` and the p447w2-N06 driver.

The tests are split into

* analytic unit tests (vapour indicators, ``L_eq`` scaling, generalised
  Richardson order recovery, ANOVA bookkeeping, telescoping error budget);
* one integration test that runs the smallest possible cavitating-airfoil case
  through the driver's ``run_grid`` and checks the conservation residuals, the
  criterion matrix and bit-for-bit reproducibility of the whole pipeline.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorlbm.cavitation_airfoil import CavitationFlowConfig, airfoil_surface
from tensorlbm.cavitation_lcav_attribution import (
    ALPHA_THRESHOLDS,
    CRITERIA,
    alpha_relative_thresholds,
    block_means,
    criterion_lengths,
    equivalent_length,
    error_budget,
    isobaric_indicator,
    relative_error,
    richardson_generalized,
    two_way_anova,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# vapour indicators / criterion family
# ---------------------------------------------------------------------------


def test_isobaric_indicator_is_the_p_lt_pv_cut():
    p_v = -0.5
    p = torch.tensor([[-1.0, -0.5, -0.25, 0.0, 0.5]], dtype=torch.float64)
    ind = isobaric_indicator(p, p_v)
    # I > 0.5 exactly where p < p_v
    assert [bool(x) for x in (ind > 0.5).flatten()] == [True, False, False, False, False]
    # zero crossing / symmetry about p_v
    # symmetry about p_v: dp = -0.5 (index 0) and dp = +0.5 (index 3) pair up
    assert abs(float(ind[0, 0] + ind[0, 3]) - 1.0) < 1e-12
    assert abs(float(ind[0, 1]) - 0.5) < 1e-12


def test_isobaric_indicator_rejects_bad_pv():
    with pytest.raises(ValueError):
        isobaric_indicator(torch.zeros(2, 2), 0.0)


def test_alpha_relative_thresholds_scale_with_field_max():
    a = torch.full((4, 4), 0.02, dtype=torch.float64)
    taus = alpha_relative_thresholds(a, (0.1, 0.5, 0.9))
    assert taus == pytest.approx((0.002, 0.01, 0.018))
    zeros = alpha_relative_thresholds(torch.zeros(3, 3), (0.1, 0.5, 0.9))
    assert zeros == (0.0, 0.0, 0.0)


def test_equivalent_length_is_linear_in_alpha():
    """Scale-free footprint threshold => ``L_eq`` scales linearly with alpha."""
    cfg = CavitationFlowConfig(airfoil="NACA0015", cells_per_chord=20,
                               domain_x_chords=6.0, domain_y_chords=3.0,
                               le_x_chords=1.5, le_y_chords=1.5)
    geom = airfoil_surface(cfg.airfoil)
    nx, ny = cfg.nx, cfg.ny
    a = torch.zeros(nx, ny, dtype=torch.float64)
    # a vapour patch in the middle of the domain (away from the wall sampling
    # band, so the footprint thickness is set by the patch itself)
    a[nx // 3 : nx // 2, ny // 2 : ny // 2 + int(0.06 * cfg.cells_per_chord)] = 0.2
    eq1 = equivalent_length(a, cfg.h, cfg=cfg, geometry=geom)
    eq2 = equivalent_length(2.0 * a, cfg.h, cfg=cfg, geometry=geom)
    assert eq1["volume"] > 0.0
    assert eq2["volume"] == pytest.approx(2.0 * eq1["volume"], rel=1e-12)
    assert eq1["t_effective"] >= cfg.h


def test_equivalent_length_zero_field_is_zero():
    cfg = CavitationFlowConfig(airfoil="NACA0015", cells_per_chord=16,
                               domain_x_chords=6.0, domain_y_chords=3.0,
                               le_x_chords=1.5, le_y_chords=1.5)
    eq = equivalent_length(torch.zeros(cfg.nx, cfg.ny), cfg.h, cfg=cfg)
    assert eq["l_eq"] == 0.0
    assert eq["volume"] == 0.0


def test_criterion_lengths_reports_whole_family():
    cfg = CavitationFlowConfig(airfoil="NACA0015", cells_per_chord=16,
                               domain_x_chords=6.0, domain_y_chords=3.0,
                               le_x_chords=1.5, le_y_chords=1.5, sigma=1.0)
    nx, ny = cfg.nx, cfg.ny
    p = torch.zeros(nx, ny, dtype=torch.float64) + 0.1
    p[: nx // 2, ny // 2] = cfg.p_v - 0.1  # a small sub-vapour-pressure pocket
    a = torch.full((nx, ny), 1e-3, dtype=torch.float64)
    a[: nx // 3, ny // 2 : ny // 2 + 3] = 0.6  # a resolved vapour pocket
    out = criterion_lengths(p, a, cfg)
    for crit in CRITERIA:
        assert crit in out["criterion_lengths"]
        assert math.isfinite(out["criterion_lengths"][crit])
    for tau in ALPHA_THRESHOLDS:
        assert f"l_alpha_v>={tau:g}" in out
    assert out["l_isobaric"] > 0.0  # the sub-p_v pocket is detected
    assert out["l_alpha_v>=0.5"] > 0.0  # the resolved pocket is detected
    assert any(k.startswith("l_alpha_rel>=") for k in out)


# ---------------------------------------------------------------------------
# generalised Richardson / GCI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_richardson_recovers_known_order_and_limit(order):
    """``f(h) = f0 + C h^p`` must give back ``p`` and ``f0`` exactly."""
    f0, c = 0.6, 4.0
    h = [1.0 / 30, 1.0 / 25, 1.0 / 20]  # non-uniform ratios 1.2 / 1.25
    f = [f0 + c * x**order for x in h]
    r = richardson_generalized(h, f)
    assert r["valid"], r["reason"]
    assert r["p"] == pytest.approx(order, rel=1e-8)
    assert r["f_extrap"] == pytest.approx(f0, rel=1e-8)
    assert r["gci_fine"] > 0.0
    assert r["asymptotic"] is True


def test_richardson_flags_non_monotone_sequence():
    r = richardson_generalized([1 / 30, 1 / 25, 1 / 20], [0.1, 0.2, 0.15])
    assert r["valid"] is False
    assert "monotone" in r["reason"]
    assert math.isnan(r["p"]) and math.isnan(r["gci_fine"])


def test_richardson_flags_degenerate_sequence():
    r = richardson_generalized([1 / 30, 1 / 25, 1 / 20], [0.25, 0.25, 0.25])
    assert r["valid"] is False
    assert "degenerate" in r["reason"]


def test_richardson_requires_three_grids():
    with pytest.raises(ValueError):
        richardson_generalized([1.0, 0.5], [1.0, 1.1])


# ---------------------------------------------------------------------------
# two-factor variance decomposition
# ---------------------------------------------------------------------------


def _anova_table(a_effect: dict, b_effect: dict, ab: float, n: int = 4):
    table = {}
    grids = sorted(b_effect)
    crits = sorted(a_effect)
    for i, g in enumerate(grids):
        for j, c in enumerate(crits):
            base = 0.5 + a_effect[c] + b_effect[g] + ab * (i - 1) * (j - 1)
            table[(g, c)] = [base] * n
    return table


def test_anova_pure_criterion_effect():
    table = _anova_table({"iso": 0.0, "a50": 0.2, "leq": -0.2}, {20: 0.0, 25: 0.0, 30: 0.0, 40: 0.0}, 0.0)
    r = two_way_anova(table)
    assert r["pct_criterion"] == pytest.approx(100.0, abs=1e-9)
    assert r["pct_grid"] == pytest.approx(0.0, abs=1e-9)
    assert r["pct_interaction"] == pytest.approx(0.0, abs=1e-9)


def test_anova_pure_grid_effect_and_percentages_sum_to_100():
    table = _anova_table({"iso": 0.0, "a50": 0.0, "leq": 0.0},
                         {20: 0.0, 25: 0.1, 30: 0.2, 40: 0.3}, 0.0)
    r = two_way_anova(table)
    assert r["pct_grid"] == pytest.approx(100.0, abs=1e-9)
    total = r["pct_criterion"] + r["pct_grid"] + r["pct_interaction"] + r["pct_residual"]
    assert total == pytest.approx(100.0, abs=1e-9)


def test_anova_detects_interaction_with_replicates():
    table = _anova_table({"iso": 0.0, "a50": 0.0}, {20: 0.0, 25: 0.0}, ab=0.3, n=3)
    r = two_way_anova(table)
    assert r["ss_interaction"] > 0.0
    assert r["df_interaction"] == 1
    assert r["df_residual"] == 2 * 2 * 2


def test_anova_rejects_ragged_design():
    with pytest.raises(ValueError):
        two_way_anova({(20, "iso"): [1.0, 2.0], (25, "iso"): [1.0]})


# ---------------------------------------------------------------------------
# additive budget
# ---------------------------------------------------------------------------


def test_error_budget_telescopes_and_sums_to_100():
    b = error_budget(l_base=0.0, l_criterion=0.0376, l_extrap=0.05, l_reference=0.25,
                     l_disc_fine=0.0376)
    assert b["closure_check"] == pytest.approx(0.0, abs=1e-15)
    pct = b["pct_criterion"] + b["pct_discretisation"] + b["pct_residual"]
    assert pct == pytest.approx(100.0, abs=1e-9)
    assert b["delta_discretisation"] == pytest.approx(0.05 - 0.0376, rel=1e-12)
    assert b["delta_residual"] == pytest.approx(0.25 - 0.05, rel=1e-12)


def test_relative_error_and_block_means():
    assert relative_error(0.05, 0.25) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        relative_error(0.05, 0.0)
    assert block_means([1.0, 1.0, 3.0, 3.0], 2) == [1.0, 3.0]
    with pytest.raises(ValueError):
        block_means([1.0, 2.0], 3)


# ---------------------------------------------------------------------------
# integration: the driver pipeline on the smallest admissible case
# ---------------------------------------------------------------------------


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "p447w2_n06_driver", ROOT / "scripts" / "p447w2_n06_cavitation_attribution.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_grid_pipeline_is_deterministic_and_conservative(tmp_path, monkeypatch):
    drv = _load_driver()
    monkeypatch.setattr(drv, "ART", tmp_path)
    spec = {"cells_per_chord": 10, "sigma": 1.0, "physical_time": 1.0}
    a = drv.run_grid(spec)
    b = drv.run_grid(spec)
    # (h) bit-for-bit reproducibility of the integrated pipeline
    for crit in CRITERIA:
        assert a["matrix"][crit] == b["matrix"][crit]
    assert a["residuals"]["mass_balance_max"] <= 1e-6
    assert a["residuals"]["div_residual_max"] <= 1e-2  # loose: 10 cells/chord
    assert set(a["matrix"]) == set(CRITERIA)
    assert all(math.isfinite(v) for v in a["matrix"].values())
    assert len(a["replicates"]["isobaric"]) == drv.N_BLOCKS
    assert a["n_steps"] == round(1.0 / (0.2 * a["h"]))
    assert (tmp_path / "mean_fields_10_sigma1.npz").exists()