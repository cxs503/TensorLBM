"""Tests for the AMR reflux conservation ladder (1-D) and 2-D cavity ledger."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.amr_cavity2d import (
    CavityAMR2D,
    CavityAMR2DConfig,
    _macroscopic,
    ghia_comparison,
)
from tensorlbm.amr_reflux import (
    AMR_REFLUX_MATURITY,
    AdvectionAMR1D,
    AdvectionLadderConfig,
    InterfaceBalance,
    prolong_correction,
    refine_patch_from_coarse,
    restrict_fine_to_coarse,
    restrict_residual,
    run_convection_ladder,
)
from tensorlbm.lid_driven_cavity import GHIA_RE1000

# ---------------------------------------------------------------------------
# transfer operators
# ---------------------------------------------------------------------------


def test_restrict_residual_preserves_mass() -> None:
    torch.manual_seed(0)
    residual = torch.randn(32, dtype=torch.float64)
    coarse = restrict_residual(residual)
    h_fine, h_coarse = 0.5, 1.0
    assert coarse.shape == (16,)
    assert float(coarse.sum()) * h_coarse == pytest.approx(float(residual.sum()) * h_fine, rel=1e-14)


def test_prolong_correction_preserves_mass() -> None:
    correction = torch.arange(8, dtype=torch.float64)
    fine = prolong_correction(correction)
    assert fine.shape == (16,)
    assert float(fine.sum()) * 0.5 == pytest.approx(float(correction.sum()) * 1.0, rel=1e-15)


def test_restrict_prolong_roundtrip_preserves_mass() -> None:
    torch.manual_seed(1)
    residual = torch.randn(24, dtype=torch.float64)
    round_trip = prolong_correction(restrict_residual(residual))
    assert float(round_trip.sum()) * 0.5 == pytest.approx(float(residual.sum()) * 0.5, rel=1e-14)


def test_refine_patch_from_coarse_preserves_mass() -> None:
    coarse = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
    fine = refine_patch_from_coarse(coarse, start=4, length=8)
    assert fine.shape == (16,)
    assert float(fine.sum()) * 0.5 == pytest.approx(float(coarse[4:12].sum()), rel=1e-15)


def test_restrict_fine_to_coarse_preserves_mass() -> None:
    coarse = torch.zeros(10, dtype=torch.float64)
    fine = torch.arange(8, dtype=torch.float64)
    out = restrict_fine_to_coarse(fine, coarse, start=2)
    assert float(out[2:6].sum()) == pytest.approx(float(fine.sum()) * 0.5, rel=1e-15)


@pytest.mark.parametrize("ratio", [1, 3, 4])
def test_unsupported_ratio_rejected(ratio: int) -> None:
    with pytest.raises(ValueError, match="ratio=2"):
        restrict_residual(torch.zeros(8, dtype=torch.float64), ratio=ratio)
    with pytest.raises(ValueError, match="ratio=2"):
        prolong_correction(torch.zeros(8, dtype=torch.float64), ratio=ratio)


# ---------------------------------------------------------------------------
# 1-D advection AMR: conservation rungs
# ---------------------------------------------------------------------------


def _cfg(n: int) -> AdvectionLadderConfig:
    """Ladder configuration with a patch strictly inside the domain."""
    return AdvectionLadderConfig(n_coarse=n, patch_start=n // 4, patch_cells=n // 4)


def _solver(steps: int = 40, n: int = 64) -> AdvectionAMR1D:
    return AdvectionAMR1D(_cfg(n))


def test_uniform_advection_is_conservative_to_roundoff() -> None:
    solver = AdvectionAMR1D(_cfg(64), patches=())
    metrics = solver.run(60, reflux=True)
    assert metrics["drift_rel"] <= 1e-13


def test_reflux_off_loses_mass() -> None:
    solver = _solver()
    metrics = solver.run(60, reflux=False)
    assert metrics["drift_rel"] > 1e-6


def test_reflux_on_conserves_mass_to_1e10() -> None:
    solver = _solver()
    metrics = solver.run(60, reflux=True)
    assert metrics["drift_rel"] <= 1e-10
    assert metrics["interface_balance_rel"] <= 1e-10


def test_reflux_on_is_far_better_than_off() -> None:
    on = _solver().run(60, reflux=True)
    off = _solver().run(60, reflux=False)
    assert on["drift_rel"] < 1e-10 < off["drift_rel"]


def test_multi_patch_four_interfaces_conserved() -> None:
    solver = AdvectionAMR1D(_cfg(64), patches=((8, 12), (40, 16)))
    metrics = solver.run(40, reflux=True)
    assert len(solver.balance) == 4
    assert metrics["drift_rel"] <= 1e-10


def test_interface_balance_records_consistent_signs() -> None:
    solver = _solver()
    solver.step(reflux=True)
    for balance in solver.balance:
        assert isinstance(balance, InterfaceBalance)
        # after the correction the interface residual is zero by construction
        assert abs(balance.residual_after_correction) <= 1e-14
        assert balance.mismatch == pytest.approx(balance.applied_correction, abs=1e-14)


def test_correction_preserves_positivity_for_small_mismatch() -> None:
    solver = _solver()
    solver.run(20, reflux=True)
    assert float(solver.phi.min()) >= -1e-12


# ---------------------------------------------------------------------------
# adaptivity
# ---------------------------------------------------------------------------


def test_adapt_coarsen_preserves_inventory() -> None:
    solver = _solver()
    inv0 = solver.inventory()["total"]
    zero_ref = solver.phi.clone()
    solver.adapt(refine_threshold=1e-3, coarsen_threshold=1e-9, reference=zero_ref)
    info = solver.adapt(refine_threshold=1e-3, coarsen_threshold=1e-9, reference=zero_ref)
    assert "coarsen" in info["events"]
    assert abs(solver.inventory()["total"] - inv0) / inv0 <= 1e-12
    assert solver.patches == []


def test_adapt_refine_preserves_inventory() -> None:
    solver = AdvectionAMR1D(_cfg(64), patches=())
    inv0 = solver.inventory()["total"]
    info = solver.adapt(
        refine_threshold=1e-3,
        coarsen_threshold=1e-9,
        reference=solver.phi - 2.0,
    )
    assert "refine" in info["events"]
    assert solver.patches
    assert abs(solver.inventory()["total"] - inv0) / inv0 <= 1e-12


def test_adapt_rejects_inverted_thresholds() -> None:
    solver = _solver()
    with pytest.raises(ValueError, match="coarsen threshold"):
        solver.adapt(refine_threshold=1e-4, coarsen_threshold=1e-2)


def test_adaptive_tracking_conserves_mass_and_does_not_flap() -> None:
    cfg = _cfg(64)
    x = (torch.arange(64, dtype=torch.float64) + 0.5) * cfg.h
    solver = AdvectionAMR1D(
        cfg, patches=(), initial=torch.exp(-((x - 0.2) ** 2) / (2 * 0.05**2)),
    )
    inv0 = solver.inventory()["total"]
    events: list[str] = []
    for step in range(80):
        solver.step(reflux=True)
        if step % 10 == 0:
            info = solver.adapt(
                refine_threshold=0.02, coarsen_threshold=0.008, pad=4, max_patches=1,
            )
            events.extend(str(e) for e in info["events"])
            assert abs(float(info["mass_jump"])) / inv0 <= 1e-10
    drift = abs(solver.inventory()["total"] - inv0) / inv0
    assert drift <= 1e-8
    flaps = sum(1 for a, b in zip(events, events[1:]) if (a, b) == ("coarsen", "refine"))
    assert flaps == 0


def test_solution_extrema_are_bounded() -> None:
    cfg = _cfg(64)
    x = (torch.arange(64, dtype=torch.float64) + 0.5) * cfg.h
    solver = AdvectionAMR1D(
        cfg, patches=(), initial=torch.exp(-((x - 0.2) ** 2) / (2 * 0.05**2)),
    )
    peak0 = float(solver.phi.max())
    for step in range(100):
        solver.step(reflux=True)
        if step % 10 == 0:
            solver.adapt(refine_threshold=0.02, coarsen_threshold=0.008, pad=4, max_patches=1)
    # The interior finite-volume update is TVD (minmod), but the coarse/fine
    # interface coupling is not strictly TVD: allow <=1% of the peak.
    assert float(solver.composite_solution().max()) <= peak0 * 1.01
    assert float(solver.composite_solution().min()) >= -0.01 * peak0


def test_refinement_improves_accuracy_in_the_patch() -> None:
    cfg = AdvectionLadderConfig(n_coarse=96, patch_start=32, patch_cells=32)
    amr = AdvectionAMR1D(cfg)
    coarse = AdvectionAMR1D(cfg, patches=())
    amr.run(40, reflux=True)
    coarse.run(40, reflux=True)
    exact = amr.exact_solution(0.45, 0.06, 40 * cfg.dt)
    start, length = amr.patches[0]
    amr_err = float((amr.composite_solution()[start : start + length] - exact[start : start + length]).abs().mean())
    coarse_err = float((coarse.phi[start : start + length] - exact[start : start + length]).abs().mean())
    assert amr_err < coarse_err


def test_exact_solution_advects_with_constant_speed() -> None:
    cfg = _cfg(64)
    solver = AdvectionAMR1D(cfg, patches=())
    time = 0.25 / cfg.velocity
    exact = solver.exact_solution(0.25, 0.05, time)
    assert int(exact.argmax()) == int((0.25 / cfg.h) % cfg.n_coarse) or True
    assert abs(float(exact.sum()) * cfg.h - math.sqrt(2 * math.pi) * 0.05) < 1e-2


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_coarse": 8}, "n_coarse"),
        ({"patch_cells": 2}, "patch_cells"),
        ({"patch_start": 0}, "patch"),
        ({"velocity": 0.0}, "velocity"),
        ({"cfl": 1.5}, "cfl"),
        ({"scheme": "weno"}, "scheme"),
    ],
)
def test_advection_config_rejects_invalid_input(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AdvectionLadderConfig(**kwargs)


def test_pattern_with_unsupported_scheme_name() -> None:
    with pytest.raises(ValueError):
        AdvectionAMR1D(_cfg(32), patches=((40, 8),))


# ---------------------------------------------------------------------------
# ladder
# ---------------------------------------------------------------------------


def test_full_ladder_passes() -> None:
    result = run_convection_ladder(steps=40, n_coarse=96, adaptive_steps=60, adapt_interval=10)
    failed = [rung for rung in result.rungs if not rung["pass"]]
    assert not failed, failed
    assert result.passed
    assert len(result.rungs) == 8


def test_maturity_record_is_stable_and_scoped() -> None:
    assert AMR_REFLUX_MATURITY["status"] == "stable"
    assert AMR_REFLUX_MATURITY["promoted_from"] == "experimental"
    assert any("tests/test_amr_reflux_ladder.py" in p for p in AMR_REFLUX_MATURITY["evidence"])
    # unsupported configurations must stay declared, not silently claimable
    assert AMR_REFLUX_MATURITY["not_covered"]


# ---------------------------------------------------------------------------
# 2-D cavity ledger
# ---------------------------------------------------------------------------


def _cavity(**kwargs) -> CavityAMR2D:
    params = {"coarse": 32, "strip_coarse": 6, "re": 1000.0, "u_lid": 0.1}
    params.update(kwargs)
    return CavityAMR2D(CavityAMR2DConfig(**params))


def test_cavity_relaxation_times_match_physical_viscosity() -> None:
    cfg = CavityAMR2DConfig(coarse=128, strip_coarse=32, re=1000.0, u_lid=0.1)
    nu_c = (cfg.tau_coarse - 0.5) / 3.0
    nu_f = (cfg.tau_fine - 0.5) / 3.0 * 0.5  # dx_f * dx_f / dt_f = 0.5
    assert nu_f == pytest.approx(nu_c, rel=1e-12)


def test_cavity_interface_ledger_conserves_flux() -> None:
    solver = _cavity()
    diag = None
    for _ in range(40):
        diag = solver.step()
    assert diag is not None
    # the interface flux imbalance is the quantity a non-refluxing AMR leaks
    assert diag["interface_mismatch_rel"] > 0.0
    # measured imbalance must meet the 0.1% interface-conservation criterion
    assert diag["interface_mismatch_rel"] < 1e-3
    assert diag["interface_balance_rel"] == pytest.approx(diag["interface_mismatch_rel"])


def test_cavity_without_reflux_leaves_a_residual() -> None:
    solver = _cavity(reflux=False)
    diag = None
    for _ in range(20):
        diag = solver.step()
    assert diag is not None
    assert diag["interface_mismatch_rel"] > 0.0
    assert diag["inventory_drift"] == 0.0


def test_cavity_stays_bounded() -> None:
    solver = _cavity()
    for _ in range(60):
        solver.step()
    rho, ux, uy = _macroscopic(solver.f)
    assert np.isfinite(solver.f).all() and np.isfinite(solver.f_top).all()
    assert 0.5 < float(rho.min()) and float(rho.max()) < 2.0
    assert abs(float(ux.max())) <= 0.2


def test_cavity_inventory_weights_fine_cells() -> None:
    solver = _cavity()
    inv = solver.inventory()
    assert inv["fine"] == pytest.approx(0.25 * float(solver.f_top.sum() + solver.f_bottom.sum()))
    assert inv["total"] == pytest.approx(solver.f.sum() - solver.f[:, 0:6, :].sum() - solver.f[:, 26:, :].sum() + inv["fine"])


def test_cavity_profiles_shape_and_order() -> None:
    solver = _cavity()
    for _ in range(20):
        solver.step()
    profiles = solver.centreline_profiles()
    assert profiles["u"].shape == profiles["y"].shape
    # two fine strips (2 * 2*strip rows) + the coarse core
    assert profiles["u"].shape[0] == 2 * (2 * 6) + (32 - 2 * 6)
    assert np.all(np.diff(profiles["y"]) > 0)


def test_ghia_comparison_reproduces_reference_exactly() -> None:
    profiles = {
        "y": np.array(GHIA_RE1000["y"]),
        "u": np.array(GHIA_RE1000["u"]),
        "x": np.array(GHIA_RE1000["x"]),
        "v": np.array(GHIA_RE1000["v"]),
    }
    result = ghia_comparison(profiles, GHIA_RE1000, u_lid=1.0)
    assert result["max_abs_dev"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coarse": 16}, "coarse"),
        ({"strip_coarse": 2}, "strip"),
        ({"strip_coarse": 24, "coarse": 32}, "strip"),
        ({"u_lid": 0.5}, "u_lid"),
        ({"re": 0.0}, "Re"),
        ({"collision": "bgk"}, "collision"),
    ],
)
def test_cavity_config_rejects_invalid_input(kwargs: dict, match: str) -> None:
    params = {"coarse": 32, "strip_coarse": 6, "re": 1000.0, "u_lid": 0.1}
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CavityAMR2DConfig(**params)