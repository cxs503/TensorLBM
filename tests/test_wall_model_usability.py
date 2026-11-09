"""Wall-modelled-LES usability criteria (node IN61-N06).

The audited numbers live in ``artifacts/in61_n06_wall_model_usability/``; these
tests pin the properties that must not regress:

* the composite-law inversion is exact to solver precision on manufactured
  states, for both laws, over the full ``u_tau`` / ``y+`` envelope used by the
  ship wind case;
* the analytic law derivatives agree with finite differences;
* the verbatim MKM1999 tables parse and the legacy ``mkm180.csv`` shipped in the
  repository matches them column-by-column;
* the a-priori wall-stress reconstruction and the log-layer slope criteria are
  met by the audited law (and *not* met by Spalding's classic constants, which is
  a recorded finding, not a tolerated drift);
* the three negative controls each break the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorlbm.turbulence import wall_model_usability as wmu
from tensorlbm.turbulence.wall_model_usability import (
    REICHARDT,
    SPALDING,
    WallLawParams,
    assert_wall_model_usable,
    assess_wall_model_usability,
    channel_level_deviation,
    channel_slope_deviation,
    channel_wall_stress_consistency,
    dns_wall_stress_recovery,
    equilibrium_wall_modelled_channel,
    grid_ladder_verdict,
    inversion_relative_error,
    load_mkm_profile,
    manufactured_wall_state,
    richardson_ratio,
    solve_friction_velocity,
    solve_friction_velocity_bisection,
    wall_law_du_plus_dy_plus,
    wall_law_u_plus,
    wall_law_y_plus,
    wall_stress_budget,
    wall_stress_residual,
)

WORK = Path(__file__).resolve().parents[1]
REF = WORK / "benchmarks" / "pending" / "channel_turb_3d" / "dns_ref"
NU_AIR = 1.2e-5
LAWS = (SPALDING, REICHARDT)
Y_PLUS = (0.05, 0.5, 1.0, 5.0, 11.0, 30.0, 100.0, 500.0, 2000.0, 20000.0, 1e5)


@pytest.fixture(scope="module")
def profiles():
    return {
        "mkm180": load_mkm_profile(REF / "mkm180.means"),
        "mkm590": load_mkm_profile(REF / "mkm590.means"),
    }


@pytest.fixture(scope="module")
def channel(profiles):
    """Channel solution with the van Driest constant fitted at Re_tau = 587."""
    best, best_deviation = None, None
    for van_driest in (26.0, 27.0, 28.0, 29.0, 30.0):
        candidate = equilibrium_wall_modelled_channel(
            profiles["mkm590"].re_tau, law=REICHARDT, van_driest_a=van_driest
        )
        deviation = channel_level_deviation(profiles["mkm590"], candidate)["rms_relative"]
        if best_deviation is None or deviation < best_deviation:
            best, best_deviation = candidate, deviation
    return best


# ---------------------------------------------------------------------------
# criterion 1: inversion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_manufactured_states_are_inverted_to_solver_precision(law):
    y_plus = torch.tensor(Y_PLUS, dtype=torch.float64)
    for u_tau_true in (1e-6, 1e-4, 0.05, 1.0):
        u_t, y = manufactured_wall_state(u_tau_true, y_plus, NU_AIR, law=law)
        solution = solve_friction_velocity(u_t, y, NU_AIR, law=law)
        assert solution.converged
        error = inversion_relative_error(
            solution.u_tau, torch.full_like(solution.u_tau, u_tau_true)
        )
        # criterion 1 asks for < 1e-8; the measured floor is ~1e-15
        assert float(error.max().item()) < 1e-8
        assert float(error.max().item()) < 1e-12


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_inversion_matches_independent_bisection(law):
    y_plus = torch.tensor([1.0, 12.0, 60.0, 400.0, 5000.0], dtype=torch.float64)
    u_t, y = manufactured_wall_state(0.03, y_plus, NU_AIR, law=law)
    newton = solve_friction_velocity(u_t, y, NU_AIR, law=law).u_tau
    reference = solve_friction_velocity_bisection(u_t, y, NU_AIR, law=law)
    assert float(((newton - reference) / reference).abs().max().item()) < 1e-10


def test_inversion_agrees_with_the_existing_spalding_module():
    """The repo's bisection Spalding solver and this Newton agree to 1e-9."""
    from tensorlbm.spalding_wall_model import (
        solve_spalding_friction_velocity as legacy_solve,
    )

    y_plus = torch.tensor([5.0, 30.0, 120.0, 900.0], dtype=torch.float64)
    u_t, y = manufactured_wall_state(0.02, y_plus, NU_AIR, law=SPALDING)
    ours = solve_friction_velocity(u_t, y, NU_AIR, law=SPALDING).u_tau
    legacy = legacy_solve(u_t, y, NU_AIR).to(dtype=torch.float64)
    assert float(((ours - legacy) / legacy).abs().max().item()) < 1e-8


def test_inversion_is_overflow_free_beyond_the_log_layer():
    """A 1e5 y+ sample with a 1e-6 u_tau must not produce NaN/inf."""
    y_plus = torch.tensor([1e3, 1e4, 1e5], dtype=torch.float64)
    u_t, y = manufactured_wall_state(1e-6, y_plus, NU_AIR, law=SPALDING)
    solution = solve_friction_velocity(u_t, y, NU_AIR, law=SPALDING)
    assert torch.isfinite(solution.u_tau).all()
    assert float(inversion_relative_error(
        solution.u_tau, torch.full_like(solution.u_tau, 1e-6)).max().item()) < 1e-8


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_law_derivative_matches_finite_difference(law):
    y_plus = torch.tensor([2.0, 10.0, 30.0, 100.0, 500.0], dtype=torch.float64)
    step = 1e-6
    fd = (wall_law_u_plus(y_plus + step, law) - wall_law_u_plus(y_plus - step, law)) / (
        2 * step
    )
    analytic = wall_law_du_plus_dy_plus(y_plus, law)
    assert float(((fd - analytic) / analytic).abs().max().item()) < 1e-5


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_law_roundtrip(law):
    y_plus = torch.tensor([0.5, 5.0, 30.0, 200.0, 1000.0], dtype=torch.float64)
    back = wall_law_y_plus(wall_law_u_plus(y_plus, law), law)
    assert float(((back - y_plus) / y_plus).abs().max().item()) < 1e-10


# ---------------------------------------------------------------------------
# DNS tables
# ---------------------------------------------------------------------------
def test_dns_tables_parse_verbatim(profiles):
    assert profiles["mkm180"].re_tau == pytest.approx(178.12, abs=1e-2)
    assert profiles["mkm590"].re_tau == pytest.approx(587.19, abs=1e-2)
    assert len(profiles["mkm180"].y_plus) == 65
    assert len(profiles["mkm590"].y_plus) == 129
    # wall-normal shear normalisation: dU+/dy+ = 1 at the wall
    assert float(profiles["mkm590"].du_plus_dy_plus[0].item()) == pytest.approx(1.0, abs=1e-3)


def test_legacy_180_csv_matches_the_verbatim_table(profiles):
    import numpy as np

    legacy = np.loadtxt(REF / "mkm180.csv", delimiter=",", skiprows=1)
    profile = profiles["mkm180"]
    n = min(len(legacy), len(profile.y_plus))
    assert np.allclose(legacy[:n, 1], profile.y_plus[:n].numpy(), atol=1e-9)
    assert np.allclose(legacy[:n, 2], profile.u_plus[:n].numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# criterion 2: wall-stress budget
# ---------------------------------------------------------------------------
def test_stress_residual_definition():
    residual = wall_stress_residual(
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([0.4, 0.5], dtype=torch.float64),
        torch.tensor([0.6, 0.5], dtype=torch.float64),
    )
    assert float(residual[0].item()) == pytest.approx(0.0, abs=1e-15)
    assert float(residual[1].item()) == pytest.approx(0.0, abs=1e-15)


def _stencil_residual(law, y1_plus, y2_plus, u_tau=0.05):
    y1 = torch.tensor([y1_plus], dtype=torch.float64)
    y2 = torch.tensor([y2_plus], dtype=torch.float64)
    u1 = wall_law_u_plus(y1, law) * u_tau
    u2 = wall_law_u_plus(y2, law) * u_tau
    return wall_stress_budget(
        u1,
        u2,
        y1 * NU_AIR / u_tau,
        y2 * NU_AIR / u_tau,
        torch.full_like(u1, u_tau**2),
        NU_AIR,
        law=law,
    )["max_residual"]


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_stress_budget_meets_two_percent_with_a_local_stencil(law):
    """A dy+ = 5 stencil satisfies the 2 % budget; a dy+ = 30 one cannot."""
    local = _stencil_residual(law, 15.0, 20.0)
    coarse = _stencil_residual(law, 15.0, 45.0)
    assert local < 0.02
    assert coarse > local
    # the coarse stencil exceeds the criterion: this is the quantitative
    # resolution constraint recorded in the report
    assert coarse > 0.02


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.name)
def test_wall_stress_recovery_is_bounded_on_dns(profiles, law):
    for tag in ("mkm180", "mkm590"):
        recovery = dns_wall_stress_recovery(profiles[tag], law)
        # 4 % is the recorded Spalding-in-buffer-layer penalty; a regression
        # beyond it would mean the inversion or the table parsing broke
        assert recovery["max_tau_w_rel_error"] < 0.05
        assert recovery["mean_tau_w_rel_error"] < 0.04


def test_reichardt_meets_the_two_percent_stress_target(profiles):
    for tag in ("mkm180", "mkm590"):
        recovery = dns_wall_stress_recovery(profiles[tag], REICHARDT)
        assert recovery["max_tau_w_rel_error"] < 0.02


def test_spalding_classic_constants_miss_the_stress_target(profiles):
    """Recorded finding: Spalding's kappa=0.4/A=0.1108 fails 2 % at Re_tau~180."""
    recovery = dns_wall_stress_recovery(profiles["mkm180"], SPALDING)
    assert recovery["max_tau_w_rel_error"] > 0.02


# ---------------------------------------------------------------------------
# criterion 3: channel validation
# ---------------------------------------------------------------------------
def test_channel_profile_reproduces_dns_level(channel, profiles):
    level = channel_level_deviation(profiles["mkm590"], channel)
    assert level["rms_relative"] < 0.02


def test_channel_slope_deviation_within_criterion(channel, profiles):
    for tag in ("mkm180", "mkm590"):
        solution = equilibrium_wall_modelled_channel(
            profiles[tag].re_tau, law=REICHARDT, van_driest_a=channel.van_driest_a
        )
        slope = channel_slope_deviation(profiles[tag], solution)
        assert slope["slope_deviation"] < 0.05


def test_channel_slope_is_not_a_tautology(channel, profiles):
    """The DNS slope and the channel slope are independently computed numbers."""
    slope = channel_slope_deviation(profiles["mkm590"], channel)
    assert slope["kappa_dns"] != pytest.approx(slope["kappa_model"], abs=1e-6)
    assert 0.3 < slope["kappa_dns"] < 0.5
    assert 0.3 < slope["kappa_model"] < 0.5


def test_channel_wall_stress_consistency_is_reported(channel):
    consistency = channel_wall_stress_consistency(channel)
    assert consistency["samples"]
    # the equilibrium channel is not fitted to the wall law: the residual is a
    # measured number (few per cent), recorded in the artifact
    assert 0.0 < consistency["max_tau_w_rel_error"] < 0.10


def test_channel_requires_a_band(channel):
    with pytest.raises(ValueError):
        channel_slope_deviation(
            wmu.MkmProfile(
                re_tau=60.0,
                y_plus=torch.linspace(60.0, 60.0, 10, dtype=torch.float64),
                u_plus=torch.linspace(1.0, 2.0, 10, dtype=torch.float64),
                du_plus_dy_plus=torch.ones(10, dtype=torch.float64),
            ),
            channel,
        )


# ---------------------------------------------------------------------------
# criterion 4: grid convergence arithmetic
# ---------------------------------------------------------------------------
def test_richardson_ratio_and_verdict():
    # changes shrink with refinement -> monotone (R = 0.05 / 0.10 = 0.5)
    assert richardson_ratio(1.20, 1.10, 1.05) == pytest.approx(0.5)
    converged = grid_ladder_verdict(1.20, 1.10, 1.05, ratio=2.0)
    assert converged["richardson_ratio"] == pytest.approx(0.5)
    assert converged["monotone"] and converged["passed"]

    # oscillating ladder: the change grows towards the fine grid (R = 2)
    diverging = grid_ladder_verdict(1.10, 1.05, 1.15, ratio=1.41)
    assert diverging["richardson_ratio"] == pytest.approx(2.0)
    assert not diverging["monotone"]


def test_grid_verdict_accepts_a_converged_fine_pair():
    verdict = grid_ladder_verdict(1.0, 0.500, 0.505, ratio=2.0)
    assert verdict["fine_pair_relative_change"] < 0.05
    assert verdict["passed"]


# ---------------------------------------------------------------------------
# criterion 5: the gate and its negative controls
# ---------------------------------------------------------------------------
def _reference_evidence(profiles, channel):
    slope = {}
    for tag in ("mkm180", "mkm590"):
        solution = equilibrium_wall_modelled_channel(
            profiles[tag].re_tau, law=REICHARDT, van_driest_a=channel.van_driest_a
        )
        slope[profiles[tag].re_tau] = channel_slope_deviation(profiles[tag], solution)[
            "slope_deviation"
        ]
    stress = max(
        dns_wall_stress_recovery(profiles[tag], REICHARDT)["max_tau_w_rel_error"]
        for tag in ("mkm180", "mkm590")
    )
    return {
        "inversion_max_rel_error": float(
            inversion_relative_error(
                solve_friction_velocity(
                    *manufactured_wall_state(0.05, torch.tensor([30.0]), NU_AIR), NU_AIR
                ).u_tau,
                torch.tensor([0.05], dtype=torch.float64),
            ).max()
        ),
        "stress_max_rel_error": stress,
        "channel_slope_deviation": slope,
        "richardson": grid_ladder_verdict(1.30, 1.10, 1.05, ratio=1.41),
    }


def test_gate_passes_on_the_audited_configuration(profiles, channel):
    report = assert_wall_model_usable(
        assess_wall_model_usability(**_reference_evidence(profiles, channel))
    )
    assert report.passed
    assert report.failures == ()


def test_gate_rejects_a_missing_channel_leg(profiles, channel):
    evidence = _reference_evidence(profiles, channel)
    evidence["channel_slope_deviation"] = {
        profiles["mkm590"].re_tau: evidence["channel_slope_deviation"][
            profiles["mkm590"].re_tau
        ]
    }
    report = assess_wall_model_usability(**evidence)
    assert not report.passed
    assert "channel_slope_deviation" in report.failures


def test_negative_control_wrong_kappa_breaks_the_gate(profiles, channel):
    evidence = _reference_evidence(profiles, channel)
    wrong = WallLawParams("spalding", 0.55, 0.1108)
    y_plus = torch.tensor(Y_PLUS, dtype=torch.float64)
    u_t, y = manufactured_wall_state(0.05, y_plus, NU_AIR, law=SPALDING)
    recovered = solve_friction_velocity(u_t, y, NU_AIR, law=wrong)
    evidence["inversion_max_rel_error"] = float(
        inversion_relative_error(recovered.u_tau, torch.full_like(recovered.u_tau, 0.05))
        .max()
        .item()
    )
    report = assess_wall_model_usability(**evidence)
    assert evidence["inversion_max_rel_error"] > 1e-3
    assert not report.passed
    with pytest.raises(AssertionError):
        assert_wall_model_usable(report)


def test_negative_control_sampling_offset_breaks_the_gate(profiles, channel):
    evidence = _reference_evidence(profiles, channel)
    y_plus = torch.tensor(Y_PLUS, dtype=torch.float64)
    u_t, _ = manufactured_wall_state(0.05, y_plus, NU_AIR, law=REICHARDT)
    _, y_shifted = manufactured_wall_state(0.05, y_plus + 15.0, NU_AIR, law=REICHARDT)
    recovered = solve_friction_velocity(u_t, y_shifted, NU_AIR, law=REICHARDT)
    evidence["inversion_max_rel_error"] = float(
        inversion_relative_error(recovered.u_tau, torch.full_like(recovered.u_tau, 0.05))
        .max()
        .item()
    )
    report = assess_wall_model_usability(**evidence)
    assert evidence["inversion_max_rel_error"] > 1e-3
    assert not report.passed
    with pytest.raises(AssertionError):
        assert_wall_model_usable(report)


def test_negative_control_disabled_wall_law_breaks_the_gate(profiles, channel):
    evidence = _reference_evidence(profiles, channel)
    evidence["stress_max_rel_error"] = 1.0  # no wall shear reconstructed at all
    report = assess_wall_model_usability(**evidence)
    assert not report.passed
    assert "stress_rel_error" in report.failures
    with pytest.raises(AssertionError):
        assert_wall_model_usable(report)


def test_negative_control_helper_flags_only_failing_controls():
    triggered = wmu.negative_control("wrong_kappa", measured=4e-2, limit=1e-8)
    assert triggered.passed
    dormant = wmu.negative_control("wrong_kappa", measured=1e-12, limit=1e-8)
    assert not dormant.passed