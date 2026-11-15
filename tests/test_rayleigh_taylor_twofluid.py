"""Invariants and reference values of the finite-density-ratio RT capability.

The fast tests below pin the *model* (pseudopotential, force sign, mass
conservation, wall closure, interface geometry, closed-form reference); the
gated physics result lives in
``benchmarks/case_pool/rayleigh_taylor_atwood05/run_twofluid.py`` and is checked
against its stored artifacts by :func:`test_case_pool_twofluid_artifacts`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.rayleigh_taylor_twofluid import (
    ATWOOD_NOMINAL,
    SCMPRTConfig,
    atwood_from_ratio,
    init_scmp_rt,
    interface_crossings,
    layzer_velocity,
    psi_exp,
    ratio_from_atwood,
    column_front_state,
    column_height,
    precondition_scmp_rt,
    run_scmp_rt,
    scmp_coexistence,
    scmp_forces,
    scmp_step,
)

ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "case_pool"
    / "rayleigh_taylor_atwood05"
    / "artifacts_twofluid"
)


# ---------------------------------------------------------------------------
# Pseudopotential and Atwood algebra
# ---------------------------------------------------------------------------


def test_psi_exp_bounded_monotone():
    rho = torch.linspace(1e-3, 20.0, 101, dtype=torch.float64)
    psi = psi_exp(rho)
    assert torch.all(psi > 0.0) and torch.all(psi < 1.0)
    assert torch.all(psi[1:] > psi[:-1])
    # psi -> 1 for large rho (the bounded branch of the exp pseudopotential)
    assert pytest.approx(float(psi_exp(torch.tensor(30.0, dtype=torch.float64))), abs=1e-12) == 1.0


def test_atwood_ratio_roundtrip():
    for ratio in (1.2, 2.0, 3.0, 12.3):
        a = atwood_from_ratio(ratio)
        assert pytest.approx(ratio_from_atwood(a), rel=1e-12) == ratio
    # the anchor Atwood number is exactly the density ratio 3
    assert pytest.approx(atwood_from_ratio(ratio_from_atwood(ATWOOD_NOMINAL)), rel=1e-12) == 0.5


# ---------------------------------------------------------------------------
# Force direction: sc_g > 0 must be attractive (points up the psi gradient)
# ---------------------------------------------------------------------------


def test_scmp_force_is_attractive_and_vanishes_on_uniform_density():
    ny, nx = 24, 8
    rho = torch.full((ny, nx), 1.0, dtype=torch.float64)
    # uniform field -> no interaction force anywhere but the (mirrored) walls
    fx, fy = scmp_forces(rho, sc_g=0.5)
    assert float(fx.abs().max()) < 1e-12
    assert float(fy.abs().max()) < 1e-12

    # dense band in the middle -> the force on the band must point towards it,
    # i.e. the outermost rows are pushed inward (up on the bottom half)
    rho = torch.full((ny, nx), 0.3, dtype=torch.float64)
    rho[8:16, :] = 1.6
    fx, fy = scmp_forces(rho, sc_g=0.5)
    # the stencil is nearest-neighbour, so the force lives on the rows adjacent
    # to each interface: row 7 is below the band (pushed up, towards it), row 16
    # is above it (pushed down)
    assert float(fy[7, :].mean()) > 0.0
    assert float(fy[15, :].mean()) < 0.0
    assert float(fx.abs().max()) < 1e-10  # translational symmetry in x


def test_gravity_is_downward_and_proportional_to_mass():
    rho = torch.full((16, 4), 2.0, dtype=torch.float64)
    _, fy = scmp_forces(rho, sc_g=0.0, gravity_y=-1e-3)
    assert float(fy.mean()) == pytest.approx(-2.0e-3, rel=1e-12)


# ---------------------------------------------------------------------------
# Conservation / closure invariants of one step
# ---------------------------------------------------------------------------


def test_no_force_step_conserves_mass_and_keeps_uniform_state():
    torch.manual_seed(0)
    ny, nx = 16, 8
    rho = torch.full((ny, nx), 1.0, dtype=torch.float64)
    rho[0, :] = 0.0
    rho[-1, :] = 0.0  # solid rows store no mass
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    mass0 = float(f.sum())
    for _ in range(20):
        f = scmp_step(f, tau=1.0, sc_g=0.0, gravity_y=0.0)
    assert float(f.sum()) == pytest.approx(mass0, rel=1e-12)
    rho_end = f.sum(dim=0)[1:-1, :]  # solid rows hold no mass
    assert float(rho_end.max() - rho_end.min()) < 1e-10  # stays uniform
    # bounce-back closure: the wall rows stay at rest
    c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                      [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float64)
    m = (f * c[:, 0].view(9, 1, 1)).sum(dim=0)
    assert float(m.abs().max()) < 1e-12


def test_gravity_step_conserves_mass_and_develops_no_horizontal_drift():
    ny, nx = 32, 8
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    phi = torch.tanh((y - ny / 2) / 2.0).expand(ny, nx).contiguous()
    rho = 1.0 + 0.6 * phi
    rho[0, :] = 0.0
    rho[-1, :] = 0.0  # solid rows store no mass
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    mass0 = float(f.sum())
    for _ in range(50):
        f = scmp_step(f, tau=0.8, sc_g=0.5, gravity_y=-1e-4)
    assert float(f.sum()) == pytest.approx(mass0, rel=1e-12)
    c = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                      [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float64)
    mx = (f * c[:, 0].view(9, 1, 1)).sum(dim=0)
    assert float(mx.abs().max()) < 1e-10  # no net horizontal momentum (periodic x)


# ---------------------------------------------------------------------------
# Interface geometry
# ---------------------------------------------------------------------------


def test_interface_crossing_interpolates_the_zero_level():
    ny, nx = 40, 4
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    phi = torch.tanh((y - 20.5) / 1.5).expand(ny, nx).contiguous()
    h = interface_crossings(phi)
    # exactly one crossing per column, at the zero level of the tanh profile
    assert int(torch.isnan(h).sum()) == (ny - 1) * nx - nx
    assert float(h[~torch.isnan(h)].mean()) == pytest.approx(20.5, abs=0.05)
    # phi is positive in the heavy phase (on top), so the crossing is ascending
    assert float(phi[21, 0]) > 0.0 > float(phi[20, 0])


def test_front_state_reads_spike_and_bubble_apart():
    cfg = SCMPRTConfig(lam=32, layers=6.0, steps=1)
    ny = cfg.ny
    x = torch.arange(cfg.nx, dtype=torch.float64).view(1, cfg.nx)
    y = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    # a spike (heavy tongue hanging down) in the middle of the column
    y_int = cfg.y_interface + 3.0 * torch.cos(2.0 * math.pi * x / cfg.lam)
    phi = torch.tanh((y - y_int) / 1.5)
    from tensorlbm.rayleigh_taylor_twofluid import front_state

    st = front_state(phi, cfg)
    assert st["valid"]
    assert st["spike_q10_lam"] < 0.0 < st["bubble_q90_lam"]


# ---------------------------------------------------------------------------
# Closed-form reference (two-fluid Layzer-Goncharov)
# ---------------------------------------------------------------------------


def test_layzer_two_fluid_asymmetry_is_sqrt_three_at_half():
    ref = layzer_velocity(0.5, 2e-4, 64.0)
    assert ref["asymmetry"] == pytest.approx(math.sqrt(3.0), rel=1e-12)
    # absolute value cross-checks the sibling Boussinesq case report
    # (u_spike ref = 6.383e-2 at A = 1/2, g = 2e-4, lam = 64, C = 1)
    assert ref["u_spike"] == pytest.approx(6.383076487e-2, rel=1e-6)
    assert ref["u_bubble"] == pytest.approx(ref["u_spike"] / math.sqrt(3.0), rel=1e-12)


def test_layzer_scaling_in_gravity_and_wavelength():
    a = layzer_velocity(0.5, 1e-4, 64.0)
    b = layzer_velocity(0.5, 4e-4, 64.0)
    c = layzer_velocity(0.5, 1e-4, 256.0)
    assert b["u_spike"] / a["u_spike"] == pytest.approx(2.0, rel=1e-12)  # sqrt(g)
    assert c["u_spike"] / a["u_spike"] == pytest.approx(2.0, rel=1e-12)  # sqrt(lam)


# ---------------------------------------------------------------------------
# Model behaviour: coexistence and the finite-Atwood asymmetry
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_coexistence_separates_at_strong_coupling():
    """Both couplings separate, and the measurement is *not* monotone here.

    The ratio measured at (ny=48, 1500 steps) is 1.27 at sc_g=0.35 and 1.19 at
    sc_g=0.60 -- i.e. *decreasing* with coupling.  That is not a physics claim: the
    coexistence density is a relaxation-time/box-size dependent read-out at this
    short run length (docs/CASE-RT-TWOFLUID-PRECOND.md finding L4: the same
    sc_g=2.8 gives A=0.505 at (ny=64, 3000 steps) and A=0.652 at (ny=48, 1500
    steps), a 30 % spread).  The invariant this test may assert is therefore only
    that both points separate and stay mass-conserving; the convergence of the
    calibration is an open item, so it is not asserted here.
    """
    low = scmp_coexistence(0.35, steps=1500, ny=48, tau=1.0)
    high = scmp_coexistence(0.60, steps=1500, ny=48, tau=1.0)
    assert high.stable and low.stable
    assert high.ratio > 1.0 and low.ratio > 1.0
    assert abs(high.mass_drift_pct) < 1.0


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason=(
    "known limitation (docs/CASE-RT-TWOFLUID-PRECOND.md): at lam=48/layers=8/"
    "g=3e-4 the staged start-up still ends on a guard before the front window, and "
    "the finite-density-ratio spike/bubble asymmetry this capability exists for is "
    "not yet demonstrated (probe P4 + finding L2/L5).  Executable record of the "
    "open item: it flips to XPASS when the asymmetry is actually measured."))
def test_finite_atwood_run_produces_asymmetric_fronts():
    """The whole point of this capability: the fronts are *not* mirror images."""
    co = scmp_coexistence(0.60, steps=1500, ny=48, tau=1.0)
    cfg = SCMPRTConfig(lam=48, layers=8.0, gravity=3.0e-4, tau_relax=0.8,
                       a0_over_lam=0.04, steps=900, sample_every=10, name="ut")
    # staged start-up: the single-shot start-up of m21799 put Ma = 0.28 in the
    # first step (force impulse of a non-equilibrium interface profile, finding
    # L1) and aborted on the guard before the interface moved at all.
    res = run_scmp_rt(cfg, rho_high=co.rho_high, rho_low=co.rho_low,
                      precondition_steps=200, ramp_steps=100, ma_guard="bulk")
    # the box is deliberately small (lam=48, layers=8, g=3e-4): the spike may reach
    # the wall inside `steps`, and the front *estimator* is itself under test
    # elsewhere (finding L2), so this test asserts only that the staged start-up
    # gets past the impurity of step 1 and runs to a physical stopping reason
    assert res["stop_reason"] in {"completed", "wall"}, res["stop_reason"]
    assert res["atwood_measured"] > 0.3
    assert math.isfinite(res["ma_peak_bulk"])
    # a Boussinesq solver returns exactly 1.0 here (see the case report)
    assert res["asym_sim"] > 1.05
    assert math.isfinite(res["err_spike_pct"])


# ---------------------------------------------------------------------------
# Gated case artifacts (skipped when the campaign has not been run here)
# ---------------------------------------------------------------------------


def test_case_pool_twofluid_artifacts():
    """Consistency of the stored campaign, including its *honest* failure state.

    The finite-Atwood arm did not reach its front-position window (both runs were
    stopped by the Mach guard, see ``docs/CASE-RT-TWOFLUID.md`` sec. 4).  The test
    therefore asserts self-consistency and *forbids a silent pass*: a case that
    did not reach the window must not be reported as a gated PASS, and a PASS must
    carry a finite error inside the tolerance.
    """
    summary_path = ARTIFACT_ROOT / "summary.json"
    if not summary_path.exists():
        pytest.skip("two-fluid campaign artifacts not present in this checkout")
    cases = json.loads(summary_path.read_text())["cases"]
    assert cases, "empty case list"
    for case in cases:
        assert case["stop_reason"] in {"completed", "wall", "mach"}, case["name"]
        assert abs(case["mass_drift_pct"]) < 5.0, case["name"]
        # the calibration of this arm is pinned at the anchor Atwood number
        assert abs(case["atwood_measured"] - 0.5) < 0.05, case["name"]
        if case["gate_pass"]:
            assert math.isfinite(case["err_spike_pct"])
            assert abs(case["err_spike_pct"]) <= 15.0
            assert case["stop_reason"] in {"completed", "wall"}
        else:
            # no window reached -> no fabricated velocity
            assert not math.isfinite(case["u_spike_sim"]) or abs(case["err_spike_pct"]) > 15.0
    incomplete = [c for c in cases if not c["gate_pass"]]
    assert incomplete, (
        "the two-fluid arm has no PASS: if this changes, the case report "
        "(docs/CASE-RT-TWOFLUID.md) must be updated with the new numbers"
    )


# ---------------------------------------------------------------------------
# Staged start-up, split Mach guards, column-height estimator
# (CASE-RT-INSTABILITY-ATWOOD, staged-start-up arm -- see
#  docs/CASE-RT-TWOFLUID-PRECOND.md)
# ---------------------------------------------------------------------------


def test_ma_split_reports_all_iface_and_bulk():
    """The three Mach numbers are returned per sample and the guard selects one."""
    cfg = SCMPRTConfig(lam=32, layers=4.0, sc_g=2.8, gravity=2.0e-4, tau_relax=1.0,
                       interface_width=2.0, steps=40, sample_every=10, ma_limit=1e9,
                       name="ut_ma")
    res = run_scmp_rt(cfg, rho_high=1.172, rho_low=0.386, ramp_steps=20,
                      ma_guard="bulk")
    assert res["ma_guard"] == "bulk"
    for key in ("ma_peak", "ma_peak_iface", "ma_peak_bulk"):
        assert math.isfinite(res[key])
    row = res["history"][-1]
    for key in ("ma_all", "ma_iface", "ma_bulk", "gravity_y"):
        assert key in row
    with pytest.raises(ValueError):
        run_scmp_rt(cfg, rho_high=1.172, rho_low=0.386, ma_guard="nonsense")


def test_gravity_scale_zero_is_a_null_dynamics_control():
    """``gravity_scale = 0`` keeps the body force off for the whole run."""
    cfg = SCMPRTConfig(lam=32, layers=4.0, sc_g=2.8, gravity=2.0e-4, tau_relax=1.0,
                       interface_width=2.0, steps=40, sample_every=10, ma_limit=1e9,
                       name="ut_null")
    res = run_scmp_rt(cfg, rho_high=1.172, rho_low=0.386, gravity_scale=0.0,
                      precondition_steps=60, ramp_steps=0, ma_guard="bulk")
    assert all(row["gravity_y"] == 0.0 for row in res["history"])
    assert res["gravity_scale"] == 0.0


def test_preconditioning_lowers_the_startup_mach():
    """Staged start-up removes the initial force impulse (measured defect fix).

    The single-shot start-up (tanh(width=2) profile + gravity at step 1) carries a
    force impulse |F|/rho that reads as Ma = 0.28 in the *first* step -- above the
    0.20 guard, which is why the campaign aborted at step 5 (docs/...-PRECOND.md
    finding L1 / probe P1).  Relaxing a flat interface at g = 0 first must bring
    the interface-band Mach at the moment the mode is seeded below that impulse.
    """
    cfg = SCMPRTConfig(lam=32, layers=4.0, sc_g=2.8, gravity=2.0e-4, tau_relax=1.0,
                       interface_width=2.0, steps=1, sample_every=1, ma_limit=1e9,
                       name="ut_pre")
    legacy = run_scmp_rt(cfg, rho_high=1.172, rho_low=0.386)
    _, info = precondition_scmp_rt(cfg, rho_high=1.172, rho_low=0.386,
                                   relax_steps=200, sample_every=50)
    assert legacy["history"][0]["ma"] > 0.15          # the impulse that aborted it
    assert info["spurious_ma_floor"] < legacy["history"][0]["ma"]
    assert info["relax_trace"][-1]["step"] == 200


def test_column_height_is_a_clamped_mass_moment():
    """``column_height`` sums the clamped heavy fraction over the free rows."""
    cfg = SCMPRTConfig(lam=16, layers=4.0)
    rho = torch.zeros((cfg.ny, cfg.nx), dtype=torch.float64)
    rho[1:cfg.ny // 2, :] = 0.2       # vapour below the interface
    rho[cfg.ny // 2:-1, :] = 1.2      # liquid above the interface
    h = column_height(rho, rho_high=1.172, rho_low=0.386)
    liquid_rows = cfg.ny // 2 - 1
    assert torch.allclose(h, torch.full_like(h, float(liquid_rows)), atol=2.0)
    # an all-vapour and an all-liquid box clamp to the two extremes
    assert float(column_height(torch.full_like(rho, 0.2), rho_high=1.172,
                               rho_low=0.386).max()) == 0.0
    assert float(column_height(torch.full_like(rho, 1.2), rho_high=1.172,
                               rho_low=0.386).min()) == float(cfg.ny - 2)


@pytest.mark.xfail(strict=False, reason=(
    "known limitation L2 (docs/CASE-RT-TWOFLUID-PRECOND.md): the column-height "
    "moment does not resolve an interface displacement -- a known 1.28-row cosine "
    "shift reads back ~0 instead of a0/lambda = 0.02.  Kept as an executable "
    "record of the defect: once the estimator is fixed this test flips to XPASS."))
def test_column_estimator_resolves_a_known_interface_shift():
    cfg = SCMPRTConfig(lam=64, layers=4.0, sc_g=2.8, gravity=2.0e-4, tau_relax=1.0,
                       a0_over_lam=0.02, interface_width=2.0)
    y = torch.arange(cfg.ny, dtype=torch.float64).view(cfg.ny, 1)
    x = torch.arange(cfg.nx, dtype=torch.float64).view(1, cfg.nx)
    # a diffuse equilibrium-like profile, displaced by exactly a0 = 0.02 lambda
    base = 0.779 + 0.393 * torch.tanh((y - 0.5 * cfg.ny) / 2.0)
    moved = 0.779 + 0.393 * torch.tanh(
        (y - (0.5 * cfg.ny + cfg.a0_over_lam * cfg.lam
              * torch.cos(2.0 * math.pi * x / cfg.lam))) / 2.0)
    ref = 0.5 * cfg.ny
    st = column_front_state(moved, cfg, rho_high=1.172, rho_low=0.386,
                            reference_row=ref)
    assert abs(st["amplitude_lam"] - cfg.a0_over_lam) < 0.2 * cfg.a0_over_lam
