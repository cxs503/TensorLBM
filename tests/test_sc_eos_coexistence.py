"""G3 ledger criteria: Shan–Chen coexistence of the cavitation pseudopotential.

Each test is a machine-checkable reading of ``benchmarks/gaps.md`` entry G3:

* C1 — the ledger's pressure anchor ``p(ρ=2.65) = −4.435`` at classic ``G = −5.5``;
* C2 — no liquid/vapour coexistence at that coupling (``ρ_l = 2.65`` is not a
  Maxwell coexistence density);
* C3 — the ledger's "valid coexistence" coupling ``G_c = −1.0``: the ledger
  numbers ``ρ_v=0.0296 / ρ_l=4.330 / ratio=146`` reproduce **only** with the
  dimensionally inconsistent ``∫(p−p_sat)dρ`` rule, whereas the correct
  ``∫(p−p_sat)dv`` construction gives a different pair, independently confirmed
  by ``μ(ρ_v) = μ(ρ_l)``;
* the landing fixes (default coupling, sign guard, density-vs-ψ force call).

Sign convention: ``sc_single_component_force`` gathers the backward neighbour, so
``G > 0`` is attractive in this library and ``G_classic = −G``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import cavitation, multiphase
from tensorlbm.cavitation import (
    COEXISTENCE_G,
    COEXISTENCE_RHO_L,
    COEXISTENCE_RHO_V,
    CavitationConfig,
    apply_cavitation_force,
    psi_cavitation,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.sc_eos_coexistence import (
    CS2,
    chemical_potential,
    classic_to_library,
    coexistence_exists,
    library_to_classic,
    maxwell_coexistence,
    mechanical_pressure,
    relax_scmp,
    spinodal_bounds,
)

# Ledger numbers, verbatim from benchmarks/gaps.md (entry G3)
LEDGER_G_CLASSIC_BAD = -5.5
LEDGER_P_AT_2P65 = -4.435
LEDGER_RHO_L_CLAIMED = 2.65
LEDGER_G_CLASSIC_VALID = -1.0
LEDGER_RHO_V = 0.0296
LEDGER_RHO_L = 4.330
LEDGER_RATIO = 146


def psi(rho: torch.Tensor) -> torch.Tensor:
    """The cavitation module's pseudopotential (module defaults)."""
    return psi_cavitation(rho)


def _rho_space_rule(G: float) -> tuple[float, float]:
    """Coexistence pair from the *wrong* rule ``∫(p−p_sat)dρ = 0`` (ledger's rule).

    Deliberately mis-scaled: kept in the test suite to prove which rule produced
    the ledger's ``0.0296 / 4.330 / 146`` triple.
    """
    bounds = spinodal_bounds(G, psi, 1e-5, 2e2, 4000)
    assert bounds is not None
    r_sp_v, r_sp_l = bounds

    def p_of(r: float) -> float:
        return float(mechanical_pressure(torch.tensor(r), G, psi))

    def bisect(f, a: float, b: float) -> float:
        fa = f(a)
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = f(m)
            if (fa > 0) == (fm > 0):
                a, fa = m, fm
            else:
                b = m
        return 0.5 * (a + b)

    def area(p_sat: float) -> float:
        r_v = bisect(lambda r: p_of(r) - p_sat, 1e-5, r_sp_v)
        r_l = bisect(lambda r: p_of(r) - p_sat, r_sp_l, 2e2)
        u = torch.linspace(math.log(r_v), math.log(r_l), 800, dtype=torch.float64)
        p = mechanical_pressure(torch.exp(u), G, psi)
        return float(torch.trapezoid((p - p_sat) * torch.exp(u), u))

    rs = torch.exp(torch.linspace(math.log(1e-5), math.log(2e2), 4000))
    ps = mechanical_pressure(rs, G, psi)
    i_v = int(torch.nonzero(rs <= r_sp_v).flatten()[-1])
    i_l = int(torch.nonzero(rs >= r_sp_l).flatten()[0])
    lo = max(float(ps[: i_v + 1].min()), float(ps[i_l:].min()))
    hi = min(float(ps[: i_v + 1].max()), float(ps[i_l:].max()))
    trials = torch.linspace(lo, hi, 120, dtype=torch.float64)
    vals = [area(float(p)) for p in trials]
    for k in range(len(vals) - 1):
        if (vals[k] > 0) != (vals[k + 1] > 0):
            p_sat = bisect(area, float(trials[k]), float(trials[k + 1]))
            r_v = bisect(lambda r: p_of(r) - p_sat, 1e-5, r_sp_v)
            r_l = bisect(lambda r: p_of(r) - p_sat, r_sp_l, 2e2)
            return r_v, r_l
    raise AssertionError("rho-space rule found no coexistence pair")


# ---------------------------------------------------------------------------
# EOS bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pressure_matches_classic_shan_chen_form():
    """p = ρcs² − ½Gcs²ψ² is the classic p = ρ/3 + G_cψ²/6 with G_c = −G."""
    rho = torch.tensor([0.4, 1.0, 2.65, 6.0], dtype=torch.float64)
    for G in (0.7, 1.0, 5.5):
        p = mechanical_pressure(rho, G, psi)
        classic = rho * CS2 + library_to_classic(G) * psi(rho) ** 2 / 6.0
        assert torch.allclose(p, classic, rtol=1e-10, atol=1e-12)
        assert classic_to_library(-G) == G
        assert G == -library_to_classic(G)


@pytest.mark.unit
def test_ledger_anchor_pressure_at_rho_two_point_six_five():
    """G3 claim C1: p(ρ=2.65) = −4.435 at classic G = −5.5 (library G = +5.5)."""
    G = classic_to_library(LEDGER_G_CLASSIC_BAD)
    p = float(mechanical_pressure(torch.tensor(LEDGER_RHO_L_CLAIMED), G, psi))
    assert p == pytest.approx(LEDGER_P_AT_2P65, rel=1e-3)
    assert abs(p - LEDGER_P_AT_2P65) / abs(LEDGER_P_AT_2P65) < 1e-3
    assert p < 0.0  # the liquid branch is mechanically impossible here


# ---------------------------------------------------------------------------
# C2: no coexistence at the historical default coupling
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_no_maxwell_coexistence_at_ledger_default_coupling():
    """G3 claim C2: classic G=−5.5 admits no two-phase state, with a stated reason."""
    G = classic_to_library(LEDGER_G_CLASSIC_BAD)
    result = maxwell_coexistence(G, psi)
    assert result.found is False
    assert result.rho_liquid is None and result.rho_vapour is None
    assert "no equal-area solution" in result.reason
    # the loop is genuinely deep: p_min is far below zero
    assert result.p_min is not None and result.p_min < -7.0
    # a spinodal band does exist (so the failure is not "monotone EOS")
    bounds = spinodal_bounds(G, psi, 1e-5, 2e2, 12000)
    assert bounds is not None
    assert bounds[0] < LEDGER_RHO_L_CLAIMED < bounds[1]


@pytest.mark.physics
def test_weak_coupling_has_no_spinodal_at_all():
    """Below the coexistence threshold the EOS is monotone (different failure mode)."""
    result = maxwell_coexistence(0.3, psi)
    assert result.found is False
    assert "no spinodal" in result.reason
    assert coexistence_exists(0.3, psi) is False


@pytest.mark.physics
def test_ledger_default_coupling_diverges_in_real_relaxation():
    """The old default (classic −5.5) is not just non-coexisting: it blows up."""
    row = relax_scmp(-5.5, psi, nx=32, ny=32, rho0=1.5, steps=200)
    assert row["diverged"] is True
    assert row["rho_max"] is None


# ---------------------------------------------------------------------------
# C3: the ledger's "valid" coupling and the rule behind its numbers
# ---------------------------------------------------------------------------


@pytest.mark.physics
def test_correct_maxwell_pair_at_ledger_valid_coupling():
    """Correct ∫(p−p_sat)dv construction at classic G=−1.0 (library G=+1.0)."""
    G = classic_to_library(LEDGER_G_CLASSIC_VALID)
    result = maxwell_coexistence(G, psi)
    assert result.found is True
    assert result.rho_vapour is not None and result.rho_liquid is not None
    # the correct pair is far from the ledger's rho_v: the ledger used a wrong rule
    assert abs(result.rho_vapour - LEDGER_RHO_V) / LEDGER_RHO_V > 1.0
    assert abs(result.density_ratio - LEDGER_RATIO) / LEDGER_RATIO > 0.5
    # the liquid density lands within 20 % of the ledger's digit (same coupling)
    assert abs(result.rho_liquid - LEDGER_RHO_L) / LEDGER_RHO_L < 0.20


@pytest.mark.physics
def test_maxwell_pair_satisfies_equal_chemical_potential():
    """Independent validation: μ(ρ_v) = μ(ρ_l) at the returned pair."""
    result = maxwell_coexistence(1.0, psi)
    assert result.found is True
    mu = chemical_potential(torch.tensor([result.rho_vapour, result.rho_liquid]), 1.0, psi, n=20000)
    assert float(abs(mu[1] - mu[0]) / CS2) < 1e-4
    assert abs(result.area_residual or 0.0) < 1e-8


@pytest.mark.physics
def test_ledger_numbers_are_reproduced_by_the_rho_space_rule():
    """G3 claim C3 attribution: the ledger triple comes from ∫(p−p_sat)dρ = 0."""
    r_v, r_l = _rho_space_rule(classic_to_library(LEDGER_G_CLASSIC_VALID))
    assert r_v == pytest.approx(LEDGER_RHO_V, rel=0.01)
    assert r_l == pytest.approx(LEDGER_RHO_L, rel=0.01)
    assert r_l / r_v == pytest.approx(LEDGER_RATIO, rel=0.01)


@pytest.mark.physics
def test_simulation_matches_correct_maxwell_at_low_density_ratio():
    """Direct SCMP relaxation reproduces the correct Maxwell pair to a few %."""
    result = maxwell_coexistence(0.7, psi)
    assert result.found is True
    row = relax_scmp(0.7, psi, nx=40, ny=40, rho0=1.5, steps=900)
    assert row["diverged"] is False
    assert row["density_ratio"] > 3.0
    assert abs(row["rho_max"] - result.rho_liquid) / result.rho_liquid < 0.05
    assert abs(row["rho_min"] - result.rho_vapour) / result.rho_vapour < 0.10


@pytest.mark.physics
def test_homogeneous_relaxation_below_coexistence_threshold():
    """Where Maxwell says there is no coexistence, the relaxation stays mixed."""
    row = relax_scmp(0.5, psi, nx=24, ny=24, rho0=1.5, steps=400)
    assert row["diverged"] is False
    assert row["density_ratio"] < 1.5


# ---------------------------------------------------------------------------
# Landing fixes in cavitation.py (regression locks)
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_apply_cavitation_force_passes_density_and_cavitation_psi(monkeypatch):
    """The SC force must receive ρ (+ ψ_cav as psi_fn), not a pre-computed ψ."""
    captured: dict[str, object] = {}
    real = multiphase.sc_single_component_force

    def spy(rho, G, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        captured["rho"] = rho.clone()
        captured["G"] = G
        captured["psi_fn"] = kwargs.get("psi_fn")
        return real(rho, G, *args, **kwargs)

    monkeypatch.setattr(multiphase, "sc_single_component_force", spy)

    n = 16
    x = torch.arange(n, dtype=torch.float32) / n
    rho = 1.2 + 0.6 * x.reshape(1, n).repeat(n, 1)
    zeros = torch.zeros(n, n)
    f = equilibrium(rho, zeros, zeros)
    apply_cavitation_force(f, rho, zeros, zeros, tau=1.0)

    assert "rho" in captured, "apply_cavitation_force did not call the SC force"
    assert torch.allclose(captured["rho"], rho)
    psi_fn = captured["psi_fn"]
    assert psi_fn is not None, "psi_fn was not forwarded: the default psi_exp would be used"
    assert torch.allclose(psi_fn(rho), psi_cavitation(rho))


@pytest.mark.regression
def test_apply_cavitation_force_rejects_non_attractive_coupling():
    """G ≤ 0 is repulsive in this library: fail loudly instead of silently."""
    n = 8
    rho = torch.full((n, n), 1.5)
    zeros = torch.zeros(n, n)
    f = equilibrium(rho, zeros, zeros)
    for bad_G in (-5.5, -1.0, 0.0):
        with pytest.raises(ValueError, match="G must be > 0"):
            apply_cavitation_force(f, rho, zeros, zeros, tau=1.0, G=bad_G)
    out = apply_cavitation_force(f, rho, zeros, zeros, tau=1.0)
    assert bool(torch.isfinite(out).all())


@pytest.mark.unit
def test_module_defaults_are_a_coexistence_state():
    """Default coupling must be positive (attractive) and match the Maxwell pair."""
    assert COEXISTENCE_G > 0.0
    assert CavitationConfig().G == COEXISTENCE_G
    result = maxwell_coexistence(COEXISTENCE_G, psi)
    assert result.found is True
    assert result.rho_liquid is not None and result.rho_vapour is not None
    assert result.rho_liquid == pytest.approx(COEXISTENCE_RHO_L, rel=1e-3)
    assert result.rho_vapour == pytest.approx(COEXISTENCE_RHO_V, rel=1e-3)
    # the reference densities are pseudopotential calibration scales, not a
    # coexistence state (ledger G3): keep them out of the coexistence claim
    assert cavitation._RHO_L == pytest.approx(2.65)
    assert cavitation._RHO_V == pytest.approx(0.038)

@pytest.mark.xfail(
    strict=True,
    reason=(
        "run_cavitation_flow is still non-functional: the inlet BC indexes f[:, :, 0] "
        "while the library layout is (9, ny, nx) — locked here so the module cannot be "
        "mistaken for a validated cavitation solver (ledger G3 does not cover the runner)"
    ),
)
def test_full_cavitation_runner_is_a_known_stub(tmp_path):
    """Sentinel: the whole-flow cavitation runner must not be assumed to work."""
    from tensorlbm.cavitation import CavitationConfig, run_cavitation_flow

    run_cavitation_flow(
        CavitationConfig(
            nx=32, ny=16, n_steps=5, output_interval=5, output_root=tmp_path, overwrite=True
        )
    )
