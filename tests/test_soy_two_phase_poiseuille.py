"""Round-2 grading of anchor ``XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE``.

Grades :mod:`tensorlbm.phasefield.soy_two_phase` — the Swift--Osborn--Yeomans
free-energy two-phase D2Q9 kernel used for the phase-field MCMP Poiseuille
cross-validation against ``ebinan92/Fingering_dynamics``.

The tests are written as *invariants*, not as "the code runs":

* :func:`test_half_way_bounce_back_reflects_every_crossing_direction` — the
  population-level invariant that the round-1 wall treatment violated (it
  reflected only half of the crossing direction pairs because the assignment was
  done in place).
* :func:`test_legacy_wall_aliasing_is_detectable` — the invalidation experiment:
  the same solver with round-1 wall semantics must be *measurably worse*, so the
  accuracy test cannot be passed by a wall scheme that does nothing.
* :func:`test_uniform_viscosity_poiseuille_matches_exact` — accuracy of the
  channel solution against the exact parabolic profile (owner gate: < 15 %).
* :func:`test_layered_reference_is_not_degenerate` — the round-1 "stratified"
  reference had equal viscosities and therefore coincided with single-phase
  Poiseuille; this test pins the viscosity-contrast reference away from it.
* :func:`test_viscosity_contrast_poiseuille_matches_exact` — the non-degenerate
  arm: two layers with ``nu_low = 3 nu_up``.
* :func:`test_phase_field_at_rest_has_no_spurious_current_and_conserves_mass` —
  the equilibrium invariants the round-1 scripts never measured.
* :func:`test_interface_width_recovers_xi` — the free-energy parameterisation
  ``a = -2 kappa / xi^2`` reproduces the ``tanh`` width it claims.

Run: ``pytest tests/test_soy_two_phase_poiseuille.py -q`` (≈ 1 min, CPU only).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.phasefield.soy_two_phase import (
    CX,
    CY,
    CS2,
    CROSS_WALL,
    OPPOSITE,
    LayeredPoiseuille,
    SOYConfig,
    SOYTwoPhase,
    kappa_from_sigma_xi,
    surface_tension_from_kappa,
)

SIGMA = 0.01
XI = 4.0


def _channel(ny: int, nx: int = 2, **kw) -> SOYTwoPhase:
    cfg = SOYConfig(ny=ny, nx=nx, sigma=SIGMA, xi=XI, tau_g=0.8, **kw)
    sim = SOYTwoPhase(cfg)
    sim.set_planar(interface_y=1.0 + 0.5 * (ny - 2))
    return sim


# ---------------------------------------------------------------------------
# Free-energy parameterisation
# ---------------------------------------------------------------------------

def test_surface_tension_round_trip():
    kappa = kappa_from_sigma_xi(SIGMA, XI)
    assert surface_tension_from_kappa(kappa, XI) == pytest.approx(SIGMA, rel=1e-14)


def test_interface_width_recovers_xi():
    sim = _channel(34, 2)
    assert sim.interface_width() == pytest.approx(XI, rel=0.05)


# ---------------------------------------------------------------------------
# Analytic reference: internal consistency (no solver involved)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu_low,nu_up", [(0.1, 0.1), (0.3, 0.1), (0.1, 0.3)])
def test_layered_poiseuille_satisfies_its_own_conditions(nu_low, nu_up):
    H, h, G = 32.0, 16.0, 1e-5
    ref = LayeredPoiseuille(H=H, h=h, G=G, nu_low=nu_low, nu_up=nu_up)
    assert ref.u(0.0) == pytest.approx(0.0, abs=1e-15)
    assert ref.u(H) == pytest.approx(0.0, abs=1e-15)
    assert ref.u(h) == pytest.approx(ref.u(h + 1e-12), abs=1e-12)   # velocity continuity
    du = 1e-6
    shear_low = nu_low * (ref.u(h) - ref.u(h - du)) / du
    shear_up = nu_up * (ref.u(h + du) - ref.u(h)) / du
    # stress continuity, compared against a scale that stays non-zero when the
    # interface happens to sit on the zero-shear plane (uniform-viscosity case)
    scale = max(abs(shear_low), abs(shear_up), G * h)
    assert abs(shear_low - shear_up) < 1e-6 * scale
    # uniform-viscosity limit must be the single-phase parabola G/(2nu) y (H - y)
    if nu_low == nu_up:
        for y in (2.0, 8.0, 16.0, 24.0, 30.0):
            assert ref.u(y) == pytest.approx(G / (2 * nu_low) * y * (H - y), rel=1e-12)


def test_layered_reference_is_not_degenerate():
    """The contrast case must be well separated from the single-phase profile."""
    H, h, G = 32.0, 16.0, 1e-5
    y = torch.arange(0.5, H, dtype=torch.float64)
    layered = LayeredPoiseuille(H=H, h=h, G=G, nu_low=0.3, nu_up=0.1).profile(y)
    single = LayeredPoiseuille(H=H, h=h, G=G, nu_low=0.1, nu_up=0.1).profile(y)
    rel = float((layered - single).norm() / single.norm())
    assert rel > 0.10, "contrast arm is indistinguishable from single-phase Poiseuille"


# ---------------------------------------------------------------------------
# Wall scheme invariants
# ---------------------------------------------------------------------------

def test_in_place_reflection_schedule_aliases():
    """Population-level statement of the round-1 wall defect.

    Reflecting a wall row *in place*, ascending in the direction index, is not a
    reflection: for the opposite pairs whose members both cross the wall the
    second assignment reads a value the first one already overwrote.  On a wall
    row holding distinct values the schedule leaves directions 4, 7 and 8 equal
    to their own pre-reflection value instead of their partner's — exactly the
    three that feed the fluid row under the top wall.
    """
    src = torch.arange(9, dtype=torch.float64)

    aliased = src.clone()
    for i in range(1, 9):
        if CY[i] != 0:
            aliased[i] = aliased[OPPOSITE[i]]        # in-place, ascending i

    correct = src.clone()
    for i in range(1, 9):
        if CY[i] != 0:
            correct[i] = src[OPPOSITE[i]]            # from the untouched source

    for i in CROSS_WALL:
        assert correct[i].item() == src[OPPOSITE[i]].item()
    unreflected = [i for i in CROSS_WALL if aliased[i].item() != src[OPPOSITE[i]].item()]
    assert unreflected == [4, 7, 8]


def test_legacy_wall_aliasing_is_detectable():
    """Wall semantics must matter: corrected < legacy error by a wide margin."""
    errs = {}
    for legacy in (False, True):
        sim = _channel(26, 2, tau_a=1.0, Gx=1e-5, legacy_walls=legacy)
        for _ in range(2500):
            sim.step()
        errs[legacy] = sim.poiseuille_error(sim.layered_reference())["l2_rel"]
    assert errs[False] < 0.02
    assert errs[True] > 3.0 * errs[False]


# ---------------------------------------------------------------------------
# Accuracy arms
# ---------------------------------------------------------------------------

def test_uniform_viscosity_poiseuille_matches_exact():
    sim = _channel(34, 2, tau_a=1.0, Gx=1e-5)
    for _ in range(4000):
        sim.step()
    res = sim.poiseuille_error(sim.layered_reference())
    assert res["l2_rel"] < 0.15, res["l2_rel"]
    assert res["l2_rel"] < 0.01, res["l2_rel"]          # observed ≈ 4e-4
    assert res["u_max_sim"] == pytest.approx(res["u_max_ana"], rel=1e-2)


def test_viscosity_contrast_poiseuille_matches_exact():
    """``nu_low = 3 nu_up``: the arm that a passive scalar cannot pass."""
    sim = _channel(34, 2, tau_a=0.8, tau_b=1.4, Gx=1e-5)
    for _ in range(4000):
        sim.step()
    ref = sim.layered_reference()
    assert ref.nu_low == pytest.approx(0.3) and ref.nu_up == pytest.approx(0.1)
    res = sim.poiseuille_error(ref)
    assert res["l2_rel"] < 0.15, res["l2_rel"]          # owner gate
    assert res["linf_rel"] < 0.15, res["linf_rel"]


# ---------------------------------------------------------------------------
# Equilibrium invariants
# ---------------------------------------------------------------------------

def test_phase_field_at_rest_has_no_spurious_current_and_conserves_mass():
    sim = _channel(26, 2)
    m0 = sim.psi_mass()
    for _ in range(600):
        sim.step()
    assert abs(sim.psi_mass() - m0) / abs(m0) < 1e-9
    assert sim.max_velocity() < 1e-5


def test_rest_state_does_not_drift_when_interface_is_off_centre():
    sim = SOYTwoPhase(SOYConfig(ny=26, nx=2, sigma=SIGMA, xi=XI, tau_g=0.8))
    sim.set_planar(interface_y=1.0 + 0.35 * (26 - 2))
    m0 = sim.psi_mass()
    for _ in range(400):
        sim.step()
    assert abs(sim.psi_mass() - m0) / abs(m0) < 1e-9
    assert sim.max_velocity() < 1e-5


def test_ma_number_is_small_in_the_graded_cases():
    """Guard rail: the graded runs must stay in the incompressible limit."""
    sim = _channel(34, 2, tau_a=1.0, Gx=1e-5)
    for _ in range(400):
        sim.step()
    u_max = max(abs(v) for v in (sim.velocity_profile().tolist()))
    assert u_max / math.sqrt(CS2) < 0.1


def test_round1_archive_binding_preserves_committed_reference():
    """The actual R2 loader resolves the relocated R1 reference, without a solver run."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/xval_phasefield_mcmp_poiseuille_r2.py"
    spec = importlib.util.spec_from_file_location("phasefield_round2_pathcheck", script)
    assert spec is not None and spec.loader is not None
    round2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = round2
    try:
        spec.loader.exec_module(round2)
        round1 = round2._load_round1()
        case = root / "benchmarks/phasefield_mcmp_poiseuille"
        assert Path(round1.__file__).resolve() == case / "reference/round1.py"
        data = json.loads((case / "evidence/m18190/xval_poiseuille_results.json").read_text())
        c = data["config"]
        result = round1.analytical_stratified_poiseuille(
            c["ny"], c["H"], c["h"], c["Gx"], c["nu"], c["nu"]
        )
        assert result["u_max"] == pytest.approx(data["soy_pytorch"]["analytical_u_max"])
        assert result["Q_total"] == pytest.approx(data["soy_pytorch"]["analytical_Q_total"])
        assert round2.OUT == root / "runs/phasefield_mcmp_poiseuille_r2"
    finally:
        sys.modules.pop(spec.name, None)
