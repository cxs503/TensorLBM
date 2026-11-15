"""Analysis-level tests for the late-time RT front-velocity study of
``CASE-RT-INSTABILITY-ATWOOD`` (``benchmarks/case_pool/rayleigh_taylor_atwood05``).

These do not run the solver: they pin the *measurement* layer, which is where
the defects fixed by this study lived.

1. The stop criterion no longer fires at ``a/lam = 0.6`` (the old
   ``run_case.MAX_A_FRAC`` guard that made the terminal-velocity stage
   unreachable), and still fires at the wall / at the Mach limit.
2. The Layzer-Goncharov reference in the Boussinesq form has the closed value
   ``Fr = 1/sqrt(pi C)`` and the inversion ``C_eff`` round-trips.
3. The front-velocity fit recovers a known drift and handles short windows.
4. ``topmost_crossing`` returns the *last* light->heavy transition of a column
   while ``interface_zero_crossing`` returns the *first* -- the distinction the
   bubble-tip measurement depends on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "benchmarks" / "case_pool" / "rayleigh_taylor_atwood05"
for _p in (str(REPO_ROOT / "src"), str(CASE_DIR), str(CASE_DIR / "ref")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import late_analysis as la  # noqa: E402


# ---------------------------------------------------------------------------
# stop criterion
# ---------------------------------------------------------------------------
def test_wall_clearance_matches_case_box():
    """layers = 8 -> 4 wavelengths per side, 1.5 of which are kept clear."""
    assert la.wall_clearance_lam(8.0) == pytest.approx(2.5)
    assert la.wall_clearance_lam(10.0) == pytest.approx(3.5)


def test_stop_criterion_no_longer_fires_at_the_old_amplitude_guard():
    """Regression: ``run_case.MAX_A_FRAC = 0.60`` stopped the run at tau ~ 1.9.

    A front at 0.6 wavelengths is 3.4 wavelengths away from the wall of a
    layers = 8 box, so the run must continue -- this is the regime the
    terminal-velocity window (tau >= 4) has to reach.
    """
    assert la.late_stop_reason(0.60, ma=0.10, layers=8.0) is None
    assert la.late_stop_reason(2.40, ma=0.10, layers=8.0) is None


def test_stop_criterion_fires_on_wall_and_mach():
    assert la.late_stop_reason(2.50, ma=0.10, layers=8.0) == "wall"
    assert la.late_stop_reason(2.60, ma=0.10, layers=8.0) == "wall"
    assert la.late_stop_reason(1.00, ma=0.90, layers=8.0) == "mach"
    assert la.late_stop_reason(float("nan"), ma=0.10, layers=8.0) == "nonfinite"


# ---------------------------------------------------------------------------
# reference velocity
# ---------------------------------------------------------------------------
def test_layzer_froude_is_one_over_sqrt_pi_c():
    for atwood, g, lam in ((0.5, 2.0e-4, 64), (0.5, 3.0e-4, 96), (0.2, 1.0e-4, 32)):
        assert la.layzer_froude(atwood, g, lam, 1.0) == pytest.approx(1.0 / math.sqrt(math.pi))
    # the 3-D convention of the sibling case is a factor sqrt(3) slower
    assert la.layzer_froude(0.5, 2.0e-4, 64, 3.0) == pytest.approx(
        1.0 / math.sqrt(3.0 * math.pi))


def test_effective_c_inverts_layzer_velocity():
    atwood, g, lam = 0.5, 2.0e-4, 64
    for c in (0.5, 1.0, 3.0):
        u = la.layzer_velocity(atwood, g, lam, c)
        assert la.effective_c(atwood, g, lam, u) == pytest.approx(c, rel=1e-12)
    assert math.isnan(la.effective_c(atwood, g, lam, 0.0))


def test_two_fluid_asymmetry_is_sqrt3_at_half():
    assert la.two_fluid_asymmetry(0.5) == pytest.approx(math.sqrt(3.0))
    assert la.two_fluid_asymmetry(1.0 / 3.0) == pytest.approx(math.sqrt(2.0))


# ---------------------------------------------------------------------------
# trajectory analysis
# ---------------------------------------------------------------------------
def test_front_velocity_recovers_a_known_drift():
    taus = [0.1 * i for i in range(80)]
    fronts = [0.25 + 0.4 * t for t in taus]          # constant drift, lam per tau
    fit = la.front_velocity(taus, fronts, 4.0, 6.0)
    assert fit["slope"] == pytest.approx(0.4, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["n"] >= 3


def test_front_velocity_reports_nan_for_short_windows():
    fit = la.front_velocity([0.0, 1.0], [0.0, 1.0], 4.0, 6.0)
    assert math.isnan(fit["slope"])


def test_lattice_velocity_unit_conversion():
    """``tau = t sqrt(A g / lam)`` -> ``u = (d h/dtau) sqrt(A g / lam)``."""
    atwood, g, lam = 0.5, 2.0e-4, 64
    slope = 0.4                                       # lattice lengths per tau
    u = la.lattice_velocity(slope, atwood, g, lam)
    assert u == pytest.approx(0.4 * math.sqrt(atwood * g / lam))
    # a slope measured in wavelengths per tau is lam times larger
    assert la.lattice_velocity(la.TAU_HI - la.TAU_LO, atwood, g, lam) == pytest.approx(
        (la.TAU_HI - la.TAU_LO) * math.sqrt(atwood * g / lam))


def test_rel_pct():
    assert la.rel_pct(1.1, 1.0) == pytest.approx(10.0)
    assert math.isnan(la.rel_pct(float("nan"), 1.0))


# ---------------------------------------------------------------------------
# crossing extraction (the defect behind the biased h_up / h_down)
# ---------------------------------------------------------------------------
def test_topmost_crossing_returns_the_last_light_heavy_transition():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorlbm")
    import run_late
    from tensorlbm.rayleigh_taylor import interface_zero_crossing

    # one column, heavy (phi > 0) at the very top: light -> heavy at rows 2-3
    # (first crossing) and again at rows 6-7 (last crossing)
    phi = torch.tensor([[1.0], [1.0], [-1.0], [1.0], [1.0], [1.0], [-1.0], [1.0]],
                       dtype=torch.float64)
    low = interface_zero_crossing(phi)[0]
    top = run_late.topmost_crossing(phi)[0]
    assert float(low) == pytest.approx(2.5)
    assert float(top) == pytest.approx(6.5)
    assert float(top) > float(low)


def test_topmost_crossing_matches_lowest_for_a_single_valued_interface():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorlbm")
    import run_late
    from tensorlbm.rayleigh_taylor import interface_zero_crossing

    y = torch.arange(10, dtype=torch.float64).view(10, 1)
    phi = torch.tanh((y - 4.5) / 1.0)                 # single crossing at y = 4.5
    assert float(run_late.topmost_crossing(phi)[0]) == pytest.approx(
        float(interface_zero_crossing(phi)[0]))