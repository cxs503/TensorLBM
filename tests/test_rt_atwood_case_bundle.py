"""Regression tests for the CASE-RT-INSTABILITY-ATWOOD evidence bundle.

These tests are solver-free (no run is executed): they pin the *measurement
semantics* that the case's gates depend on.

1. ``run_case.velocity_in_window`` must return a **lattice** velocity, i.e. the
   tau-slope multiplied by ``d tau / dt``.  The pre-fix version returned the
   tau-slope, which is off by ``1/tau_unit`` = 800 lattice units per step at the
   reference point.  The defect was masked by the fact that the window
   ``tau in [4, 6]`` was never reached by that suite (result ``nan`` either way),
   so it is pinned here instead.
2. The pre-declared analysis windows are **amplitude** windows on ``a / lam``,
   not time windows; fitting the same data over a ``tau`` window of the same
   numeric bounds biases sigma by ~16 %.  ``window_slope`` must therefore filter
   on the amplitude.
3. The Layzer-Goncharov reference is consumed in one place: ``Fr = 1/sqrt(pi C)``
   with ``C = 1`` in 2-D, and the two-fluid asymmetry is
   ``sqrt((1+A)/(1-A))``.  A silent change of the geometric constant would move
   every gate of this case by ~11 %.
4. The similarity ladder of ``run_simladder.py`` really holds its dimensionless
   groups (``Ma ~ sqrt(A g lam)``, ``zeta = nu k^2 / sqrt(A g k)``, ``k a0``)
   fixed -- otherwise the ladder cannot separate grid from model-form effects.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "case_pool" / "rayleigh_taylor_atwood05"
for _p in (str(BENCH), str(BENCH / "ref")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ATWOOD, G0, LAM0 = 0.5, 2.0e-4, 64


def _tau_unit(atwood: float = ATWOOD, g: float = G0, lam: float = LAM0) -> float:
    return math.sqrt(atwood * g / lam)


# ---------------------------------------------------------------------------
# 1. unit semantics of the front-velocity estimator
# ---------------------------------------------------------------------------
def test_velocity_in_window_returns_lattice_units():
    from run_case import velocity_in_window

    tau_unit = _tau_unit()
    u_true = 0.0123                                   # lattice cells per step
    steps = [float(i) for i in range(1, 8001)]
    fronts = [u_true * s for s in steps]
    taus = [s * tau_unit for s in steps]

    fixed = velocity_in_window(taus, fronts, 4.0, 6.0, tau_unit)
    assert fixed == pytest.approx(u_true, rel=1e-9)

    # the legacy (no tau_unit) signature stays callable for reproducibility, and
    # reproduces exactly the mis-scaled number: the raw tau-slope
    raw = velocity_in_window(taus, fronts, 4.0, 6.0)
    # ``fronts`` is in cells and ``taus`` in tau, so the raw slope is u/tau_unit,
    # i.e. the mis-scaled number is 1/tau_unit = 800 times too large
    assert raw == pytest.approx(u_true / tau_unit, rel=1e-9)
    assert raw / fixed == pytest.approx(1.0 / tau_unit, rel=1e-9)
    assert 1.0 / tau_unit == pytest.approx(800.0, rel=1e-12)


def test_velocity_in_window_needs_three_points():
    from run_case import velocity_in_window

    assert math.isnan(velocity_in_window([1.0, 2.0], [1.0, 2.0], 0.0, 10.0, 1.0))


# ---------------------------------------------------------------------------
# 2. window semantics: amplitude windows of a/lam, not tau windows
# ---------------------------------------------------------------------------
def test_window_slope_filters_on_amplitude():
    from run_case import window_slope

    # exponential growth a = a0 exp(sigma t): the fitted slope must be sigma
    sigma, a0 = 3.0e-3, 0.01
    times = [float(i) for i in range(0, 2000)]
    amps = [a0 * math.exp(sigma * t) for t in times]
    fit = window_slope(times, amps, 0.02, 0.08)
    assert fit["n"] > 10
    assert fit["sigma"] == pytest.approx(sigma, rel=1e-6)
    # the filter is on the amplitude: every selected sample lies inside the window
    lo, hi = fit["a_range"]
    assert lo >= 0.02 - 1e-12 and hi <= 0.08 + 1e-12

    # a tau-window interpretation of the same bounds selects the startup transient
    wrong = window_slope(times, amps, 0.02, 0.08 * 1e9)   # 0.02 <= a is also true
    assert wrong["n"] == len([a for a in amps if a >= 0.02])


# ---------------------------------------------------------------------------
# 3. reference forms used by the gates
# ---------------------------------------------------------------------------
def test_layzer_goncharov_reference_forms():
    import late_analysis as la
    import ref_lock

    lam, g = LAM0, G0
    k = 2.0 * math.pi / lam
    fr_lz = la.layzer_froude(ATWOOD, g, lam, la.C_2D)
    assert fr_lz == pytest.approx(1.0 / math.sqrt(math.pi * la.C_2D), rel=1e-12)
    assert fr_lz == pytest.approx(0.5641895835, rel=1e-9)

    u_b = ref_lock.goncharov(ATWOOD, g, lam, "b", ref_lock.C_2D)
    u_s = ref_lock.goncharov(ATWOOD, g, lam, "s", ref_lock.C_2D)
    assert u_b == pytest.approx(math.sqrt(2 * ATWOOD * g / ((1 + ATWOOD) * k)), rel=1e-12)
    assert u_s == pytest.approx(math.sqrt(2 * ATWOOD * g / ((1 - ATWOOD) * k)), rel=1e-12)
    # product invariant: u_b u_s = 2 A g / (C k) / sqrt(1 - A^2)
    assert u_b * u_s == pytest.approx(
        2 * ATWOOD * g / (la.C_2D * k) / math.sqrt(1 - ATWOOD ** 2), rel=1e-12)
    # the two-fluid asymmetry that a Boussinesq solver cannot produce
    assert la.two_fluid_asymmetry(ATWOOD) == pytest.approx(math.sqrt(3.0), rel=1e-12)


def test_late_and_case_specs_share_the_declared_windows():
    import late_analysis as la
    import run_case

    assert la.TAU_LO == run_case.TAU_LO
    assert la.TAU_HI == run_case.TAU_HI
    assert la.C_2D == run_case.ref_lock.C_2D
    assert la.GATE_PCT == run_case.GATE_PCT


# ---------------------------------------------------------------------------
# 4. the similarity ladder holds its dimensionless groups
# ---------------------------------------------------------------------------
def test_simladder_holds_similarity_groups():
    import run_simladder as sl

    specs = {s.name: s for s in sl.sim_specs()}
    ladder = [specs[n] for n in sl.LADDER_NAMES if n in specs]
    assert len(ladder) == 3, "ladder arms missing from sim_specs()"

    k = lambda s: 2.0 * math.pi / s.lam                                  # noqa: E731
    mach = [math.sqrt(s.atwood * s.g * s.lam) for s in ladder]           # Ma group
    zeta = [s.nu * k(s) ** 2 / math.sqrt(s.atwood * s.g * k(s)) for s in ladder]
    ka0 = [k(s) * s.eta0_frac * s.lam for s in ladder]

    assert max(mach) / min(mach) == pytest.approx(1.0, rel=1e-9)
    assert max(zeta) / min(zeta) == pytest.approx(1.0, rel=1e-9)
    assert max(ka0) / min(ka0) == pytest.approx(1.0, rel=1e-12)
    # the one group that is allowed to change: the interface thickness in k*xi
    kxi = [k(s) * s.ac_width for s in ladder]
    assert kxi[0] > kxi[1] > kxi[2]
    # and it must shrink like 1/lam at fixed ac_width
    assert kxi[0] / kxi[2] == pytest.approx(ladder[2].lam / ladder[0].lam, rel=1e-9)

    # the k*xi control arm doubles the interface thickness of the reference arm
    # (k*xi = 0.393 vs 0.196): one finite-difference point in the k*xi direction at
    # fixed grid, i.e. dFr/d(k*xi) can be measured while the ladder drifts by
    # kxi[0] - kxi[2] = 0.098 in the same parameter
    w4 = specs["S_w4_l064_g2e-4"]
    assert k(w4) * w4.ac_width == pytest.approx(2.0 * kxi[0], rel=1e-9)

    # the Mach arm is the reference grid with half gravity
    ref, mach_arm = specs["S_ref_l064_g2e-4"], specs["S_mach_l064_g1e-4"]
    assert mach_arm.lam == ref.lam
    assert mach_arm.g == pytest.approx(0.5 * ref.g, rel=1e-12)
    assert mach_arm.tau == ref.tau