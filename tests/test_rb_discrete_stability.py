"""Tests for the discrete lattice linear-stability analysis of the Rayleigh-Bénard
operator (node IN31-N12, part 1) and its continuum reference.

The heavy production runs live in ``benchmarks/rb_discrete_stability/run_bench.py``;
the cases here use deliberately small grids so the suite stays fast, but every
assertion is a real numeric check (no hard-coded expectations beyond the literature
values 1707.76 / 3.11632).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.rb_continuum_stability import (
    A_CRIT_CONTINUOUS,
    RA_CRIT_CONTINUOUS,
    critical_at,
    growth as continuum_growth,
    richardson,
)
from tensorlbm.rb_discrete_stability import (
    RBLatticeParams,
    conductive_base_state,
    growth_rate,
    measure_growth_rate,
    mode_block_matrix,
    split_spectrum,
)


# ---------------------------------------------------------------------------
# discrete operator
# ---------------------------------------------------------------------------
def test_base_state_is_the_discrete_conduction_profile():
    """The base state is the fixed point of the library operator, with the walls
    pinned at the wall *nodes* (measured effective gap ``H_eff = ny - 1``)."""
    ny = 8
    p = RBLatticeParams(ny=ny, nx=2 * ny, Ra=1500.0)
    base = conductive_base_state(p, steps=4000)
    assert base.residual < 1e-7
    T = base.g.sum(dim=0)[:, 0].numpy()
    assert T[0] == pytest.approx(p.T_hot, abs=1e-10)
    assert T[-1] == pytest.approx(p.T_cold, abs=1e-10)
    slope = np.polyfit(np.arange(1, ny - 1), T[1:-1], 1)[0]
    H_eff = p.dT / abs(slope)
    # the discrete conduction profile spans the wall-node distance ny - 1, not the
    # half-way gap H = ny that the benchmark's non-dimensionalisation assumes
    assert H_eff == pytest.approx(ny - 1.0, abs=0.15)


def test_unit_circle_artifacts_are_separated_from_the_physical_mode():
    """Boundary substitution rows produce exact unit-circle multipliers; the
    physical roll mode must be identified past them."""
    ny = 8
    p = RBLatticeParams(ny=ny, nx=2 * ny, Ra=6000.0)
    base = conductive_base_state(p, steps=3000)
    J, _ = mode_block_matrix(p, base=base, eps=1e-6)
    phys, artifacts = split_spectrum(J, ny)
    assert len(artifacts) >= 4
    for lam, _w in artifacts:
        assert abs(abs(lam) - 1.0) < 1e-7
    assert abs(abs(phys[0][0]) - 1.0) > 1e-5
    # the physical mode must be well separated from the rest of the spectrum
    assert abs(phys[0][0]) > abs(phys[1][0]) + 1e-4


def test_growth_rate_reproduces_the_nonlinear_library_solver():
    """Eigenvalue growth rate vs a time-march of the same (non-linear) operator."""
    ny = 8
    p = RBLatticeParams(ny=ny, nx=2 * ny, Ra=6000.0)
    base = conductive_base_state(p, steps=3000)
    eig = growth_rate(p, base=base, eps=1e-6)
    sim = measure_growth_rate(p, steps=900, amp=1e-7, sample_every=20)
    assert eig["gamma_step"] < 0.0
    assert sim["gamma_sim"] == pytest.approx(eig["gamma_step"], rel=0.05)


def test_growth_rate_responds_to_rayleigh_number():
    """Destabilisation with Ra.  Restricted to a short interval on purpose: at
    ny = 8 the leading mode switches branch around Ra ~ 5e3 (measured), so a wide
    sweep is not expected to be monotone on such a coarse grid."""
    ny = 8
    gammas = []
    for Ra in (4000.0, 6000.0):
        p = RBLatticeParams(ny=ny, nx=2 * ny, Ra=Ra)
        base = conductive_base_state(p, steps=3000)
        gammas.append(growth_rate(p, base=base, eps=1e-6)["gamma_step"])
    assert gammas[0] < gammas[1] < 0.0


def test_unresolved_rayleigh_number_raises():
    """A Rayleigh number that the grid cannot carry must fail loudly, not return NaN."""
    p = RBLatticeParams(ny=8, nx=16, Ra=2.0e5)
    with pytest.raises((FloatingPointError, ValueError)):
        growth_rate(p, eps=1e-6)


# ---------------------------------------------------------------------------
# continuum reference
# ---------------------------------------------------------------------------
def test_continuum_critical_rayleigh_matches_chandrasekhar():
    """Second-order finite differences converge to Ra_c = 1707.76 at a_c = 3.11632."""
    ra = [critical_at(A_CRIT_CONTINUOUS, M=M) for M in (50, 100)]
    assert ra[0] < ra[1] < RA_CRIT_CONTINUOUS + 1.0
    extrap = richardson(ra)
    assert extrap == pytest.approx(RA_CRIT_CONTINUOUS, rel=2e-3)
    # quadratic convergence: the error must shrink by roughly 4x per grid doubling
    err = [abs(x / RA_CRIT_CONTINUOUS - 1.0) for x in ra]
    assert err[1] < 0.4 * err[0]


def test_continuum_growth_is_prandtl_independent_at_onset():
    """The marginal state of (1)-(2) does not depend on Pr, so Ra_c(a) must not."""
    ra_1 = critical_at(A_CRIT_CONTINUOUS, Pr=0.71, M=100)
    ra_2 = critical_at(A_CRIT_CONTINUOUS, Pr=5.0, M=100)
    assert ra_1 == pytest.approx(ra_2, rel=1e-3)


def test_continuum_growth_changes_sign_across_onset():
    below = continuum_growth(A_CRIT_CONTINUOUS, 1600.0, Pr=0.71, M=100)["sigma"]
    above = continuum_growth(A_CRIT_CONTINUOUS, 1900.0, Pr=0.71, M=100)["sigma"]
    assert below < 0.0 < above
    # the leading mode is separated from the rest of the spectrum by a wide margin
    out = continuum_growth(A_CRIT_CONTINUOUS, 2500.0, Pr=0.71, M=100)
    assert out["sigma"] - out["sigma_second_real"] > 5.0


def test_coarse_lattice_is_more_stable_than_the_continuum_at_equal_ra():
    """At equal *nominal* Ra (benchmark normalisation H = ny) the coarse lattice is
    far more stable than the continuum: at ny = 8 the roll mode still decays at
    Ra = 6000 while the continuum grows there.  The origin is documented in the
    report (the discrete conduction profile pins the walls at the wall nodes, so
    the realised gap is ny-1, plus coarse-grid truncation error)."""
    from tensorlbm.rb_continuum_stability import growth as continuum_growth

    ny = 8
    p = RBLatticeParams(ny=ny, nx=2 * ny, Ra=6000.0)
    base = conductive_base_state(p, steps=3000)
    lat = growth_rate(p, base=base, eps=1e-6)["gamma_hat"]
    cont = continuum_growth(p.a, 6000.0, Pr=p.Pr, M=100)["sigma"]
    assert lat < 0.0 < cont
    assert cont - lat > 10.0