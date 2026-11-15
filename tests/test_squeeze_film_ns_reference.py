"""First-principles verification of the squeeze-film LBM against an independent N-S solve.

The historical acceptance of ``tensorlbm.squeeze_film_lbm`` compared the momentum-exchange
load with the leading-order lubrication asymptote ``8 mu V a^3/h^3`` and reported a
12 % - 37 % "error".  This file pins down what those two numbers actually mean:

A. **The reference is right.**  In the thin limit (``h/a`` small) an independent 2D
   incompressible Navier-Stokes solve must reproduce the asymptote, be mass conserving
   and divergence free, and not depend on its symmetry-breaking seed.
B. **The lattice solver is right where it is resolved.**  Compared with that reference -
   not with the asymptote - the LBM agrees to a few percent for ``h >= 6`` and the
   residual decays monotonically as the film is resolved (``h = 3, 4, 6, 8, 12``).
C. **The 12 % - 37 % is the asymptote.**  ``reference / asymptote - 1`` reproduces the
   historical error column almost exactly, so the old verdict measured the caliber, not
   the solver.

Checks A use a live (small) reference solve; B and C read the archived sweep
``benchmarks/verified/squeeze_film_ns/results.json`` produced by ``run_compare.py``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(HERE, "benchmarks", "verified", "squeeze_film_ns")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)

import reference_ns as REF  # noqa: E402

RESULTS = os.path.join(BENCH, "results.json")


# --------------------------------------------------------------------------- A
@pytest.fixture(scope="module")
def thin():
    """Live reference solve in the thin-film limit (h/a = 0.125)."""
    return REF.solve(a=16.0, h=2.0, nu=0.1, V=0.005, ny=8, max_iter=200_000)


def test_reference_is_mass_conserving_and_divergence_free(thin):
    assert thin["converged"], "reference did not reach steady state"
    # injected 2a*V per unit depth must leave through the two vents
    assert thin["mass_imbalance_rel"] < 1e-5
    assert thin["div_scaled"] < 1e-4            # max|div u| / (V/h)


def test_reference_reproduces_lubrication_asymptote_when_it_is_valid(thin):
    """h/a = 0.125: the asymptote is good to a few percent, the reference must land there."""
    err = (thin["normal_load_top"] / thin["reynolds_load"] - 1.0) * 100.0
    assert -1.0 < err < 12.0, f"reference vs asymptote {err:+.2f} % is out of family"
    # ... and it must sit *above* the asymptote (finite-gap correction is positive here)
    assert thin["normal_load_top"] > thin["reynolds_load"]


def test_viscous_normal_stress_is_negligible_at_the_plate(thin):
    """Lubrication predicts dv/dy = 0 at both plates, i.e. a pure pressure load."""
    share = (thin["normal_load_top"] - thin["pressure_load_top"]) / thin["normal_load_top"]
    assert abs(share) < 0.03


def test_reference_is_seed_independent():
    a = REF.solve(a=16.0, h=4.0, nu=0.1, V=0.005, ny=8, seed=1e-3, max_iter=200_000)
    b = REF.solve(a=16.0, h=4.0, nu=0.1, V=0.005, ny=8, seed=5e-3, max_iter=200_000)
    assert a["converged"] and b["converged"]
    assert abs(b["normal_load_top"] / a["normal_load_top"] - 1.0) < 1e-3


# --------------------------------------------------------------------------- B/C
def _sweep():
    if not os.path.exists(RESULTS):
        pytest.skip("run benchmarks/verified/squeeze_film_ns/run_compare.py first")
    with open(RESULTS) as fh:
        return {r["h"]: r for r in json.load(fh)["sweep"]}


def test_lbm_matches_first_principles_where_the_film_is_resolved():
    """Verification question: LBM vs the independent N-S solve, for h >= 6 (h/a >= 0.375)."""
    sweep = _sweep()
    for h in (6, 8, 12):
        assert h in sweep, f"missing sweep row h={h}"
        err = sweep[h]["err_pct"]["lbm_me_vs_ref_normal"]
        assert abs(err) < 3.0, f"h={h}: LBM vs N-S {err:+.2f} % exceeds the 3 % band"


def test_lbm_error_decays_monotonically_with_gap_resolution():
    """The residual at h = 3 - 4 is discretisation (3 - 4 cells across the film), not physics."""
    sweep = _sweep()
    hs = sorted(sweep)
    errs = [abs(sweep[h]["err_pct"]["lbm_me_vs_ref_normal"]) for h in hs]
    for lo, hi, e_lo, e_hi in zip(hs, hs[1:], errs, errs[1:]):
        assert e_hi <= e_lo + 0.5, f"LBM error not decaying: h={lo} {e_lo:.2f} -> h={hi} {e_hi:.2f}"


def test_the_historical_12_to_37_percent_is_the_asymptote_not_the_solver():
    """The old error column must be reproduced by the *reference* against the asymptote."""
    sweep = _sweep()
    # h = 3 / 4 are deliberately excluded from the reproduction band: there the archived
    # number mixes the caliber error with a genuine 3-4 cell discretisation error
    # (+11.55 % / +5.21 %), so it is *not* a pure asymptote artefact.
    for h, recorded in ((6, 14.2), (8, 19.9), (12, 37.2)):
        aged = sweep[h]["err_pct"]["ref_normal_vs_reynolds"]
        assert abs(aged - recorded) < 6.0, (
            f"h={h}: N-S vs asymptote {aged:+.2f} % does not reproduce the archived "
            f"caliber error {recorded:+.1f} %")
        # where the film is resolved the "error" must be far larger than the actual
        # solver deviation - i.e. it was the caliber, not the solver (h=3-4 are excluded
        # here: there the 3-4 cell film genuinely does carry a discretisation error)
        assert abs(aged) > 3.0 * abs(sweep[h]["err_pct"]["lbm_me_vs_ref_normal"]) + 1.0