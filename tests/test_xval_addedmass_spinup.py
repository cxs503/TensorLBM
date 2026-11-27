"""Gate tests for the added-mass spin-up criterion and the analytic references.

These are the invariants behind XVAL-LBM-ADDEDMASS-KC-RESPONSE-ENVELOPE:

* the concentric-container closed form equals an independent radial-FD +
  energy-integral solve of the same BVP (algebra check, not a tautology);
* the closed form reduces to the unbounded value 1 in the wide-container limit;
* the shipped spin-up of ``oscillatory_cylinder_inertia`` (1.5 periods) is
  provably *not* enough at a declared point: the first-harmonic residual is
  large and the per-period phase drifts, while the same point spun up to the
  declared criterion ``N_spin = beta*(box/2pi)^2`` has a residual < 0.05 and a
  phase drift < 1 deg/period.  If someone later "fixes" the module by changing
  the numbers without converging the physics, this test still fails.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.oscillatory_cylinder_inertia import (
    ArmCase,
    harmonic_decompose,
    run_arm,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "development" / "verification"
                       / "addedmass_kc_response_envelope"))

import reference as ref  # noqa: E402


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def test_annulus_closed_form_matches_independent_numeric_solve():
    for R, Ro in ((6.0, 24.0), (8.0, 64.0)):
        num = ref.c_a_annulus_numeric(R, Ro, n=8000)["c_a_numeric"]
        clo = ref.c_a_annulus_closed(R, Ro)
        assert num == pytest.approx(clo, rel=0.01), (R, Ro, num, clo)


def test_annulus_reduces_to_unbounded_and_is_monotone():
    assert ref.c_a_annulus_closed(1.0, 1e7) == pytest.approx(1.0, abs=1e-6)
    vals = [ref.c_a_annulus_closed(8.0, ro) for ro in (16.0, 32.0, 64.0, 128.0, 256.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert ref.c_a_unbounded() == 1.0


def test_spinup_criterion_formula():
    # documented criterion: N_spin = beta * (box/(2*pi))^2 periods
    assert ref.spinup_periods(50.0, 4.0) == pytest.approx(50.0 * (4.0 / (2 * math.pi)) ** 2)
    assert ref.spinup_periods(100.0, 12.0) > 300.0  # the shipped defaults are infeasible


# --------------------------------------------------------------------------- #
# the arm itself (cheap declared point: D=8, box=4, beta=50 -> N_spin = 20)
# --------------------------------------------------------------------------- #
def _arm(transient: float, measure: float = 3.0):
    case = ArmCase("gate", D=8, kc=0.2, beta=50.0, u0=0.02, box=4.0,
                   transient=transient, measure=measure)
    res, trace = run_arm(case, threads=2)
    T = res["numerics"]["T_steps"]
    phis = []
    for j in range(int(measure)):
        a = res["numerics"]["measure_start_step"] + j * T
        d = harmonic_decompose(trace["t"], trace["F_mem"], trace["omega"], t0=a, t1=a + T, nh=1)
        phis.append(math.degrees(math.atan2(d["sin"], d["cos"])))
    drift = float(np.polyfit(np.arange(len(phis)), phis, 1)[0])
    return res, drift


def test_shipped_spinup_is_not_converged_and_declared_criterion_is():
    n_spin = ref.spinup_periods(50.0, 4.0)          # 20.3 periods
    shipped, drift_ship = _arm(1.5)
    converged, drift_conv = _arm(round(1.5 * n_spin, 1))
    assert shipped["harmonic"]["E1_mem"]["resid_rel"] > 0.10, "shipped spin-up unexpectedly clean"
    assert converged["harmonic"]["E1_mem"]["resid_rel"] < 0.05, (
        f"residual {converged['harmonic']['E1_mem']['resid_rel']:.3f} did not collapse")
    assert abs(drift_conv) < 1.0, drift_conv
    assert abs(drift_ship) > 3.0 * max(abs(drift_conv), 0.05), (drift_ship, drift_conv)
    # a converged harmonic fit is phase-stable: C_I must not move period to period
    k = converged["kpi"]["E1_mem_area"]
    assert math.isfinite(k["C_I"]) and k["C_I"] > 0.0
    assert converged["err_inphase_pct_area"] == pytest.approx(
        100.0 * (k["C_I"] - converged["reference"]["potential_same_mask_area"])
        / converged["reference"]["potential_same_mask_area"], rel=1e-9)