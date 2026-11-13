"""Regression pins for the literature-anchored DEM validation (XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY).

The OSS cache holds no DEM twin (see ``xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/survey_oss.md``),
so the DEM leg is pinned against closed-form references instead:

* restitution and contact duration of the Kelvin-Voigt contact must stay inside the
  owner precision standard (15 %) at a resolved time step -- this is the invariant
  that a future change to ``LinearSpringDashpot`` or to the integrator must not break;
* the tighter 2 % bound is recorded as an expected-failure, because the ``max(0, .)``
  no-tension clamp in ``LinearSpringDashpot`` biases the restitution by ~9 %
  (``REPORT.md`` section 6, finding 1).  The control run with the clamp removed
  passes 2 % at the same time step, so the xfail flips to xpass as soon as the
  clamp is fixed -- that is the intended signal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_DRIVER = _REPO / "xval" / "XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY" / "run_xval.py"

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location("xval_dem_driver", _DRIVER)
assert _spec is not None and _spec.loader is not None
xval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xval)


@pytest.fixture(scope="module")
def clipped():
    return xval.contact_arm(80)


@pytest.fixture(scope="module")
def unclipped():
    return xval.contact_arm(80, clip=False)


def test_contact_duration_matches_closed_form(clipped):
    """t_c = pi / (omega_n sqrt(1-zeta^2)) must be reproduced to <5 %."""
    assert abs(clipped["t_c_rel_err"]) < 0.05, clipped["t_c_rel_err"]


def test_peak_overlap_matches_closed_form(clipped):
    """The damped-oscillator peak x_max must be reproduced to <2 %."""
    assert abs(clipped["delta_max_rel_err"]) < 0.02, clipped["delta_max_rel_err"]


def test_restitution_within_owner_standard(clipped):
    """e = exp(-pi zeta / sqrt(1-zeta^2)) must be reproduced to <15 %."""
    assert abs(clipped["e_rel_err"]) < 0.15, clipped["e_rel_err"]


def test_restitution_within_2pct_without_no_tension_clamp(unclipped):
    """Control: remove the max(0, .) clamp and the 2 % bound is met.

    This isolates the clamp (not the integrator) as the source of the bias.
    """
    assert abs(unclipped["e_rel_err"]) < 0.02, unclipped["e_rel_err"]


@pytest.mark.xfail(strict=False, reason="no-tension clamp inflates e by ~9 % (REPORT.md finding 1)")
def test_restitution_within_2pct_with_clamp(clipped):
    assert abs(clipped["e_rel_err"]) < 0.02, clipped["e_rel_err"]


def test_settling_matches_literature_clift_gauvin():
    """Schiller-Naumann settling must sit on the Clift-Gauvin terminal velocity."""
    row = xval.settling_arm(2.0e-4, "schiller_naumann", 25)
    assert abs(row["rel_err_vs_reference"]) < 0.01, row
    assert row["converged_tail_drift_rel"] < 1e-5, row


def test_momentum_exchange_ledger_closes():
    """Particle impulse and the independently recomputed water reaction cancel."""
    res = xval.ledger_arm()
    assert res["relative_residual"] < 1.0e-10, res
    assert res["n_events"] == 50