"""Invariants of the Zou-He channel h-refinement study.

The records are produced by
``xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/run_ladder.sh``
(``refinement.py``).  These tests do not re-run the solver -- they assert the
*properties* a refinement study must have, so a regression that silently
degrades convergence (e.g. a broken reconstruction that still "runs") is caught
by the record set:

* every level closes the streamwise mass flux to round-off-ish level;
* the profile error decreases monotonically with refinement;
* the observed order of accuracy is at least first order and inside the band
  the closure can plausibly deliver;
* doubling the averaging window does not move the metrics (temporal sensitivity);
* the analytic reference stored in each record is recomputable from the
  record's own fitted gradient (record/report consistency).

They skip (not pass) when the artifact set is absent.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "artifacts", "xval_zouhe")
LADDER = ["H16_t12k", "H24_t12k", "H32_t14k"]
TEMPORAL_PAIRS = [("H16_t6k", "H16_t12k")]


def _load():
    recs = {}
    for path in glob.glob(os.path.join(DATA, "refine_*.json")):
        with open(path) as fh:
            recs[os.path.basename(path)[len("refine_"):-len(".json")]] = json.load(fh)
    return recs


@pytest.fixture(scope="module")
def recs():
    r = _load()
    missing = [t for t in LADDER if t not in r]
    if missing:
        pytest.skip(f"refinement records missing {missing}; run run_ladder.sh")
    return r


@pytest.fixture(scope="module")
def levels(recs):
    return [recs[t] for t in LADDER]


def test_every_level_closes_the_mass_flux(levels):
    for r in levels:
        assert r["mass_defect"] < 1e-4, (r["height"], r["mass_defect"])
        assert r["q_interior_spread"] < 1e-3, (r["height"], r["q_interior_spread"])


def test_profile_error_decreases_monotonically(levels):
    linf = [r["metrics_vs_measured_gradient"]["rel_linf"] for r in levels]
    shape = [r["metrics_vs_measured_gradient"]["shape_rel_l2"] for r in levels]
    for a, b in zip(linf, linf[1:]):
        assert b < a, linf
    for a, b in zip(shape, shape[1:]):
        assert b < a, shape


def test_observed_order_of_accuracy(levels):
    h = np.array([1.0 / r["height"] for r in levels])
    for key in ("rel_linf", "shape_rel_l2"):
        e = np.array([r["metrics_vs_measured_gradient"][key] for r in levels])
        # e ~ h**p, so the slope of log e vs log h is the order p
        p = np.polyfit(np.log(h), np.log(e), 1)[0]
        assert 1.0 < p < 3.0, (key, p)


def test_temporal_averaging_is_not_the_limiting_error(recs):
    for a, b in TEMPORAL_PAIRS:
        if a not in recs or b not in recs:
            pytest.skip(f"temporal pair {a}->{b} not recorded")
        ea = recs[a]["metrics_vs_measured_gradient"]["rel_linf"]
        eb = recs[b]["metrics_vs_measured_gradient"]["rel_linf"]
        assert abs(eb - ea) / ea < 0.2, (a, b, ea, eb)


def test_finest_level_inside_the_acceptance_bound(levels):
    m = levels[-1]["metrics_vs_measured_gradient"]
    assert m["rel_linf"] < 0.03, m            # project-wide 3 % target
    assert m["shape_rel_l2"] < 0.03, m


def test_stored_analytic_reference_is_recomputable(levels):
    """The record's own numbers must reproduce its own analytic profile."""
    for r in levels:
        yt = np.asarray(r["yt_over_H"]) * r["height"]
        h = float(r["height"])
        u_ex = r["g_measured"] / (2.0 * r["nu"]) * yt * (h - yt)
        u_rec = np.asarray(r["u_profile_exact"])
        assert np.allclose(u_ex, u_rec, rtol=1e-9, atol=0), r["height"]


def test_records_declare_a_fixed_aspect_ratio(levels):
    for r in levels:
        assert abs(r["aspect"] - 8.0) < 1e-9, (r["height"], r["aspect"])