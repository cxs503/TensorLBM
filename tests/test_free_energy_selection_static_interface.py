"""Guards for the round-5 free-energy selection study (anchor XVAL-LBM-R4-...).

Two groups:

* **behaviour** (always run, ~15 s): the quiescent flat interface built by
  ``benchmarks/pending/free_energy_selection/static_interface.py`` is periodic-consistent, stays
  finite, reproduces the analytic tanh width / surface tension inside the owner 15 %
  band, does not drift, and its residual velocity decays (no steady parasitic current);
* **selection record** (skipped when the run artifacts are absent): the committed
  decision record really says what the round-5 document claims -- free-energy passes
  D1-D4, the pseudopotential arm fails criterion ③.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from benchmarks.pending.free_energy_selection.static_interface import (  # noqa: E402
    TANH_10_90,
    _params,
    run_static_interface,
)
from tensorlbm.phasefield.free_energy_pressure import tanh_profile  # noqa: E402

SELECTION = os.path.join(_ROOT, "artifacts", "free_energy_selection", "selection.json")


def test_band_seed_is_periodic_consistent():
    """The seeded profile must be continuous across the periodic wrap.

    A single monotone ``tanh`` across a periodic box would put an infinitely sharp
    seam at the wrap-around and diverge the run within a few hundred steps; this test
    pins the band construction that avoids it.
    """
    p = _params("matched", 2.0, 0.05)
    ny, nx, band = 64, 4, 32
    ys = torch.arange(ny, dtype=torch.float64).view(ny, 1).expand(ny, nx)
    signed = (band / 2.0) - (ys - ny / 2.0).abs()
    phi0 = tanh_profile(signed, 0.0, p.xi)
    seam = float((phi0[0] - phi0[-1]).abs().max())
    assert seam < 1e-3, f"periodic seam jump {seam:.3e} in the interface seed"
    assert float(phi0.max()) > 0.99 and float(phi0.min()) < -0.99


@pytest.fixture(scope="module")
def short_run():
    return run_static_interface(arm="matched", ny=64, nx=4, band=32, steps=600,
                                sample_every=100)


def test_static_interface_remains_finite(short_run):
    s = short_run["params"]
    assert s["finite"], "free-energy flat interface diverged"
    assert abs(s["drho"]) < 1e-4
    assert abs(s["dphi_per_cell"]) < 1e-6


def test_width_and_tension_inside_owner_band(short_run):
    s = short_run["params"]
    assert abs(s["w_rel_err"]) < 0.15, f"w_10_90 error {s['w_rel_err']:+.3%}"
    assert abs(s["sigma_rel_err"]) < 0.15, f"sigma error {s['sigma_rel_err']:+.3%}"
    # the matched arm must sit on the Shan-Chen coexistence width
    assert abs(s["w_10_90_measured"] - 4.645) / 4.645 < 0.15


def test_residual_decays_instead_of_plateauing(short_run):
    """No steady parasitic current: the band residual keeps falling with time."""
    sf = [h for h in short_run["series"] if h["S_ifc"] == h["S_ifc"]]
    assert len(sf) >= 3
    assert sf[-1]["S_ifc"] < 0.1 * sf[0]["S_ifc"], (
        f"residual did not decay: {sf[0]['S_ifc']:.2e} -> {sf[-1]['S_ifc']:.2e}")


def test_interface_does_not_drift(short_run):
    assert short_run["params"]["y_drift"] < 0.05


@pytest.mark.skipif(not os.path.exists(SELECTION),
                    reason="selection artifacts not present in this checkout")
def test_selection_record_rules():
    rec = json.load(open(SELECTION))
    assert rec["anchor"] == "XVAL-LBM-R4-COEXISTENCE-CALIBRATION-AND-FREE-ENER"
    for key, row in rec["free_energy"].items():
        assert row["D1_w_ok"] and row["D2_sigma_ok"] and row["D4_comparable"], key
        assert row["ratio_r4"] < rec["rules"]["gate"], key
    # honest head-to-head: the pseudopotential arm must still be recorded as failing
    for name, arm in rec["pseudopotential"].items():
        assert arm["ratio"] > rec["rules"]["gate"], name
        assert arm["pass_gate"] is False
    assert rec["selection"] == "free-energy"
    assert rec["improvement_decades"] > 2.0
    assert math.isfinite(rec["improvement_by_run"][rec["primary_run"]])