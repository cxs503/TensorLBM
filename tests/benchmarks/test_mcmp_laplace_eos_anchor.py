"""Regression locks for the independent sigma_EOS anchor (Route C, non-circular).

Round 1's acceptance (radius-independence of ``sigma_i = dp_i R_i``) is
*circular*: ``sigma_fit`` is a through-origin least-squares fit of the same
``dp_i`` data, so any common multiplicative error in the model stress passes
it unchanged.  Round 2 adds an anchor that does not come from the droplet
pressure jump: the EOS-profile (gradient-theory) integral on a flat static
interface,

    sigma_EOS = (1/n_int) * sum_y [ p0 - p_iso(rho1(y), rho2(y)) ],
    p_iso = cs^2 (rho1+rho2) + G12 cs^2 rho1 rho2.

These tests pin the anchor's arithmetic (bulk-pressure subtraction, per-
interface division, positivity of the deficit) so it cannot silently drift.
No solver is run in the loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "case_pool" / "mcmp_laplace"))

from run_case import eos_sigma_from_fields  # noqa: E402


def _flat_interface(n: int = 96, rho_l: float = 0.9, rho_g: float = 0.1, w: float = 2.0):
    """A relaxed-looking flat interface: two parallel tanh edges in a slab."""
    ys = torch.arange(n, dtype=torch.float32).view(n, 1)
    c = n / 2.0
    half = 0.25 * n
    s = 0.5 * (1.0 + torch.tanh((ys - (c - half)) / w)) * 0.5 * (1.0 - torch.tanh((ys - (c + half)) / w))
    s = torch.clamp(s, 0.0, 1.0).expand(n, n).clone()
    rho1 = rho_g + s * (rho_l - rho_g)
    rho2 = rho_l - s * (rho_l - rho_g)
    return rho1, rho2


def test_eos_sigma_positive_and_finite():
    rho1, rho2 = _flat_interface()
    r = eos_sigma_from_fields(rho1, rho2, G12=-2.5, n_int=2)
    assert r["sigma_eos"] > 0.0
    assert torch.isfinite(torch.tensor(r["sigma_eos"]))
    # bulk pressures are close (coexistence, no curvature)
    assert abs(r["bulk_dp"]) < 1e-2
    # the pressure deficit is dominated by positive rows
    assert r["max_deficit"] > 0.0


def test_eos_sigma_scales_with_coupling():
    """Stronger separation (more negative G12) => larger surface tension."""
    rho1, rho2 = _flat_interface()
    weak = eos_sigma_from_fields(rho1, rho2, G12=-2.5, n_int=2)["sigma_eos"]
    strong = eos_sigma_from_fields(rho1, rho2, G12=-3.0, n_int=2)["sigma_eos"]
    assert strong > weak > 0.0


def test_eos_sigma_per_interface_division():
    """n_int divides the total deficit: one interface halves the value."""
    rho1, rho2 = _flat_interface()
    two = eos_sigma_from_fields(rho1, rho2, G12=-2.5, n_int=2)["sigma_eos"]
    one = eos_sigma_from_fields(rho1, rho2, G12=-2.5, n_int=1)["sigma_eos"]
    assert one == pytest.approx(2.0 * two, rel=1e-12)


def test_eos_sigma_independent_of_droplet():
    """The anchor uses only the profile + bulk EOS: no dp or R enters."""
    rho1, rho2 = _flat_interface()
    a = eos_sigma_from_fields(rho1, rho2, G12=-2.5, n_int=2)
    b = eos_sigma_from_fields(rho1.clone(), rho2.clone(), G12=-2.5, n_int=2)
    assert a["sigma_eos"] == pytest.approx(b["sigma_eos"], rel=1e-12)
    # a uniform (no-interface) field has zero deficit => zero tension
    uni = torch.full((64, 64), 0.5, dtype=torch.float32)
    z = eos_sigma_from_fields(uni, uni, G12=-2.5, n_int=2)
    assert z["sigma_eos"] == pytest.approx(0.0, abs=1e-6)
