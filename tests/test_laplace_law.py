"""Three-radius Laplace gate for the SC-LBM interface tension (node P143W2-N01).

Two levels:

* a short self-contained relaxation (a few thousand steps, ``L=64``) that proves
  the pipeline runs and that the momentum-flux pressure jump is positive and
  consistent with the independent Korteweg anchor ``sigma_K``;
* the campaign artifact ``artifacts/sc_surface_tension/laplace_gate.json``, which
  carries the three-radius consistency numbers (this is the machine-checkable
  form of acceptance criterion (a) and it is *not* recomputed here: the test only
  asserts that the stored numbers satisfy the criterion).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "sc_surface_tension"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.sc_surface_tension import (  # noqa: E402
    build_stencil,
    droplet_rho,
    equivalent_radius,
    pressure_flux,
    pressure_kinetic,
    radius_field,
    sigma_korteweg,
    step_sc,
)

TOL_PCT = 3.0


import functools  # noqa: E402


@functools.lru_cache(maxsize=None)
def _relax_cached(R: float, L: int, steps: int) -> dict:
    return _relax(R, L, steps)


def _relax(R: float, L: int, steps: int) -> dict:
    st = build_stencil("d2q9")
    rho0 = droplet_rho(L, R)
    z = torch.zeros_like(rho0)
    f = equilibrium(rho0, z, z)
    mass0 = float(rho0.sum())
    for _ in range(steps):
        f = step_sc(f, stencil=st)
    rho = f.sum(dim=0)
    assert bool(torch.isfinite(rho).all())
    r = radius_field(L)
    Req = equivalent_radius(rho)
    inn, out = r < 0.5 * Req, r > 1.5 * Req
    p = pressure_flux(f)
    pk = pressure_kinetic(f)
    _, ux, uy = macroscopic(f)
    return {
        "R_eq": Req,
        "dp": float(p[inn].mean() - p[out].mean()),
        "dp_kin": float(pk[inn].mean() - pk[out].mean()),
        "mass_drift": abs(float(f.sum()) - mass0) / mass0,
        "u_max": float(torch.sqrt(ux ** 2 + uy ** 2).max()),
        "sigma_L": float(p[inn].mean() - p[out].mean()) * Req,
        "sigma_K": sigma_korteweg(rho, L, st)["sigma_K"],
    }


@pytest.mark.parametrize("R,L,steps", [(12.0, 64, 600)])
def test_short_relaxation_builds_a_pressure_jump(R, L, steps):
    res = _relax_cached(R, L, steps)
    assert res["dp"] > 0.0, res
    assert res["mass_drift"] < 1e-3, res
    # the raw kinetic readout sees the *ideal-gas* density difference, which is
    # two orders larger than the capillary jump -- it is blind to the physics
    assert abs(res["dp_kin"]) > 10.0 * res["dp"], res
    # sigma from Laplace vs sigma from the Korteweg integral: loose here because
    # this run is deliberately short (the gate uses the converged campaign runs)
    ratio = res["sigma_L"] / res["sigma_K"]
    assert 0.4 < ratio < 1.8, res


def test_pressure_jump_decreases_with_radius():
    a = _relax_cached(12.0, 64, 600)
    b = _relax_cached(16.0, 72, 600)
    assert a["dp"] > b["dp"], (a["dp"], b["dp"])
    # Laplace: Delta p * R is radius independent (2-D cylinder)
    assert abs(a["sigma_L"] - b["sigma_L"]) / b["sigma_L"] < 0.25, (a["sigma_L"], b["sigma_L"])


def _gate() -> dict | None:
    p = REPO / "artifacts" / "sc_surface_tension" / "laplace_gate.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_three_radius_gate_from_artifact():
    gate = _gate()
    if gate is None:
        pytest.skip("campaign artifact not present")
    d2 = gate["laplace_d2q9"]
    sh = gate["laplace_shell4"]
    assert d2["n_radii"] == 3, d2["n_radii"]
    assert d2["max_dev_pct"] < TOL_PCT, d2["max_dev_pct"]
    assert sh["n_radii"] == 3, sh["n_radii"]
    assert sh["max_dev_pct"] < TOL_PCT, sh["max_dev_pct"]
    # the fitted sigma must be positive and of the expected order
    assert 0.01 < d2["sigma_mean"] < 0.2
    # every radius converged
    for row in d2["per_radius"]:
        assert row["converged"], row
        assert row["mass_drift"] < 1e-3, row


def test_korteweg_anchor_reported_from_artifact():
    gate = _gate()
    if gate is None:
        pytest.skip("campaign artifact not present")
    for key in ("laplace_d2q9", "laplace_shell4"):
        rows = gate[key]["per_radius"]
        for row in rows:
            # the anchor is independent of the radius, so it must not drift by
            # more than the published tolerance *between* radii
            assert 0.5 < row["ratio_sigma_laplace_to_K"] < 1.5, row


def test_negative_control_from_artifact():
    gate = _gate()
    if gate is None:
        pytest.skip("campaign artifact not present")
    neg = gate["negative"]
    assert neg["pass_kappa3_zero"], neg
    assert neg["kappa3_zero_dp"] < 1e-3, neg
    # and the control still has phase separation (so it is not a trivial zero)
    rho_in, rho_out = neg["kappa3_zero_rho_separation"]
    assert rho_in > 1.0 and rho_out < 0.5, neg