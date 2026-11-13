"""Invariants for the declarative example runtime (tensorlbm.example_harness).

These assert physics and contracts, not "the code ran":

* the runtime composes the library kernels -- with no body force one harness
  step must equal ``collide_bgk`` + ``stream`` from :mod:`tensorlbm.solver`
  exactly, so it cannot silently redefine the scheme;
* the force-driven channel reproduces the closed form ``u_max = g H^2/(8 nu)``
  and conserves mass (a truncated Guo source term loses ~2.5e-3 of the density,
  see the module docstring);
* the Taylor-Green energy decays at ``4 nu k^2`` (Laplacian factor 2), not
  ``2 nu k^2``;
* identical specs give bit-identical fields, a perturbed spec does not.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.example_harness import (
    CaseSpec,
    analytic_reference,
    available_cases,
    get_case,
    record_digest,
    run_case,
)
from tensorlbm.solver import collide_bgk, stream


def _spec(**kw):
    base = dict(kind="channel_poiseuille", ny=16, nx=4, tau=0.8, steps=6000, body_force=1e-5)
    base.update(kw)
    return CaseSpec(**base)


def test_spec_rejects_unphysical_inputs():
    with pytest.raises(ValueError):
        _spec(tau=0.5)  # nu <= 0
    with pytest.raises(ValueError):
        _spec(body_force=0.0)
    with pytest.raises(ValueError):
        CaseSpec(kind="taylor_green_2d", ny=16, nx=16, tau=0.8, steps=10, u0=0.5)
    with pytest.raises(ValueError):
        CaseSpec(kind="nope", ny=16, nx=16, tau=0.8, steps=10)
    with pytest.raises(KeyError):
        get_case("not_a_case")


def test_wall_height_matches_half_way_bounce_back():
    # walls at y = 1/2 and y = ny - 3/2 -> width ny - 2
    assert _spec(ny=32).wall_height == 30
    assert _spec(ny=16).wall_height == 14


def test_no_force_step_equals_library_kernels():
    """The runtime must not re-implement what the solver already provides."""
    spec = _spec(body_force=0.0, kind="taylor_green_2d", ny=16, nx=16, steps=1, u0=0.05)
    from tensorlbm.example_harness import _step, _inward_masks, _initial_state

    f0 = _initial_state(spec)
    got = _step(f0, spec, *_inward_masks(spec))
    want = stream(collide_bgk(f0, spec.tau))
    assert torch.equal(got, want), "harness step diverged from collide_bgk+stream"


def test_channel_poiseuille_hits_closed_form_and_conserves_mass():
    rec = run_case(_spec(ny=16), keep_history=False)
    ref = rec["analytic"]["u_max"]
    got = rec["kpis"]["u_max"]
    assert abs(got - ref) / ref < 0.03, f"u_max {got} vs {ref}"
    assert rec["kpis"]["profile_rel_l2"] < 0.03
    assert abs(rec["kpis"]["mass_drift"]) < 1e-10
    assert rec["kpis"]["rho_min"] > 0.999 and rec["kpis"]["rho_max"] < 1.001
    assert rec["kpis"]["finite"]


def test_channel_analytic_reference_formula():
    spec = _spec(ny=32, body_force=2e-5)
    a = analytic_reference(spec)
    assert a["H"] == 30
    assert a["nu"] == pytest.approx((0.8 - 0.5) / 3.0)
    assert a["u_max"] == pytest.approx(2e-5 * 30 * 30 / (8 * (0.8 - 0.5) / 3.0))


def test_taylor_green_decays_at_four_nu_k_squared():
    spec = get_case("taylor_green_2d", ny=32, nx=32, steps=400, sample_every=50)
    rec = run_case(spec)
    measured = rec["kpis"]["decay_rate_measured"]
    ref = rec["analytic"]["decay_rate"]
    assert ref == pytest.approx(4 * spec.nu * spec.k**2)
    assert abs(measured - ref) / ref < 0.03, f"decay {measured} vs {ref}"
    # energy must be monotonically decreasing for a free-decaying vortex
    energies = [h["kinetic_energy"] for h in rec["history"]]
    assert all(a >= b for a, b in zip(energies, energies[1:]))


def test_runtime_is_deterministic_and_digest_is_sensitive():
    spec = _spec(ny=12, steps=400)
    r1 = run_case(spec, keep_history=False)
    r2 = run_case(spec, keep_history=False)
    assert r1["kpis"]["field_digest"] == r2["kpis"]["field_digest"]
    assert record_digest(r1) == record_digest(r2)

    r3 = run_case(_spec(ny=12, steps=400, body_force=2.5e-5), keep_history=False)
    assert record_digest(r3) != record_digest(r1)
    assert r3["kpis"]["u_max"] > r1["kpis"]["u_max"]


def test_registry_override_does_not_leak():
    a = get_case("channel_poiseuille")
    b = get_case("channel_poiseuille", ny=64)
    assert a.ny == 48 and b.ny == 64
    assert a.digest() != b.digest()
    assert available_cases() == ["channel_poiseuille", "taylor_green_2d"]
    # the label is not part of the physical identity
    assert a.digest() == _spec(ny=48, nx=8, steps=12000, steady_window=200, sample_every=500).digest()


def test_record_is_json_serialisable_and_carries_provenance():
    import json

    rec = run_case(_spec(ny=10, steps=200), keep_history=False)
    blob = json.dumps(rec)
    assert len(blob) > 100
    prov = rec["provenance"]
    assert prov["spec_digest"] == _spec(ny=10, steps=200).digest()
    assert "torch" in prov and prov["torch"] == torch.__version__
    assert math.isfinite(rec["kpis"]["mlups"])