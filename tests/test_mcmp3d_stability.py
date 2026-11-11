"""Contract tests for the 3D MCMP stability harness and the 3D Guo forcing fix.

Checks:
1. ``G_self=0`` / default pseudopotentials are bit-identical to the historical
   cross-coupling-only 3D force and collision (no silent behaviour change).
2. The D3Q19 Guo path injects exactly the applied force per step (regression
   test for the over-forcing defect: the pre-fix code injected
   ``(2 - 1/(2τ))·F``, i.e. 1.5x at τ=1).
3. The self-coupling sign convention (G>0 attractive, same as the
   single-component kernel) shows up as growth/decay of a density perturbation.
4. The stability harness produces JSON-serialisable records, classifies a
   diverging case as ``diverged``, and orders the boundary interval correctly.
"""

from __future__ import annotations

import json

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.mcmp3d_stability import (
    DEFAULT_CRITERIA,
    classify,
    init_mcmp_slab,
    run_case,
    stability_boundary,
)
from tensorlbm.multiphase import psi_linear
from tensorlbm.multiphase3d import collide_sc_two_component_3d, sc_two_component_force_3d
from tensorlbm.solver3d import stream3d_cat


def test_gself_and_psi_defaults_are_bit_identical():
    torch.manual_seed(3)
    r1 = torch.rand(6, 6, 6) + 0.5
    r2 = torch.rand(6, 6, 6) + 0.5
    base = sc_two_component_force_3d(r1, r2, -1.0)
    new = sc_two_component_force_3d(r1, r2, -1.0, G_self=0.0, psi1_fn=psi_linear, psi2_fn=psi_linear)
    for a, b in zip(base, new):
        assert torch.equal(a, b)

    f1 = torch.rand(19, 6, 6, 6) * 0.2
    f2 = torch.rand(19, 6, 6, 6) * 0.2
    o0 = collide_sc_two_component_3d(f1, f2, G_12=-1.0)
    o1 = collide_sc_two_component_3d(f1, f2, G_12=-1.0, G_self=(0.0, 0.0))
    assert torch.equal(o0[0], o1[0]) and torch.equal(o0[1], o1[1])


def test_guo_forcing_injects_exactly_the_applied_force():
    """u_x must equal g*t: the Guo branch may not double-count the force."""
    n, g, steps = 8, 1e-4, 200
    r = torch.ones(n, n, n)
    z = torch.zeros_like(r)
    for guo, tol in ((False, 1e-3), (True, 2e-3)):
        f1, f2 = equilibrium3d(r, z, z, z), equilibrium3d(r, z, z, z)
        for _ in range(steps):
            f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=0.0, gx=g, use_guo=guo)
            f1, f2 = stream3d_cat(f1), stream3d_cat(f2)
        u = float(macroscopic3d(f1)[1].mean())
        assert abs(u - g * steps) / (g * steps) < tol, (guo, u, g * steps)


def test_self_coupling_sign_convention():
    """G_self > 0 attracts (perturbation grows), G_self < 0 repels (decays)."""
    n, steps = 8, 60
    amp = 0.05
    prof = (1.0 + amp * torch.cos(2 * torch.pi * torch.arange(n) / n)).view(n, 1, 1)
    rho = prof.expand(n, n, n).contiguous()
    rho2 = torch.full_like(rho, 0.5)
    z = torch.zeros_like(rho)

    def growth(g_self: float) -> float:
        f1 = equilibrium3d(rho.clone(), z, z, z)
        f2 = equilibrium3d(rho2.clone(), z, z, z)
        a0 = None
        for s in range(steps):
            f1, f2 = collide_sc_two_component_3d(f1, f2, G_12=0.0, G_self=g_self)
            f1, f2 = stream3d_cat(f1), stream3d_cat(f2)
            if s == steps - 1:
                r = macroscopic3d(f1)[0]
                a0 = float((r - r.mean()).abs().max())
        return a0 / amp

    assert growth(0.5) > 1.0  # attraction amplifies the perturbation
    assert growth(-0.5) < 1.0  # repulsion damps it


def test_harness_record_is_json_serialisable_and_classified():
    rec = run_case(
        dict(name="unit-stable", G_12=-1.0, steps=200, nz=24, ny=8, nx=8,
             rho1_in=0.92, rho1_out=0.14, rho2_in=0.08, rho2_out=0.86)
    )
    json.dumps(rec)
    assert rec["steps_run"] == 200 and rec["diverged"] is False
    assert rec["classification"]["state"] in ("stable", "unstable")
    assert rec["bulk"]["delta_rho"] > 0.0
    assert abs(rec["bulk"]["rho_mix_dense"] - 1.0) < 0.1  # equal-density family

    blown = dict(rec, diverged=True, diverged_reason="velocity burst")
    assert classify(blown)["state"] == "diverged"


def test_stability_boundary_orders_by_magnitude():
    records = [
        dict(G_12=-0.5, classification={"state": "stable"}),
        dict(G_12=-2.0, classification={"state": "stable"}),
        dict(G_12=-2.5, classification={"state": "diverged"}),
        dict(G_12=-3.0, classification={"state": "diverged"}),
    ]
    b = stability_boundary(records)
    assert b["last_stable"] == -2.0 and b["first_unstable"] == -2.5
    assert b["boundary_estimate"] == -2.25 and b["boundary_kind"] == "interval"
    assert b["states"]["G_12=-2.5"] == "diverged"  # falls back to classify() when absent
    assert DEFAULT_CRITERIA["u_burst"] == 0.30