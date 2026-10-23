"""Adiabatic-wall conservation regression for the 2D D2Q5 thermal BC fix.

The pre-fix rule reflected the wrong population on the adiabatic walls: it
copied the post-stream arrival from the interior row (g_pre[4, 1, :])
instead of the wall node own outgoing distribution (g_pre[4, 0, :]) that
periodic ``temperature_stream`` had wrapped around to the opposite wall
row.  Under convection the two differ by the advective non-equilibrium, so
the old rule acted as a spurious O(1e-2)/step heat sink (benchmark
campaign evidence 2026-09: de Vahl Davis cavity Ra=1e4, N=64 developed
state, leak -2.2e-2/step, worsening x5.3 under refinement).  The fixed
rule unwraps the wrapped-around distribution and reflects it back, so the
two adiabatic walls exchange energy in exact antisymmetry.

This test drives the temperature pipeline alone (no flow solver) with a
deterministic velocity field so advection -- the leak driver -- is nonzero
from the first step, and asserts (a) exact budget identity closure and
(b) net-zero adiabat exchange.  On the old rule (b) fails by ~1e-2/step.
"""

import math

import torch

from tensorlbm.thermal import (
    apply_temperature_boundaries,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
    thermal_params,
)

RA, PR, TAU, TH, TC = 1e4, 0.71, 0.71, 1.0, 0.0
KICK = 0.05
STEPS = 20
TOL = 1e-12


def _adiabat_budget(nx=32):
    tau_T = thermal_params(nx, RA, PR, TAU)["tau_T"]
    y = torch.arange(nx, dtype=torch.float64).view(nx, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, nx)
    # deterministic cellular velocity field: advection nonzero everywhere,
    # wall rows included (worst case for the adiabat reflection rule)
    ux = KICK * torch.sin(2 * math.pi * y / nx) * torch.sin(math.pi * (x + 1) / nx)
    uy = -KICK * torch.cos(2 * math.pi * y / nx) * torch.cos(math.pi * (x + 1) / nx)
    T0 = TC + (TH - TC) * (x / (nx - 1)).expand(nx, nx).contiguous()
    g = temperature_equilibrium(T0, ux, uy)
    tb, ident = 0.0, 0.0
    for _ in range(STEPS):
        g = temperature_collision(g, tau_T, ux, uy)
        g = temperature_stream(g)
        g_ps = g
        g = apply_temperature_boundaries(g, TH, TC)
        dl = (g[:, :, 0].double() - g_ps[:, :, 0].double()).sum().item()
        dr = (g[:, :, -1].double() - g_ps[:, :, -1].double()).sum().item()
        db = (g[:, 0, 1:-1].double() - g_ps[:, 0, 1:-1].double()).sum().item()
        du = (g[:, -1, 1:-1].double() - g_ps[:, -1, 1:-1].double()).sum().item()
        tot = (g.double().sum() - g_ps.double().sum()).item()
        tb += db + du
        ident = max(ident, abs(tot - dl - dr - db - du))
    return tb / STEPS, ident / STEPS


def test_adiabat_exchange_net_zero():
    tb, _ = _adiabat_budget()
    assert abs(tb) < TOL, (
        f"net adiabatic-wall leak {tb:+.3e}/step -- the reflection rule "
        "uses the wrapped-around outgoing distribution (old rule: ~1e-2/step)"
    )


def test_budget_identity_closes():
    _, ident = _adiabat_budget()
    assert ident < TOL, f"budget decomposition leak {ident:.3e}/step"
