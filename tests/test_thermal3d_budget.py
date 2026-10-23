"""Adiabatic-wall conservation regression for the 3D D3Q7 thermal BC fix.

The pre-fix rule copied the whole post-stream distribution of the adjacent
interior slab onto the four adiabatic walls (y = 0/ny-1, z = 0/nz-1) of the
periodic ``stream_thermal_3d``: a zero-gradient condition referencing the
interior arrivals instead of the wall node's own outgoing distributions
that the periodic stream had wrapped around to the opposite wall.  Under
convection (and even in pure diffusion with wall-normal temperature
structure) the two differ, so the old rule acted as a spurious heat
source/sink (2026-09 evidence, 32^3 fp64 budget: ~1e-2/step in modulated
pure diffusion, -7.8e-3/step in developed Ra=1e3 convection, -0.46/step at
Ra=1e4, worsening under refinement; same disease family as the 2D D2Q5 fix,
PR #300).  The fixed rule unwraps the wrapped-around distribution and
reflects it back, so each adiabatic wall pair exchanges energy in exact
antisymmetry.

This test drives the temperature pipeline alone (no flow solver) with a
deterministic velocity field so advection -- the leak driver -- is nonzero
from the first step, and asserts (a) exact budget identity closure and
(b) net-zero adiabat exchange.  On the old rule (b) fails by ~0.5/step.
"""

import math

import torch

from tensorlbm import thermal3d
from tensorlbm.thermal3d import (
    collide_thermal_bgk_3d,
    equilibrium_thermal_3d,
    macroscopic_thermal_3d,
    stream_thermal_3d,
)

KICK = 0.05
STEPS = 20
TOL = 1e-12
TAU_T = 0.8  # thermal3d driver value
WALLS = ("x0", "xn", "y0", "yn", "z0", "zn")

# the fixed module exports the public alias; the pre-fix module only has
# the private implementation (getattr fallback keeps this test runnable
# against both generations -- it must FAIL on the old rule, not error)
apply_bc = getattr(
    thermal3d, "apply_temperature_boundaries_3d", thermal3d._apply_temperature_boundaries_3d
)


def _wall_deltas(g: torch.Tensor, g_ps: torch.Tensor) -> dict[str, float]:
    return {
        "x0": (g[:, :, :, 0] - g_ps[:, :, :, 0]).sum().item(),
        "xn": (g[:, :, :, -1] - g_ps[:, :, :, -1]).sum().item(),
        "y0": (g[:, :, 0, 1:-1] - g_ps[:, :, 0, 1:-1]).sum().item(),
        "yn": (g[:, :, -1, 1:-1] - g_ps[:, :, -1, 1:-1]).sum().item(),
        "z0": (g[:, 0, 1:-1, 1:-1] - g_ps[:, 0, 1:-1, 1:-1]).sum().item(),
        "zn": (g[:, -1, 1:-1, 1:-1] - g_ps[:, -1, 1:-1, 1:-1]).sum().item(),
    }


def _adiabat_budget(n=16):
    nz = ny = nx = n
    z = torch.arange(nz, dtype=torch.float64).view(nz, 1, 1)
    y = torch.arange(ny, dtype=torch.float64).view(1, ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, nx)
    # deterministic cellular velocity field: advection nonzero everywhere,
    # wall slabs included (worst case for the adiabat reflection rule)
    ux = (
        (KICK * torch.sin(2 * math.pi * y / ny) * torch.sin(math.pi * (x + 1) / nx))
        .expand(nz, ny, nx)
        .contiguous()
    )
    uy = (
        (-KICK * torch.cos(2 * math.pi * y / ny) * torch.cos(math.pi * (x + 1) / nx))
        .expand(nz, ny, nx)
        .contiguous()
    )
    uz = (
        (KICK * torch.sin(2 * math.pi * z / nz) * torch.cos(math.pi * (x + 1) / nx))
        .expand(nz, ny, nx)
        .contiguous()
    )
    T0 = (x / (nx - 1)).expand(nz, ny, nx).contiguous()
    g = equilibrium_thermal_3d(T0, ux, uy, uz)
    tb, ident = 0.0, 0.0
    for _ in range(STEPS):
        g = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), ux, uy, uz, tau_T=TAU_T)
        g = stream_thermal_3d(g)
        g_ps = g
        g = apply_bc(g, 1.0, 0.0)
        d = _wall_deltas(g, g_ps)
        tot = (g.sum() - g_ps.sum()).item()
        tb += d["y0"] + d["yn"] + d["z0"] + d["zn"]
        ident = max(ident, abs(tot - sum(d[k] for k in WALLS)))
    return tb / STEPS, ident / STEPS


def test_3d_adiabat_exchange_net_zero():
    tb, _ = _adiabat_budget()
    assert abs(tb) < TOL, (
        f"net adiabatic-wall leak {tb:+.3e}/step -- the reflection rule "
        "must use the wrapped-around outgoing distribution (old rule: "
        "~5e-1/step at 16^3)"
    )


def test_3d_budget_identity_closes():
    _, ident = _adiabat_budget()
    assert ident < TOL, f"budget decomposition leak {ident:.3e}/step"


def test_3d_uniform_field_is_fixed_point():
    nz = ny = nx = 10
    T0 = torch.full((nz, ny, nx), 1.0, dtype=torch.float64)
    zeros = torch.zeros_like(T0)
    g = equilibrium_thermal_3d(T0, zeros, zeros, zeros)
    g = apply_bc(g, 1.0, 1.0)
    for _ in range(5):
        g = collide_thermal_bgk_3d(g, T0.clone(), zeros, zeros, zeros, tau_T=TAU_T)
        g = stream_thermal_3d(g)
        g = apply_bc(g, 1.0, 1.0)
    assert torch.equal(g, equilibrium_thermal_3d(T0, zeros, zeros, zeros)), (
        "uniform pure-diffusion state must stay bit-identical"
    )
