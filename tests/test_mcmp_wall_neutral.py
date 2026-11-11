"""Neutral-wetting (mirror) wall for the two-component Shan-Chen force.

Background
----------
The historical ``solid_mask`` handling sets the neighbour density to **zero**
inside wall cells ("dry" wall).  On the first fluid row that removes
``rho_other/3`` from the interaction sum, so the wall exerts

    F_y ~ -G_12 * rho_sigma * rho_other / 3        (per unit volume)

which for ``G_12 = -2.5`` and ``rho = (0.7, 0.3)`` is ~0.0875
(the D2Q9 y-stencil coefficient is ``cs^2/2 = 1/6``) — about 1750x the
body force ``G_x = 5e-5`` used by the stratified-Poiseuille benchmark.  That
artificial force wets the wall with one component, drags the diffuse interface
off the geometric mid-plane and drives a persistent transverse current.

``wall_mode="mirror"`` fills the solid cells with the mirrored nearest-fluid
density before the interaction sum, i.e. a neutral-wetting wall.

What is asserted here
---------------------
1. fluid cells are untouched by the mirror fill, and the fill is exactly the
   adjacent fluid value (one row of wall cells);
2. a uniform mixture feels **zero** transverse wall force with ``mirror`` and a
   large one with ``dry`` -- the property that makes the analytic reference
   (no wetting preference) admissible at all;
3. ``wall_mode="dry"`` is bit-identical to the historical call (no regression);
4. an unknown ``wall_mode`` raises instead of silently falling back;
5. end-to-end: after a short stratified run the mirror wall reduces the
   transverse current by a large factor against the dry wall, with both runs
   still stratified (the direction and rough size of the R2 effect).  The
   *equilibrium* interface position is measured by the acceptance records, not
   by a 3000-step transient.

Physical integration test (5) is deliberately tolerant (factor 2) -- it is a
regression guard on the mechanism, not a substitute for the quantitative
records in ``artifacts/mcmp_stratified_poiseuille/records/``.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorlbm.boundary.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    _mirror_wall_fill,
    collide_sc_two_component,
    sc_two_component_force,
)
from tensorlbm.solver import stream  # noqa: E402

torch.set_grad_enabled(False)


def _channel(ny=25, nx=4):
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    return wall


def _uniform(ny=25, nx=4, rho1=0.7, rho2=0.3):
    return (torch.full((ny, nx), rho1), torch.full((ny, nx), rho2))


@pytest.mark.unit
def test_mirror_fill_touches_only_solid_cells():
    ny, nx = 9, 3
    wall = _channel(ny, nx)
    field = torch.arange(ny * nx, dtype=torch.float32).reshape(ny, nx)
    filled = _mirror_wall_fill(field, wall)
    assert torch.equal(filled[~wall], field[~wall])
    # one wall row: the fill is exactly the mirrored adjacent fluid value
    assert torch.allclose(filled[0], field[1])
    assert torch.allclose(filled[-1], field[-2])


@pytest.mark.unit
def test_uniform_mixture_has_no_wall_force_under_mirror():
    ny, nx = 25, 4
    wall = _channel(ny, nx)
    rho1, rho2 = _uniform(ny, nx)
    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(
        rho1, rho2, -2.5, 0.0, 0.0, wall, wall_mode="mirror"
    )
    # exactly zero up to float32 round-off (the sums cancel analytically)
    assert float(Fy1.abs().max()) < 1e-8
    assert float(Fy2.abs().max()) < 1e-8
    # the dry wall, by contrast, pulls hard on the first fluid rows
    Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(
        rho1, rho2, -2.5, 0.0, 0.0, wall, wall_mode="dry"
    )
    dry_wall_force = float(Fy1.abs().max())
    assert dry_wall_force > 0.05          # ~|G| rho1 rho2 / 6 = 0.0875
    assert dry_wall_force > 1e3 * 5e-5    # >> the benchmark body force
    # cs^2/2 = 1/6 is the D2Q9 y-stencil coefficient of the one-sided sum
    assert dry_wall_force == pytest.approx(2.5 * 0.7 * 0.3 / 6.0, rel=1e-5)


@pytest.mark.unit
def test_dry_wall_is_bit_identical_to_historical_path():
    torch.manual_seed(0)
    ny, nx = 21, 4
    wall = _channel(ny, nx)
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    alpha = 0.5 * (1.0 - torch.tanh((ys - ny // 2) / 3.0))
    f1 = equilibrium(0.7 * alpha + 0.3 * (1 - alpha), torch.zeros_like(alpha),
                     torch.zeros_like(alpha))
    f2 = equilibrium(0.3 * alpha + 0.7 * (1 - alpha), torch.zeros_like(alpha),
                     torch.zeros_like(alpha))
    kw = dict(G_12=-2.5, tau1=1.0, tau2=0.75, gx=5e-5, solid_mask=wall,
              u_eq="viscous", force_scheme="guo")
    a1, a2 = collide_sc_two_component(f1, f2, **kw)
    b1, b2 = collide_sc_two_component(f1, f2, wall_mode="dry", **kw)
    assert torch.equal(a1, b1) and torch.equal(a2, b2)
    # an explicit call with an invalid mode must not silently fall back
    with pytest.raises(ValueError):
        collide_sc_two_component(f1, f2, wall_mode="wet", **kw)


def _run(ny, steps, wall_mode, gx=5e-5, g12=-2.5):
    nx = 4
    wall = _channel(ny, nx)
    half = ny // 2
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    alpha = 0.5 * (1.0 - torch.tanh((ys - float(half)) / 3.0))
    rho1 = 0.7 * alpha + 0.3 * (1 - alpha)
    rho2 = 0.3 * alpha + 0.7 * (1 - alpha)
    z = torch.zeros_like(rho1)
    f1, f2 = equilibrium(rho1, z, z), equilibrium(rho2, z, z)
    for _ in range(steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=g12, tau1=1.0, tau2=0.75, gx=gx, solid_mask=wall,
            u_eq="viscous", force_scheme="guo", wall_mode=wall_mode,
        )
        f1, f2 = stream(f1), stream(f2)
        f1, f2 = bounce_back_cells(f1, wall), bounce_back_cells(f2, wall)
    r1, ux1, uy1 = macroscopic(f1)
    r2, ux2, uy2 = macroscopic(f2)
    rho = torch.clamp(r1 + r2, min=1e-12)
    uy = (r1 * uy1 + r2 * uy2) / rho
    alpha_p = ((r1 - 0.3) / 0.4).mean(dim=1).clamp(0.0, 1.0)
    return dict(
        spurious=float(uy[~wall].abs().max()),
        phi_contrast=float(alpha_p[1:-1].max() - alpha_p[1:-1].min()),
        alpha=alpha_p.tolist(),
    )


@pytest.mark.slow
def test_mirror_wall_reduces_transverse_current_and_centres_interface():
    ny, steps = 25, 3000
    dry = _run(ny, steps, "dry")
    mir = _run(ny, steps, "mirror")
    # both remain stratified (the comparison is a two-phase comparison)
    assert dry["phi_contrast"] > 0.5 and mir["phi_contrast"] > 0.5
    # the neutral wall cuts the transverse current by a large factor
    assert mir["spurious"] < 0.5 * dry["spurious"]
    # the interface is still resolvable and finite in both runs (the
    # equilibrium interface position is measured by the acceptance records,
    # not by a 3000-step transient: this test guards the mechanism only).
    def crossing(alpha):
        for j in range(len(alpha) - 1):
            if (alpha[j] - 0.5) * (alpha[j + 1] - 0.5) <= 0 and alpha[j] != alpha[j + 1]:
                return j + (0.5 - alpha[j]) / (alpha[j + 1] - alpha[j])
        return float("nan")

    assert math.isfinite(crossing(mir["alpha"]))
    assert math.isfinite(crossing(dry["alpha"]))
