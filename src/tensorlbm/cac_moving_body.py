"""Prescribed-motion solid bodies inside a CAC two-phase simulation.

Purpose
-------
``tensorlbm.cac_lbm`` handles *static* solids only.  A water-entry benchmark
needs a body that crosses the air/water interface while the force on it is
measured.  This module adds exactly that, and nothing else:

* a rigid body with a **prescribed** velocity (no 6-DOF integration),
* **moving-wall bounce-back** (Ladd 1994) on the hydrodynamic distribution,
* **Ladd momentum-exchange** force on the body (normalised per unit volume),
* fresh/cleared node treatment when the mask changes,
* a wedge and a disc/circle geometry helper for the entry cases.

Physics ownership
-----------------
The collision, streaming and the static-wall bounce are **not** re-implemented
here: :meth:`CACSim.collide` / :meth:`CACSim.stream` /
:meth:`CACSim.apply_boundaries` are reused unchanged.  This module only
replaces the post-stream boundary step with the moving-wall variant and
accumulates the momentum-exchange force.

Conventions and limitations (read before trusting a number)
-----------------------------------------------------------
* The phase-field distribution ``f`` gets a **plain** bounce-back at the body
  (no advective correction): a solid surface is a zero-normal-flux boundary
  for ``phi``.  ``move_phase_field=True`` switches on the Ladd correction with
  ``phi`` in place of ``rho``; it is off by default because the CAC ``f`` is an
  advection-reaction distribution, not a mass distribution.
* The hydrodynamic distribution ``g`` gets the moving-wall correction
  ``g_q = g_q^post(opp) + 2 w_q rho (c_q . u_w) / cs^2`` with the **local**
  ``rho(phi)``; inside a solid the caller must keep ``phi`` filled by the
  fresh-node refill (see :func:`refill_new_fluid`).
* The body is a rigid *staircase* mask re-rasterised every step.  Sub-cell
  wall position is therefore lost; the wedge benchmark must be read as
  "staircase body at resolution dx", not as a sharp wedge.
* There is no contact-angle boundary condition: the fluid-solid contact line
  is neutral by construction.  See ``pending/phasefield_water_entry/README.md``
  blocker B7.

Reference
---------
Ladd, A. J. C. (1994). Numerical simulations of particulate suspensions via a
discretized Boltzmann equation. *J. Fluid Mech.* 271, 285-309.
"""

from __future__ import annotations

import math

import torch

from .cac_lbm import CACSim, f_equilibrium, g_equilibrium, rho_of_phi
from .d3q19 import C as C3
from .d3q19 import W as W3

CS2 = 1.0 / 3.0
OPP = None  # resolved lazily from d3q19 to avoid a stale import


def _opposite(device: torch.device) -> torch.Tensor:
    global OPP
    if OPP is None:
        from .d3q19 import OPPOSITE

        OPP = OPPOSITE
    return OPP.to(device)


# ---------------------------------------------------------------------------
# Geometry helpers (all masks are (nz, ny, nx) boolean)
# ---------------------------------------------------------------------------


def wall_box(shape: tuple[int, int, int], *, z_periodic: bool = True) -> torch.Tensor:
    """Closed tank: y and x faces solid, z faces solid only if not periodic."""
    w = torch.zeros(shape, dtype=torch.bool)
    w[:, 0, :] = w[:, -1, :] = True
    w[:, :, 0] = w[:, :, -1] = True
    if not z_periodic:
        w[0, :, :] = w[-1, :, :] = True
    return w


def wedge_mask(
    shape: tuple[int, int, int],
    *,
    apex_xy: tuple[float, float],
    deadrise_rad: float,
    half_length: float,
    thickness: float,
) -> torch.Tensor:
    """2-D symmetric wedge (apex pointing **down**) extruded over z.

    The wetted surface is the inverted V

        ``y_surf(x) = y_apex + |x - x_apex| * tan(deadrise)``

    and the solid is everything **above** it, ``y_surf(x) <= y <= y_apex +
    thickness``, restricted to ``|x - x_apex| <= half_length``.  ``apex_xy`` is
    therefore the lowest point of the body (the tip that enters the water
    first); ``thickness`` is the body height in cells.

    Note: the mask is staircased on the lattice, so a 2 deg wedge is a
    staircase whose local facet angles alternate between 0 deg and 90 deg; see
    ``pending/phasefield_water_entry/NOTES.md`` for why the 2 deg arm is read
    as "staircase wedge" and not as a sharp wedge.
    """
    nz, ny, nx = shape
    y = torch.arange(ny, dtype=torch.float64).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, -1)
    x_apex, y_apex = apex_xy
    surf = y_apex + (x - x_apex).abs() * math.tan(deadrise_rad)
    body2d = (y >= surf) & (y <= y_apex + thickness) & ((x - x_apex).abs() <= half_length)
    return body2d.expand(nz, ny, nx).contiguous()


def disc_mask(
    shape: tuple[int, int, int],
    *,
    centre_xy: tuple[float, float],
    radius: float,
) -> torch.Tensor:
    """Circular disc (cylinder cross-section) extruded over z."""
    nz, ny, nx = shape
    y = torch.arange(ny, dtype=torch.float64).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, -1)
    cx, cy = centre_xy
    body2d = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return body2d.expand(nz, ny, nx).contiguous()


def translated_disc(
    shape: tuple[int, int, int],
    *,
    centre_xy: tuple[float, float],
    radius: float,
    displacement: tuple[float, float],
) -> torch.Tensor:
    cx, cy = centre_xy
    dx, dy = displacement
    return disc_mask(shape, centre_xy=(cx + dx, cy + dy), radius=radius)


# ---------------------------------------------------------------------------
# Link bookkeeping (mirrors the validated helpers in multiphase_water_entry)
# ---------------------------------------------------------------------------


def _shifted(mask: torch.Tensor, sign: int) -> torch.Tensor:
    """``(19, nz, ny, nx)`` boolean: ``mask(x + sign*c_q)`` for every q."""
    c = C3.to(mask.device)
    out = [mask]
    for q in range(1, 19):
        out.append(
            torch.roll(
                mask,
                shifts=(-sign * int(c[q, 2]), -sign * int(c[q, 1]), -sign * int(c[q, 0])),
                dims=(0, 1, 2),
            )
        )
    return torch.stack(out)


def links_into(body: torch.Tensor) -> torch.Tensor:
    """``x + c_q`` lies inside ``body`` (downstream neighbour test)."""
    return _shifted(body, +1)


def links_from(body: torch.Tensor) -> torch.Tensor:
    """``x - c_q`` lies inside ``body`` (upstream / pull-source test)."""
    return _shifted(body, -1)


# ---------------------------------------------------------------------------
# Node refill (fresh solid, cleared solid)
# ---------------------------------------------------------------------------


def _neighbour_mean(field: torch.Tensor, fluid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``field`` over the fluid neighbours of every ``mask`` node."""
    src = torch.where(fluid, field, torch.zeros_like(field))
    nb = _shifted(src, +1).sum(dim=0)
    cnt = _shifted(fluid, +1).sum(dim=0).clamp(min=1.0)
    return torch.where(mask, nb / cnt, field)


def refill_new_fluid(sim: CACSim, changed: torch.Tensor) -> None:
    """Re-seed ``f`` and ``g`` where the mask went solid -> fluid."""
    if not bool(changed.any()):
        return
    phi = _neighbour_mean(sim.f.sum(dim=0), sim.fluid_mask, changed)
    zero = torch.zeros(sim.shape, dtype=sim.dtype, device=sim.device)
    feq = f_equilibrium(phi, zero, sim.wv, sim.cf)
    sim.f = torch.where(changed.unsqueeze(0), feq, sim.f)
    rho = rho_of_phi(phi, sim.rho_l, sim.rho_g)
    p = torch.full(sim.shape, sim.p0, dtype=sim.dtype, device=sim.device)
    geq = g_equilibrium(p, rho, zero, sim.wv, sim.cf, sim.w0)
    sim.g = torch.where(changed.unsqueeze(0), geq, sim.g)


def fill_new_solid(sim: CACSim, changed: torch.Tensor) -> None:
    """Give freshly covered nodes fluid-like content (ghosts for the stencils)."""
    if not bool(changed.any()):
        return
    phi = _neighbour_mean(sim.f.sum(dim=0), sim.fluid_mask, changed)
    zero = torch.zeros(sim.shape, dtype=sim.dtype, device=sim.device)
    feq = f_equilibrium(phi, zero, sim.wv, sim.cf)
    sim.f = torch.where(changed.unsqueeze(0), feq, sim.f)
    rho = rho_of_phi(phi, sim.rho_l, sim.rho_g)
    p = torch.full(sim.shape, sim.p0, dtype=sim.dtype, device=sim.device)
    geq = g_equilibrium(p, rho, zero, sim.wv, sim.cf, sim.w0)
    sim.g = torch.where(changed.unsqueeze(0), geq, sim.g)


# ---------------------------------------------------------------------------
# Moving-wall bounce-back + Ladd momentum exchange
# ---------------------------------------------------------------------------


def moving_wall_bounce(
    dist: torch.Tensor,
    dist_pre: torch.Tensor,
    *,
    body: torch.Tensor,
    fluid: torch.Tensor,
    u_wall: tuple[float, float, float],
    density: torch.Tensor,
    move: bool = True,
) -> torch.Tensor:
    """Ladd moving bounce-back on links whose pull source is inside ``body``.

    ``dist`` is post-stream, ``dist_pre`` post-collision (pre-stream) of the
    same distribution.  For a fluid node whose upstream neighbour lies in the
    body::

        dist_q = dist_pre[opp(q)] + 2 w_q density (c_q . u_wall) / cs^2

    With ``move=False`` this degenerates to the plain bounce-back swap.

    .. warning::
       This is the *fluid-side* (half-way) form.  It is **not** mass
       conserving on its own: the body nodes' own ghosts are never refreshed,
       so every step destroys the populations that stream into them.  The
       :class:`MovingBodyCAC` stepper therefore uses
       :func:`moving_wall_bounce_full`, which swaps inside the solid exactly
       like the validated ``bounce_back_cells_3d`` used for static walls.
       Kept for reference and for the unit tests that pin the formula.
    """
    if not bool(body.any()):
        return dist
    link = fluid.unsqueeze(0) & links_from(body)
    if not bool(link.any()):
        return dist
    opp = _opposite(dist.device)
    refl = dist_pre[opp]
    if move:
        c = C3.to(dist.device).to(dist.dtype)
        w = W3.to(dist.device).to(dist.dtype).view(19, 1, 1, 1)
        cuw = (
            c[:, 0].view(19, 1, 1, 1) * u_wall[0]
            + c[:, 1].view(19, 1, 1, 1) * u_wall[1]
            + c[:, 2].view(19, 1, 1, 1) * u_wall[2]
        )
        refl = refl + 2.0 * w * density.unsqueeze(0) * cuw / CS2
    return torch.where(link, refl, dist)


def moving_wall_bounce_full(
    dist: torch.Tensor,
    *,
    body: torch.Tensor,
    fluid: torch.Tensor,
    u_wall: tuple[float, float, float],
    density: torch.Tensor,
    move: bool = True,
) -> torch.Tensor:
    """Mass-conserving bounce-back at a *moving* body (full-way convention).

    Same convention as :func:`tensorlbm.boundary.boundaries3d.bounce_back_cells_3d`
    used for the static walls: the swap happens **inside the solid**, so a
    body node keeps the population it received and hands it back one step
    later.  The wall velocity enters as the standard Ladd term on the
    *outgoing* direction ``q`` of the swapped population::

        dist_q(x_b) = dist_opp(q)(x_b) + 2 w_q density (c_q . u_wall) / cs^2

    ``dist`` is post-stream.  Because nothing is destroyed, the phase-field
    integral ``sum(f)`` is conserved to machine precision (see
    ``tests/test_cac_moving_body.py::test_moving_body_conserves_phase_mass``).

    ``density`` is sampled at the **fluid** node the reflected population
    streams into (``x_b + c_q``), not at the body node itself: the ghost value
    inside the body is an extrapolation and, in air above the water line, can
    be O(1) where the fluid is O(10^-2).  Using the body value injects
    momentum ~100x too large at the contact line.
    """
    if not bool(body.any()):
        return dist
    opp = _opposite(dist.device)
    refl = dist[opp]
    surface = body.unsqueeze(0) & links_from(fluid)  # q with x_b - c_q in fluid
    if move and (u_wall[0] != 0.0 or u_wall[1] != 0.0 or u_wall[2] != 0.0):
        c = C3.to(dist.device).to(dist.dtype)
        w = W3.to(dist.device).to(dist.dtype).view(19, 1, 1, 1)
        cuw = (
            c[:, 0].view(19, 1, 1, 1) * u_wall[0]
            + c[:, 1].view(19, 1, 1, 1) * u_wall[1]
            + c[:, 2].view(19, 1, 1, 1) * u_wall[2]
        )
        dens_out = _shifted(density, +1)  # density at x_b + c_q
        corr = 2.0 * w * dens_out * cuw / CS2
        # Only boundary links carry the wall velocity.  Applying the term on
        # interior solid nodes as well makes each swap add momentum that no
        # fluid ever takes away, and the body interior pumps until it blows up.
        refl = refl + torch.where(surface, corr, torch.zeros_like(corr))
    return torch.where(body.unsqueeze(0), refl, dist)


def momentum_exchange(
    dist_pre: torch.Tensor,
    *,
    body: torch.Tensor,
    fluid: torch.Tensor,
    u_wall: tuple[float, float, float],
    density: torch.Tensor,
    move: bool = True,
) -> tuple[float, float, float]:
    """Ladd (1994) momentum-exchange force on the body from one distribution.

    Uses the *post-collision, pre-stream* populations and the links that run
    from a fluid node into the body::

        F = 2 sum_x sum_{q: x+c_q in body} c_q [ f_q - w_q rho (c_q . u_w) / cs^2 ]

    This is the textbook one-sided form; the stepper uses
    :func:`momentum_exchange_delta` instead (exact momentum bookkeeping,
    no convention ambiguity).  Kept as the reference/oracle used by the tests.
    """
    if not bool(body.any()):
        return 0.0, 0.0, 0.0
    link = fluid.unsqueeze(0) & links_into(body)
    if not bool(link.any()):
        return 0.0, 0.0, 0.0
    c = C3.to(dist_pre.device).to(dist_pre.dtype)
    contrib = dist_pre
    if move:
        w = W3.to(dist_pre.device).to(dist_pre.dtype).view(19, 1, 1, 1)
        cuw = (
            c[:, 0].view(19, 1, 1, 1) * u_wall[0]
            + c[:, 1].view(19, 1, 1, 1) * u_wall[1]
            + c[:, 2].view(19, 1, 1, 1) * u_wall[2]
        )
        contrib = dist_pre - w * density.unsqueeze(0) * cuw / CS2
    contrib = contrib * link.to(contrib.dtype)
    return tuple(2.0 * float((c[:, a].view(19, 1, 1, 1) * contrib).sum().item()) for a in range(3))


def momentum_exchange_delta(
    dist_before: torch.Tensor,
    dist_after: torch.Tensor,
    *,
    body: torch.Tensor,
    fluid: torch.Tensor,
) -> tuple[float, float, float]:
    """Force on the body from the momentum the boundary actually removed.

    ``dist_before`` is post-stream / pre-bounce, ``dist_after`` post-bounce.
    For every body node and every direction whose source ``x_b - c_q`` is
    fluid, the boundary replaced the incoming by the outgoing population, so
    the momentum given up to the body is::

        F = - sum_q c_q [ dist_after,q(x_b) - dist_before,q(x_b) ]

    The overall factor 2 is the Ladd factor for *full-way* bounce-back: the
    swap inside the solid reverses one population per step, but the
    reflected population only reaches the fluid on the next stream, so the
    momentum transfer of a link is spread over two steps and the steady
    force is twice the single-step bookkeeping.  It is fixed (not fitted) by
    the Archimedes gate: a static fully submerged body must feel
    ``rho_l g V`` -- ``probe/p2_mem_buoyancy.py``, ratio 1.00 at 2000 steps.
    """
    if not bool(body.any()) or not bool(fluid.any()):
        return 0.0, 0.0, 0.0
    link = body.unsqueeze(0) & links_from(fluid)
    if not bool(link.any()):
        return 0.0, 0.0, 0.0
    c = C3.to(dist_after.device).to(dist_after.dtype)
    diff = (dist_after - dist_before) * link.to(dist_after.dtype)
    return tuple(-2.0 * float((c[:, a].view(19, 1, 1, 1) * diff).sum().item()) for a in range(3))


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


class MovingBodyCAC:
    """Drive a :class:`CACSim` with a prescribed-motion solid body.

    Parameters
    ----------
    sim
        Configured ``CACSim``; its ``wall_mask`` is replaced by
        ``static_wall | body`` on every step.
    static_wall
        Fixed walls (tank), or ``None``.
    """

    def __init__(
        self,
        sim: CACSim,
        *,
        static_wall: torch.Tensor | None = None,
        move_phase_field: bool = True,
    ) -> None:
        self.sim = sim
        self.static_wall = static_wall
        self.move_phase_field = move_phase_field
        self.body: torch.Tensor | None = None
        self.force = (0.0, 0.0, 0.0)
        self.force_phase = (0.0, 0.0, 0.0)

    # -- geometry ----------------------------------------------------------

    def set_body(self, body: torch.Tensor) -> None:
        """Install a new body mask, refilling cleared and fresh nodes."""
        sim = self.sim
        old = self.body
        if self.static_wall is None:
            old_solid = old
            new_solid = body
        else:
            old_solid = self.static_wall if old is None else (self.static_wall | old)
            new_solid = self.static_wall | body
        if old is not None:
            cleared = old_solid & ~new_solid
            fresh = new_solid & ~old_solid
            sim.set_wall_mask(new_solid)
            # Cleared nodes become fluid -> must look like fluid.  Fresh solid
            # nodes need no reset: with the full-way swap they keep handing
            # back exactly what they received, and overwriting them with a
            # rest state injects mass/momentum into the fluid every step.
            refill_new_fluid(sim, cleared)
            del fresh
        else:
            sim.set_wall_mask(new_solid)
        self.body = body

    # -- one step ----------------------------------------------------------

    def step(self, body: torch.Tensor, velocity: tuple[float, float, float]) -> None:
        """Advance one step with ``body`` at ``body``-mask and wall velocity."""
        sim = self.sim
        self.set_body(body)
        body_mask = body if self.static_wall is None else (self.static_wall | body)
        fluid = ~body_mask
        density = rho_of_phi(sim.f.sum(dim=0), sim.rho_l, sim.rho_g)

        # Neumann ghosts for the *static* walls only; the body must keep the
        # populations the bounce-back swap put there.
        sim.refresh_solid_ghosts(self.static_wall)
        sim.collide()
        f_pre = sim.f
        g_pre = sim.g
        sim.stream()

        from .boundaries3d import bounce_back_cells_3d

        # static walls: the validated plain swap inside the solid
        if self.static_wall is not None and bool(self.static_wall.any()):
            sim.f = bounce_back_cells_3d(sim.f, self.static_wall)
            sim.g = bounce_back_cells_3d(sim.g, self.static_wall)

        # moving body: same full-way swap, plus the Ladd wall-velocity term
        if bool(body.any()):
            f_before = sim.f
            g_before = sim.g
            sim.f = moving_wall_bounce_full(
                f_before,
                body=body,
                fluid=fluid,
                u_wall=velocity,
                density=f_before.sum(dim=0),
                move=self.move_phase_field,
            )
            sim.g = moving_wall_bounce_full(
                g_before,
                body=body,
                fluid=fluid,
                u_wall=velocity,
                density=density,
                move=True,
            )
            # Force reader: Ladd's momentum-exchange sum on the *post-collision*
            # populations, with the equilibrium term subtracted so that a
            # hydrostatic field reads rho g V (Archimedes gate) and a moving
            # wall does not read a volume-proportional offset.  The
            # ``momentum_exchange_delta`` variant is kept for bookkeeping tests
            # only: with a wall velocity it counts both sides of every surface
            # cell and the result scales with the body volume, not its area.
            self.force = momentum_exchange(
                g_pre, body=body, fluid=fluid, u_wall=velocity, density=density, move=True
            )
            self.force_phase = momentum_exchange(
                f_pre,
                body=body,
                fluid=fluid,
                u_wall=velocity,
                density=f_pre.sum(dim=0),
                move=self.move_phase_field,
            )
        else:
            self.force = (0.0, 0.0, 0.0)
            self.force_phase = (0.0, 0.0, 0.0)
        sim.step_count += 1
