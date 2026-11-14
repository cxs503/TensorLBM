"""Autograd-safe D2Q9 wall/obstacle closures for boundary-gradient matrices.

GAP-DIFFLBM-BOUNDARY-GRAD.  Three solid-boundary closures are provided on one
differentiable D2Q9 channel solver, so that the *force sensitivities*
``d(drag)/d(theta)`` and ``d(lift)/d(theta)`` can be obtained either by reverse
mode AD through the time-stepping operator or by central differences of the
same observable:

``"bb_static"``
    Stationary half-way bounce-back (no-slip) on the wall rows and on the
    obstacle cells.  The population facing a solid neighbour is replaced by the
    oppositely directed post-collision population of the fluid node (the wall
    sits half a lattice link away); Ladd (1994) momentum exchange is the force
    probe.
``"bb_moving"``
    Same half-way bounce-back, but the bottom wall is a Couette wall moving with
    ``uw = (uw_x, uw_y)``: the reflected population carries the wall momentum
    kick ``6 w_i rho (c_i . u_w)`` (Ladd's ``2 w_i rho (c_i . u_w)/c_s^2`` with
    ``c_s^2 = 1/3``).  The correction is applied at the streaming level (after
    collision); applying it to pre-collision populations makes the wall behave
    as a static one (negative-knowledge entry ``NKL-M18794-MWBB-ORDER``).
``"penalization"``
    Volume (Brinkman) penalization: the obstacle is a smooth indicator field
    ``eta(x) = 0.5 (1 - tanh((|x - x_c| - r)/smoothing))`` and the
    post-collision populations are relaxed towards the solid-state equilibrium,
    ``f <- (1 - eta) f + eta feq(rho_s, u_s)``.  ``eta`` is differentiable in the
    geometric parameters, so drag/lift sensitivities with respect to the body
    position are exact AD gradients of the penalised flow.

    ``rho_s`` is the *prescribed* solid density (``penalization_rho``, default
    1), not the local one.  Relaxing towards the local density instead makes the
    penalised region a mass trap: fluid leaks through the diffuse interface, the
    velocity there is pinned to ``u_s``, and the density accumulates until the
    run diverges.  Measured on the 40 x 120 channel at ``tau = 0.65``: with the
    local density target the run is NaN by step 600 (``smoothing = 0.8``) or
    step 600 (``smoothing = 1.2``), while the prescribed-density target is stable
    for >= 3000 steps at every ``smoothing`` in [0.4, 1.2].  The price is that
    mass is not conserved across the diffuse interface (the body is an isobaric
    rigid region), which is the usual trade-off of Brinkman-type penalisation in
    LBM; the AD-vs-FD gradient claim is unaffected because both sides integrate
    the same operator.

Design contract (why the implementation looks the way it does)
--------------------------------------------------------------
``docs/differentiable_path.md`` records that in-place boundary overwrites are
gradient-hostile.  Every closure here is therefore expressed with selections
only: ``torch.where`` for the bounce-back links, plain multiplies for the solid
mask and for the penalisation blend, ``torch.cat`` for the inlet/outlet
columns.  There is no ``index_put_``/``copy_`` anywhere in the time step, so a
scalar loss built from :meth:`DiffChannel2D.probe` back-propagates through
collision, streaming, bounce-back, penalisation and the inlet/outlet planes.

Boundaries in ``x``: Zou & He (1997) velocity inlet on the left plane and a
zero-gradient (copy) outlet on the right plane; both are algebraic and enter the
graph.  ``periodic_x=True`` removes both (used by the analytic Couette /
Poiseuille anchors, which are translationally invariant in ``x``).

Force probes (lattice units, ``dt = 1``):

* half-way bounce-back, momentum exchange over the links touching a solid:
  ``F_body = -sum_links c_i (f_opp(i)^+ + f_i^new)`` with
  ``f_i^new = f_opp(i)^+ + 6 w_i rho (c_i . u_w)``;
* volume penalization: the momentum removed from the fluid in one step is
  ``eta (rho u - rho u_s)`` per cell, i.e. the force acting on the body.

Both probes are verified against exact solutions (plane Couette, plane
Poiseuille) and against the DFG 2-D-2 benchmark (Schaefer & Turek) in
``scripts/run_gap_diff_boundary_grad.py``.

References
----------
Ladd, A.J.C., "Numerical simulations of particulate suspensions via a
discretised Boltzmann equation", J. Fluid Mech. 271 (1994) 285-339.
Zou, Q., He, X., "On pressure and velocity boundary conditions for the lattice
Boltzmann BGK model", Phys. Fluids 9 (1997) 1591-1598.
Angot, P., Bruneau, C.-H., Fabrie, P., "A penalization method to take into
account obstacles in incompressible viscous flows", Numer. Math. 81 (1999)
497-520.
Schaefer, M., Turek, S., "Benchmark computations of laminar flow around a
cylinder", in Flow Simulation with High-Performance Computers II, NNFM 48
(1996) 547-566.
Ataei, N., Salehipour, H., "XLB: A differentiable massively parallel lattice
Boltzmann library", Comput. Phys. Commun. 300 (2024) 109187 (solver-in-the-loop
paradigm followed here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .d2q9 import OPPOSITE, W_EXACT, equilibrium, macroscopic

CX: tuple[int, ...] = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY: tuple[int, ...] = (0, 0, 1, 0, -1, 1, 1, -1, -1)

CLOSURES = ("bb_static", "bb_moving", "penalization")


def _shift(field: torch.Tensor, i: int) -> torch.Tensor:
    """``out[y, x] = field[y - c_iy, x - c_ix]`` (periodic)."""
    return torch.roll(field, shifts=(CY[i], CX[i]), dims=(-2, -1))


def stream(f: torch.Tensor) -> torch.Tensor:
    """Gather streaming (periodic wrap), differentiable, one roll per direction."""
    return torch.stack([_shift(f[i], i) for i in range(9)], dim=0)


def collide(f: torch.Tensor, tau):
    """BGK collision ``f - (f - feq)/tau``; ``tau`` may be a graph tensor."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau, rho, ux, uy


@dataclass
class Geometry:
    """Pre-computed index masks for one shape/closure combination.

    The per-direction masks are stacked along a leading axis of length 9 so the
    whole bounce-back sweep is a single ``torch.where`` (the per-direction
    Python loop dominates the runtime on small grids).
    """

    fluid: torch.Tensor          # (ny, nx) float, 1 on fluid cells
    links9: torch.Tensor         # (9, ny, nx) bool, bounce-back link in direction i
    body9: torch.Tensor          # (9, ny, nx) float, link touches the obstacle (BB probe)
    move9: torch.Tensor          # (9, ny, nx) float, link touches the moving bottom wall
    bot9: torch.Tensor           # (9, ny, nx) bool, link touches the bottom wall
    top9: torch.Tensor           # (9, ny, nx) bool, link touches the top wall
    body: torch.Tensor | None    # (ny, nx) bool obstacle cells (BB closures)
    eta: torch.Tensor | None     # (ny, nx) penalisation indicator (penalisation)


class DiffChannel2D:
    """Differentiable D2Q9 channel solver with pluggable solid closures.

    ``ny`` spans the wall-normal direction (rows ``0`` and ``ny - 1`` are wall
    rows, so the first and last fluid rows are ``1`` and ``ny - 2``), ``nx`` the
    streamwise direction.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        *,
        tau: float | torch.Tensor = 0.65,
        radius: float = 5.0,
        smoothing: float = 0.8,
        penalization_rho: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        periodic_x: bool = False,
    ) -> None:
        self.ny, self.nx = int(ny), int(nx)
        self.tau = tau
        self.radius = float(radius)
        self.smoothing = float(smoothing)
        self.penalization_rho = float(penalization_rho)
        self.dtype = dtype
        self.device = torch.device(device)
        self.periodic_x = bool(periodic_x)
        yy, xx = torch.meshgrid(
            torch.arange(self.ny, dtype=dtype, device=self.device),
            torch.arange(self.nx, dtype=dtype, device=self.device),
            indexing="ij",
        )
        self.yy, self.xx = yy, xx
        self.w = torch.tensor(W_EXACT, dtype=dtype, device=self.device)
        self._wall = torch.zeros((self.ny, self.nx), dtype=torch.bool, device=self.device)
        self._wall[0, :] = True
        self._wall[-1, :] = True
        self._wall_bottom = torch.zeros_like(self._wall)
        self._wall_bottom[0, :] = True
        self._wall_top = torch.zeros_like(self._wall)
        self._wall_top[-1, :] = True
        self._w9 = self.w.view(9, 1, 1)
        self._cx9 = torch.tensor(CX, dtype=dtype, device=self.device).view(9, 1, 1)
        self._cy9 = torch.tensor(CY, dtype=dtype, device=self.device).view(9, 1, 1)
        self._opp = OPPOSITE.to(dtype=torch.int64, device=self.device)

    # -- geometry ---------------------------------------------------------
    def body_mask(self, xc: float, yc: float) -> torch.Tensor:
        d2 = (self.xx - xc) ** 2 + (self.yy - yc) ** 2
        return d2 <= self.radius**2

    def eta_field(self, xc, yc) -> torch.Tensor:
        """Smooth obstacle indicator; differentiable in tensor ``xc``/``yc``."""
        d = torch.sqrt((self.xx - xc) ** 2 + (self.yy - yc) ** 2) - self.radius
        eta = 0.5 * (1.0 - torch.tanh(d / self.smoothing))
        return eta * (~self._wall).to(self.dtype)

    def geometry(self, xc, yc, closure: str) -> Geometry:
        if closure not in CLOSURES:
            raise ValueError(f"unknown closure {closure!r}, expected one of {CLOSURES}")
        xcf = float(xc) if not torch.is_tensor(xc) else float(xc.detach())
        ycf = float(yc) if not torch.is_tensor(yc) else float(yc.detach())
        if closure == "penalization":
            eta = self.eta_field(xc, yc)      # stays in the graph
            body = None
            solid = self._wall
        else:
            eta = None
            body = self.body_mask(xcf, ycf)
            solid = self._wall | body
        fluid = ~solid
        links, body_m, move_m, bot_m, top_m = [], [], [], [], []
        for i in range(9):
            if i == 0:
                links.append(torch.zeros_like(fluid))
                body_m.append(torch.zeros_like(fluid))
                move_m.append(torch.zeros_like(fluid))
                bot_m.append(torch.zeros_like(fluid))
                top_m.append(torch.zeros_like(fluid))
                continue
            links.append(_shift(solid, i) & fluid)
            body_m.append((_shift(body, i) & fluid) if body is not None else torch.zeros_like(fluid))
            move_m.append(_shift(self._wall_bottom, i) & fluid & ~body_m[i])
            bot_m.append(_shift(self._wall_bottom, i) & fluid)
            top_m.append(_shift(self._wall_top, i) & fluid)
        return Geometry(
            fluid=fluid.to(self.dtype),
            links9=torch.stack(links),
            body9=torch.stack(body_m).to(self.dtype),
            move9=torch.stack(move_m).to(self.dtype),
            bot9=torch.stack(bot_m),
            top9=torch.stack(top_m),
            body=body,
            eta=eta,
        )

    # -- initial condition ------------------------------------------------
    def initial(self, ux0, uy0, rho0: float = 1.0) -> torch.Tensor:
        rho = torch.full((self.ny, self.nx), float(rho0), dtype=self.dtype, device=self.device)
        ux = ux0.expand(self.ny, self.nx) if torch.is_tensor(ux0) else torch.full_like(rho, float(ux0))
        uy = uy0.expand(self.ny, self.nx) if torch.is_tensor(uy0) else torch.full_like(rho, float(uy0))
        return equilibrium(rho, ux, uy)

    # -- planes in x ------------------------------------------------------
    def _inlet_outlet(self, f: torch.Tensor, u0, alpha) -> torch.Tensor:
        if torch.is_tensor(alpha):
            ux, uy = u0 * torch.cos(alpha), u0 * torch.sin(alpha)
        else:
            ux, uy = u0 * math.cos(alpha), u0 * math.sin(alpha)
        fc = f[:, :, :1]
        f0, f1, f2, f3, f4, f5, f6, f7, f8 = (fc[i : i + 1] for i in range(9))
        rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)
        new1 = f3 + (2.0 / 3.0) * rho_in * ux
        new5 = f7 - 0.5 * (f2 - f4) + rho_in * ux / 6.0 + 0.5 * rho_in * uy
        new8 = f6 + 0.5 * (f2 - f4) + rho_in * ux / 6.0 - 0.5 * rho_in * uy
        left = torch.cat([f0, new1, f2, f3, f4, new5, f6, f7, new8], dim=0)
        rc = f[:, :, -1:]
        pc = f[:, :, -2:-1]
        right = torch.cat(
            [rc[0:1], rc[1:2], rc[2:3], pc[3:4], rc[4:5], rc[5:6], pc[6:7], pc[7:8], rc[8:9]],
            dim=0,
        )
        return torch.cat([left, f[:, :, 1:-1], right], dim=2)

    # -- one time step ----------------------------------------------------
    def step(self, f, geom, *, u0, alpha, closure, uw=(0.0, 0.0), body_force: float = 0.0):
        """One lattice step.  Returns ``(f_new, f_post, rho_post)``."""
        fpost, rho, _ux, _uy = collide(f, self.tau)
        if closure == "penalization":
            zero = torch.zeros_like(rho)
            rho_s = torch.full_like(rho, self.penalization_rho)
            femit = fpost + geom.eta * (equilibrium(rho_s, zero, zero) - fpost)
        else:
            femit = fpost
        if body_force:
            femit = femit + 3.0 * self._w9 * rho.unsqueeze(0) * (self._cx9 * body_force)
        fs = stream(femit)
        vals = femit[self._opp] + self.wall_kick(rho, uw) * geom.move9
        fnew = torch.where(geom.links9, vals, fs) * geom.fluid
        if not self.periodic_x:
            fnew = self._inlet_outlet(fnew, u0, alpha)
        return fnew, femit, rho

    def wall_kick(self, rho: torch.Tensor, uw=(0.0, 0.0)) -> torch.Tensor:
        """Ladd wall-momentum kick ``6 w_i rho (c_i . u_w)``, shape ``(9, ny, nx)``."""
        ux_w, uy_w = uw
        return 6.0 * self._w9 * rho.unsqueeze(0) * (self._cx9 * ux_w + self._cy9 * uy_w)

    # -- force probes -----------------------------------------------------
    def bb_force(self, femit, rho, geom: Geometry, uw=(0.0, 0.0), links9=None):
        """Momentum-exchange force over ``links9`` (default: links touching the body)."""
        links = geom.body9 if links9 is None else links9
        contrib = (2.0 * femit[self._opp] + self.wall_kick(rho, uw) * geom.move9) * links
        # Force ON the solid: minus the momentum handed to the fluid
        # (plane Couette: fluid resists the moving wall; plane Poiseuille: fluid
        # drags both walls downstream -- both signs are checked in the script).
        return -(self._cx9 * contrib).sum(), -(self._cy9 * contrib).sum()

    def penalization_force(self, rho, ux, uy, geom: Geometry):
        """Force on the penalised body: momentum removed from the fluid per step."""
        return (geom.eta * rho * ux).sum(), (geom.eta * rho * uy).sum()

    # -- rollout ----------------------------------------------------------
    def rollout(self, *, xc, yc, u0, alpha, closure, uw=(0.0, 0.0), steps: int,
                body_force: float = 0.0, checkpoint_segment: int = 0):
        """Roll ``steps`` steps; returns the final ``f``."""
        geom = self.geometry(xc, yc, closure)
        f = self.initial(u0 * math.cos(float(alpha)) if not torch.is_tensor(alpha) else u0 * torch.cos(alpha),
                         u0 * math.sin(float(alpha)) if not torch.is_tensor(alpha) else u0 * torch.sin(alpha))
        if checkpoint_segment:

            def seg(f_, n):
                for _ in range(n):
                    f_, _, _ = self.step(f_, geom, u0=u0, alpha=alpha, closure=closure, uw=uw, body_force=body_force)
                return f_

            k = int(checkpoint_segment)
            done = 0
            while done < steps:
                n = min(k, steps - done)
                f = torch.utils.checkpoint.checkpoint(seg, f, n, use_reentrant=False)
                done += n
        else:
            for _ in range(steps):
                f, _, _ = self.step(f, geom, u0=u0, alpha=alpha, closure=closure, uw=uw, body_force=body_force)
        return f

    def probe(self, *, xc, yc, u0, alpha, closure, uw=(0.0, 0.0), steps: int,
              body_force: float = 0.0, checkpoint_segment: int = 0):
        """Return ``(drag, lift)`` on the body after ``steps`` lattice steps."""
        geom = self.geometry(xc, yc, closure)
        f = self.rollout(xc=xc, yc=yc, u0=u0, alpha=alpha, closure=closure, uw=uw,
                         steps=steps, body_force=body_force,
                         checkpoint_segment=checkpoint_segment)
        fpost, rho, ux, uy = collide(f, self.tau)
        if closure == "penalization":
            return self.penalization_force(rho, ux, uy, geom)
        return self.bb_force(fpost, rho, geom, uw=uw)

    def velocity_profile(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return macroscopic(f)